#--- 0. Load Libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
import random
import argparse
from pathlib import Path
import matplotlib.pyplot as plt # Keep for potential future plotting
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, mean_squared_error # Keep MSE
from joblib import Parallel, delayed
from functools import partial # Keep for joblib
from tqdm import tqdm
import math # For positional encoding
import sys
import multiprocessing # Added for dynamic worker count
import signal # Added for graceful exit
import optuna # Added for HPT
import traceback # For detailed error printing
import csv # Added for HPT logging

# Fix imports - add the parent directory to sys.path to find 'models' and 'config'
# Assuming 'models.py' and 'config.py' are in the same directory as this script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Import Model Definitions with absolute import relative to script location
try:
    from models import PositionalEmbedding, TransformerEncoderBlock, TransformerForecastingModel
except ImportError:
    print(f"ERROR: Could not import models from {SCRIPT_DIR / 'models.py'}. Make sure the file exists.")
    sys.exit(1)

# --- Import Config ---
try:
    import config
    # Define HPT result file names relative to the study output dir
    HPT_RESULTS_CSV = "hpt_results.csv"
    BEST_HPARAMS_PKL = "best_hparams.pkl"
except ImportError:
    print(f"ERROR: config.py not found in {SCRIPT_DIR}. Make sure it's in the same directory as the script.")
    sys.exit(1)

# --- Global flag for graceful exit ---
stop_training_flag = False

# --- Signal Handler ---
def signal_handler(sig, frame):
    """Sets the stop_training_flag when SIGINT (Ctrl+C) is received."""
    global stop_training_flag
    if not stop_training_flag: # Prevent multiple messages if Ctrl+C is pressed again
        print("\nCtrl+C detected! Attempting graceful stop after current epoch/operation...")
        print("Press Ctrl+C again to force exit (may corrupt state).")
        stop_training_flag = True
    else:
        print("Second Ctrl+C detected. Exiting forcefully.")
        sys.exit(1) # Force exit on second Ctrl+C

# --- Worker Initialization Function ---
def worker_init_fn(worker_id):
    """Sets worker processes (DataLoader) to ignore SIGINT (Ctrl+C). The main process handles it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# --- PyTorch Device Setup ---
def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # More robust check for MPS
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Selected device: {device.type}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device)}")
    return device

DEVICE = get_device()

# --- Sequence Generation Functions (Output NumPy arrays) ---
def create_sequences_for_person_py(person_group_tuple, seq_len, features, pad_val, date_col):
    """
    Generates sequences for a single person's data. Ensures chronological order.
    Args:
        person_group_tuple: A tuple (person_id, person_df) from pandas groupby.
        seq_len: Length of sequences.
        features: List of feature column names.
        pad_val: Value used for padding.
        date_col: Name of the date column for sorting.
    Returns:
        List of tuples [(sequence_array, target_value), ...], or None if too short.
    """
    person_id, person_df = person_group_tuple
    # Ensure chronological order *within* the person's data
    person_df = person_df.sort_values(by=date_col)
    n_obs = len(person_df)

    # Need at least 2 observations: one for the sequence end, one for the target
    if n_obs <= 1:
        return None

    person_features = person_df[features].values.astype(np.float32)
    # Assuming integer targets (class indices)
    person_targets = person_df['target_state'].values.astype(np.int64) # Use int64 for potential LongTensor conversion

    sequences = []
    # Iterate up to the second-to-last observation to have a target for the sequence ending at index i
    for i in range(n_obs - 1):
        # Sequence ends at index i, target is at index i+1
        end_index = i + 1 # Exclusive index for slicing features
        start_index = max(0, end_index - seq_len)
        sequence_data = person_features[start_index : end_index, :] # Slice features up to current obs i

        target = person_targets[i + 1] # Target is the state at the *next* time step (i+1)

        actual_len = sequence_data.shape[0]
        pad_len = seq_len - actual_len

        if pad_len < 0:
             # This should theoretically not happen with the max(0, ...) logic
             print(f"Warning: Negative padding ({pad_len}) calculated for person {person_id}, index {i}. Using last {seq_len} elements.")
             padded_sequence = sequence_data[-seq_len:, :]
        elif pad_len == 0:
            padded_sequence = sequence_data
        else: # pad_len > 0
            # Create padding matrix with the specified pad_val
            padding_matrix = np.full((pad_len, len(features)), pad_val, dtype=np.float32)
            padded_sequence = np.vstack((padding_matrix, sequence_data))

        # Final check for shape consistency (defensive programming)
        if padded_sequence.shape != (seq_len, len(features)):
            print(f"ERROR: Sequence shape mismatch for person {person_id} at index {i}. Expected {(seq_len, len(features))}, got {padded_sequence.shape}. Skipping sequence.")
            continue # Skip this sequence

        sequences.append((padded_sequence, target))

    return sequences if sequences else None # Return None if no valid sequences were generated for the person

def generate_sequences_py(data_df, group_col, seq_len, features, pad_val, date_col, n_workers):
    """
    Generates sequences in parallel using joblib.
    """
    if not features:
        raise ValueError("Feature list cannot be empty for sequence generation.")
    if group_col not in data_df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")
    if date_col not in data_df.columns:
         raise ValueError(f"Date column '{date_col}' not found in dataframe.")
    if 'target_state' not in data_df.columns:
         raise ValueError(f"Target column 'target_state' not found in dataframe.")

    print(f"Generating sequences with seq_len={seq_len}, pad_val={pad_val} using {n_workers} workers...")
    start_time = time.time()

    # Group data by person
    grouped_data = data_df.groupby(group_col)
    num_groups = len(grouped_data)
    if num_groups == 0:
        print("Warning: No groups found in the data.")
        return np.array([], dtype=np.float32).reshape(0, seq_len, len(features)), np.array([], dtype=np.int64)

    # partial applies fixed arguments to the function
    func = partial(create_sequences_for_person_py, seq_len=seq_len, features=features, pad_val=pad_val, date_col=date_col)

    # Process groups in parallel using joblib with loky backend (more robust)
    results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(func)(group) for group in tqdm(grouped_data, total=num_groups, desc="Processing groups")
    )

    # Flatten the list of lists and filter out None results (from people with too few observations)
    all_sequences = []
    for person_result in results:
        if person_result: # Check if not None (i.e., sequences were generated for this person)
            all_sequences.extend(person_result)

    end_time = time.time()
    print(f"Sequence generation took {end_time - start_time:.2f} seconds.")

    if not all_sequences:
        print("Warning: No sequences were generated from the provided data.")
        # Return empty arrays with the correct dimensions
        return np.array([], dtype=np.float32).reshape(0, seq_len, len(features)), np.array([], dtype=np.int64)

    # Unzip sequences and targets
    x_list, y_list = zip(*all_sequences)

    # Convert to NumPy arrays with specific dtypes
    x_array = np.array(x_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.int64) # Use int64 for targets -> LongTensor

    print(f"Generated {x_array.shape[0]} sequences.")
    print(f"Shape of X: {x_array.shape}") # Should be (num_sequences, seq_len, n_features)
    print(f"Shape of y: {y_array.shape}") # Should be (num_sequences,)

    return x_array, y_array

# --- PyTorch Dataset ---
class SequenceDataset(Dataset):
    def __init__(self, x_data, y_data, pad_value):
        if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
             raise TypeError("Input data must be NumPy arrays")
        # Convert NumPy arrays to PyTorch tensors upon initialization
        self.x_data = torch.from_numpy(x_data.astype(np.float32))
        # Ensure target type is Long (Int64) for CrossEntropyLoss
        self.y_data = torch.from_numpy(y_data.astype(np.int64))
        self.pad_value = pad_value
        self.seq_len = x_data.shape[1] # Store sequence length

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        # Create padding mask: True for padded time steps, False for real data
        # A time step is considered padded if ALL features at that step match the pad_value
        # Mask shape should be (batch_size, seq_len) for transformer's src_key_padding_mask
        padding_mask = torch.all(x == self.pad_value, dim=-1) # Check across the feature dimension

        # Ensure mask has the correct shape (seq_len,)
        if padding_mask.shape != (self.seq_len,):
             # This case should ideally not happen if sequence generation is correct
             print(f"Warning: Unexpected padding mask shape {padding_mask.shape} for index {idx}. Expected ({self.seq_len},).")
             # Attempt to reshape or handle appropriately, e.g., create a default mask
             padding_mask = torch.zeros(self.seq_len, dtype=torch.bool) # Default to no padding if shape is wrong

        return x, y, padding_mask

# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Disable tqdm in worker processes to avoid multiple bars
    is_main_process = not torch.utils.data.get_worker_info()
    data_iterator = tqdm(dataloader, desc="Training", leave=False, disable=not is_main_process)

    for x_batch, y_batch, mask_batch in data_iterator:
        # Check for stop signal *during* batch iteration (more responsive)
        if stop_training_flag:
            print("Stop signal detected during training batch iteration.")
            break # Exit the inner loop

        x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()
        # Pass the padding mask to the model
        outputs = model(x_batch, src_key_padding_mask=mask_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.data, 1)
        total_samples += batch_size
        correct_predictions += (predicted == y_batch).sum().item()

        # Update tqdm description dynamically if it's the main process
        if is_main_process:
             current_loss = total_loss / total_samples if total_samples > 0 else 0
             current_acc = correct_predictions / total_samples if total_samples > 0 else 0
             data_iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    # Avoid division by zero if epoch finishes with no samples or was interrupted early
    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluates the model for one epoch."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    # Disable tqdm in worker processes
    is_main_process = not torch.utils.data.get_worker_info()
    data_iterator = tqdm(dataloader, desc="Evaluating", leave=False, disable=not is_main_process)

    with torch.no_grad():
        for x_batch, y_batch, mask_batch in data_iterator:
             # Check for stop signal (less critical here, but good practice)
             if stop_training_flag:
                 print("Stop signal detected during evaluation iteration.")
                 break

             x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

             # Pass the padding mask
             outputs = model(x_batch, src_key_padding_mask=mask_batch)
             loss = criterion(outputs, y_batch)

             batch_size = x_batch.size(0)
             total_loss += loss.item() * batch_size
             _, predicted = torch.max(outputs.data, 1)
             total_samples += batch_size
             correct_predictions += (predicted == y_batch).sum().item()
             all_preds.extend(predicted.cpu().numpy())
             all_targets.extend(y_batch.cpu().numpy())

             # Update tqdm description dynamically
             if is_main_process:
                 current_loss = total_loss / total_samples if total_samples > 0 else 0
                 current_acc = correct_predictions / total_samples if total_samples > 0 else 0
                 data_iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})


    if total_samples == 0:
         print("Warning: Evaluation completed with zero samples.")
         return 0.0, 0.0, np.array([]), np.array([])

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy, np.array(all_targets), np.array(all_preds)

def evaluate_aggregate_unemployment_error(model, dataloader, device, metadata):
    """
    Evaluates the model based on the aggregate unemployment rate error using deterministic (argmax) forecast.
    Args:
        model: The trained PyTorch model.
        dataloader: Validation or Test DataLoader.
        device: PyTorch device.
        metadata: Dictionary containing metadata (not strictly needed here if classes are assumed 0=E, 1=U, 2=N).
                  Kept for potential future use or different class mappings.
    Returns:
        Mean Squared Error (MSE) between predicted and actual aggregate unemployment rates.
        Returns float('inf') if calculation is not possible.
    """
    # Define class indices based on common convention (adjust if metadata differs)
    employed_idx = 0
    unemployed_idx = 1
    inactive_idx = 2 # Or whatever the third class is

    model.eval()
    all_preds = []
    all_targets = []

    is_main_process = not torch.utils.data.get_worker_info()
    data_iterator = tqdm(dataloader, desc="Agg. Err Eval", leave=False, disable=not is_main_process)

    with torch.no_grad():
        for x_batch, y_batch, mask_batch in data_iterator:
            x_batch, mask_batch = x_batch.to(device), mask_batch.to(device)
            # y_batch stays on CPU as we only need it for final calculation

            outputs = model(x_batch, src_key_padding_mask=mask_batch)
            # Deterministic forecast using argmax
            predicted = torch.argmax(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy()) # y_batch was already on CPU

    if not all_targets:
        print("Warning: No targets found in dataset for aggregate error calculation.")
        return float('inf') # Return infinity if no data

    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)

    # Calculate actual aggregate unemployment rate
    actual_unemployed_count = np.sum(all_targets_np == unemployed_idx)
    actual_labor_force = np.sum((all_targets_np == employed_idx) | (all_targets_np == unemployed_idx))
    if actual_labor_force == 0:
         print("Warning: Actual labor force size is zero. Cannot calculate actual rate.")
         actual_agg_rate = 0.0 # Or handle as error? Assign 0 for now.
    else:
         actual_agg_rate = actual_unemployed_count / actual_labor_force

    # Calculate predicted aggregate unemployment rate
    predicted_unemployed_count = np.sum(all_preds_np == unemployed_idx)
    predicted_labor_force = np.sum((all_preds_np == employed_idx) | (all_preds_np == unemployed_idx))
    if predicted_labor_force == 0:
         print("Warning: Predicted labor force size is zero. Cannot calculate predicted rate.")
         predicted_agg_rate = 0.0 # Assign 0 for now.
    else:
         predicted_agg_rate = predicted_unemployed_count / predicted_labor_force

    # Calculate Mean Squared Error between the single rate values
    error = mean_squared_error([actual_agg_rate], [predicted_agg_rate])
    print(f"  Agg. Rate Eval - Actual: {actual_agg_rate:.4f} ({actual_unemployed_count}/{actual_labor_force}), "
          f"Predicted: {predicted_agg_rate:.4f} ({predicted_unemployed_count}/{predicted_labor_force}), MSE: {error:.6f}")

    return error


# --- Refactored Helper Functions ---

def load_and_prepare_data(train_file, val_file, metadata_file, date_col, group_id_col, train_start_date=None, train_end_date=None):
    """Loads data, metadata, performs checks, and applies date filters."""
    print("\n===== STEP 1: Loading Preprocessed Data & Metadata =====")
    # --- Input Checks ---
    required_files = [train_file, val_file, metadata_file]
    print("Checking for required input files...")
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("ERROR: Missing required preprocessed input file(s):")
        for f in missing_files: print(f" - {f}")
        raise FileNotFoundError(f"Missing input files: {missing_files}")
    print("All required input files found.")

    # --- Load Data ---
    try:
        train_data_baked = pd.read_parquet(train_file)
        val_data_baked = pd.read_parquet(val_file)
        with open(metadata_file, 'rb') as f: metadata = pickle.load(f)
        print("Loaded baked data (train, val) & metadata.")
    except Exception as e:
        print(f"ERROR loading data/metadata: {e}")
        traceback.print_exc()
        raise

    # --- Extract Metadata ---
    try:
        feature_names = metadata['feature_names']
        n_features = metadata['n_features']
        n_classes = metadata['n_classes']
        target_col = 'target_state' # Expected target column name
        if target_col not in train_data_baked.columns or target_col not in val_data_baked.columns:
             raise ValueError(f"Target column '{target_col}' not found in train or validation data.")
        if date_col not in train_data_baked.columns or group_id_col not in train_data_baked.columns:
             raise ValueError(f"Date ('{date_col}') or Group ID ('{group_id_col}') column not found in data.")
        print(f" - n_features: {n_features}, n_classes: {n_classes}")
        if n_classes != 3: print(f"WARNING: Loaded n_classes = {n_classes}. Ensure this is expected.")
    except KeyError as e:
        print(f"ERROR: Metadata file {metadata_file} is missing required key: {e}")
        raise
    except Exception as e:
         print(f"ERROR processing metadata: {e}")
         traceback.print_exc()
         raise

    # --- Filter Baked Data by Date ---
    print("\nFiltering data based on date range (if specified)...")
    try:
        train_data_baked[date_col] = pd.to_datetime(train_data_baked[date_col])
        val_data_baked[date_col] = pd.to_datetime(val_data_baked[date_col])
    except Exception as e:
        print(f"ERROR converting date column '{date_col}' to datetime: {e}")
        traceback.print_exc()
        raise

    initial_train_rows = len(train_data_baked)
    initial_val_rows = len(val_data_baked)

    if train_start_date:
        try:
            train_start_dt = pd.to_datetime(train_start_date)
            train_data_baked = train_data_baked[train_data_baked[date_col] >= train_start_dt].copy()
            val_data_baked = val_data_baked[val_data_baked[date_col] >= train_start_dt].copy()
            print(f"Applied TRAIN_START_DATE >= {train_start_date}")
        except Exception as e:
            print(f"Warning: Could not apply train_start_date filter: {e}")

    if train_end_date:
        try:
            train_end_dt = pd.to_datetime(train_end_date)
            train_data_baked = train_data_baked[train_data_baked[date_col] <= train_end_dt].copy()
            val_data_baked = val_data_baked[val_data_baked[date_col] <= train_end_dt].copy()
            print(f"Applied TRAIN_END_DATE <= {train_end_date}")
        except Exception as e:
            print(f"Warning: Could not apply train_end_date filter: {e}")

    print(f"Train data rows: {initial_train_rows} -> {len(train_data_baked)}")
    print(f"Validation data rows: {initial_val_rows} -> {len(val_data_baked)}")
    if len(train_data_baked) == 0 or len(val_data_baked) == 0:
         raise ValueError("No data remaining after date filtering.")

    return train_data_baked, val_data_baked, metadata, feature_names, n_features, n_classes


def setup_sequence_generation(hparams, train_data_baked, val_data_baked, processed_data_dir, group_id_col, date_col, feature_names, n_features):
    """Handles sequence generation including caching and parallelization."""
    print("\n===== STEP 2: Generating Sequences =====")
    parallel_workers = hparams['parallel_workers']
    refresh_sequences = hparams['refresh_sequences']
    pad_value = hparams['pad_value']
    seq_len = hparams['sequence_length']

    # Determine parallel workers dynamically
    if parallel_workers <= 0:
        try:
            num_cores = multiprocessing.cpu_count()
            parallel_workers = min(max(1, num_cores // 2), 8)
            print(f"Parallel workers dynamically set to {parallel_workers} (detected {num_cores} cores)")
        except NotImplementedError:
            print("Could not detect CPU count. Setting parallel workers to 1.")
            parallel_workers = 1
    else:
        print(f"Using specified parallel workers for sequence generation: {parallel_workers}")

    # --- Sequence cache setup ---
    sequence_cache_base_dir = processed_data_dir / config.SEQUENCE_CACHE_DIR_NAME
    sequence_cache_dir = sequence_cache_base_dir / f"seqlen_{seq_len}"
    sequence_cache_dir.mkdir(parents=True, exist_ok=True)
    train_seq_cache_file = sequence_cache_dir / "train_sequences.npz"
    val_seq_cache_file = sequence_cache_dir / "val_sequences.npz"

    # Helper function for sequence generation/caching for a split
    def process_split_py(split_name, baked_data, cache_file):
        # nonlocal seq_len, n_features # Access outer scope variables needed for validation
        if not refresh_sequences and cache_file.exists():
            print(f"Attempting to load cached sequences for '{split_name}' from: {cache_file}")
            try:
                cache_data = np.load(cache_file)
                x_array = cache_data['x']
                y_array = cache_data['y']
                if x_array.ndim == 3 and x_array.shape[1] == seq_len and x_array.shape[2] == n_features and y_array.ndim == 1 and x_array.shape[0] == y_array.shape[0]:
                    print(f"Loaded {x_array.shape[0]} sequences from cache. Shape X: {x_array.shape}, y: {y_array.shape}")
                    return x_array, y_array
                else:
                    print(f"Cached sequence dimensions mismatch parameters. Regenerating.")
            except Exception as e:
                print(f"Error loading or validating cache file {cache_file}: {e}. Regenerating.")

        print(f"Generating sequences for '{split_name}'...")
        x_array, y_array = generate_sequences_py(
            baked_data, group_col=group_id_col, date_col=date_col,
            seq_len=seq_len, features=feature_names, pad_val=pad_value, n_workers=parallel_workers
        )

        if stop_training_flag:
            print(f"Stop signal detected after generating sequences for '{split_name}'.")
            raise KeyboardInterrupt

        if x_array.shape[0] > 0:
             print(f"Saving generated sequences for '{split_name}' ({x_array.shape[0]} sequences) to cache: {cache_file}")
             try: np.savez_compressed(cache_file, x=x_array, y=y_array)
             except Exception as e: print(f"Error saving sequence cache {cache_file}: {e}")
        else: print(f"No sequences generated for '{split_name}', cache not saved.")
        return x_array, y_array

    try:
        x_train_np, y_train_np = process_split_py("train", train_data_baked, train_seq_cache_file)
        if stop_training_flag: raise KeyboardInterrupt
        x_val_np, y_val_np = process_split_py("validation", val_data_baked, val_seq_cache_file)
        if stop_training_flag: raise KeyboardInterrupt
    except KeyboardInterrupt:
         print("Sequence generation interrupted.")
         raise
    except Exception as e:
        print(f"ERROR during sequence generation: {e}")
        traceback.print_exc()
        raise

    if x_train_np.shape[0] == 0: raise ValueError("No training sequences were generated or loaded.")
    if x_val_np.shape[0] == 0: raise ValueError("No validation sequences were generated or loaded.")

    return x_train_np, y_train_np, x_val_np, y_val_np, parallel_workers # Return workers for DataLoader setup


def create_dataloaders(x_train_np, y_train_np, x_val_np, y_val_np, hparams, n_classes, device, parallel_workers):
    """Creates Datasets, Sampler, Loss Weights, and DataLoaders."""
    print("\nCreating PyTorch Datasets and DataLoaders...")
    batch_size = hparams['batch_size']
    pad_value = hparams['pad_value']

    try:
         train_dataset = SequenceDataset(x_train_np, y_train_np, pad_value)
         val_dataset = SequenceDataset(x_val_np, y_val_np, pad_value)
    except Exception as e:
         print(f"ERROR creating SequenceDataset objects: {e}")
         traceback.print_exc()
         raise

    # --- WeightedRandomSampler ---
    train_sampler = None
    if hparams['use_weighted_sampling']:
        print("Calculating sample weights for WeightedRandomSampler...")
        if len(y_train_np) > 0:
            class_counts = np.bincount(y_train_np, minlength=n_classes)
            print(f"Training target class counts (for sampler): {class_counts}")
            if np.any(class_counts == 0):
                print("Warning: Zero samples found for some classes. Adjusting counts to 1 for weighting.")
                class_counts = np.maximum(class_counts, 1)
            class_weights_inv = 1.0 / class_counts
            sample_weights = class_weights_inv[y_train_np]
            sample_weights_tensor = torch.from_numpy(sample_weights).double()
            train_sampler = WeightedRandomSampler(sample_weights_tensor, len(sample_weights_tensor), replacement=True)
            print("WeightedRandomSampler created for training data.")
        else: print("Training data is empty, cannot create weighted sampler.")
    else: print("WeightedRandomSampler is disabled.")

    # --- Class Weights for Loss ---
    class_weights_tensor = None
    if hparams['use_weighted_loss']:
        print("Calculating class weights for loss function...")
        if len(y_train_np) > 0:
            class_counts = np.bincount(y_train_np, minlength=n_classes)
            print(f"Training target class counts (for loss): {class_counts}")
            total_samples_train = len(y_train_np)
            if np.any(class_counts == 0):
                 print("Warning: Zero samples for some classes in loss weight calc. Using adjusted counts.")
                 weights = total_samples_train / (n_classes * np.maximum(class_counts, 1))
            else: weights = total_samples_train / (n_classes * class_counts)
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Calculated class weights for loss: {class_weights_tensor.cpu().numpy()}")
        else: print("Training data empty, cannot calculate weighted loss. Using standard loss.")
    else: print("Weighted loss is disabled. Using standard CrossEntropyLoss.")

    # --- DataLoaders ---
    dataloader_workers = min(4, parallel_workers) if parallel_workers > 0 else 0
    print(f"Using {dataloader_workers} workers for DataLoaders.")
    pin_memory = device.type == 'cuda'
    persistent_workers = dataloader_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=dataloader_workers,
        worker_init_fn=worker_init_fn if dataloader_workers > 0 else None,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader_params = {
        'batch_size': batch_size, 'shuffle': False, 'num_workers': dataloader_workers,
        'pin_memory': pin_memory, 'persistent_workers': persistent_workers,
        'worker_init_fn': worker_init_fn if dataloader_workers > 0 else None
    }
    val_loader = DataLoader(val_dataset, **val_loader_params)

    print(f"DataLoaders created (Training uses {'WeightedRandomSampler' if train_sampler else 'standard shuffling'}).")
    if dataloader_workers > 0: print("DataLoader workers configured to ignore SIGINT.")

    return train_loader, val_loader, val_loader_params, class_weights_tensor, val_dataset # Return val_dataset for final eval


def build_model(hparams, n_features, n_classes, device):
    """Builds the Transformer model."""
    print("\n===== STEP 3: Building Transformer Model =====")
    try:
        model = TransformerForecastingModel(
            input_dim=n_features, seq_len=hparams['sequence_length'],
            embed_dim=hparams['embed_dim'], num_heads=hparams['num_heads'],
            ff_dim=hparams['ff_dim'], num_transformer_blocks=hparams['num_transformer_blocks'],
            mlp_units=hparams['mlp_units'], dropout=hparams['dropout'],
            mlp_dropout=hparams['mlp_dropout'], n_classes=n_classes
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model built successfully.")
        print(f"Total trainable parameters: {total_params:,}")
        return model
    except Exception as e:
        print(f"ERROR building model: {e}")
        traceback.print_exc()
        raise


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, hparams, device, checkpoint_path, trial=None):
    """Runs the main training loop with validation, checkpointing, and early stopping."""
    print("\n===== STEP 4: Training the Model =====")
    epochs = hparams['epochs']
    early_stopping_patience = hparams['early_stopping_patience']
    max_grad_norm = hparams['max_grad_norm']

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    training_interrupted = False
    last_epoch = 0 # Track the last epoch number run

    print(f"Starting training loop: Max epochs={epochs}, Batch size={hparams['batch_size']}, LR={hparams['learning_rate']:.6f}")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print(f"Best model checkpoint path: {checkpoint_path}")

    for epoch in range(epochs):
        last_epoch = epoch
        epoch_start_time = time.time()

        if stop_training_flag:
            print(f"\nStop signal detected before starting epoch {epoch+1}. Stopping training.")
            training_interrupted = True
            break

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)

        if stop_training_flag:
            print(f"\nStop signal detected after training epoch {epoch+1}. Stopping training.")
            training_interrupted = True
            history['loss'].append(train_loss); history['accuracy'].append(train_acc)
            history['val_loss'].append(float('nan')); history['val_accuracy'].append(float('nan'))
            break

        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        epoch_end_time = time.time()

        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_end_time - epoch_start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.7f}")

        history['loss'].append(train_loss); history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_accuracy'].append(val_acc)

        if np.isnan(val_loss):
            print("Warning: Validation loss is NaN. Skipping LR scheduler step and early stopping check.")
        else:
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                try:
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"  -> Val loss improved to {val_loss:.4f}. Saved best model checkpoint.")
                except Exception as e: print(f"  -> ERROR saving best model checkpoint: {e}")
            else: epochs_no_improve += 1

        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch+1} based on intermediate validation loss.")
                raise optuna.TrialPruned()

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    if training_interrupted: print("Training loop was interrupted by user (Ctrl+C).")
    elif epochs_no_improve < early_stopping_patience: print("Training loop finished after completing all epochs.")

    return history, best_val_loss, training_interrupted, last_epoch, epochs_no_improve


# --- Main Train/Evaluate Function ---

def train_and_evaluate(hparams: dict, trial: optuna.Trial = None):
    """
    Main function to load data, build model, train, and evaluate based on hyperparameters.
    Handles graceful exit and Optuna integration.
    Args:
        hparams (dict): Dictionary containing all hyperparameters.
        trial (optuna.Trial, optional): Optuna trial object for reporting intermediate values and pruning.
    Returns:
        dict: Dictionary containing evaluation results:
              {'agg_error': float, 'best_val_loss': float, 'final_val_acc': float, 'best_epoch': int, 'status': str}
              Returns status='Failed' or 'Interrupted' or 'Pruned' on issues.
    """
    global stop_training_flag
    stop_training_flag = False # Reset flag at the start of each run/trial
    training_was_interrupted = False # Track if training loop specifically was interrupted

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    start_run_time = time.time()
    final_agg_error = float('inf')
    best_val_loss_final = float('inf')
    final_val_acc_final = float('nan')
    best_epoch_final = -1
    run_status = "Started"
    model = None # Initialize model to None

    try:
        print("=" * 60)
        run_type = f"Optuna Trial {trial.number}" if trial else "Standard Run"
        print(f"Starting Transformer Model: {run_type}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Hyperparameters:")
        for key, value in hparams.items(): print(f"  - {key}: {value}")
        print("-" * 60)

        # --- Setup Output Dirs ---
        processed_data_dir = Path(config.PREPROCESS_OUTPUT_DIR)
        run_id = f"trial_{trial.number}" if trial else "standard_run"
        base_output_dir = Path(config.TRAIN_OUTPUT_SUBDIR)
        model_output_dir = base_output_dir / run_id
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run output directory: {model_output_dir}")
        history_file = model_output_dir / "training_history.pkl"
        checkpoint_path = model_output_dir / "best_model_val_loss.pt"
        params_path = model_output_dir / "model_params.pkl" # Define params path

        # --- Set Random Seeds ---
        seed_value = hparams['random_seed']
        os.environ['PYTHONHASHSEED'] = str(seed_value); random.seed(seed_value)
        np.random.seed(seed_value); torch.manual_seed(seed_value)
        if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed_value)
        print(f"Random seeds set to: {seed_value}")

        # --- Step 1: Load Data ---
        train_file = processed_data_dir / config.TRAIN_DATA_FILENAME
        val_file = processed_data_dir / config.VAL_DATA_FILENAME
        metadata_file = processed_data_dir / config.METADATA_FILENAME
        train_data_baked, val_data_baked, metadata, feature_names, n_features, n_classes = load_and_prepare_data(
            train_file, val_file, metadata_file, config.DATE_COL, config.GROUP_ID_COL,
            hparams.get('train_start_date'), hparams.get('train_end_date')
        )

        # --- Add derived params to hparams for saving --- # Added
        hparams['n_features'] = n_features
        hparams['n_classes'] = n_classes

        # --- Step 2: Generate Sequences ---
        x_train_np, y_train_np, x_val_np, y_val_np, parallel_workers = setup_sequence_generation(
            hparams, train_data_baked, val_data_baked, processed_data_dir,
            config.GROUP_ID_COL, config.DATE_COL, feature_names, n_features
        )
        # Clear large dataframes if no longer needed
        del train_data_baked, val_data_baked

        # --- Step 3: Create DataLoaders ---
        train_loader, val_loader, val_loader_params, class_weights_tensor, val_dataset = create_dataloaders(
            x_train_np, y_train_np, x_val_np, y_val_np, hparams, n_classes, DEVICE, parallel_workers
        )
        # Clear large numpy arrays if no longer needed
        del x_train_np, y_train_np, x_val_np, y_val_np

        # --- Step 4: Build Model ---
        model = build_model(hparams, n_features, n_classes, DEVICE)

        # --- Save Model Parameters Immediately --- # Added
        try:
            # Save hyperparameters needed for model reconstruction
            # Filter hparams if necessary, but saving all is often fine
            with open(params_path, 'wb') as f:
                pickle.dump(hparams, f)
            print(f"Model parameters saved to: {params_path}")
        except Exception as e:
            print(f"Warning: Could not save model parameters: {e}")
            # Decide if this should be a fatal error depending on requirements
            # raise # Uncomment to make it fatal

        # --- Step 5: Setup Training Components ---
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=hparams['lr_scheduler_factor'],
            patience=hparams['lr_scheduler_patience'], verbose=True
        )

        # --- Step 6: Run Training Loop ---
        history, best_val_loss, training_interrupted_flag, last_epoch, epochs_no_improve = run_training_loop(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            hparams, DEVICE, checkpoint_path, trial
        )
        best_val_loss_final = best_val_loss # Store best loss from training
        training_was_interrupted = training_interrupted_flag # Store if training loop was interrupted

        # --- Save Training History ---
        try:
            with open(history_file, 'wb') as f: pickle.dump(history, f)
            print(f"Training history saved to: {history_file}")
        except Exception as e: print(f"Warning: Could not save training history: {e}")

        # --- Reset stop flag BEFORE final evaluation ---
        print("Resetting stop flag before final evaluation.")
        stop_training_flag = False
        # Restore original handler temporarily during evaluation? No, keep custom one for now.

        # --- Step 7: Final Evaluation ---
        print("\n===== STEP 5: Evaluating Model (Aggregate Unemployment Error) =====") # Renumbering comment is minor

        # Load best weights if checkpoint exists
        if checkpoint_path.exists():
            print(f"Loading best model weights from {checkpoint_path} for final evaluation.")
            try: model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            except Exception as e: print(f"Warning: Could not load best weights from {checkpoint_path}: {e}. Using last weights.")
        else: print("Warning: Best model checkpoint not found. Using last weights from training for evaluation.")

        # Recreate validation loader for clean evaluation
        print("Recreating validation DataLoader for final evaluation...")
        if 'val_dataset' not in locals() or val_dataset is None: # Check if val_dataset exists
             raise RuntimeError("Validation dataset not available for final evaluation.")
        if 'val_loader_params' not in locals(): # Check if params exist
             raise RuntimeError("Validation DataLoader parameters not available for final evaluation.")
        final_val_loader = DataLoader(val_dataset, **val_loader_params)
        print("Validation DataLoader recreated.")

        # Run evaluation - this should now complete even if training was interrupted
        final_agg_error = evaluate_aggregate_unemployment_error(model, final_val_loader, DEVICE, metadata)
        print(f"Final Aggregate Unemployment Rate Error (MSE) on Validation Set: {final_agg_error:.6f}")

        # Extract final metrics from history
        final_val_acc_final = next((h for h in reversed(history.get('val_accuracy', [])) if not np.isnan(h)), float('nan'))
        if not np.isnan(best_val_loss_final) and best_val_loss_final != float('inf'):
             try: best_epoch_final = history.get('val_loss', []).index(best_val_loss_final) + 1
             except (ValueError, IndexError): best_epoch_final = last_epoch + 1 # Fallback
        else: best_epoch_final = last_epoch + 1 # Fallback

        # Determine final status
        run_successful = not np.isinf(final_agg_error) and not np.isnan(final_agg_error)
        if training_was_interrupted:
            run_status = "Interrupted" # Mark as interrupted even if evaluation succeeded
        elif run_successful:
            run_status = "Completed"
        else:
            run_status = "Finished with Error" # e.g., if evaluation failed for other reasons


    except optuna.TrialPruned as e:
        print(f"Optuna Trial Pruned: {e}")
        final_agg_error = float('inf'); run_status = "Pruned"
        training_was_interrupted = True # Treat pruning like an interruption for status
    except KeyboardInterrupt:
        print("\nRun interrupted by user (KeyboardInterrupt caught in train_and_evaluate).")
        final_agg_error = float('inf'); run_status = "Interrupted"
        training_was_interrupted = True # Explicitly set flag
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as e: # Catch specific expected errors
        print(f"\nA known error occurred during the run: {e}")
        traceback.print_exc()
        final_agg_error = float('inf'); run_status = "Failed"
    except Exception as e: # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred during the run: {e}")
        traceback.print_exc()
        final_agg_error = float('inf'); run_status = "Failed"
    finally:
        # --- Cleanup ---
        signal.signal(signal.SIGINT, original_sigint_handler) # Restore original handler at the very end
        print("Restored original SIGINT handler.")
        # Clean up GPU memory if model was loaded
        if model is not None:
            del model
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            print("Cleaned up model and GPU memory.")

        end_run_time = time.time()
        elapsed_mins = (end_run_time - start_run_time) / 60
        print("-" * 60)
        # Report status based on whether training was interrupted or other errors occurred
        print(f"Run Status: {run_status} | Elapsed: {elapsed_mins:.2f} minutes")
        # Report the calculated aggregate error, even if interrupted
        print(f"Final Aggregate Error Returned: {final_agg_error}")
        print("=" * 60)

    results = {
        'agg_error': final_agg_error, 'best_val_loss': best_val_loss_final,
        'final_val_acc': final_val_acc_final, 'best_epoch': best_epoch_final,
        'status': run_status # Use the final determined status
    }
    return results


# --- Optuna Objective Function ---

def objective(trial: optuna.Trial):
    """Optuna objective function to minimize aggregate unemployment error."""
    # Construct study_dir relative to the base output dir defined in config
    study_dir = Path(config.TRAIN_OUTPUT_SUBDIR) / trial.study.study_name
    hpt_results_file = study_dir / HPT_RESULTS_CSV
    best_hparams_file = study_dir / BEST_HPARAMS_PKL

    # --- Define Hyperparameter Search Space ---
    hparams = {
        'sequence_length': trial.suggest_int("sequence_length", config.HPT_SEQ_LEN_MIN, config.HPT_SEQ_LEN_MAX),
        'embed_dim': trial.suggest_categorical("embed_dim", config.HPT_EMBED_DIM_OPTIONS),
        'num_heads': trial.suggest_categorical("num_heads", config.HPT_NUM_HEADS_OPTIONS),
        'ff_dim': trial.suggest_int("ff_dim", config.HPT_FF_DIM_MIN, config.HPT_FF_DIM_MAX, step=config.HPT_FF_DIM_STEP),
        'num_transformer_blocks': trial.suggest_int("num_transformer_blocks", config.HPT_NUM_BLOCKS_MIN, config.HPT_NUM_BLOCKS_MAX),
        'mlp_units': [trial.suggest_int("mlp_units_1", config.HPT_MLP_UNITS_MIN, config.HPT_MLP_UNITS_MAX, step=config.HPT_MLP_UNITS_STEP)],
        'dropout': trial.suggest_float("dropout", config.HPT_DROPOUT_MIN, config.HPT_DROPOUT_MAX),
        'mlp_dropout': trial.suggest_float("mlp_dropout", config.HPT_MLP_DROPOUT_MIN, config.HPT_MLP_DROPOUT_MAX),
        'learning_rate': trial.suggest_float("learning_rate", config.HPT_LR_MIN, config.HPT_LR_MAX, log=True),
        'batch_size': trial.suggest_categorical("batch_size", config.HPT_BATCH_SIZE_OPTIONS),
        'use_weighted_loss': trial.suggest_categorical("use_weighted_loss", config.HPT_USE_WEIGHTED_LOSS_OPTIONS),
        # Fixed params for HPT run
        'epochs': config.HPT_EPOCHS,
        'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        'lr_scheduler_factor': config.LR_SCHEDULER_FACTOR,
        'lr_scheduler_patience': config.LR_SCHEDULER_PATIENCE,
        'max_grad_norm': config.MAX_GRAD_NORM,
        'pad_value': config.PAD_VALUE,
        'parallel_workers': config.PARALLEL_WORKERS,
        'refresh_sequences': config.REFRESH_SEQUENCES,
        'train_start_date': config.TRAIN_START_DATE,
        'train_end_date': config.TRAIN_END_DATE,
        'random_seed': config.RANDOM_SEED,
        'use_weighted_sampling': config.USE_WEIGHTED_SAMPLING
    }

    # --- Constraint Check ---
    if hparams['embed_dim'] % hparams['num_heads'] != 0:
        print(f"Pruning trial {trial.number}: embed_dim {hparams['embed_dim']} not divisible by num_heads {hparams['num_heads']}")
        raise optuna.TrialPruned(f"embed_dim ({hparams['embed_dim']}) must be divisible by num_heads ({hparams['num_heads']}).")

    # --- Run Training and Evaluation ---
    results = train_and_evaluate(hparams, trial)
    aggregate_error = results['agg_error']

    # --- Log results to CSV ---
    params_flat = hparams.copy()
    params_flat['mlp_units'] = str(params_flat['mlp_units'])
    fieldnames = ['trial_number', 'status', 'agg_error', 'best_val_loss', 'final_val_acc', 'best_epoch'] + list(params_flat.keys())
    log_entry = {
        'trial_number': trial.number, 'status': results['status'],
        'agg_error': f"{aggregate_error:.6f}" if not np.isinf(aggregate_error) and not np.isnan(aggregate_error) else 'inf',
        'best_val_loss': f"{results['best_val_loss']:.6f}" if not np.isinf(results['best_val_loss']) and not np.isnan(results['best_val_loss']) else 'inf',
        'final_val_acc': f"{results['final_val_acc']:.4f}" if not np.isnan(results['final_val_acc']) else 'nan',
        'best_epoch': results['best_epoch'], **params_flat
    }
    try:
        study_dir.mkdir(parents=True, exist_ok=True)
        write_header = not hpt_results_file.exists()
        with open(hpt_results_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header: writer.writeheader()
            writer.writerow(log_entry)
        print(f"Logged trial {trial.number} results to {hpt_results_file}")
    except Exception as e: print(f"ERROR logging trial {trial.number} results to CSV: {e}")

    # --- Update best hyperparameters file ---
    current_best_value = float('inf')
    try: # Handle case where study has no completed trials yet
        current_best_value = trial.study.best_value
    except ValueError: pass # Keep current_best_value as inf

    if results['status'] == "Completed" and aggregate_error < current_best_value:
         print(f"Trial {trial.number} improved aggregate error to {aggregate_error:.6f}. Updating best hyperparameters.")
         try:
             with open(best_hparams_file, 'wb') as f: pickle.dump(trial.params, f)
             print(f"Best hyperparameters updated in {best_hparams_file}")
         except Exception as e: print(f"ERROR saving best hyperparameters for trial {trial.number}: {e}")

    return aggregate_error


# --- HPT and Standard Run Functions ---

def run_hyperparameter_tuning(args, base_output_dir):
    """Sets up and runs the Optuna hyperparameter tuning study."""
    print(f"\n--- Starting Hyperparameter Tuning ({args.n_trials} trials) ---")
    study_name = args.study_name
    # study_dir is now relative to the base_output_dir (e.g., models/transformer_pytorch/hpt_study_name)
    study_dir = base_output_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    storage_path = study_dir / f"{study_name}.db"
    storage_name = f"sqlite:///{storage_path}"
    hpt_results_file = study_dir / HPT_RESULTS_CSV
    best_hparams_file = study_dir / BEST_HPARAMS_PKL

    print(f"Optuna study name: {study_name}")
    print(f"Optuna storage: {storage_name}")
    print(f"HPT results log: {hpt_results_file}")
    print(f"Best hyperparameters file: {best_hparams_file}")

    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.HPT_PRUNER_STARTUP, n_warmup_steps=config.HPT_PRUNER_WARMUP, interval_steps=1
        )
    )
    print(f"Optuna pruner enabled: MedianPruner (startup={config.HPT_PRUNER_STARTUP}, warmup={config.HPT_PRUNER_WARMUP})")
    try: print(f"Existing best value in study: {study.best_value}")
    except ValueError: print("No completed trials yet in study.")

    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=config.HPT_TIMEOUT_SECONDS, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nOptuna optimization interrupted by user (Ctrl+C).")
        print("Results for completed trials (and the current best parameters) should be saved.")
    except Exception as e:
        print(f"\nAn critical error occurred during Optuna optimization: {e}")
        traceback.print_exc()

    # --- Print HPT Summary ---
    print("\n--- HPT Finished ---")
    try:
        print(f"Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        states = [t.state for t in study.trials]
        print(f"  States: Complete({states.count(optuna.trial.TrialState.COMPLETE)}), "
              f"Pruned({states.count(optuna.trial.TrialState.PRUNED)}), "
              f"Fail({states.count(optuna.trial.TrialState.FAIL)})")

        if any(s == optuna.trial.TrialState.COMPLETE for s in states):
            best_trial = study.best_trial
            print(f"\nBest trial overall:")
            print(f"  Trial Number: {best_trial.number}")
            print(f"  Value (Min Aggregate Error): {best_trial.value:.6f}")
            print("  Best Params: ")
            for key, value in best_trial.params.items(): print(f"    {key}: {value}")
        else: print("\nNo trials completed successfully.")

        print(f"\nDetailed results logged to: {hpt_results_file}")
        if best_hparams_file.exists(): print(f"Best hyperparameters saved to: {best_hparams_file}")
        else: print(f"Best hyperparameters file not created (no successful trials or error saving).")
    except Exception as e: print(f"Error retrieving final study results: {e}")


def run_standard_training(base_output_dir):
    """Runs a standard training process, potentially using best HPT params."""
    print("\n--- Starting Standard Training Run ---")
    study_name = config.HPT_STUDY_NAME
    # Locate the study directory where best hparams might be saved
    study_dir = base_output_dir / study_name
    best_hparams_file = study_dir / BEST_HPARAMS_PKL

    # Start with default hyperparameters
    standard_hparams = {
        'sequence_length': config.SEQUENCE_LENGTH, 'embed_dim': config.EMBED_DIM,
        'num_heads': config.NUM_HEADS, 'ff_dim': config.FF_DIM,
        'num_transformer_blocks': config.NUM_TRANSFORMER_BLOCKS, 'mlp_units': config.MLP_UNITS,
        'dropout': config.DROPOUT, 'mlp_dropout': config.MLP_DROPOUT,
        'learning_rate': config.LEARNING_RATE, 'batch_size': config.BATCH_SIZE,
        'use_weighted_loss': config.USE_WEIGHTED_LOSS, 'use_weighted_sampling': config.USE_WEIGHTED_SAMPLING,
        'epochs': config.EPOCHS, 'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        'lr_scheduler_factor': config.LR_SCHEDULER_FACTOR, 'lr_scheduler_patience': config.LR_SCHEDULER_PATIENCE,
        'max_grad_norm': config.MAX_GRAD_NORM, 'pad_value': config.PAD_VALUE,
        'parallel_workers': config.PARALLEL_WORKERS, 'refresh_sequences': config.REFRESH_SEQUENCES,
        'train_start_date': config.TRAIN_START_DATE, 'train_end_date': config.TRAIN_END_DATE,
        'random_seed': config.RANDOM_SEED
    }

    # Load best HPT params if available
    if best_hparams_file.exists():
        print(f"\nFound best hyperparameters file: {best_hparams_file}")
        try:
            with open(best_hparams_file, 'rb') as f: best_params = pickle.load(f)
            print("Successfully loaded tuned hyperparameters.")
            original_epochs = standard_hparams['epochs'] # Keep standard epochs
            standard_hparams.update(best_params)
            standard_hparams['epochs'] = original_epochs
            print("Applied tuned hyperparameters (epochs kept from config.EPOCHS).")
        except Exception as e:
            print(f"Warning: Failed to load or apply best hyperparameters from {best_hparams_file}: {e}")
            print("Proceeding with default hyperparameters from config.py.")
    else:
        print(f"\nBest hyperparameters file not found at {best_hparams_file}.")
        print("Proceeding with default hyperparameters from config.py.")

    # Run training
    try:
        results = train_and_evaluate(standard_hparams, trial=None)
        print(f"\nStandard run finished with status: {results['status']}")
        print(f"Final Aggregate Unemployment Error (Validation Set): {results['agg_error']:.6f}")
        # Consider adding final evaluation on the *test* set here
    except KeyboardInterrupt: print("\nStandard training run interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn critical error occurred during the standard training run: {e}")
        traceback.print_exc()


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Model or Run Hyperparameter Tuning")
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning using Optuna')
    parser.add_argument('--n_trials', type=int, default=config.HPT_N_TRIALS, help='Number of trials for Optuna tuning')
    parser.add_argument('--study_name', type=str, default=config.HPT_STUDY_NAME, help='Name for the Optuna study')
    args = parser.parse_args() # Corrected this line

    print("\n===== Transformer Training Script =====")
    print(f"Selected device: {DEVICE.type}") # Ensure device is printed early

    # base_output_dir is now models/transformer_pytorch
    base_output_dir = Path(config.TRAIN_OUTPUT_SUBDIR)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base output directory: {base_output_dir}")

    if args.tune:
        # Pass the base directory; run_hyperparameter_tuning will create the study subdir inside it
        run_hyperparameter_tuning(args, base_output_dir)
    else:
        # Pass the base directory; run_standard_training will use it to find hparams
        # and train_and_evaluate will create the 'standard_run' subdir inside it
        run_standard_training(base_output_dir)

    print("\n--- Python Script Finished ---")

# --- End of Script ---