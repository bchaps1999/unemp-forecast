import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
import random
import signal
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
import traceback
import sys

# --- Global flag for graceful exit ---
stop_training_flag = False

# --- Signal Handler ---
def signal_handler(sig, frame):
    """Sets the stop_training_flag when SIGINT (Ctrl+C) is received."""
    global stop_training_flag
    if not stop_training_flag: # Prevent multiple messages
        print("\nCtrl+C detected! Attempting graceful stop...")
        print("Press Ctrl+C again to force exit (may corrupt state).")
        stop_training_flag = True
    else:
        print("Second Ctrl+C detected. Exiting forcefully.")
        sys.exit(1)

# --- Worker Initialization Function ---
def worker_init_fn(worker_id):
    """Initialize DataLoader workers to ignore SIGINT and set unique random seeds."""
    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore Ctrl+C in workers
    os.environ['PYTHONMALLOC'] = 'default' # Mitigate potential macOS memory warnings

    # Set unique random seeds for each worker based on initial seed + worker_id
    base_seed = torch.initial_seed() % (2**32)
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# --- PyTorch Device Setup ---
def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Selected device: CUDA ({torch.cuda.get_device_name(device)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Selected device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Selected device: CPU")
    return device

# --- Sequence Generation Functions (Output NumPy arrays) ---
def create_sequences_for_person_py(person_group_tuple, seq_len, features, pad_val, date_col, weight_col):
    """
    Generates sequences for a single person's data. Ensures chronological order.
    Returns list of tuples [(sequence_array, target_value, target_date, target_weight), ...], or None.
    """
    person_id, person_df = person_group_tuple
    # Data should be pre-sorted by date within the group
    n_obs = len(person_df)

    # Need at least 2 observations for one sequence/target pair
    if n_obs <= 1:
        return None

    # Extract data efficiently
    person_features = person_df[features].values.astype(np.float32)
    person_targets = person_df['target_state'].values.astype(np.int64)
    person_dates = person_df[date_col].values
    person_weights = person_df[weight_col].values.astype(np.float32)

    sequences = []
    num_features = len(features)
    padding_matrix = np.full((seq_len, num_features), pad_val, dtype=np.float32) # Pre-allocate padding

    # Iterate to create sequences ending at index i, predicting state at i+1
    for i in range(n_obs - 1):
        end_index = i + 1 # Sequence includes data up to observation i
        start_index = max(0, end_index - seq_len)
        sequence_data = person_features[start_index : end_index, :]

        target = person_targets[i + 1]
        target_date = person_dates[i + 1]
        target_weight = person_weights[i + 1]

        actual_len = sequence_data.shape[0]
        pad_len = seq_len - actual_len

        if pad_len < 0: # Should not happen with max(0, ...)
             print(f"Warning: Negative padding ({pad_len}) for person {person_id}, index {i}. Clipping sequence.")
             padded_sequence = sequence_data[-seq_len:, :]
        elif pad_len == 0:
            padded_sequence = sequence_data
        else: # pad_len > 0
            # Use pre-allocated padding matrix slice for efficiency
            padded_sequence = np.vstack((padding_matrix[:pad_len], sequence_data))

        # Final shape check
        if padded_sequence.shape != (seq_len, num_features):
            print(f"ERROR: Sequence shape mismatch for person {person_id} at index {i}. Expected {(seq_len, num_features)}, got {padded_sequence.shape}. Skipping.")
            continue

        sequences.append((padded_sequence, target, target_date, target_weight))

    return sequences if sequences else None

def generate_sequences_py(baked_data: pd.DataFrame, group_col: str, date_col: str, weight_col: str,
                          seq_len: int, features: list, pad_val: float, n_workers: int):
    """
    Generates sequences from baked data using Python multiprocessing (joblib).

    Args:
        baked_data: DataFrame with preprocessed features, IDs, date, and weight.
        group_col: Name of the individual identifier column.
        date_col: Name of the date column.
        weight_col: Name of the weight column.
        seq_len: Length of the sequences.
        features: List of feature column names.
        pad_val: Value used for padding.
        n_workers: Number of parallel workers.

    Returns:
        Tuple of NumPy arrays: (x_array, y_array, date_array, weight_array)
    """
    if baked_data.empty:
        print("Warning: Input data for sequence generation is empty.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    print(f"Generating sequences with seq_len={seq_len}, pad_val={pad_val} using {n_workers} workers...")

    # Ensure data is sorted for correct sequence generation within groups
    baked_data = baked_data.sort_values([group_col, date_col])

    grouped_data = baked_data.groupby(group_col)
    num_groups = len(grouped_data)

    # Use functools.partial to pre-fill arguments for the worker function
    func = partial(create_sequences_for_person_py, seq_len=seq_len, features=features, pad_val=pad_val, date_col=date_col, weight_col=weight_col)

    try:
        # Use joblib for parallel processing
        # Pass the tuple (name, group) to the delayed function
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(func)((name, group)) for name, group in tqdm(grouped_data, total=num_groups, desc="Processing groups")
        )

        # Filter out None results and flatten the list of lists into a single list of tuples
        flat_results = [item for sublist in results if sublist is not None for item in sublist]

        if not flat_results:
            print("Warning: No valid sequences generated.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Unpack the flattened list of tuples
        x_list, y_list, date_list, weight_list = zip(*flat_results)

        # Stack results into NumPy arrays
        x_array = np.array(x_list, dtype=np.float32)
        y_array = np.array(y_list, dtype=np.int64) # Target should be int64 for CrossEntropyLoss
        date_array = np.array(date_list) # Dates might be objects or specific types
        weight_array = np.array(weight_list, dtype=np.float32)

        print(f"Generated sequences: X shape={x_array.shape}, Y shape={y_array.shape}, Dates shape={date_array.shape}, Weights shape={weight_array.shape}")
        return x_array, y_array, date_array, weight_array

    except Exception as e:
        print(f"ERROR during sequence generation: {e}")
        traceback.print_exc()
        raise # Re-raise the exception after printing traceback

# --- PyTorch Dataset ---
class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence data with padding masks."""
    def __init__(self, x_data, y_data, weight_data, pad_value):
        if not all(isinstance(arr, np.ndarray) for arr in [x_data, y_data, weight_data]):
             raise TypeError("Input data must be NumPy arrays")
        if not (x_data.shape[0] == y_data.shape[0] == weight_data.shape[0]):
             raise ValueError("Input arrays (x, y, weight) must have the same number of samples")

        # Convert NumPy arrays to PyTorch tensors
        self.x_data = torch.from_numpy(x_data.astype(np.float32))
        self.y_data = torch.from_numpy(y_data.astype(np.int64)) # Targets as Long
        self.weight_data = torch.from_numpy(weight_data.astype(np.float32))
        self.pad_value = pad_value
        self.seq_len = x_data.shape[1]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        w = self.weight_data[idx]

        # Create padding mask: True for padded time steps, False for real data
        # Mask shape: (seq_len,) for transformer's src_key_padding_mask
        # A time step is padded if ALL features at that step match the pad_value
        padding_mask = torch.all(x == self.pad_value, dim=-1) # Check across feature dimension

        # Ensure mask has the correct shape (defensive check)
        if padding_mask.shape != (self.seq_len,):
             print(f"Warning: Unexpected padding mask shape {padding_mask.shape} for index {idx}. Expected ({self.seq_len},). Creating default mask.")
             padding_mask = torch.zeros(self.seq_len, dtype=torch.bool) # Default to no padding

        return x, y, w, padding_mask # Return sequence, target, weight, mask

# --- Data Loading and Preparation ---
def load_and_prepare_data(train_file: Path, val_file: Path, hpt_val_file: Path, metadata_file: Path, date_col: str, group_id_col: str):
    """Loads data, metadata, performs checks. Returns train, val, hpt_val (optional), metadata, features, n_features, n_classes."""
    print("\n===== STEP 1: Loading Preprocessed Data & Metadata =====")
    # --- Input Checks ---
    required_files = [train_file, val_file, metadata_file]
    optional_files = [hpt_val_file]
    print("Checking for required input files...")
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required preprocessed input file(s): {missing_files}")
    print("All required input files found.")
    print("Checking for optional files...")
    hpt_val_exists = hpt_val_file and hpt_val_file.exists()
    if hpt_val_exists: print(f" - Found HPT validation data: {hpt_val_file}")
    else: print(f" - Optional HPT validation data not found or path not provided: {hpt_val_file}")

    # --- Load Data ---
    try:
        train_data_baked = pd.read_parquet(train_file)
        val_data_baked = pd.read_parquet(val_file)
        hpt_val_data_baked = pd.read_parquet(hpt_val_file) if hpt_val_exists else None
        with open(metadata_file, 'rb') as f: metadata = pickle.load(f)
        print(f"Loaded baked data: train {train_data_baked.shape}, val {val_data_baked.shape}")
        if hpt_val_data_baked is not None: print(f"Loaded HPT validation data: {hpt_val_data_baked.shape}")
        print("Loaded metadata.")
    except Exception as e:
        print(f"ERROR loading data/metadata: {e}")
        traceback.print_exc()
        raise

    # --- Extract and Validate Metadata ---
    try:
        feature_names = metadata['feature_names']
        n_features = metadata['n_features']
        n_classes = metadata['n_classes']
        target_col = 'target_state'
        required_cols_in_data = [target_col, date_col, group_id_col]
        # Check required columns in train/val
        for df_name, df in [("train", train_data_baked), ("val", val_data_baked)]:
             missing = [col for col in required_cols_in_data if col not in df.columns]
             if missing: raise ValueError(f"Column(s) {missing} not found in {df_name} data.")
        # Check in optional HPT val data
        if hpt_val_data_baked is not None:
             missing_hpt = [col for col in required_cols_in_data if col not in hpt_val_data_baked.columns]
             if missing_hpt: print(f"Warning: Column(s) {missing_hpt} not found in HPT validation data.")

        print(f" - n_features: {n_features}, n_classes: {n_classes}")
        if n_classes != 3: print(f"WARNING: Loaded n_classes = {n_classes}. Ensure this is expected.")
    except KeyError as e:
        raise KeyError(f"Metadata file {metadata_file} is missing required key: {e}")
    except Exception as e:
         print(f"ERROR processing metadata: {e}")
         traceback.print_exc()
         raise

    # --- Convert Date Columns ---
    try:
        train_data_baked[date_col] = pd.to_datetime(train_data_baked[date_col])
        val_data_baked[date_col] = pd.to_datetime(val_data_baked[date_col])
        if hpt_val_data_baked is not None:
             hpt_val_data_baked[date_col] = pd.to_datetime(hpt_val_data_baked[date_col])
    except Exception as e:
        raise ValueError(f"ERROR converting date column '{date_col}' to datetime: {e}")

    # --- Check for Empty DataFrames ---
    if train_data_baked.empty or val_data_baked.empty:
         raise ValueError("Train or validation data is empty after loading.")
    if hpt_val_data_baked is not None and hpt_val_data_baked.empty:
         print("Warning: HPT validation data is empty after loading.")

    return train_data_baked, val_data_baked, hpt_val_data_baked, metadata, feature_names, n_features, n_classes

def setup_sequence_generation(hparams, train_data_baked, val_data_baked, hpt_val_data_baked, processed_data_dir, group_id_col, date_col, feature_names, n_features, weight_col, sequence_cache_dir_name):
    """Handles sequence generation including caching and parallelization for train, val, and HPT val splits."""
    print("\n===== STEP 2: Generating Sequences =====")
    parallel_workers = hparams['parallel_workers']
    refresh_sequences = hparams['refresh_sequences']
    pad_value = hparams['pad_value']
    seq_len = hparams['sequence_length']

    # Determine number of workers dynamically if needed
    if parallel_workers <= 0:
        try:
            num_cores = multiprocessing.cpu_count()
            parallel_workers = max(1, num_cores - 1) # Leave one core free
            print(f"Parallel workers dynamically set to {parallel_workers} (detected {num_cores} cores)")
        except NotImplementedError:
            print("Could not detect CPU count. Setting parallel workers to 1.")
            parallel_workers = 1
    else:
        print(f"Using specified parallel workers for sequence generation: {parallel_workers}")

    # --- Sequence cache setup ---
    sequence_cache_base_dir = processed_data_dir / sequence_cache_dir_name
    sequence_cache_dir = sequence_cache_base_dir / f"seqlen_{seq_len}"
    sequence_cache_dir.mkdir(parents=True, exist_ok=True)

    # Helper to get unique individuals count safely
    def count_individuals(df, col):
        return df[col].nunique() if df is not None and not df.empty else 0

    # Construct cache filenames including individual counts
    num_train_ind = count_individuals(train_data_baked, group_id_col)
    num_val_ind = count_individuals(val_data_baked, group_id_col)
    num_hpt_val_ind = count_individuals(hpt_val_data_baked, group_id_col)

    train_seq_cache_file = sequence_cache_dir / f"train_sequences_N{num_train_ind}.npz"
    val_seq_cache_file = sequence_cache_dir / f"val_sequences_N{num_val_ind}.npz"
    hpt_val_seq_cache_file = sequence_cache_dir / f"hpt_val_sequences_N{num_hpt_val_ind}.npz"
    print(f"Using cache directory: {sequence_cache_dir}")
    print(f" - Train cache file: {train_seq_cache_file.name}")
    print(f" - Val cache file: {val_seq_cache_file.name}")
    print(f" - HPT Val cache file: {hpt_val_seq_cache_file.name}")

    # Helper function for sequence generation/caching for a split
    def process_split_py(split_name, baked_data, cache_file):
        # Define empty arrays with correct shapes
        empty_shape_x = (0, seq_len, n_features)
        empty_x = np.zeros(empty_shape_x, dtype=np.float32)
        empty_y = np.zeros(0, dtype=np.int64)
        empty_dates = np.zeros(0, dtype='datetime64[ns]')
        empty_weights = np.zeros(0, dtype=np.float32)

        # Attempt to load from cache if not refreshing
        if not refresh_sequences and cache_file.exists():
            print(f"Attempting to load cached sequences for '{split_name}' from: {cache_file}")
            try:
                cache_data = np.load(cache_file)
                # Check if all required keys exist
                required_keys = ['x', 'y', 'dates', 'weights']
                if not all(key in cache_data for key in required_keys):
                     print("Cached data missing required keys ('x', 'y', 'dates', 'weights'). Regenerating.")
                else:
                    x_array = cache_data['x']
                    y_array = cache_data['y']
                    date_array = cache_data['dates']
                    weight_array = cache_data['weights']
                    # Validate shapes and dimensions
                    if x_array.ndim == 3 and x_array.shape[1] == seq_len and x_array.shape[2] == n_features and \
                       y_array.ndim == 1 and date_array.ndim == 1 and weight_array.ndim == 1 and \
                       x_array.shape[0] == y_array.shape[0] == date_array.shape[0] == weight_array.shape[0]:
                        print(f"Loaded {x_array.shape[0]} sequences from cache. Shapes: X={x_array.shape}, y={y_array.shape}, dates={date_array.shape}, weights={weight_array.shape}")
                        return x_array, y_array, date_array, weight_array
                    else:
                        print(f"Cached sequence dimensions mismatch parameters. Regenerating.")
            except Exception as e:
                print(f"Error loading or validating cache file {cache_file}: {e}. Regenerating.")

        # Generate sequences if cache loading failed or refresh is True
        print(f"Generating sequences for '{split_name}'...")
        if baked_data is None or baked_data.empty:
             print(f"Data for '{split_name}' is empty. Skipping sequence generation.")
             return empty_x, empty_y, empty_dates, empty_weights

        x_array, y_array, date_array, weight_array = generate_sequences_py(
            baked_data, group_col=group_id_col, date_col=date_col, weight_col=weight_col,
            seq_len=seq_len, features=feature_names, pad_val=pad_value, n_workers=parallel_workers
        )

        # Check for stop signal after generation
        if stop_training_flag:
            print(f"Stop signal detected after generating sequences for '{split_name}'.")
            raise KeyboardInterrupt # Propagate interruption

        # Save generated sequences to cache if any were generated
        if x_array.shape[0] > 0:
             print(f"Saving generated sequences for '{split_name}' ({x_array.shape[0]} sequences) to cache: {cache_file}")
             try:
                 np.savez_compressed(cache_file, x=x_array, y=y_array, dates=date_array, weights=weight_array)
             except Exception as e: print(f"Error saving sequence cache {cache_file}: {e}")
        else: print(f"No sequences generated for '{split_name}', cache not saved.")
        return x_array, y_array, date_array, weight_array

    try:
        # Process train, validation, and optionally HPT validation splits
        x_train_np, y_train_np, date_train_np, weight_train_np = process_split_py("train", train_data_baked, train_seq_cache_file)
        if stop_training_flag: raise KeyboardInterrupt
        x_val_np, y_val_np, date_val_np, weight_val_np = process_split_py("validation", val_data_baked, val_seq_cache_file)
        if stop_training_flag: raise KeyboardInterrupt
        x_hpt_val_np, y_hpt_val_np, date_hpt_val_np, weight_hpt_val_np = process_split_py("hpt_validation", hpt_val_data_baked, hpt_val_seq_cache_file)
        if stop_training_flag: raise KeyboardInterrupt

    except KeyboardInterrupt:
         print("Sequence generation interrupted.")
         raise # Re-raise to be caught by the main script
    except Exception as e:
        print(f"ERROR during sequence generation: {e}")
        traceback.print_exc()
        raise

    # Check if essential splits are empty
    if x_train_np.shape[0] == 0: raise ValueError("No training sequences were generated or loaded.")
    if x_val_np.shape[0] == 0: raise ValueError("No validation sequences were generated or loaded.")
    if x_hpt_val_np.shape[0] == 0: print("Warning: No HPT validation sequences were generated or loaded.")

    return x_train_np, y_train_np, date_train_np, weight_train_np, \
           x_val_np, y_val_np, date_val_np, weight_val_np, \
           x_hpt_val_np, y_hpt_val_np, date_hpt_val_np, weight_hpt_val_np, \
           parallel_workers

def period_to_date(period):
    """Converts YYYYMM integer to datetime.date object. Returns None on failure."""
    if pd.isna(period): return None
    try:
        period_int = int(period)
        year = period_int // 100
        month = period_int % 100
        if 1 <= month <= 12:
            return datetime(year, month, 1).date() # Return date object
        else:
            # print(f"Warning: Invalid month encountered in period {period}. Returning None.")
            return None
    except (ValueError, TypeError):
        # print(f"Warning: Could not convert period '{period}' to date. Returning None.")
        return None
