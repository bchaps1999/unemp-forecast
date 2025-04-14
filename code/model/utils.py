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
    if not stop_training_flag: # Prevent multiple messages if Ctrl+C is pressed again
        print("\nCtrl+C detected! Attempting graceful stop after current epoch/operation...")
        print("Press Ctrl+C again to force exit (may corrupt state).")
        stop_training_flag = True
    else:
        print("Second Ctrl+C detected. Exiting forcefully.")
        sys.exit(1) # Force exit on second Ctrl+C

# --- Worker Initialization Function ---
def worker_init_fn(worker_id):
    """Initialize workers to ignore KeyboardInterrupt and set a different random seed for each."""
    # Ignore SIGINT in workers so they don't capture the KeyboardInterrupt
    import signal
    import os
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Set environment variable to disable MallocStackLogging warnings on macOS
    os.environ['PYTHONMALLOC'] = 'default'
    
    # Set different random seeds for each worker
    import numpy as np
    import random
    import torch
    
    # Use base_seed + worker_id as the seed for this worker
    base_seed = torch.initial_seed() % (2**32)  # Get base seed from PyTorch
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
    
    # Set PyTorch seed for this worker
    torch.manual_seed(base_seed + worker_id)

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

# --- Sequence Generation Functions (Output NumPy arrays) ---
def create_sequences_for_person_py(person_group_tuple, seq_len, features, pad_val, date_col, weight_col):
    """
    Generates sequences for a single person's data. Ensures chronological order.
    Args:
        person_group_tuple: A tuple (person_id, person_df) from pandas groupby.
        seq_len: Length of sequences.
        features: List of feature column names.
        pad_val: Value used for padding.
        date_col: Name of the date column for sorting.
        weight_col: Name of the weight column.
    Returns:
        List of tuples [(sequence_array, target_value, target_date, target_weight), ...], or None if too short.
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
    person_dates = person_df[date_col].values # Get dates
    person_weights = person_df[weight_col].values.astype(np.float32) # Get weights

    sequences = []
    # Iterate up to the second-to-last observation to have a target for the sequence ending at index i
    for i in range(n_obs - 1):
        # Sequence ends at index i, target is at index i+1
        end_index = i + 1 # Exclusive index for slicing features
        start_index = max(0, end_index - seq_len)
        sequence_data = person_features[start_index : end_index, :] # Slice features up to current obs i

        target = person_targets[i + 1] # Target is the state at the *next* time step (i+1)
        target_date = person_dates[i + 1] # Date of the target observation
        target_weight = person_weights[i + 1] # Weight of the target observation

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

        sequences.append((padded_sequence, target, target_date, target_weight)) # Add target_weight

    return sequences if sequences else None # Return None if no valid sequences were generated for the person

def generate_sequences_py(data_df, group_col, seq_len, features, pad_val, date_col, weight_col, n_workers):
    """
    Generates sequences in parallel using joblib. Returns X, y, target dates, and target weights.
    """
    if not features:
        raise ValueError("Feature list cannot be empty for sequence generation.")
    if group_col not in data_df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")
    if date_col not in data_df.columns:
         raise ValueError(f"Date column '{date_col}' not found in dataframe.")
    if 'target_state' not in data_df.columns:
         raise ValueError(f"Target column 'target_state' not found in dataframe.")
    if weight_col not in data_df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in dataframe.")

    print(f"Generating sequences with seq_len={seq_len}, pad_val={pad_val} using {n_workers} workers...")
    start_time = time.time()

    # Group data by person
    grouped_data = data_df.groupby(group_col)
    num_groups = len(grouped_data)
    if num_groups == 0:
        print("Warning: No groups found in the data.")
        # Return empty arrays with correct dimensions
        return np.array([], dtype=np.float32).reshape(0, seq_len, len(features)), \
               np.array([], dtype=np.int64), \
               np.array([], dtype='datetime64[ns]'), \
               np.array([], dtype=np.float32) # Empty weight array

    # partial applies fixed arguments to the function
    func = partial(create_sequences_for_person_py, seq_len=seq_len, features=features, pad_val=pad_val, date_col=date_col, weight_col=weight_col)

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
        return np.array([], dtype=np.float32).reshape(0, seq_len, len(features)), \
               np.array([], dtype=np.int64), \
               np.array([], dtype='datetime64[ns]'), \
               np.array([], dtype=np.float32)

    # Unzip sequences, targets, dates, and weights
    x_list, y_list, date_list, weight_list = zip(*all_sequences)

    # Convert to NumPy arrays with specific dtypes
    x_array = np.array(x_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.int64) # Use int64 for targets -> LongTensor
    date_array = np.array(date_list, dtype='datetime64[ns]') # Store dates
    weight_array = np.array(weight_list, dtype=np.float32) # Store weights

    print(f"Generated {x_array.shape[0]} sequences.")
    print(f"Shape of X: {x_array.shape}") # Should be (num_sequences, seq_len, n_features)
    print(f"Shape of y: {y_array.shape}") # Should be (num_sequences,)
    print(f"Shape of dates: {date_array.shape}") # Should be (num_sequences,)
    print(f"Shape of weights: {weight_array.shape}") # Print weight shape

    return x_array, y_array, date_array, weight_array # Return weights as well

# --- PyTorch Dataset ---
class SequenceDataset(Dataset):
    def __init__(self, x_data, y_data, weight_data, pad_value):
        if not isinstance(x_data, np.ndarray) or \
           not isinstance(y_data, np.ndarray) or \
           not isinstance(weight_data, np.ndarray):
             raise TypeError("Input data must be NumPy arrays")
        if x_data.shape[0] != y_data.shape[0] or x_data.shape[0] != weight_data.shape[0]:
             raise ValueError("Input arrays (x, y, weight) must have the same number of samples")

        # Convert NumPy arrays to PyTorch tensors upon initialization
        self.x_data = torch.from_numpy(x_data.astype(np.float32))
        # Ensure target type is Long (Int64) for CrossEntropyLoss
        self.y_data = torch.from_numpy(y_data.astype(np.int64))
        self.weight_data = torch.from_numpy(weight_data.astype(np.float32)) # Store weights
        self.pad_value = pad_value
        self.seq_len = x_data.shape[1] # Store sequence length

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        w = self.weight_data[idx] # Get weight

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

        return x, y, w, padding_mask # Return weight

# --- Data Loading and Preparation ---
def load_and_prepare_data(train_file, val_file, hpt_val_file, metadata_file, date_col, group_id_col, train_start_date=None, train_end_date=None):
    """Loads data, metadata, performs checks, and applies date filters."""
    print("\n===== STEP 1: Loading Preprocessed Data & Metadata =====")
    # --- Input Checks ---
    required_files = [train_file, val_file, metadata_file]
    optional_files = [hpt_val_file] # HPT val file is optional for standard runs
    print("Checking for required input files...")
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("ERROR: Missing required preprocessed input file(s):")
        for f in missing_files: print(f" - {f}")
        raise FileNotFoundError(f"Missing input files: {missing_files}")
    print("All required input files found.")
    print("Checking for optional files...")
    if hpt_val_file and hpt_val_file.exists():
        print(f" - Found HPT validation data: {hpt_val_file}")
    else:
        print(f" - Optional HPT validation data not found or path not provided: {hpt_val_file}")

    # --- Load Data ---
    try:
        train_data_baked = pd.read_parquet(train_file)
        val_data_baked = pd.read_parquet(val_file)
        hpt_val_data_baked = None
        if hpt_val_file and hpt_val_file.exists():
            hpt_val_data_baked = pd.read_parquet(hpt_val_file)
        with open(metadata_file, 'rb') as f: metadata = pickle.load(f)
        print("Loaded baked data (train, val) & metadata.")
        if hpt_val_data_baked is not None:
            print(f"Loaded HPT validation data: {hpt_val_data_baked.shape}")
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
        if hpt_val_data_baked is not None and target_col not in hpt_val_data_baked.columns:
             print(f"Warning: Target column '{target_col}' not found in HPT validation data.")
             # Decide if this is fatal or just a warning
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
        if hpt_val_data_baked is not None:
             hpt_val_data_baked[date_col] = pd.to_datetime(hpt_val_data_baked[date_col])
    except Exception as e:
        print(f"ERROR converting date column '{date_col}' to datetime: {e}")
        traceback.print_exc()
        raise

    initial_train_rows = len(train_data_baked)
    initial_val_rows = len(val_data_baked)
    initial_hpt_val_rows = len(hpt_val_data_baked) if hpt_val_data_baked is not None else 0

    if train_start_date:
        try:
            train_start_dt = pd.to_datetime(train_start_date)
            train_data_baked = train_data_baked[train_data_baked[date_col] >= train_start_dt].copy()
            val_data_baked = val_data_baked[val_data_baked[date_col] >= train_start_dt].copy()
            # Apply start date to HPT val data as well (though less common use case)
            if hpt_val_data_baked is not None:
                hpt_val_data_baked = hpt_val_data_baked[hpt_val_data_baked[date_col] >= train_start_dt].copy()
            print(f"Applied TRAIN_START_DATE >= {train_start_date}")
        except Exception as e:
            print(f"Warning: Could not apply train_start_date filter: {e}")

    if train_end_date:
        try:
            train_end_dt = pd.to_datetime(train_end_date)
            train_data_baked = train_data_baked[train_data_baked[date_col] <= train_end_dt].copy()
            val_data_baked = val_data_baked[val_data_baked[date_col] <= train_end_dt].copy()
            # Apply end date filter to HPT validation data
            if hpt_val_data_baked is not None:
                hpt_val_data_baked = hpt_val_data_baked[hpt_val_data_baked[date_col] <= train_end_dt].copy()
            print(f"Applied TRAIN_END_DATE <= {train_end_date}")
        except Exception as e:
            print(f"Warning: Could not apply train_end_date filter: {e}")

    print(f"Train data rows: {initial_train_rows} -> {len(train_data_baked)}")
    print(f"Validation data rows: {initial_val_rows} -> {len(val_data_baked)}")
    if hpt_val_data_baked is not None:
        print(f"HPT Validation data rows: {initial_hpt_val_rows} -> {len(hpt_val_data_baked)}") # Show filtered count
    if len(train_data_baked) == 0 or len(val_data_baked) == 0:
         raise ValueError("No data remaining after date filtering for train/val splits.")
    if hpt_val_data_baked is not None and len(hpt_val_data_baked) == 0:
         print("Warning: No data remaining in HPT validation set after date filtering.")

    return train_data_baked, val_data_baked, hpt_val_data_baked, metadata, feature_names, n_features, n_classes

def setup_sequence_generation(hparams, train_data_baked, val_data_baked, hpt_val_data_baked, processed_data_dir, group_id_col, date_col, feature_names, n_features, weight_col, sequence_cache_dir_name):
    """Handles sequence generation including caching and parallelization for train, val, and HPT val splits. Returns X, y, dates, and weights."""
    print("\n===== STEP 2: Generating Sequences =====")
    parallel_workers = hparams['parallel_workers']
    refresh_sequences = hparams['refresh_sequences']
    pad_value = hparams['pad_value']
    seq_len = hparams['sequence_length']

    if parallel_workers <= 0:
        try:
            num_cores = multiprocessing.cpu_count()
            # Use num_cores - 1 to leave one core for the main process/OS, ensure at least 1 worker
            parallel_workers = max(1, num_cores - 1)
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
    train_seq_cache_file = sequence_cache_dir / "train_sequences.npz"
    val_seq_cache_file = sequence_cache_dir / "val_sequences.npz"
    hpt_val_seq_cache_file = sequence_cache_dir / "hpt_val_sequences.npz" # Cache for HPT val

    # Helper function for sequence generation/caching for a split
    def process_split_py(split_name, baked_data, cache_file):
        empty_x = np.array([], dtype=np.float32).reshape(0, seq_len, n_features)
        empty_y = np.array([], dtype=np.int64)
        empty_dates = np.array([], dtype='datetime64[ns]')
        empty_weights = np.array([], dtype=np.float32)

        if not refresh_sequences and cache_file.exists():
            print(f"Attempting to load cached sequences for '{split_name}' from: {cache_file}")
            try:
                cache_data = np.load(cache_file)
                x_array = cache_data['x']
                y_array = cache_data['y']
                # Load dates if they exist in the cache
                date_array = cache_data.get('dates')
                weight_array = cache_data.get('weights') # Load weights
                if date_array is None or weight_array is None:
                     print("Cached data missing 'dates' or 'weights'. Regenerating.")
                elif x_array.ndim == 3 and x_array.shape[1] == seq_len and x_array.shape[2] == n_features and \
                     y_array.ndim == 1 and date_array.ndim == 1 and weight_array.ndim == 1 and \
                     x_array.shape[0] == y_array.shape[0] == date_array.shape[0] == weight_array.shape[0]:
                    print(f"Loaded {x_array.shape[0]} sequences from cache. Shape X: {x_array.shape}, y: {y_array.shape}, dates: {date_array.shape}, weights: {weight_array.shape}")
                    return x_array, y_array, date_array, weight_array
                else:
                    print(f"Cached sequence dimensions mismatch parameters or missing dates/weights. Regenerating.")
            except Exception as e:
                print(f"Error loading or validating cache file {cache_file}: {e}. Regenerating.")

        print(f"Generating sequences for '{split_name}'...")
        if baked_data is None or baked_data.empty:
             print(f"Data for '{split_name}' is empty. Skipping sequence generation.")
             return empty_x, empty_y, empty_dates, empty_weights

        # Generate sequences including dates and weights
        x_array, y_array, date_array, weight_array = generate_sequences_py(
            baked_data, group_col=group_id_col, date_col=date_col, weight_col=weight_col,
            seq_len=seq_len, features=feature_names, pad_val=pad_value, n_workers=parallel_workers
        )

        if stop_training_flag:
            print(f"Stop signal detected after generating sequences for '{split_name}'.")
            raise KeyboardInterrupt

        if x_array.shape[0] > 0:
             print(f"Saving generated sequences for '{split_name}' ({x_array.shape[0]} sequences) to cache: {cache_file}")
             try:
                 # Save weights along with X, y, and dates
                 np.savez_compressed(cache_file, x=x_array, y=y_array, dates=date_array, weights=weight_array)
             except Exception as e: print(f"Error saving sequence cache {cache_file}: {e}")
        else: print(f"No sequences generated for '{split_name}', cache not saved.")
        return x_array, y_array, date_array, weight_array

    try:
        x_train_np, y_train_np, date_train_np, weight_train_np = process_split_py("train", train_data_baked, train_seq_cache_file)
        if stop_training_flag: raise KeyboardInterrupt
        x_val_np, y_val_np, date_val_np, weight_val_np = process_split_py("validation", val_data_baked, val_seq_cache_file)
        if stop_training_flag: raise KeyboardInterrupt
        # Generate sequences for HPT validation set including dates
        x_hpt_val_np, y_hpt_val_np, date_hpt_val_np, weight_hpt_val_np = process_split_py("hpt_validation", hpt_val_data_baked, hpt_val_seq_cache_file)
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
    if x_hpt_val_np.shape[0] == 0: print("Warning: No HPT validation sequences were generated or loaded.") # Allow continuation

    # Return all generated sequences, dates, and weights
    return x_train_np, y_train_np, date_train_np, weight_train_np, \
           x_val_np, y_val_np, date_val_np, weight_val_np, \
           x_hpt_val_np, y_hpt_val_np, date_hpt_val_np, weight_hpt_val_np, \
           parallel_workers

def period_to_date(period):
    """Converts YYYYMM integer to datetime.date object for plotting."""
    if pd.isna(period): return None
    try:
        period_int = int(period)
        year = period_int // 100
        month = period_int % 100
        if 1 <= month <= 12:
            # Return as datetime object first, then extract date part
            return datetime(year, month, 1).date()
        else:
            print(f"Warning: Invalid month encountered in period {period}. Skipping.")
            return None # Invalid month
    except (ValueError, TypeError):
        print(f"Warning: Could not convert period '{period}' to date. Skipping.")
        return None
