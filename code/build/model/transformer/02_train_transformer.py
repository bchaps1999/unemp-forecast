#--- 0. Load Libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
# Added WeightedRandomSampler
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
import math # For positional encoding
import sys
import multiprocessing # Added for dynamic worker count

# Fix imports - add the parent directory to sys.path
# This allows absolute imports to work properly
sys.path.append(str(Path(__file__).parent))

# Import Model Definitions with absolute import
from models import PositionalEmbedding, TransformerEncoderBlock, TransformerForecastingModel

# --- Import Config --- # Added
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Make sure it's in the same directory or sys.path is configured correctly.")
    sys.exit(1)

# --- PyTorch Device Setup ---
def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

DEVICE = get_device()

# --- Sequence Generation Functions (Keep as is, output NumPy arrays) ---
def create_sequences_for_person_py(person_group_tuple, seq_len, features, pad_val):
    """
    Generates sequences for a single person's data.
    Args:
        person_group_tuple: A tuple (person_id, person_df) from pandas groupby.
        seq_len: Length of sequences.
        features: List of feature column names.
        pad_val: Value used for padding.
    Returns:
        List of tuples [(sequence_array, target_value), ...], or None if too short.
    """
    person_id, person_df = person_group_tuple
    person_df = person_df.sort_values(by='date') # Ensure chronological order
    n_obs = len(person_df)

    if n_obs <= 1: # Need at least 2 observations for one sequence/target pair
        return None

    person_features = person_df[features].values.astype(np.float32)
    person_targets = person_df['target_state'].values.astype(np.int32) # Assuming integer targets

    sequences = []
    for i in range(n_obs - 1): # Iterate up to the second-to-last observation
        start_index = max(0, i - seq_len + 1)
        sequence_data = person_features[start_index : i + 1, :] # Slice includes current obs i
        target = person_targets[i + 1] # Target is the state at the *next* time step

        actual_len = sequence_data.shape[0]
        pad_len = seq_len - actual_len

        if pad_len < 0: # Should not happen if start_index logic is correct
             print(f"Warning: Negative padding for person {person_id}, index {i}?")
             padded_sequence = sequence_data[-seq_len:, :] # Take the last seq_len elements
        elif pad_len == 0:
            padded_sequence = sequence_data
        else: # pad_len > 0
            padding_matrix = np.full((pad_len, len(features)), pad_val, dtype=np.float32)
            padded_sequence = np.vstack((padding_matrix, sequence_data))

        # Final check for shape consistency
        if padded_sequence.shape != (seq_len, len(features)):
            print(f"ERROR: Sequence shape mismatch for person {person_id} at index {i}. Expected {(seq_len, len(features))}, got {padded_sequence.shape}")
            # Handle error: skip sequence, use fallback, etc. For now, skip.
            continue

        sequences.append((padded_sequence, target))

    return sequences

def generate_sequences_py(data_df, group_col, seq_len, features, pad_val, n_workers):
    """
    Generates sequences in parallel using joblib.
    """
    print(f"Generating sequences with seq_len={seq_len}, pad_val={pad_val} using {n_workers} workers...")
    start_time = time.time()

    # Group data by person
    grouped_data = data_df.groupby(group_col)
    num_groups = len(grouped_data)

    # Use joblib for parallel processing with progress bar (requires tqdm installed)
    # partial applies fixed arguments (seq_len, features, pad_val) to the function
    func = partial(create_sequences_for_person_py, seq_len=seq_len, features=features, pad_val=pad_val)

    # Process groups in parallel
    # n_jobs=-1 uses all available cores, adjust if needed
    # backend="loky" is often more robust than "multiprocessing"
    results = Parallel(n_jobs=n_workers, backend="loky")( # Changed backend
        delayed(func)(group) for group in tqdm(grouped_data, total=num_groups, desc="Processing groups")
    )

    # Flatten the list of lists and filter out None results
    all_sequences = []
    for person_result in results:
        if person_result: # Check if not None
            all_sequences.extend(person_result)

    end_time = time.time()
    print(f"Sequence generation took {end_time - start_time:.2f} seconds.")

    if not all_sequences:
        print("Warning: No sequences were generated.")
        return np.array([], dtype=np.float32).reshape(0, seq_len, len(features)), np.array([], dtype=np.int32)

    # Unzip sequences and targets
    x_list, y_list = zip(*all_sequences)

    # Convert to NumPy arrays
    x_array = np.array(x_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.int32)

    print(f"Generated {x_array.shape[0]} sequences.")
    print(f"Shape of X: {x_array.shape}")
    print(f"Shape of y: {y_array.shape}")

    return x_array, y_array

# --- PyTorch Dataset ---
class SequenceDataset(Dataset):
    def __init__(self, x_data, y_data, pad_value):
        self.x_data = torch.from_numpy(x_data.astype(np.float32))
        self.y_data = torch.from_numpy(y_data.astype(np.int64)) # CrossEntropyLoss expects Long type
        self.pad_value = pad_value

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        # Create padding mask: True where value is pad_value
        # Check across all features in the last dimension
        padding_mask = torch.all(x == self.pad_value, dim=-1)
        return x, y, padding_mask

# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Set tqdm to disable in worker processes and be less verbose
    for x_batch, y_batch, mask_batch in tqdm(dataloader, desc="Training", leave=False, 
                                            disable=not torch.utils.data.get_worker_info() is None):
        x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch, src_key_padding_mask=mask_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += y_batch.size(0)
        correct_predictions += (predicted == y_batch).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        # Set tqdm to disable in worker processes and be less verbose
        for x_batch, y_batch, mask_batch in tqdm(dataloader, desc="Evaluating", leave=False, 
                                                disable=not torch.utils.data.get_worker_info() is None):
            x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

            outputs = model(x_batch, src_key_padding_mask=mask_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += y_batch.size(0)
            correct_predictions += (predicted == y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy, np.array(all_targets), np.array(all_preds)


# --- Main Pipeline Function (PyTorch) --- # Modified signature
def run_transformer_pipeline():
    # --- Start --- Use config parameters
    start_run_time = time.time()
    print("=" * 60)
    print("Starting Transformer Model Pipeline (PyTorch)")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Parameters (from config.py):")
    print(f"  - sequence_length: {config.SEQUENCE_LENGTH}, embed_dim: {config.EMBED_DIM}, num_heads: {config.NUM_HEADS}")
    print(f"  - ff_dim: {config.FF_DIM}, num_transformer_blocks: {config.NUM_TRANSFORMER_BLOCKS}, mlp_units: {config.MLP_UNITS}")
    print(f"  - dropout: {config.DROPOUT}, mlp_dropout: {config.MLP_DROPOUT}, epochs: {config.EPOCHS}, batch_size: {config.BATCH_SIZE}")
    print(f"  - pad_value: {config.PAD_VALUE}, learning_rate: {config.LEARNING_RATE}")
    print(f"  - refresh_model: {config.REFRESH_MODEL}, refresh_sequences: {config.REFRESH_SEQUENCES}")
    print(f"  - parallel_workers (config): {config.PARALLEL_WORKERS}")
    print(f"  - use_weighted_sampling: {config.USE_WEIGHTED_SAMPLING}") # Log new config
    print(f"  - train_start_date: {config.TRAIN_START_DATE}, train_end_date: {config.TRAIN_END_DATE}") # Log new config
    print(f"  - use_weighted_loss: {config.USE_WEIGHTED_LOSS}") # Log new config
    print("-" * 60)

    # --- Setup Output --- Use config paths
    processed_data_dir = config.PREPROCESS_OUTPUT_DIR
    model_output_dir = config.TRAIN_OUTPUT_SUBDIR
    model_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model output directory: {model_output_dir}")
    model_file = model_output_dir / "transformer_model.pt" # PyTorch model state_dict
    history_file = model_output_dir / "training_history.pkl"
    params_file = model_output_dir / "model_params.pkl"
    plot_file = model_output_dir / "training_history_plot.png"
    checkpoint_path = model_output_dir / "best_model.pt"

    # Sequence cache setup - Use config paths
    sequence_cache_dir = processed_data_dir / config.SEQUENCE_CACHE_DIR_NAME
    sequence_cache_dir.mkdir(parents=True, exist_ok=True)
    train_seq_cache_file = sequence_cache_dir / "train_sequences.npz"
    val_seq_cache_file = sequence_cache_dir / "val_sequences.npz"
    test_seq_cache_file = sequence_cache_dir / "test_sequences.npz"

    # --- Input File Paths --- Use config paths
    train_file = processed_data_dir / config.TRAIN_DATA_FILENAME
    val_file = processed_data_dir / config.VAL_DATA_FILENAME
    test_file = processed_data_dir / config.TEST_DATA_FILENAME
    metadata_file = processed_data_dir / config.METADATA_FILENAME
    recipe_file = processed_data_dir / config.RECIPE_FILENAME

    # --- Input Checks ---
    required_files = [train_file, val_file, test_file, metadata_file]
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("ERROR: Missing required preprocessed input file(s):")
        for f in missing_files: print(f" - {f}")
        raise FileNotFoundError("Missing input files.")
    print("All required input files found.")

    # --- Check for Existing Model --- Use config flag
    if not config.REFRESH_MODEL and model_file.exists():
        print("\n===== Existing Model File Found =====")
        print(f"Found existing model file: {model_file}")
        print("Pipeline will terminate early. To retrain, use --refresh_model.")
        # Load params to return paths correctly if needed
        saved_params = {}
        if params_file.exists():
            try:
                with open(params_file, 'rb') as f: saved_params = pickle.load(f)
            except Exception as e: print(f"Warning: Could not load params file {params_file}: {e}")

        return {
            "model_path": str(model_file),
            "history_path": str(history_file) if history_file.exists() else None,
            "params_path": str(params_file) if params_file.exists() else None,
            "recipe_path": saved_params.get("recipe_path", str(recipe_file) if recipe_file.exists() else None),
            "plot_path": str(plot_file) if plot_file.exists() else None
        }
    elif not config.REFRESH_MODEL:
        print("No existing model file found. Training a new model.")
    else:
        print("`refresh_model` is TRUE. Training a new model.")

    # --- Reproducibility --- Use config seed
    seed_value = config.RANDOM_SEED
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # if use multi-GPU
        # Potentially make CuDNN deterministic (can impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print("Random seeds set.")

    # --- 1. Load Preprocessed Data and Metadata ---
    print("\n===== STEP 1: Loading Preprocessed Data & Metadata =====")
    try:
        train_data_baked = pd.read_parquet(train_file)
        val_data_baked = pd.read_parquet(val_file)
        test_data_baked = pd.read_parquet(test_file)
        with open(metadata_file, 'rb') as f: metadata = pickle.load(f)
    except Exception as e:
        print(f"Error loading data/metadata: {e}")
        raise

    # --- Filter Baked Data by Date (Before Sequence Generation) --- # New Step
    print("\nFiltering baked data based on TRAIN_START_DATE and TRAIN_END_DATE...")
    date_col = config.DATE_COL # Use date column from config

    # Ensure date column is datetime type
    try:
        train_data_baked[date_col] = pd.to_datetime(train_data_baked[date_col])
        val_data_baked[date_col] = pd.to_datetime(val_data_baked[date_col])
        # Test data is usually not filtered by training dates, but ensure type consistency
        test_data_baked[date_col] = pd.to_datetime(test_data_baked[date_col])
    except KeyError:
        print(f"ERROR: Date column '{date_col}' not found in baked data.")
        raise
    except Exception as e:
        print(f"ERROR converting date column '{date_col}' to datetime: {e}")
        raise

    initial_train_rows = len(train_data_baked)
    initial_val_rows = len(val_data_baked)

    if config.TRAIN_START_DATE:
        train_start_dt = pd.to_datetime(config.TRAIN_START_DATE)
        train_data_baked = train_data_baked[train_data_baked[date_col] >= train_start_dt]
        val_data_baked = val_data_baked[val_data_baked[date_col] >= train_start_dt]
        print(f"Applied TRAIN_START_DATE >= {config.TRAIN_START_DATE}")

    if config.TRAIN_END_DATE:
        train_end_dt = pd.to_datetime(config.TRAIN_END_DATE)
        train_data_baked = train_data_baked[train_data_baked[date_col] <= train_end_dt]
        val_data_baked = val_data_baked[val_data_baked[date_col] <= train_end_dt]
        print(f"Applied TRAIN_END_DATE <= {config.TRAIN_END_DATE}")

    print(f"Train data rows reduced from {initial_train_rows} to {len(train_data_baked)}")
    print(f"Validation data rows reduced from {initial_val_rows} to {len(val_data_baked)}")
    # --- End Filtering Step ---

    try:
        feature_names = metadata['feature_names']
        n_features = metadata['n_features']
        n_classes = metadata['n_classes']
    except KeyError as e:
        print(f"Error: Metadata file {metadata_file} is missing key: {e}")
        raise

    print("Loaded baked data & metadata.")
    print(f" - n_features: {n_features}, n_classes: {n_classes}")
    if n_classes != 3: print(f"WARNING: Loaded n_classes = {n_classes} (expected 3).")
    target_col = 'target_state'  # Use the known target column name
    if target_col not in train_data_baked.columns:
         raise ValueError(f"Target column '{target_col}' not found in training data.")

    # --- 2. Sequence Generation (with Caching & Parallelization) --- Use config flags/params
    print("\n===== STEP 2: Generating Sequences =====")

    # Determine parallel workers dynamically if needed
    if config.PARALLEL_WORKERS <= 0:
        try:
            num_cores = multiprocessing.cpu_count()
            parallel_workers = max(1, num_cores // 2)
            print(f"Parallel workers for sequence generation set to {parallel_workers} (detected {num_cores} cores, using half)")
        except ImportError:
            print("Could not import multiprocessing. Setting parallel workers to 1.")
            parallel_workers = 1
    else:
        parallel_workers = config.PARALLEL_WORKERS
        print(f"Using specified parallel workers: {parallel_workers}")

    def process_split_py(split_name, baked_data, cache_file):
        if not config.REFRESH_SEQUENCES and cache_file.exists(): # Use config flag
            print(f"Loading cached sequences for '{split_name}' from: {cache_file}")
            try:
                cache_data = np.load(cache_file)
                x_array = cache_data['x']
                y_array = cache_data['y']
                # Use config for sequence length
                if x_array.shape[1] == config.SEQUENCE_LENGTH and x_array.shape[2] == n_features:
                    print(f"Loaded {x_array.shape[0]} sequences from cache.")
                    return x_array, y_array
                else:
                    print("Cached sequence dimensions mismatch parameters. Regenerating.")
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}. Regenerating.")

        print(f"Generating sequences for '{split_name}'...")
        x_array, y_array = generate_sequences_py(
            baked_data, group_col=config.GROUP_ID_COL, # Use config
            seq_len=config.SEQUENCE_LENGTH, # Use config
            features=feature_names,
            pad_val=config.PAD_VALUE, # Use config
            n_workers=parallel_workers # Use determined worker count
        )
        print(f"Saving generated sequences for '{split_name}' to cache: {cache_file}")
        try:
            np.savez_compressed(cache_file, x=x_array, y=y_array)
        except Exception as e:
            print(f"Error saving sequence cache {cache_file}: {e}")
        return x_array, y_array

    x_train_np, y_train_np = process_split_py("train", train_data_baked, train_seq_cache_file)
    x_val_np, y_val_np = process_split_py("validation", val_data_baked, val_seq_cache_file)
    x_test_np, y_test_np = process_split_py("test", test_data_baked, test_seq_cache_file)

    if x_train_np.shape[0] == 0 or x_val_np.shape[0] == 0:
        raise ValueError("Error: No valid training or validation sequences generated/loaded.")
    if x_test_np.shape[0] == 0: print("Warning: Test set contains no sequences.")

    # --- Calculate Sample Weights for WeightedRandomSampler --- # Modified section
    print("\nCalculating sample weights for WeightedRandomSampler...")
    train_sampler = None # Initialize sampler to None
    class_weights_for_sampler_log = np.ones(n_classes) / n_classes # Default log value
    sampling_strategy = "Standard" # Default strategy

    # Only calculate weights and create sampler if enabled in config
    if config.USE_WEIGHTED_SAMPLING:
        if len(y_train_np) > 0:
            class_counts = np.bincount(y_train_np, minlength=n_classes)
            print(f"Training target class counts: {class_counts}")

            # Check if any class has zero count
            if np.any(class_counts == 0):
                print("Warning: One or more classes have zero samples in the training data. WeightedRandomSampler might behave unexpectedly or error.")
                # Handle this case: maybe skip sampling or assign a very small weight?
                # For now, we'll proceed but the weights for zero-count classes will be inf.
                # Let's replace 0 counts with 1 to avoid division by zero, effectively giving them high weight if they appear later.
                class_counts = np.maximum(class_counts, 1)
                print(f"Adjusted class counts (minimum 1): {class_counts}")


            # Compute weight for each class: 1 / (number of samples of that class)
            class_weights_inv = 1. / class_counts
            # Assign weight to each sample based on its class
            sample_weights = class_weights_inv[y_train_np]
            sample_weights_tensor = torch.from_numpy(sample_weights).double() # Sampler expects Double

            print(f"Created sample weights tensor of shape: {sample_weights_tensor.shape}")
            # Create the sampler: samples are drawn with probability proportional to weights
            train_sampler = WeightedRandomSampler(
                weights=sample_weights_tensor,
                num_samples=len(sample_weights_tensor), # Draw this many samples in total per epoch
                replacement=True # Samples are drawn with replacement
            )
            print("WeightedRandomSampler created for training data.")
            # Save the inverse class weights for reference
            class_weights_for_sampler_log = class_weights_inv
            sampling_strategy = "WeightedRandomSampler" # Update strategy name
        else:
            print("Training data is empty, cannot create sampler. Proceeding without sampler.")
    else:
        print("USE_WEIGHTED_SAMPLING is False. Standard shuffling will be used for training.")

    print("\nCreating PyTorch Datasets and DataLoaders...")
    train_dataset = SequenceDataset(x_train_np, y_train_np, config.PAD_VALUE)
    val_dataset = SequenceDataset(x_val_np, y_val_np, config.PAD_VALUE)
    test_dataset = SequenceDataset(x_test_np, y_test_np, config.PAD_VALUE)

    # Use determined parallel_workers for DataLoader num_workers
    dataloader_workers = min(4, parallel_workers)
    # Use sampler for train_loader, set shuffle=False # Modified
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler, # Use the sampler if created
        num_workers=dataloader_workers,
        # shuffle must be False when a sampler is provided
        shuffle=False if train_sampler else True
    )
    # Validation and Test loaders do NOT use the sampler
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=dataloader_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=dataloader_workers)
    print(f"DataLoaders created (using {'WeightedRandomSampler' if train_sampler else 'standard shuffling'} for training).")

    # --- 3. Build Transformer Model --- Use config params
    print("\n===== STEP 3: Building Transformer Model =====")
    model = TransformerForecastingModel(
        input_dim=n_features,
        seq_len=config.SEQUENCE_LENGTH,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM,
        num_transformer_blocks=config.NUM_TRANSFORMER_BLOCKS,
        mlp_units=config.MLP_UNITS,
        dropout=config.DROPOUT,
        mlp_dropout=config.MLP_DROPOUT,
        n_classes=n_classes
    ).to(DEVICE)
    print("Model built successfully. Structure:")
    print(model)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")


    # --- 4. Train the Model --- Use config params
    print("\n===== STEP 4: Training the Model =====")

    # --- Calculate Class Weights for Loss Function --- # New Section
    class_weights_tensor = None
    class_weights_for_loss_log = np.ones(n_classes) / n_classes # Default log value
    loss_weighting_strategy = "Standard"

    if config.USE_WEIGHTED_LOSS:
        print("Calculating class weights for loss function...")
        if len(y_train_np) > 0:
            class_counts = np.bincount(y_train_np, minlength=n_classes)
            print(f"Training target class counts (for loss): {class_counts}")

            if np.any(class_counts == 0):
                print("Warning: One or more classes have zero samples in the training data. Assigning weight 1.0 to avoid division by zero.")
                # Avoid division by zero, assign a default weight (e.g., 1.0) or a very small weight
                # Using inverse count: weight = total_samples / (n_classes * count)
                total_samples = len(y_train_np)
                # Replace 0 counts with 1 for calculation, effectively giving them high weight
                adjusted_counts = np.maximum(class_counts, 1)
                class_weights = total_samples / (n_classes * adjusted_counts)
                print(f"Adjusted counts for weight calculation: {adjusted_counts}")
            else:
                # Standard inverse frequency weighting: weight = total_samples / (n_classes * count)
                total_samples = len(y_train_np)
                class_weights = total_samples / (n_classes * class_counts)

            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
            print(f"Calculated class weights for loss: {class_weights_tensor.cpu().numpy()}")
            class_weights_for_loss_log = class_weights # Save for logging
            loss_weighting_strategy = "WeightedLoss"
        else:
            print("Training data is empty, cannot calculate loss weights. Using standard loss.")
    else:
        print("USE_WEIGHTED_LOSS is False. Using standard CrossEntropyLoss.")
    # --- End Class Weights for Loss ---

    # Initialize Loss Function (potentially with weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # Pass weights tensor (or None)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Use config parameters for scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        factor=config.LR_SCHEDULER_FACTOR, 
        patience=config.LR_SCHEDULER_PATIENCE, 
        verbose=True
    )
    
    # Use config parameter for early stopping
    early_stopping_patience = config.EARLY_STOPPING_PATIENCE
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Use config parameter for gradient clipping
    max_grad_norm = config.MAX_GRAD_NORM
    
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    print(f"Starting training: Max epochs={config.EPOCHS}, Batch size={config.BATCH_SIZE}, LR={config.LEARNING_RATE}")

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, max_grad_norm)
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE) # Ignore preds/targets here
        epoch_end_time = time.time()

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Time: {epoch_end_time - epoch_start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # LR Scheduler Step
        scheduler.step(val_loss)

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model state
            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> Validation loss improved by {improvement:.2f}%. Saved best model to {checkpoint_path}")
            except Exception as e:
                print(f"  -> ERROR saving best model checkpoint: {e}")
        else:
            epochs_no_improve += 1
            print(f"  -> Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print("Training finished.")
    # Load the best weights found during training
    if checkpoint_path.exists():
        print(f"Loading best model weights from {checkpoint_path}")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        except Exception as e:
            print(f"Warning: Could not load best weights from {checkpoint_path}: {e}. Using last weights.")
    else:
        print("Warning: Best model checkpoint not found. Using last weights.")


    # --- 5. Evaluate Model ---
    print("\n===== STEP 5: Evaluating Model on Test Set =====")
    if len(test_dataset) > 0:
        test_loss, test_accuracy, y_test_true, y_test_pred = evaluate_epoch(model, test_loader, criterion, DEVICE)
        print(f"Test Set Eval: Loss: {test_loss:.4f} | Acc: {test_accuracy:.4f}")

        # Generate classification report and confusion matrix
        class_labels_map = {0: "Employed", 1: "Unemployed", 2: "NILF"} # Assuming 0,1,2 map like this
        class_names = [class_labels_map[i] for i in sorted(class_labels_map.keys())]

        print("\nClassification Report (Test Set):")
        present_classes = sorted(np.unique(np.concatenate([y_test_true, y_test_pred])))
        target_names = [class_labels_map[i] for i in present_classes]
        print(classification_report(y_test_true, y_test_pred, labels=present_classes, target_names=target_names, zero_division=0))

        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test_true, y_test_pred, labels=sorted(class_labels_map.keys()))
        print(cm)

        # Plot confusion matrix
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(7, 6))
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            plt.title("Confusion Matrix (Test Set)")
            plt.tight_layout()
            cm_plot_file = model_output_dir / "confusion_matrix_test.png"
            plt.savefig(cm_plot_file)
            print(f"Confusion matrix plot saved to: {cm_plot_file}")
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")

    else:
        print("Test set is empty. Skipping evaluation.")
        test_loss, test_accuracy = None, None

    # Plot training history
    if history and history['loss']:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.plot(history['accuracy'], label='Training Accuracy', linestyle='--')
            plt.plot(history['val_accuracy'], label='Validation Accuracy', linestyle='--')
            plt.title('Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss / Accuracy')
            plt.legend()
            plt.grid(True)
            plt.ylim(bottom=0)
            subtitle = f"Test Acc: {test_accuracy:.3f}" if test_accuracy is not None else "Test set empty"
            plt.text(0.5, 0.95, subtitle, ha='center', va='top', transform=plt.gca().transAxes)
            plt.savefig(plot_file)
            print(f"Training history plot saved to: {plot_file}")
            plt.close()
        except Exception as e:
            print(f"Error plotting training history: {e}")

    # --- 6. Save Final Model and Results --- Use config params
    print("\n===== STEP 6: Saving Final Model and Results =====")
    final_model_path_str = None
    try:
        # Save the final model state (best weights should already be loaded)
        torch.save(model.state_dict(), model_file)
        final_model_path_str = str(model_file)
        print(f"Model state_dict saved successfully: {model_file}")
    except Exception as e:
        print(f"ERROR saving final model state_dict: {e}")
        final_model_path_str = str(checkpoint_path) if checkpoint_path.exists() else None # Fallback to best checkpoint path if saving final fails

    # Save history
    if history and history['loss']:
        try:
            with open(history_file, 'wb') as f: pickle.dump(history, f)
            print(f"History saved: {history_file}")
        except Exception as e: print(f"Error saving history: {e}")
    else: print("History not available to save.")

    # Prepare parameters dictionary using config values
    epochs_completed = len(history['loss']) if history and history['loss'] else None
    # Get final LR from optimizer
    final_lr = optimizer.param_groups[0]['lr'] if optimizer else None

    params = {
        "framework": "PyTorch",
        "processed_data_dir": str(processed_data_dir),
        "model_output_dir": str(model_output_dir),
        "saved_model_path": final_model_path_str,
        "sequence_length": config.SEQUENCE_LENGTH,
        "embed_dim": config.EMBED_DIM,
        "num_heads": config.NUM_HEADS,
        "ff_dim": config.FF_DIM,
        "num_transformer_blocks": config.NUM_TRANSFORMER_BLOCKS,
        "mlp_units": config.MLP_UNITS,
        "dropout": config.DROPOUT,
        "mlp_dropout": config.MLP_DROPOUT,
        "max_epochs_set": config.EPOCHS,
        "epochs_completed": epochs_completed,
        "batch_size": config.BATCH_SIZE,
        "pad_value": config.PAD_VALUE,
        "initial_learning_rate": config.LEARNING_RATE,
        "final_learning_rate": final_lr,
        # Log the inverse class weights used for sampling, not direct loss weights
        "class_weights_for_sampler": class_weights_for_sampler_log.tolist(),
        "sampling_strategy": sampling_strategy, # Use the determined strategy name
        "use_weighted_loss": config.USE_WEIGHTED_LOSS, # Add flag
        "class_weights_for_loss": class_weights_for_loss_log.tolist(), # Add calculated weights
        "loss_weighting_strategy": loss_weighting_strategy, # Add strategy name
        "n_features": n_features,
        "n_classes": n_classes,
        "feature_names": feature_names,
        "input_shape": (config.SEQUENCE_LENGTH, n_features), # Input shape to model
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "timestamp": start_run_time,
        "parallel_workers_used": parallel_workers,
        "recipe_path": str(recipe_file) # Save recipe path used
    }
    try:
        with open(params_file, 'wb') as f: pickle.dump(params, f)
        print(f"Params saved: {params_file}")
    except Exception as e: print(f"Error saving parameters: {e}")

    # --- Completion ---
    end_run_time = time.time()
    elapsed_mins = (end_run_time - start_run_time) / 60
    print("-" * 60)
    print(f"Transformer Model Pipeline (PyTorch) Completed | Elapsed: {elapsed_mins:.2f} minutes")
    print("=" * 60)

    return {
        "model_path": final_model_path_str,
        "history_path": str(history_file) if history_file.exists() else None,
        "params_path": str(params_file) if params_file.exists() else None,
        "recipe_path": str(recipe_file) if recipe_file.exists() else None,
        "plot_path": str(plot_file) if plot_file.exists() else None
    }


# --- Main Execution Block --- # Modified
if __name__ == "__main__":
    print("\n===== Auto-Executing Transformer Training Pipeline (PyTorch) =====")
    
    # Print device info once here
    device_name = "Unknown"
    if DEVICE.type == 'cuda':
        device_name = f"CUDA/GPU ({torch.cuda.get_device_name(DEVICE)})"
    elif DEVICE.type == 'mps':
        device_name = "Apple Silicon GPU (MPS)"
    elif DEVICE.type == 'cpu':
        device_name = "CPU"
    print(f"Using device: {device_name}")

    # Execute Pipeline - No arguments needed
    print("Calling run_transformer_pipeline()...")
    pipeline_results = run_transformer_pipeline()

    print("\n--- Python Script Processing Finished ---")

# --- End of Script ---