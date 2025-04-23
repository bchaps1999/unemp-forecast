import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
import random
from pathlib import Path
from tqdm import tqdm
import math
import sys
import gc # Import garbage collector
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression # Add import for trend calculation

# Assuming utils.py contains get_device, SequenceDataset, period_to_date, worker_init_fn
from utils import get_device, SequenceDataset, period_to_date, worker_init_fn
# Assuming models.py contains the model definition
from models import TransformerForecastingModel
# Assuming config.py contains necessary constants
import config

# --- Helper: Map Group IDs to Integers ---
def map_group_ids_to_int(group_series: pd.Series):
    """Maps unique group IDs from a pandas Series to contiguous integers (0 to N-1)."""
    unique_groups = group_series.unique()
    group_to_int_map = {group: i for i, group in enumerate(unique_groups)}
    int_ids = group_series.map(group_to_int_map).values
    num_groups = len(unique_groups)
    print(f"Mapped {len(group_series)} group IDs to {num_groups} unique integer indices.")
    return torch.tensor(int_ids, dtype=torch.long), num_groups, group_to_int_map


# --- Loading Functions ---
def load_pytorch_model_and_params(model_dir: Path, processed_data_dir: Path, device):
    """Loads the PyTorch model, parameters, and metadata."""
    print(f"Loading model artifacts from: {model_dir}")
    # Use the checkpoint name saved during training
    model_path = model_dir / "best_model_val_loss.pt" # Standardized filename
    params_path = model_dir / "model_params.pkl"
    metadata_path = processed_data_dir / config.METADATA_FILENAME # Use config

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights file '{model_path.name}' in {model_dir}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing model params file '{params_path.name}' in {model_dir}")
    if not metadata_path.exists():
         raise FileNotFoundError(f"Missing metadata file at {metadata_path}. Ensure preprocessing was run.")

    # Load parameters used to build the model
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    print(f"Loaded model parameters from {params_path}.")

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded preprocessing metadata from {metadata_path}.")

    # Validate required metadata keys
    required_meta_keys = ['n_features', 'n_classes', 'target_state_map_inverse', 'feature_names', 'pad_value', 'original_identifier_columns']
    missing_keys = [k for k in required_meta_keys if k not in metadata]
    if missing_keys:
        raise KeyError(f"Metadata is missing required keys: {missing_keys}")

    # Validate required parameter keys (ensure training saved them)
    required_param_keys = ['sequence_length', 'embed_dim', 'num_heads', 'ff_dim', 'num_transformer_blocks', 'mlp_units', 'dropout', 'mlp_dropout']
    missing_param_keys = [k for k in required_param_keys if k not in params]
    if missing_param_keys:
         raise KeyError(f"Model parameters file is missing required keys used for model instantiation: {missing_param_keys}")

    # Rebuild model architecture using params from training run
    # Use metadata['n_features'] and metadata['n_classes'] for consistency
    model = TransformerForecastingModel(
        input_dim=metadata['n_features'],
        seq_len=params['sequence_length'],
        embed_dim=params['embed_dim'],
        num_heads=params['num_heads'],
        ff_dim=params['ff_dim'],
        num_transformer_blocks=params['num_transformer_blocks'],
        mlp_units=params['mlp_units'],
        dropout=params['dropout'],
        mlp_dropout=params['mlp_dropout'],
        n_classes=metadata['n_classes']
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode
    print(f"Model loaded successfully from {model_path} and set to eval mode.")

    return model, params, metadata

def load_forecasting_data(processed_data_dir: Path, metadata: dict):
    """Loads the FULL TEST baked data (which includes lookback history).
       Returns the full dataframe and the list of available forecast start periods.
    """
    print("Loading FULL test data for forecasting initialization...")
    # --- Load TEST Baked Data (includes history) ---
    test_baked_path = processed_data_dir / config.TEST_DATA_FILENAME
    if not test_baked_path.exists():
         raise FileNotFoundError(f"Required test data file not found at {test_baked_path}. Please ensure the preprocessing script ran successfully and saved the file.")

    print(f"Loading FULL TEST baked data from: {test_baked_path}")
    try:
        test_data_df = pd.read_parquet(test_baked_path)
    except Exception as e:
        print(f"Error loading Parquet file {test_baked_path}: {e}")
        raise
    print(f"Loaded FULL TEST baked data (including history): {test_data_df.shape}")
    if test_data_df.empty:
        raise ValueError("Test data file is empty. Cannot initialize forecast.")
    # --- End Load Baked Data ---

    # Ensure original identifier columns are present
    original_id_cols = metadata.get('original_identifier_columns', [])
    missing_orig_cols = [col for col in original_id_cols if col not in test_data_df.columns]
    if missing_orig_cols:
        raise ValueError(f"Baked data is missing required original identifier columns: {missing_orig_cols}")
    print(f"Found original identifier columns in baked data: {original_id_cols}")

    # Calculate period
    test_data_df[config.DATE_COL] = pd.to_datetime(test_data_df[config.DATE_COL])
    test_data_df['period'] = test_data_df[config.DATE_COL].dt.year * 100 + test_data_df[config.DATE_COL].dt.month

    # Determine the actual start date of the test observations from metadata
    test_start_dt_str = metadata.get('train_end_date_preprocess')
    if not test_start_dt_str:
        raise ValueError("Metadata missing 'train_end_date_preprocess', cannot determine test start date.")
    test_start_dt = pd.to_datetime(test_start_dt_str) + pd.Timedelta(days=1)
    actual_test_start_period = test_start_dt.year * 100 + test_start_dt.month

    # Filter available periods to only include those >= actual_test_start_period
    available_periods = sorted([p for p in test_data_df['period'].unique() if p >= actual_test_start_period])
    if not available_periods:
        raise ValueError(f"No periods found in the loaded test data on or after the expected test start period {actual_test_start_period}.")

    print(f"Available forecast start periods in test data: {available_periods}")

    # Return the full dataframe and the list of available periods
    return test_data_df, available_periods

def get_sequences_for_simulation(sim_data_df: pd.DataFrame, group_col: str, seq_len: int, features: list, original_id_cols: list, pad_val: float, end_period: int):
    """
    Extracts the sequence ending at `end_period` for each individual active in that period.
    Also extracts the original identifier columns for the last observation of each individual.
    Assumes sim_data_df contains history *up to* end_period for relevant individuals.
    """
    print(f"Extracting sequences and identifiers ending at period {end_period} for each individual...")
    sequences = {}
    original_identifiers = {} # Store original IDs for the last time step
    weights = {} # Store weights for the last time step

    # Ensure data is sorted correctly before grouping
    # Grouping by the main ID should be sufficient if data is pre-sorted
    grouped = sim_data_df.groupby(group_col)

    valid_ids = 0
    skipped_ids = 0
    weight_col = config.WEIGHT_COL # Get from config

    for person_id, person_df in tqdm(grouped, desc="Extracting Sequences"):
        # The input df is already filtered, but double-check the person was present in the end period
        if person_df['period'].max() != end_period:
            # This shouldn't happen if initial_sim_data was filtered correctly, but good safety check
            skipped_ids += 1
            continue

        # person_df is already sorted by DATE_COL from load_and_prepare_data

        # Extract features for sequence
        person_features = person_df[features].values.astype(np.float32)
        # Extract original identifiers for the *last* observation
        last_obs_identifiers = person_df.iloc[-1][original_id_cols].to_dict()
        # Extract weight for the *last* observation
        last_obs_weight = person_df.iloc[-1][weight_col]

        n_obs = person_features.shape[0]
        if n_obs == 0:
            skipped_ids += 1
            continue # Should not happen with filtering, but check

        # Extract the last 'seq_len' observations for features
        sequence_data = person_features[-seq_len:, :] # Take the tail up to seq_len
        actual_len = sequence_data.shape[0] # How many observations we actually got

        if actual_len < seq_len:
            # Need padding
            pad_len = seq_len - actual_len
            padding_matrix = np.full((pad_len, len(features)), pad_val, dtype=np.float32)
            padded_sequence = np.vstack((padding_matrix, sequence_data))
        else:
            # We have enough (or exactly) seq_len observations
            padded_sequence = sequence_data # No padding needed, or already sliced to seq_len

        # Final check on shape
        if padded_sequence.shape == (seq_len, len(features)):
            sequences[person_id] = padded_sequence
            original_identifiers[person_id] = last_obs_identifiers # Store the identifiers
            weights[person_id] = last_obs_weight # Store the weight
            valid_ids += 1
        else:
            print(f"Warning: Final shape mismatch for {person_id}. Got {padded_sequence.shape}, expected {(seq_len, len(features))}. Skipping.")
            skipped_ids += 1

    print(f"Finished sequence extraction: Processed {valid_ids + skipped_ids} individuals.")
    print(f"  -> Extracted sequences for {valid_ids} individuals ending at period {end_period}.")
    if skipped_ids > 0:
        print(f"  -> Skipped {skipped_ids} individuals (not present in end period or data issue).")

    if not sequences:
        # This is a critical error, simulation cannot start
        raise ValueError(f"FATAL: No valid sequences could be extracted for simulation start period {end_period}. Check data filtering and availability.")

    # Stack sequences into a NumPy array for batch processing
    ids_list = list(sequences.keys())
    sequences_np = np.stack([sequences[pid] for pid in ids_list], axis=0)

    # Create a DataFrame for original identifiers, indexed by person_id (matching ids_list order)
    original_identifiers_df = pd.DataFrame.from_dict(original_identifiers, orient='index')
    # Ensure the DataFrame index matches the order of sequences_np
    original_identifiers_df = original_identifiers_df.reindex(ids_list)

    # Create a Series for weights, indexed by person_id
    weights_series = pd.Series(weights, name=weight_col)
    weights_series = weights_series.reindex(ids_list)

    return sequences_np, ids_list, original_identifiers_df, weights_series


def forecast_next_period_pytorch(sequences_tensor: torch.Tensor, model: nn.Module, device, metadata: dict):
    """
    Forecasts the next labor market state probability distribution for each individual.

    Args:
        sequences_tensor: Tensor of shape (n_individuals, seq_len, n_features) on target device.
        model: PyTorch model (already on target device and in eval mode).
        device: PyTorch device.
        metadata: Dictionary with metadata including n_classes, pad_value.

    Returns:
        probabilities: Tensor of shape (n_individuals, n_classes) with predicted probabilities.
    """
    model.eval() # Ensure model is in eval mode
    n_individuals = sequences_tensor.shape[0]
    pad_value = metadata.get('pad_value', config.PAD_VALUE)

    # Create padding mask: True where padded
    # Check if the model's forward method explicitly accepts src_key_padding_mask
    # This requires inspecting the model's forward method signature or knowing its design.
    # Let's assume the model CAN accept it if needed.
    padding_mask = None
    try:
        import inspect
        sig = inspect.signature(model.forward)
        if 'src_key_padding_mask' in sig.parameters:
             # Create mask: True for positions that should be masked (i.e., padded)
             # Mask needs shape (N, S) where N=batch_size, S=seq_len
             # We check if ALL features in a time step match the pad value
             padding_mask = torch.all(sequences_tensor == pad_value, dim=-1).to(device)
             # print("DEBUG: Created padding mask for model.") # Optional debug
    except Exception as e:
        print(f"Warning: Could not inspect model signature for padding mask. Assuming not used. Error: {e}")


    # Predict next state logits
    with torch.no_grad():
        if padding_mask is not None:
            # print("DEBUG: Calling model WITH padding mask.") # Optional debug
            logits = model(sequences_tensor, src_key_padding_mask=padding_mask)
        else:
            # print("DEBUG: Calling model WITHOUT padding mask.") # Optional debug
            logits = model(sequences_tensor) # Assume model handles padding internally if mask not needed

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1) # Softmax over the class dimension

    if probabilities.shape != (n_individuals, metadata['n_classes']):
         raise ValueError(f"Output probability shape mismatch. Expected {(n_individuals, metadata['n_classes'])}, got {probabilities.shape}")

    return probabilities


def sample_states_and_calc_rates(probabilities: torch.Tensor, # Shape (N, C) on GPU
                                 weights: torch.Tensor, # Shape (N,) on GPU
                                 metadata: dict):
    """
    Samples states based on probabilities and calculates WEIGHTED E/U rates for the sample.

    Args:
        probabilities: Tensor of shape (n_individuals, n_classes) from forecast_next_period_pytorch.
        weights: Tensor of shape (n_individuals,) containing individual weights.
        metadata: Dictionary with metadata including target_state_map_inverse.

    Returns:
        sampled_states: Tensor (GPU) with predicted state indices (shape: n_individuals).
        weighted_unemp_rate: Weighted unemployment rate across all individuals in this sample.
        weighted_emp_rate: Weighted employment rate (E / LF) across all individuals in this sample.
    """
    n_individuals = probabilities.shape[0]
    target_map_inverse = metadata.get('target_state_map_inverse', {})

    # --- Find integer indices for Employed and Unemployed states ---
    try:
        employed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'employed')
        unemployed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'unemployed')
    except StopIteration:
        raise ValueError("Metadata missing required state definitions ('employed', 'unemployed') in target_state_map_inverse")

    # Sample states based on probabilities (on the device probabilities are on)
    sampled_states_gpu = torch.multinomial(probabilities, num_samples=1).squeeze(1)

    # Calculate WEIGHTED unemployment and employment rates for this specific sample
    employed_mask = (sampled_states_gpu == employed_int)
    unemployed_mask = (sampled_states_gpu == unemployed_int)

    employed_weight_sum = torch.sum(weights[employed_mask])
    unemployed_weight_sum = torch.sum(weights[unemployed_mask])
    labor_force_weight_sum = employed_weight_sum + unemployed_weight_sum

    # Avoid division by zero
    if labor_force_weight_sum > 0:
        weighted_unemp_rate = unemployed_weight_sum / labor_force_weight_sum
        weighted_emp_rate = employed_weight_sum / labor_force_weight_sum # E / LF
    else:
        weighted_unemp_rate = torch.tensor(0.0, device=probabilities.device)
        weighted_emp_rate = torch.tensor(0.0, device=probabilities.device)

    return sampled_states_gpu, weighted_unemp_rate.item(), weighted_emp_rate.item() # Return rates as floats


# --- New Tensor-Based Group Stats Calculation ---
def calculate_group_stats_tensor(sampled_states_gpu: torch.Tensor, # Shape (N,) on GPU
                                 weights_gpu: torch.Tensor, # Shape (N,) on GPU
                                 group_ids_int_gpu: torch.Tensor, # Shape (N,) integer group IDs on GPU
                                 num_groups: int,
                                 metadata: dict):
    """Calculates WEIGHTED unemp rate, emp rate, and emp level for each group."""
    device = sampled_states_gpu.device
    target_map_inverse = metadata.get('target_state_map_inverse', {})

    # Find integer indices for Employed and Unemployed states
    try:
        employed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'employed')
        unemployed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'unemployed')
    except StopIteration:
        raise ValueError("Metadata missing required state definitions ('employed', 'unemployed') for group rate calculation")

    # Masks for employment status
    employed_mask = (sampled_states_gpu == employed_int)
    unemployed_mask = (sampled_states_gpu == unemployed_int)

    # Calculate weighted sums per group using torch.bincount
    weights_float_gpu = weights_gpu.float()

    # Sum of weights for employed individuals in each group (Employment Level)
    employed_weight_per_group = torch.bincount(
        group_ids_int_gpu[employed_mask],
        weights=weights_float_gpu[employed_mask],
        minlength=num_groups
    )

    # Sum of weights for unemployed individuals in each group
    unemployed_weight_per_group = torch.bincount(
        group_ids_int_gpu[unemployed_mask],
        weights=weights_float_gpu[unemployed_mask],
        minlength=num_groups
    )

    # Calculate labor force weight per group
    labor_force_weight_per_group = employed_weight_per_group + unemployed_weight_per_group

    # Calculate rates, handle division by zero
    epsilon = 1e-9
    unemp_rate_per_group = unemployed_weight_per_group / (labor_force_weight_per_group + epsilon)
    emp_rate_per_group = employed_weight_per_group / (labor_force_weight_per_group + epsilon) # Emp / LF

    # clamp rates into [0,1]
    unemp_rate_per_group = unemp_rate_per_group.clamp(0.0, 1.0)
    emp_rate_per_group   = emp_rate_per_group.clamp(0.0, 1.0)

    # Stack stats into a tensor: shape (num_groups, 3) -> [unemp_rate, emp_rate, emp_level]
    group_stats_tensor = torch.stack([unemp_rate_per_group, emp_rate_per_group, employed_weight_per_group], dim=1)

    return group_stats_tensor # Return tensor on GPU


# --- Helper: Find feature indices ---
def _find_feature_indices(feature_names, target_map_inverse):
    indices = {}
    try:
        # State features (using names from target_map_inverse)
        indices['state_indices'] = {}
        for state_int, state_name in target_map_inverse.items():
            feature_name = f'cat__current_state_{state_name}' # Assumes this naming convention
            if feature_name in feature_names:
                indices['state_indices'][feature_name] = feature_names.index(feature_name)
            else:
                 print(f"Warning: Expected state feature '{feature_name}' not found in feature list.")
        if not indices['state_indices']: raise ValueError("No 'cat__current_state_*' features found.")

        # Time features
        indices['age_idx'] = feature_names.index('num__age') if 'num__age' in feature_names else -1
        indices['months_last_idx'] = feature_names.index('num__months_since_last_observation') if 'num__months_since_last_observation' in feature_names else -1
        indices['durunemp_idx'] = feature_names.index('num__duration_unemployed') if 'num__duration_unemployed' in feature_names else -1
        # Add month cyclical features
        indices['mth_dim1_idx'] = feature_names.index('num__mth_dim1') if 'num__mth_dim1' in feature_names else -1
        indices['mth_dim2_idx'] = feature_names.index('num__mth_dim2') if 'num__mth_dim2' in feature_names else -1

        # Aggregate rate features (only unemployment rates remain)
        indices['nat_unemp_idx'] = feature_names.index('num__national_unemployment_rate') if 'num__national_unemployment_rate' in feature_names else -1
        indices['state_unemp_idx'] = feature_names.index('num__state_unemployment_rate') if 'num__state_unemployment_rate' in feature_names else -1

        # Add pctchg indices
        indices['ind_emp_pctchg_idx'] = feature_names.index('num__ind_emp_pctchg') if 'num__ind_emp_pctchg' in feature_names else -1

        # Categorical features for lookback (Industry, Occupation, Class)
        indices['ind_group_indices'] = {f: i for i, f in enumerate(feature_names) if f.startswith('cat__ind_group_cat_')}
        indices['occ_group_indices'] = {f: i for i, f in enumerate(feature_names) if f.startswith('cat__occ_group_cat_')}
        indices['classwkr_indices'] = {f: i for i, f in enumerate(feature_names) if f.startswith('cat__classwkr_cat_')}

        # Indices for 'Unknown' or 'NIU' within each group (adjust names based on actual features)
        unknown_niu_suffixes = ['Unknown', 'unknown', 'NIU', 'niu', 'Missing', 'missing']
        indices['ind_unknown_indices'] = {idx for name, idx in indices['ind_group_indices'].items() if any(suffix in name for suffix in unknown_niu_suffixes)}
        indices['occ_unknown_indices'] = {idx for name, idx in indices['occ_group_indices'].items() if any(suffix in name for suffix in unknown_niu_suffixes)}
        indices['classwkr_unknown_indices'] = {idx for name, idx in indices['classwkr_indices'].items() if any(suffix in name for suffix in unknown_niu_suffixes)}

        # New numeric feature indices
        indices['ind_emp_pctchg_idx'] = feature_names.index('num__ind_emp_pctchg') if 'num__ind_emp_pctchg' in feature_names else -1
        # Only include transitions starting from E or U
        for t in ['E_U','E_NE','U_E','U_NE']: # Removed NE_E, NE_U
            key = f'{t.lower()}_idx'
            feat = f'num__{t}'
            indices[key] = feature_names.index(feat) if feat in feature_names else -1

    except (ValueError, KeyError) as e:
        print(f"ERROR: Could not find expected feature indices in metadata. Feature causing error: {e}")
        print(f"Available features: {feature_names}")
        raise ValueError(f"Feature index lookup failed: {e}")

    return indices


# --- Refactored Feature Update Function (Tensor-based) ---
@torch.no_grad() # Ensure no gradients are computed here
def update_sequences_and_features_tensor(
    current_sequences_tensor: torch.Tensor, # Shape: (N, S, F), on DEVICE
    sampled_states_gpu: torch.Tensor, # Shape: (N,), on DEVICE
    weights_gpu: torch.Tensor,                     # New
    state_ids_int_gpu: torch.Tensor, # Shape: (N,), integer state IDs on GPU
    ind_ids_int_gpu: torch.Tensor, # Shape: (N,), integer industry IDs on GPU
    state_stats_tensor: torch.Tensor, # Renamed: Shape (num_states, 3) -> [ur, er, el] (el not used here)
    industry_stats_tensor: torch.Tensor, # Renamed: Shape (num_industries, 3) -> [ur, er, el]
    previous_industry_emp_levels_tensor: torch.Tensor, # New: Shape (num_industries,) - Levels from previous step
    overall_sample_unemp_rate: float, # Overall rate for this step (scalar)
    overall_sample_emp_rate: float, # Overall rate for this step (scalar)
    feature_names: list, # Original feature names list
    metadata: dict,
    current_period: int, # Period *before* prediction (e.g., 202312)
    feature_indices: dict # Pre-calculated indices
    ):
    """
    Updates sequences for the next step using tensor operations on the GPU.
    Shifts sequence, appends new features based on sampled state.
    Handles time updates, aggregate rate updates (using group/overall rates).
    Industry/Occupation/Class features are carried over implicitly.
    """
    device = current_sequences_tensor.device
    n_individuals, seq_len, n_features = current_sequences_tensor.shape
    target_map_inverse = metadata['target_state_map_inverse']

    # --- Find integer indices for states ---
    try:
        employed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'employed')
        unemployed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'unemployed')
        # nilf_int = next(k for k, v in target_map_inverse.items() if v.lower().replace(' ', '_') == 'not_in_labor_force' or v.lower() == 'nilf') # Not needed for updates
    except StopIteration:
        raise ValueError("Metadata missing required state definitions ('employed', 'unemployed')")

    # Pre-extract feature indices
    state_indices_dict = feature_indices['state_indices'] # Dict: {feature_name: index}
    state_indices_values = list(state_indices_dict.values()) # List of indices for state features
    age_idx, months_last_idx, durunemp_idx = feature_indices.get('age_idx', -1), feature_indices.get('months_last_idx', -1), feature_indices.get('durunemp_idx', -1)
    mth_dim1_idx, mth_dim2_idx = feature_indices.get('mth_dim1_idx', -1), feature_indices.get('mth_dim2_idx', -1)
    nat_unemp_idx, state_unemp_idx = feature_indices.get('nat_unemp_idx', -1), feature_indices.get('state_unemp_idx', -1)
    ind_emp_pctchg_idx = feature_indices.get('ind_emp_pctchg_idx', -1) # Get index for industry emp % change

    # --- Determine next period ---
    current_year = current_period // 100
    current_month = current_period % 100
    if current_month == 12:
        next_month = 1
    else:
        next_month = current_month + 1
    next_month_is_jan = (next_month == 1)

    # --- Get previous features ON GPU ---
    previous_features_gpu = current_sequences_tensor[:, -1, :] # Shape (N, F)

    # Create tensor for the NEW feature vectors ON GPU - initialized from previous step
    new_feature_vectors_gpu = previous_features_gpu.clone() # Use clone()

    # --- Vectorized Feature Updates ON GPU ---

    # 1. Update 'current_state' based on sampled_states_gpu
    new_feature_vectors_gpu[:, state_indices_values] = 0.0 # Reset all state flags
    # Create mapping from sampled state int to the correct feature index ONCE
    # This map can be created outside if target_map_inverse doesn't change
    state_int_to_feature_idx = {}
    for state_int, state_name in target_map_inverse.items():
        feature_name = f'cat__current_state_{state_name}'
        if feature_name in state_indices_dict:
            state_int_to_feature_idx[state_int] = state_indices_dict[feature_name]
        # else: # Warning printed during index finding if missing

    # Map sampled states (GPU tensor) to their corresponding feature indices (CPU calculation, result is numpy array)
    # Convert sampled_states_gpu to CPU only for this mapping step if necessary, or find a tensor way
    # Let's try keeping it on GPU: Create a mapping tensor
    max_state_int = max(target_map_inverse.keys())
    state_map_tensor = torch.full((max_state_int + 1,), -1, dtype=torch.long, device=device)
    valid_state_ints = []
    valid_feature_indices = []
    for state_int, feature_idx in state_int_to_feature_idx.items():
        if state_int >= 0 and state_int <= max_state_int:
             valid_state_ints.append(state_int)
             valid_feature_indices.append(feature_idx)
    state_map_tensor[torch.tensor(valid_state_ints, device=device)] = torch.tensor(valid_feature_indices, device=device)

    # Gather the target feature indices using the mapping tensor
    target_feature_indices_gpu = state_map_tensor[sampled_states_gpu]
    valid_indices_mask_gpu = (target_feature_indices_gpu != -1)

    # Use scatter_ or advanced indexing on GPU to set the correct state flag to 1.0
    if torch.any(valid_indices_mask_gpu):
        rows_to_update = torch.arange(n_individuals, device=device)[valid_indices_mask_gpu]
        cols_to_update = target_feature_indices_gpu[valid_indices_mask_gpu]
        # Ensure indices are within bounds (should be if mapping is correct)
        if torch.all(cols_to_update >= 0) and torch.all(cols_to_update < n_features):
             new_feature_vectors_gpu[rows_to_update, cols_to_update] = 1.0
        else:
             print("Warning: Invalid column indices detected during state update on GPU.")
    # if not torch.all(valid_indices_mask_gpu): # Check if any were invalid
    #     print(f"Warning: Could not map {torch.sum(~valid_indices_mask_gpu)} sampled states to feature indices on GPU.")


    # 2. Update Time Features ON GPU
    if age_idx != -1 and next_month_is_jan:
        new_feature_vectors_gpu[:, age_idx] += 1
    if months_last_idx != -1:
        new_feature_vectors_gpu[:, months_last_idx] += 1
    if durunemp_idx != -1:
        is_unemployed_now_mask_gpu = (sampled_states_gpu == unemployed_int)
        # Increment duration if unemployed now, otherwise reset to 0
        new_feature_vectors_gpu[:, durunemp_idx] = torch.where(
            is_unemployed_now_mask_gpu,
            previous_features_gpu[:, durunemp_idx] + 1,
            torch.tensor(0.0, device=device) # Ensure reset value is tensor
        )
    # Update cyclical month features based on the *next* month, explicitly using float32
    if mth_dim1_idx != -1:
        new_feature_vectors_gpu[:, mth_dim1_idx] = torch.sin(torch.tensor(next_month / 12 * 2 * np.pi, dtype=torch.float32, device=device)) + 1
    if mth_dim2_idx != -1:
        new_feature_vectors_gpu[:, mth_dim2_idx] = torch.cos(torch.tensor(next_month / 12 * 2 * np.pi, dtype=torch.float32, device=device)) + 1


    # 3. Update Aggregate Rate Features ON GPU
    # Update national rates (same for everyone)
    if nat_unemp_idx != -1: new_feature_vectors_gpu[:, nat_unemp_idx] = overall_sample_unemp_rate

    # Update state rates using gathered values from state_stats_tensor
    if state_unemp_idx != -1 and state_ids_int_gpu is not None:
        gathered_state_stats = state_stats_tensor[state_ids_int_gpu] # Shape: (N, 3)
        nan_mask_state_unemp = torch.isnan(gathered_state_stats[:, 0])
        new_feature_vectors_gpu[:, state_unemp_idx] = torch.where(nan_mask_state_unemp, previous_features_gpu[:, state_unemp_idx], gathered_state_stats[:, 0])

    # 4. Update Industry Employment Percentage Change (ind_emp_pctchg) based on LEVELS
    if ind_emp_pctchg_idx != -1 and ind_ids_int_gpu is not None and previous_industry_emp_levels_tensor is not None:
        # Get current industry employment level (per industry group)
        current_industry_emp_levels_tensor = industry_stats_tensor[:, 2] # Shape: (num_industries,)

        # Calculate ratio of levels: current_level / previous_level
        epsilon = 1e-9
        industry_emp_level_ratio = torch.where(
            previous_industry_emp_levels_tensor > epsilon,
            current_industry_emp_levels_tensor / previous_industry_emp_levels_tensor,
            torch.tensor(1.0, device=device) # Default to 1.0 if previous level was zero
        )

        # Gather the ratio for each individual based on their industry ID
        gathered_ind_emp_pctchg = industry_emp_level_ratio[ind_ids_int_gpu] # Shape: (N,)

        # Handle potential NaNs/Infs resulting from the division
        nan_inf_mask_pctchg = torch.isnan(gathered_ind_emp_pctchg) | torch.isinf(gathered_ind_emp_pctchg)
        new_feature_vectors_gpu[:, ind_emp_pctchg_idx] = torch.where(
            nan_inf_mask_pctchg,
            torch.tensor(1.0, device=device), # Replace NaN/Inf with 1.0
            gathered_ind_emp_pctchg
        )
    elif ind_emp_pctchg_idx != -1:
        # If it's the first step (previous levels tensor is None), carry over the value
        new_feature_vectors_gpu[:, ind_emp_pctchg_idx] = previous_features_gpu[:, ind_emp_pctchg_idx]


    # --- Industry/Occupation/Class features are carried over implicitly ---

    # --- Update national transition features on GPU ---
    # derive previous state from flags in previous_features_gpu
    prev_flags = previous_features_gpu[:, state_indices_values]           # shape (N,3)
    previous_state_idx = torch.argmax(prev_flags, dim=1)
    nilf_int = next(k for k,v in target_map_inverse.items() if v.lower() in ['not in labor force','nilf'])
    w = weights_gpu
    eps = 1e-9
    # masks
    mask_pE = previous_state_idx == employed_int
    mask_pU = previous_state_idx == unemployed_int
    mask_pN = previous_state_idx == nilf_int
    mask_cE = sampled_states_gpu   == employed_int
    mask_cU = sampled_states_gpu   == unemployed_int
    mask_cN = sampled_states_gpu   == nilf_int
    # denominators
    dE = torch.sum(w[mask_pE]) + eps
    dU = torch.sum(w[mask_pU]) + eps
    # dN = torch.sum(w[mask_pN]) + eps # Not needed anymore
    # transition percents
    v_EU = torch.sum(w[mask_pE & mask_cU]) / dE
    v_E_NE = torch.sum(w[mask_pE & mask_cN]) / dE
    v_UE = torch.sum(w[mask_pU & mask_cE]) / dU
    v_U_NE = torch.sum(w[mask_pU & mask_cN]) / dU
    # v_NE_E = torch.sum(w[mask_pN & mask_cE]) / dN # Removed
    # v_NE_U = torch.sum(w[mask_pN & mask_cU]) / dN # Removed
    # assign into feature vector if indices exist
    fi = feature_indices
    if fi.get('e_u_idx',-1)    >= 0: new_feature_vectors_gpu[:, fi['e_u_idx']]    = v_EU
    if fi.get('e_ne_idx',-1)   >= 0: new_feature_vectors_gpu[:, fi['e_ne_idx']]   = v_E_NE
    if fi.get('u_e_idx',-1)    >= 0: new_feature_vectors_gpu[:, fi['u_e_idx']]    = v_UE
    if fi.get('u_ne_idx',-1)   >= 0: new_feature_vectors_gpu[:, fi['u_ne_idx']]   = v_U_NE
    # if fi.get('ne_e_idx',-1)   >= 0: new_feature_vectors_gpu[:, fi['ne_e_idx']]   = v_NE_E # Removed
    # if fi.get('ne_u_idx',-1)   >= 0: new_feature_vectors_gpu[:, fi['ne_u_idx']]   = v_NE_U # Removed

    # --- Shift sequences and append ON GPU ---
    # Shift left (remove oldest time step)
    shifted_sequences_gpu = current_sequences_tensor[:, 1:, :]
    # Append the newly calculated feature vectors as the last time step
    # Add a time dimension: (N, F) -> (N, 1, F)
    new_sequences_gpu = torch.cat([shifted_sequences_gpu, new_feature_vectors_gpu.unsqueeze(1)], dim=1)

    return new_sequences_gpu # Return tensor on GPU


def forecast_multiple_periods_pytorch(initial_sequences_tensor: torch.Tensor, # Shape: (N, S, F), on DEVICE
                                      initial_identifiers_df: pd.DataFrame, # CPU DataFrame
                                      initial_weights_series: pd.Series, # CPU Series
                                      model: nn.Module, # On DEVICE
                                      device,
                                      metadata: dict,
                                      params: dict,
                                      initial_period: int, # Starting period YYYYMM
                                      periods_to_forecast: int = 12,
                                      n_samples: int = 1, # Default to 1 for HPT objective, >1 for full forecast
                                      return_raw_samples: bool = False, # Control returning raw sample data
                                      forecast_batch_size: int = 512 # Added batch size for forecasting step
                                      ):
    """
    Runs the multi-period forecast simulation using tensor operations for efficiency.
    Processes one sample fully at a time to conserve memory.
    Processes individuals in mini-batches during prediction.
    Calculates and uses group-specific rates dynamically on GPU.
    Returns aggregated forecast statistics (mean, CI) and optionally raw sample trajectories.
    """
    print(f"\nStarting multi-period forecast (Tensor Optimized):") # Updated print
    print(f" - Initial Period: {initial_period}")
    print(f" - Periods to Forecast: {periods_to_forecast}")
    print(f" - Number of Samples: {n_samples}")
    print(f" - Device: {device}")
    print(f" - Forecast Batch Size: {forecast_batch_size}")
    print(" - Strategy: Sequential samples, batched prediction, tensorized rate calculation & feature updates.")

    # Store results per sample, per period (for unemployment rate primarily)
    all_samples_period_urs = [] # List of lists: outer=samples, inner=UR per period

    n_individuals, seq_len, n_features = initial_sequences_tensor.shape
    # Move initial weights to GPU tensor for calculations
    current_weights_gpu = torch.from_numpy(initial_weights_series.values.astype(np.float32)).to(device)

    # --- Get metadata/params needed ---
    feature_names = metadata.get('feature_names', [])
    target_map_inverse = metadata.get('target_state_map_inverse', {})
    original_id_cols = metadata.get('original_identifier_columns', [])
    state_col = 'statefip' if 'statefip' in original_id_cols else None
    ind_col = 'ind_group_cat' if 'ind_group_cat' in original_id_cols else None

    if not feature_names: raise ValueError("Feature names not found in metadata.")
    if not target_map_inverse: raise ValueError("Target state map not found in metadata.")

    # --- Pre-calculate feature indices ONCE ---
    print("Pre-calculating feature indices...")
    try:
        feature_indices = _find_feature_indices(feature_names, target_map_inverse)
        # ... existing checks ...
        print("Feature indices calculated successfully.")
    except ValueError as e:
        print(f"Fatal error during feature index calculation: {e}")
        raise

    # --- Pre-map Group IDs to Integers ONCE ---
    print("Mapping group identifiers to integer indices...")
    state_ids_int_gpu, num_states, _ = map_group_ids_to_int(initial_identifiers_df[state_col]) if state_col else (None, 0, {})
    ind_ids_int_gpu, num_industries, _ = map_group_ids_to_int(initial_identifiers_df[ind_col]) if ind_col else (None, 0, {})
    # Move mapped IDs to GPU
    if state_ids_int_gpu is not None: state_ids_int_gpu = state_ids_int_gpu.to(device)
    if ind_ids_int_gpu is not None: ind_ids_int_gpu = ind_ids_int_gpu.to(device)
    print("Group ID mapping complete.")

    # --- Outer loop: Iterate through each Monte Carlo sample ---
    for s in tqdm(range(n_samples), desc="Processing Samples", disable=(n_samples <= 1)):
        sample_period_urs = [] # Store overall UR for each period within this sample

        # Initialize sequence tensor for this sample by cloning the initial state
        current_sample_sequences_gpu = initial_sequences_tensor.clone() # Already on device
        current_sim_period = initial_period # Reset start period for each sample

        # Initialize previous step's industry employment levels (needed for pctchg calc)
        previous_industry_emp_levels_tensor = None # Will be set after first step

        # --- Inner loop: Iterate through forecast periods for this sample ---
        for i in range(periods_to_forecast):
            # Calculate target period (YYYYMM) for this step (period being predicted)
            current_year = current_sim_period // 100
            current_month = current_sim_period % 100
            if current_month == 12:
                target_year = current_year + 1
                target_month = 1
            else:
                target_year = current_year
                target_month = current_month + 1
            target_period = target_year * 100 + target_month

            # --- Simulation Step ---
            # 1. Predict probability distribution for the next state IN BATCHES
            all_probabilities_gpu = [] # Initialize list to store batch probabilities
            num_individuals = current_sample_sequences_gpu.shape[0]
            # Inner loop for mini-batch prediction
            for batch_start in range(0, num_individuals, forecast_batch_size):
                batch_end = min(batch_start + forecast_batch_size, num_individuals)
                sequences_batch_gpu = current_sample_sequences_gpu[batch_start:batch_end]

                # Predict for the current mini-batch
                probabilities_batch_gpu = forecast_next_period_pytorch(
                    sequences_batch_gpu, model, device, metadata
                )
                all_probabilities_gpu.append(probabilities_batch_gpu)
                # Optional: Clear cache within batch loop if memory is extremely tight
                # if device.type == 'cuda': torch.cuda.empty_cache()
                # elif device.type == 'mps': torch.mps.empty_cache()

            # Concatenate probabilities from all batches
            probabilities_gpu = torch.cat(all_probabilities_gpu, dim=0)
            del all_probabilities_gpu # Free memory from list of tensors

            # 2. Sample states and calculate WEIGHTED OVERALL rates for this sample step
            sampled_states_gpu, weighted_overall_unemp_rate, weighted_overall_emp_rate = sample_states_and_calc_rates(
                 probabilities_gpu, current_weights_gpu, metadata
            )
            sample_period_urs.append(weighted_overall_unemp_rate)

            # 3. Calculate WEIGHTED Group-Specific Stats (State, Industry) using TENSOR function
            state_stats_tensor = torch.zeros((num_states, 3), device=device) # Shape includes level now
            if state_ids_int_gpu is not None and num_states > 0:
                state_stats_tensor = calculate_group_stats_tensor( # Use new function name
                    sampled_states_gpu, current_weights_gpu, state_ids_int_gpu, num_states, metadata
                )

            industry_stats_tensor = torch.zeros((num_industries, 3), device=device) # Shape includes level now
            if ind_ids_int_gpu is not None and num_industries > 0:
                 industry_stats_tensor = calculate_group_stats_tensor( # Use new function name
                    sampled_states_gpu, current_weights_gpu, ind_ids_int_gpu, num_industries, metadata
                )

            # Extract current industry employment levels for the *next* step's calculation
            current_industry_emp_levels_tensor = industry_stats_tensor[:, 2].clone() if ind_ids_int_gpu is not None else None

            # 4. Update this sample's sequences using TENSORIZED function
            current_sample_sequences_gpu = update_sequences_and_features_tensor(
                current_sequences_tensor=current_sample_sequences_gpu,
                sampled_states_gpu=sampled_states_gpu,
                weights_gpu=current_weights_gpu,
                state_ids_int_gpu=state_ids_int_gpu,
                ind_ids_int_gpu=ind_ids_int_gpu,
                state_stats_tensor=state_stats_tensor, # Pass full state stats
                industry_stats_tensor=industry_stats_tensor, # Pass full industry stats
                previous_industry_emp_levels_tensor=previous_industry_emp_levels_tensor, # Pass previous levels
                overall_sample_unemp_rate=weighted_overall_unemp_rate,
                overall_sample_emp_rate=weighted_overall_emp_rate,
                feature_names=feature_names,
                metadata=metadata,
                current_period=current_sim_period,
                feature_indices=feature_indices
            )

            # 5. Store current levels as previous for the next iteration
            previous_industry_emp_levels_tensor = current_industry_emp_levels_tensor # Update for next loop

            # 6. Advance time for the next period in this sample's simulation
            current_sim_period = target_period

            # Optional: Clear GPU cache periodically if memory issues arise
            # if (i + 1) % 10 == 0: # Example: clear every 10 periods
            #      if device.type == 'cuda': torch.cuda.empty_cache()
            #      elif device.type == 'mps': torch.mps.empty_cache() # If using MPS
            #      gc.collect()

        # --- End Period Loop for Sample 's' ---

        # Store the list of period URs for this completed sample
        all_samples_period_urs.append(sample_period_urs)

        # Clean up GPU memory for this sample's tensor before starting next sample
        del current_sample_sequences_gpu, probabilities_gpu, sampled_states_gpu
        if device.type == 'cuda': torch.cuda.empty_cache()
        elif device.type == 'mps': torch.mps.empty_cache()
        gc.collect()
    # --- End Sample Loop ---

    # --- Aggregate results across all samples ---
    print("\nAggregating results across periods and samples...")

    # Create forecast period list (YYYYMM) corresponding to the predictions
    forecast_periods_list = []
    temp_period = initial_period
    for _ in range(periods_to_forecast):
        current_year = temp_period // 100
        current_month = temp_period % 100
        if current_month == 12: target_year, target_month = current_year + 1, 1
        else: target_year, target_month = current_year, current_month + 1
        target_period = target_year * 100 + target_month
        forecast_periods_list.append(target_period)
        temp_period = target_period

    # Convert collected UR lists into a DataFrame for easier aggregation
    # Rows = samples, Columns = forecast periods
    if not all_samples_period_urs:
        print("Warning: No sample results were collected. Cannot aggregate.")
        empty_agg_df = pd.DataFrame(columns=['period', 'date', 'unemployment_rate_forecast', 'unemployment_rate_p10', 'unemployment_rate_p90'])
        empty_raw_df = pd.DataFrame()
        return empty_agg_df, empty_raw_df # Return empty dataframes

    sample_urs_over_time = pd.DataFrame(all_samples_period_urs, columns=forecast_periods_list)

    # Calculate mean and quantiles across samples (axis=0) for each period (column)
    forecast_agg_df = pd.DataFrame({
        'period': forecast_periods_list,
        'unemployment_rate_forecast': sample_urs_over_time.mean(axis=0).values, # Use MEAN as primary forecast
        'unemployment_rate_p10': sample_urs_over_time.quantile(0.1, axis=0).values,
        'unemployment_rate_p90': sample_urs_over_time.quantile(0.9, axis=0).values,
        # Add Employment Rate aggregation if needed (would require collecting all_samples_period_ers)
    })
    forecast_agg_df['date'] = forecast_agg_df['period'].apply(period_to_date)

    # Handle potential date conversion failures
    forecast_agg_df.dropna(subset=['date'], inplace=True)

    print("Multi-period forecast complete.")
    # Return aggregated results and optionally raw sample UR data
    if return_raw_samples:
        return forecast_agg_df, sample_urs_over_time
    else:
        # Return aggregated results and an empty DataFrame for raw samples
        return forecast_agg_df, pd.DataFrame()


# --- New Helper: Calculate Squared Slope Difference ---
def calculate_slope_error(y_forecast: np.ndarray, y_actual: np.ndarray) -> float:
    """Fit linear models to forecast vs actual series; return squared difference of slopes."""
    x = np.arange(len(y_forecast)).reshape(-1, 1)
    slope_f = LinearRegression().fit(x, y_forecast).coef_[0]
    slope_a = LinearRegression().fit(x, y_actual).coef_[0]
    return float((slope_f - slope_a) ** 2)


# --- HPT Objective Specific Functions ---

def calculate_hpt_forecast_metrics(model: nn.Module,
                                hpt_val_baked_data: pd.DataFrame, # Still needed to get start periods/sequences
                                national_rates_file: Path, # Path to the truth data
                                metadata: dict,
                                params: dict,
                                device,
                                forecast_horizon: int = 12, # Months to forecast
                                forecast_batch_size: int = 512, # Added batch size for HPT forecast
                                hpt_mc_samples: int = 1 # Number of samples for HPT forecast
                                ):
    """
    Calculates the HPT objective using the tensor-optimized forecast function.
    Uses the MEAN forecast across hpt_mc_samples for comparison.
    Returns RMSE, Std Dev, and squared‐slope‐difference (slope_error).
    """
    print("\n===== Calculating HPT Metrics (RMSE, Std Dev, Slope Error) =====")
    print(f"Using {hpt_mc_samples} MC sample(s) per forecast run.")
    default_return = {'rmse': float('inf'), 'std_dev': float('inf'), 'slope_error': float('inf')}
    if hpt_val_baked_data is None or hpt_val_baked_data.empty:
        return default_return
    if not national_rates_file.exists():
        print(f"ERROR: National rates file not found at {national_rates_file}. Cannot calculate HPT forecast metrics.")
        return default_return
    if 'hpt_validation_intervals' not in metadata or not metadata['hpt_validation_intervals']:
        print("ERROR: 'hpt_validation_intervals' not found or empty in metadata. Cannot determine forecast start periods.")
        return default_return

    model.eval() # Ensure model is in eval mode
    all_forecast_errors = []  # Store forecast minus actual errors
    all_slope_errors = []     # Store SQUARED slope_error for each interval

    # --- Load Actual National Rates ---
    try:
        actual_rates_df = pd.read_csv(national_rates_file)
        if 'date' not in actual_rates_df.columns or 'national_unemp_rate' not in actual_rates_df.columns:
            raise ValueError("National rates file missing required columns: 'date', 'national_unemp_rate'")
        actual_rates_df['date'] = pd.to_datetime(actual_rates_df['date'])
        # Calculate period column for merging
        actual_rates_df['period'] = actual_rates_df['date'].dt.year * 100 + actual_rates_df['date'].dt.month
        actual_rates_df.rename(columns={'national_unemp_rate': 'actual_unemployment_rate'}, inplace=True)
        actual_rates_df = actual_rates_df[['period', 'actual_unemployment_rate']].dropna().sort_values('period') # Sort for trend calc
        print(f"Loaded actual national rates from {national_rates_file} for {len(actual_rates_df)} periods.")
        if actual_rates_df.empty:
            print("Warning: Loaded actual rates data is empty.")
            return default_return
    except Exception as e:
        print(f"Error loading or processing actual rates file {national_rates_file}: {e}")
        return default_return

    # --- Determine Start Periods from HPT Intervals in Metadata ---
    hpt_intervals = metadata['hpt_validation_intervals']
    start_periods = []
    max_actual_period = actual_rates_df['period'].max() if not actual_rates_df.empty else -1

    # Convert interval strings to datetime and get start periods
    try:
        for start_str, end_str in hpt_intervals:
            start_dt = pd.to_datetime(start_str)
            # end_dt = pd.to_datetime(end_str) # End date not directly needed for start period
            potential_start_period = start_dt.year * 100 + start_dt.month

            # Check if enough future data exists in actual_rates_df for this start period
            required_end_period = potential_start_period
            valid_horizon = True
            for _ in range(forecast_horizon):
                 yr, mn = required_end_period // 100, required_end_period % 100
                 if mn == 12: required_end_period = (yr + 1) * 100 + 1
                 else: required_end_period += 1
                 # Check if the *next* period exists in actuals
                 if required_end_period not in actual_rates_df['period'].values:
                     valid_horizon = False
                     break

            if valid_horizon:
                 start_periods.append(potential_start_period)
            else:
                 print(f"Skipping potential start period {potential_start_period} (from interval {start_str}-{end_str}): Not enough future data ({forecast_horizon} months) available in actual rates file (max period with data: {max_actual_period}).")

    except Exception as e:
        print(f"ERROR processing HPT intervals from metadata: {e}")
        return default_return

    if not start_periods:
        print("ERROR: No valid start periods found based on HPT intervals and available actual rates data.")
        return default_return

    print(f"Selected start periods for HPT forecast evaluation based on metadata intervals: {start_periods}")
    num_forecasts = len(start_periods) # Number of forecasts is now determined by valid intervals

    # --- Run Forecasts for Each Start Period ---
    seq_len = params['sequence_length']
    group_col = config.GROUP_ID_COL
    features = metadata['feature_names']
    original_id_cols = metadata['original_identifier_columns']
    pad_val = metadata['pad_value']

    # Ensure period column exists in hpt_val_baked_data for filtering
    if 'period' not in hpt_val_baked_data.columns:
        hpt_val_baked_data['period'] = pd.to_datetime(hpt_val_baked_data[config.DATE_COL]).dt.year * 100 + pd.to_datetime(hpt_val_baked_data[config.DATE_COL]).dt.month

    for start_period in start_periods:
        print(f"\n--- Running HPT forecast starting from {start_period} ---")
        try:
            # 1. Filter HPT data up to the start period for individuals present then
            start_period_ids = hpt_val_baked_data[hpt_val_baked_data['period'] == start_period][group_col].unique()
            if len(start_period_ids) == 0:
                print(f"Warning: No individuals found in HPT data for start period {start_period}. Skipping forecast.")
                continue
            initial_sim_data_hpt = hpt_val_baked_data[
                (hpt_val_baked_data[group_col].isin(start_period_ids)) &
                (hpt_val_baked_data['period'] <= start_period)
            ].sort_values([group_col, config.DATE_COL])

            # 2. Extract initial sequences, identifiers, and weights
            initial_sequences_np, _, initial_identifiers_df, initial_weights_series = get_sequences_for_simulation(
                sim_data_df=initial_sim_data_hpt,
                group_col=group_col,
                seq_len=seq_len,
                features=features,
                original_id_cols=original_id_cols,
                pad_val=pad_val,
                end_period=start_period
            )
            initial_sequences_tensor = torch.from_numpy(initial_sequences_np).to(device)

            # 3. Run the recursive forecast (using tensor-optimized function)
            forecast_agg_df, _ = forecast_multiple_periods_pytorch( # Call the main (now optimized) function
                initial_sequences_tensor=initial_sequences_tensor,
                initial_identifiers_df=initial_identifiers_df, # Still needed for ID mapping inside
                initial_weights_series=initial_weights_series,
                model=model,
                device=device,
                metadata=metadata,
                params=params,
                initial_period=start_period,
                periods_to_forecast=forecast_horizon,
                n_samples=hpt_mc_samples, # Use specified number of samples
                return_raw_samples=False,
                forecast_batch_size=forecast_batch_size # Pass batch size
            )

            if forecast_agg_df.empty:
                print(f"Warning: Forecast from start period {start_period} returned empty results.")
                continue

            # 4. Merge with actual rates (loaded from file) and calculate errors using MEAN forecast
            comparison_df = pd.merge(
                forecast_agg_df[['period', 'unemployment_rate_forecast']].sort_values('period'), # Sort for trend calc
                actual_rates_df, # Use the dataframe loaded from national_rates_file
                on='period',
                how='inner' # Only compare months where both forecast and actual exist
            )

            if len(comparison_df) != forecast_horizon:
                 print(f"Warning: Forecast from {start_period} - Expected {forecast_horizon} comparison months, found {len(comparison_df)}. Check actual data availability in national rates file.")

            if not comparison_df.empty:
                # --- Added Diagnostic Printing ---
                print(f"  Comparison Data (Mean Forecast vs Actual) for start_period {start_period}:") # Update print label
                print(comparison_df.to_string(index=False, float_format="%.6f"))
                # --- End Added Diagnostic Printing ---

                errors = comparison_df['unemployment_rate_forecast'] - comparison_df['actual_unemployment_rate'] # Use MEAN forecast
                all_forecast_errors.extend(errors.tolist())
                run_rmse = np.sqrt(mean_squared_error(comparison_df['actual_unemployment_rate'], comparison_df['unemployment_rate_forecast'])) # Use MEAN forecast

                # Calculate SQUARED slope error for this run
                forecast_vals = comparison_df['unemployment_rate_forecast'].values
                actual_vals = comparison_df['actual_unemployment_rate'].values
                if len(forecast_vals) > 1 and len(actual_vals) > 1: # Check length again before calling
                    run_slope_error_squared = calculate_slope_error(forecast_vals, actual_vals) # This returns (slope_f - slope_a)**2
                else:
                    print(f"  Warning: Not enough data points ({len(forecast_vals)}) to calculate slope error for start period {start_period}.")
                    run_slope_error_squared = float('inf') # Assign inf if cannot calculate

                all_slope_errors.append(run_slope_error_squared)  # Store the SQUARED slope error for this run

                print(f"  -> Forecast run from {start_period} completed. RMSE: {run_rmse:.6f}, Squared Slope Error: {run_slope_error_squared:.6f}") # Updated print
            else:
                 print(f"  -> Forecast run from {start_period} - No matching actual data found for comparison in national rates file.")

            # Clean up memory
            del initial_sequences_tensor, initial_identifiers_df, initial_weights_series, forecast_agg_df, comparison_df, initial_sim_data_hpt
            if device.type == 'cuda': torch.cuda.empty_cache()
            elif device.type == 'mps': torch.mps.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"ERROR during HPT forecast run starting from {start_period}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next start period if possible, but the final RMSE might be unreliable
            all_slope_errors.append(float('inf')) # Add inf if run failed

    # --- Calculate Final Metrics ---
    if not all_forecast_errors:
        print("ERROR: No forecast errors collected across all runs. Cannot calculate final metrics.")
        return default_return # Return dict with inf values

    final_rmse = np.sqrt(np.mean(np.square(all_forecast_errors)))
    final_std_dev = np.std(all_forecast_errors) # Calculate standard deviation

    # Calculate the ROOT MEAN SQUARE of the slope errors across all successful forecast runs
    valid_squared_slopes = [e for e in all_slope_errors if not np.isinf(e) and not np.isnan(e)]
    # Calculate the square root of the mean of the squared errors
    final_slope_error = np.sqrt(np.mean(valid_squared_slopes)) if valid_squared_slopes else float('inf') # Changed calculation

    print(f"\nCalculated Final HPT Metrics (over {len(all_forecast_errors)} months across {num_forecasts} forecasts):")
    print(f"  RMSE: {final_rmse:.6f}")
    print(f"  Std Dev: {final_std_dev:.6f}")
    print(f"  Root Mean Squared Slope Error: {final_slope_error:.6f}") # Updated print description

    # Return the Root Mean Squared Slope Error under the 'slope_error' key
    return {'rmse': final_rmse, 'std_dev': final_std_dev, 'slope_error': final_slope_error}


# --- Standard Evaluation Function (from training script) ---
def evaluate_aggregate_unemployment_error(model, dataloader, device, metadata):
    """
    Evaluates the model based on the *weighted* aggregate unemployment rate error
    using a standard DataLoader (typically for the validation set during training).
    """
    # Define class indices based on common convention (adjust if metadata differs)
    target_map_inverse = metadata.get('target_state_map_inverse', {})
    try:
        employed_idx = next(k for k, v in target_map_inverse.items() if v.lower() == 'employed')
        unemployed_idx = next(k for k, v in target_map_inverse.items() if v.lower() == 'unemployed')
        inactive_idx = next(k for k, v in target_map_inverse.items() if v.lower().replace(' ', '_') == 'not_in_labor_force' or v.lower() == 'nilf')
    except StopIteration:
        raise ValueError("Metadata missing required state definitions ('employed', 'unemployed', 'nilf'/'not in labor force') for aggregate error evaluation")


    model.eval()
    all_preds = []
    all_targets = []
    all_weights = [] # To store weights

    is_main_process = True # Assume called from main process during standard eval
    data_iterator = tqdm(dataloader, desc="Agg. Err Eval", leave=False, disable=not is_main_process)

    with torch.no_grad():
        # Dataloader provides (x, y, w, mask)
        for x_batch, y_batch, w_batch, mask_batch in data_iterator:
            x_batch, mask_batch = x_batch.to(device), mask_batch.to(device)
            # y_batch and w_batch stay on CPU as we only need it for final calculation

            outputs = model(x_batch, src_key_padding_mask=mask_batch)
            # Deterministic forecast using argmax
            predicted = torch.argmax(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy()) # y_batch was already on CPU
            all_weights.extend(w_batch.numpy()) # Store weights

    if not all_targets:
        print("Warning: No targets found in dataset for aggregate error calculation.")
        return float('inf') # Return infinity if no data

    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)
    all_weights_np = np.array(all_weights) # Convert weights to numpy array

    # --- Calculate WEIGHTED aggregate unemployment rate ---
    # Actual
    actual_unemployed_weight = np.sum(all_weights_np[all_targets_np == unemployed_idx])
    actual_employed_weight = np.sum(all_weights_np[all_targets_np == employed_idx])
    actual_labor_force_weight = actual_employed_weight + actual_unemployed_weight
    if actual_labor_force_weight == 0:
         print("Warning: Actual weighted labor force size is zero. Cannot calculate actual rate.")
         actual_agg_rate = 0.0
    else:
         actual_agg_rate = actual_unemployed_weight / actual_labor_force_weight

    # Predicted
    predicted_unemployed_weight = np.sum(all_weights_np[all_preds_np == unemployed_idx])
    predicted_employed_weight = np.sum(all_weights_np[all_preds_np == employed_idx])
    predicted_labor_force_weight = predicted_employed_weight + predicted_unemployed_weight
    if predicted_labor_force_weight == 0:
         print("Warning: Predicted weighted labor force size is zero. Cannot calculate predicted rate.")
         predicted_agg_rate = 0.0
    else:
         predicted_agg_rate = predicted_unemployed_weight / predicted_labor_force_weight

    # Calculate Mean Squared Error between the single rate values
    error = mean_squared_error([actual_agg_rate], [predicted_agg_rate])
    print(f"  Weighted Agg. Rate Eval - Actual: {actual_agg_rate:.4f} (U={actual_unemployed_weight:.1f}/LF={actual_labor_force_weight:.1f}), "
          f"Predicted: {predicted_agg_rate:.4f} (U={predicted_unemployed_weight:.1f}/LF={predicted_labor_force_weight:.1f}), MSE: {error:.6f}")

    return error
