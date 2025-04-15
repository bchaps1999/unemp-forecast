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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import math
import sys
import gc # Import garbage collector
from sklearn.metrics import mean_squared_error

# Assuming utils.py contains get_device, SequenceDataset, period_to_date, worker_init_fn
from utils import get_device, SequenceDataset, period_to_date, worker_init_fn
# Assuming models.py contains the model definition
from models import TransformerForecastingModel
# Assuming config.py contains necessary constants
import config

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

def load_forecasting_data(processed_data_dir: Path, metadata: dict, start_year: int = None, start_month: int = None):
    """Loads the TEST baked data (which includes lookback history) for simulation start.
       Filters data based on specified start_year and start_month if provided,
       otherwise uses the EARLIEST period in the test set's actual forecast range.
    """
    print("Loading data for forecasting initialization...")
    # --- Load TEST Baked Data (includes history) ---
    test_baked_path = processed_data_dir / config.TEST_DATA_FILENAME
    if not test_baked_path.exists():
         raise FileNotFoundError(f"Required test data file not found at {test_baked_path}. Please ensure the preprocessing script ran successfully and saved the file.")

    print(f"Loading TEST baked data from: {test_baked_path}")
    try:
        test_data_df = pd.read_parquet(test_baked_path) # Renamed variable
    except Exception as e:
        print(f"Error loading Parquet file {test_baked_path}: {e}")
        raise
    print(f"Loaded TEST baked data (including history): {test_data_df.shape}")
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

    # Determine the simulation start period
    if start_year and start_month:
        start_period = start_year * 100 + start_month
        if start_period not in available_periods:
             raise ValueError(f"Specified start period {start_period} ({start_year}-{start_month:02d}) not found in the test data's forecast range. Available periods range from {min(available_periods)} to {max(available_periods)}.")
        print(f"Using specified start period from test data: {start_period}")
    else:
        start_period = available_periods[0] # Use the EARLIEST period in the test set's forecast range
        start_date = period_to_date(start_period)
        print(f"Using earliest period in test data's forecast range as start period: {start_period} ({start_date.strftime('%Y-%m') if start_date else 'Invalid Date'})")

    # Get individuals present in the simulation start period
    start_period_ids = test_data_df[test_data_df['period'] == start_period][config.GROUP_ID_COL].unique()
    if len(start_period_ids) == 0:
        raise ValueError(f"No individuals found in the specified start period {start_period} within the test data.")
    print(f"Found {len(start_period_ids)} unique individuals present in the start period ({start_period}).")

    # Filter the loaded test_data_df to include history *up to* the start period for these individuals
    initial_sim_data = test_data_df[
        (test_data_df[config.GROUP_ID_COL].isin(start_period_ids)) &
        (test_data_df['period'] <= start_period)
    ].sort_values([config.GROUP_ID_COL, config.DATE_COL]) # Sort is important

    # --- Delete the loaded dataframe if no longer needed (initial_sim_data is a slice/copy) ---
    del test_data_df
    gc.collect()
    print("Released memory from the loaded test dataframe.")
    # ---

    # Return the filtered data containing history + start period state
    return initial_sim_data, start_period, start_period_ids

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


def calculate_group_rates(sampled_states_np: np.ndarray, # Shape (n_individuals,) on CPU
                          grouping_series: pd.Series, # Series with group IDs (statefip or ind_group_cat), index matching sampled_states
                          weights_series: pd.Series, # Series with weights, index matching sampled_states
                          target_map_inverse: dict):
    """Calculates WEIGHTED unemployment and employment rates for each group."""
    if grouping_series is None or grouping_series.empty:
        print("Warning: Grouping series is empty, cannot calculate group rates.")
        return {}
    if weights_series is None or weights_series.empty:
        print("Warning: Weights series is empty, cannot calculate weighted group rates.")
        return {}
    if len(sampled_states_np) != len(grouping_series) or len(sampled_states_np) != len(weights_series):
        print(f"Warning: Length mismatch between states ({len(sampled_states_np)}), group IDs ({len(grouping_series)}), or weights ({len(weights_series)}). Cannot calculate group rates.")
        return {}

    # --- Find integer indices for Employed and Unemployed states ---
    try:
        employed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'employed')
        unemployed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'unemployed')
    except StopIteration:
        raise ValueError("Metadata missing required state definitions ('employed', 'unemployed') in target_state_map_inverse for group rate calculation")

    # Create a DataFrame for calculation
    df = pd.DataFrame({
        'group': grouping_series.values, # Use .values to ensure alignment if index is different
        'state': sampled_states_np,
        'weight': weights_series.values
    })

    # Calculate weighted counts for each state within each group
    df['is_employed'] = (df['state'] == employed_int)
    df['is_unemployed'] = (df['state'] == unemployed_int)

    group_weights = df.groupby('group').agg(
        employed_weight=('weight', lambda x: x[df.loc[x.index, 'is_employed']].sum()),
        unemployed_weight=('weight', lambda x: x[df.loc[x.index, 'is_unemployed']].sum())
    )

    # Calculate rates
    labor_force_weight = group_weights['employed_weight'] + group_weights['unemployed_weight']

    rates_df = pd.DataFrame(index=group_weights.index)
    rates_df['unemp_rate'] = np.where(labor_force_weight > 0, group_weights['unemployed_weight'] / labor_force_weight, 0.0)
    rates_df['emp_rate'] = np.where(labor_force_weight > 0, group_weights['employed_weight'] / labor_force_weight, 0.0) # Emp / LF

    # Convert to dictionary: {group_id: {'unemp_rate': x, 'emp_rate': y}}
    return rates_df.to_dict(orient='index')


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

        # Aggregate rate features
        indices['nat_unemp_idx'] = feature_names.index('num__national_unemployment_rate') if 'num__national_unemployment_rate' in feature_names else -1
        indices['nat_emp_idx'] = feature_names.index('num__national_employment_rate') if 'num__national_employment_rate' in feature_names else -1
        indices['state_unemp_idx'] = feature_names.index('num__state_unemployment_rate') if 'num__state_unemployment_rate' in feature_names else -1
        indices['state_emp_idx'] = feature_names.index('num__state_employment_rate') if 'num__state_employment_rate' in feature_names else -1
        indices['ind_unemp_idx'] = feature_names.index('num__industry_unemployment_rate') if 'num__industry_unemployment_rate' in feature_names else -1
        indices['ind_emp_idx'] = feature_names.index('num__industry_employment_rate') if 'num__industry_employment_rate' in feature_names else -1

        # Categorical features for lookback (Industry, Occupation, Class)
        indices['ind_group_indices'] = {f: i for i, f in enumerate(feature_names) if f.startswith('cat__ind_group_cat_')}
        indices['occ_group_indices'] = {f: i for i, f in enumerate(feature_names) if f.startswith('cat__occ_group_cat_')}
        indices['classwkr_indices'] = {f: i for i, f in enumerate(feature_names) if f.startswith('cat__classwkr_cat_')}

        # Indices for 'Unknown' or 'NIU' within each group (adjust names based on actual features)
        unknown_niu_suffixes = ['Unknown', 'unknown', 'NIU', 'niu', 'Missing', 'missing']
        indices['ind_unknown_indices'] = {idx for name, idx in indices['ind_group_indices'].items() if any(suffix in name for suffix in unknown_niu_suffixes)}
        indices['occ_unknown_indices'] = {idx for name, idx in indices['occ_group_indices'].items() if any(suffix in name for suffix in unknown_niu_suffixes)}
        indices['classwkr_unknown_indices'] = {idx for name, idx in indices['classwkr_indices'].items() if any(suffix in name for suffix in unknown_niu_suffixes)}

    except (ValueError, KeyError) as e:
        print(f"ERROR: Could not find expected feature indices in metadata. Feature causing error: {e}")
        print(f"Available features: {feature_names}")
        raise ValueError(f"Feature index lookup failed: {e}")

    return indices

# --- Helper: Lookback for categorical features ---
def _find_last_valid_category_index(sequence_history_np: np.ndarray, # Shape (seq_len, n_features)
                                   group_indices: dict, # {feature_name: index}
                                   unknown_indices: set): # {index_of_unknown, index_of_niu, ...}
    """Looks back in time (excluding the last step) to find the last non-unknown category."""
    last_valid_idx = -1
    seq_len = sequence_history_np.shape[0]
    # Look backwards from the second-to-last time step (index seq_len - 2) down to 0
    for t in range(seq_len - 2, -1, -1):
        step_features = sequence_history_np[t, :]
        # Check if any category in this group was active (value > 0.5 for robustness)
        active_indices_in_group = [idx for idx in group_indices.values() if step_features[idx] > 0.5]
        if active_indices_in_group:
            # Found an active category at step t
            found_idx = active_indices_in_group[0] # Assume only one is active due to one-hot
            # Check if it's NOT an unknown/NIU one
            if found_idx not in unknown_indices:
                last_valid_idx = found_idx
                break # Stop lookback once a valid category is found
    return last_valid_idx


def update_sequences_and_features(current_sequences_tensor: torch.Tensor, # Shape: (N, S, F), on DEVICE
                                  sampled_states: torch.Tensor, # Shape: (N,), on DEVICE
                                  original_identifiers_df: pd.DataFrame, # CPU DataFrame, index matches tensor N dim
                                  feature_names: list,
                                  metadata: dict,
                                  current_period: int, # Period *before* prediction (e.g., 202312)
                                  state_rates: dict, # Group rates for this step {state_id: {unemp_rate: x, emp_rate: y}}
                                  industry_rates: dict, # Group rates for this step {ind_id: {unemp_rate: x, emp_rate: y}}
                                  overall_sample_unemp_rate: float, # Overall rate for this step
                                  overall_sample_emp_rate: float, # Overall rate for this step
                                  feature_indices: dict # Pre-calculated indices
                                  ):
    """
    Updates sequences for the next step using vectorized operations where possible.
    Shifts sequence, appends new features based on sampled state.
    Handles time updates, aggregate rate updates (using group/overall rates), and ind/occ/class updates on E transition.
    Operates primarily on NumPy arrays derived from tensors for easier manipulation, then converts back.
    """
    device = current_sequences_tensor.device
    n_individuals, seq_len, n_features = current_sequences_tensor.shape
    pad_val = metadata.get('pad_value', config.PAD_VALUE)
    target_map_inverse = metadata['target_state_map_inverse']

    # --- Find integer indices for states ---
    try:
        employed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'employed')
        unemployed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'unemployed')
        nilf_int = next(k for k, v in target_map_inverse.items() if v.lower().replace(' ', '_') == 'not_in_labor_force' or v.lower() == 'nilf')
    except StopIteration:
        raise ValueError("Metadata missing required state definitions ('employed', 'unemployed', 'nilf'/'not in labor force')")

    # Pre-extract indices from the helper dict
    state_indices = feature_indices['state_indices'] # Dict: {feature_name: index}
    state_indices_values = list(state_indices.values()) # List of indices for state features
    age_idx, months_last_idx, durunemp_idx = feature_indices['age_idx'], feature_indices['months_last_idx'], feature_indices['durunemp_idx']
    nat_unemp_idx, nat_emp_idx = feature_indices['nat_unemp_idx'], feature_indices['nat_emp_idx']
    state_unemp_idx, state_emp_idx = feature_indices['state_unemp_idx'], feature_indices['state_emp_idx']
    ind_unemp_idx, ind_emp_idx = feature_indices['ind_unemp_idx'], feature_indices['ind_emp_idx']
    ind_group_indices, occ_group_indices, classwkr_indices = feature_indices['ind_group_indices'], feature_indices['occ_group_indices'], feature_indices['classwkr_indices']
    ind_unknown_indices, occ_unknown_indices, classwkr_unknown_indices = feature_indices['ind_unknown_indices'], feature_indices['occ_unknown_indices'], feature_indices['classwkr_unknown_indices']

    # --- Determine next period ---
    current_year = current_period // 100
    current_month = current_period % 100
    next_month_is_jan = (current_month == 12)

    # --- Get data onto CPU for processing ---
    sequence_history_np = current_sequences_tensor.cpu().numpy() # (N, S, F)
    previous_features_np = sequence_history_np[:, -1, :] # Features from t-1 (N, F)
    sampled_states_np = sampled_states.cpu().numpy() # Predicted states for next step (N,)

    # Create array for the NEW feature vectors (N, F) - initialized from previous step
    new_feature_vectors = np.copy(previous_features_np)

    # --- Vectorized Feature Updates ---

    # 1. Update 'current_state' based on sampled_state_np
    new_feature_vectors[:, state_indices_values] = 0.0 # Reset all state flags
    # Create mapping from sampled state int to the correct feature index
    state_int_to_feature_idx = {}
    for state_int, state_name in target_map_inverse.items():
        feature_name = f'cat__current_state_{state_name}'
        if feature_name in state_indices:
            state_int_to_feature_idx[state_int] = state_indices[feature_name]
        else:
            print(f"Warning: State feature '{feature_name}' for state {state_int} not found in state_indices.")

    # Map sampled states to their corresponding feature indices
    # Handle potential missing mappings gracefully (though ideally all states should map)
    target_feature_indices = np.array([state_int_to_feature_idx.get(s, -1) for s in sampled_states_np])
    valid_indices_mask = (target_feature_indices != -1)
    # Use advanced indexing to set the correct state flag to 1.0
    if np.any(valid_indices_mask):
        rows_to_update = np.arange(n_individuals)[valid_indices_mask]
        cols_to_update = target_feature_indices[valid_indices_mask]
        new_feature_vectors[rows_to_update, cols_to_update] = 1.0
    if not np.all(valid_indices_mask):
        print(f"Warning: Could not map {np.sum(~valid_indices_mask)} sampled states to feature indices.")


    # 2. Update Time Features (Age, Months Since Last, Duration Unemployed)
    if age_idx != -1 and next_month_is_jan:
        new_feature_vectors[:, age_idx] += 1
    if months_last_idx != -1:
        new_feature_vectors[:, months_last_idx] += 1
    if durunemp_idx != -1:
        is_unemployed_now_mask = (sampled_states_np == unemployed_int)
        # Increment duration if unemployed now, otherwise reset to 0
        new_feature_vectors[:, durunemp_idx] = np.where(is_unemployed_now_mask, previous_features_np[:, durunemp_idx] + 1, 0)

    # 3. Update Aggregate Rate Features (National, State, Industry)
    # Get original identifiers needed for mapping
    original_id_cols = metadata.get('original_identifier_columns', [])
    state_col = 'statefip' if 'statefip' in original_id_cols else None
    ind_col = 'ind_group_cat' if 'ind_group_cat' in original_id_cols else None

    # Map group rates - use pandas merge/map for efficiency
    # Create a temporary df with individual IDs and their group identifiers
    temp_rate_df = pd.DataFrame({
        'id_idx': np.arange(n_individuals), # Index corresponding to numpy arrays
        'state_id': original_identifiers_df[state_col].values if state_col else None,
        'ind_id': original_identifiers_df[ind_col].values if ind_col else None
    })

    # Map state rates
    if state_col and state_unemp_idx != -1 and state_emp_idx != -1 and state_rates:
        state_rates_df = pd.DataFrame.from_dict(state_rates, orient='index').rename_axis('state_id').reset_index()
        temp_rate_df = pd.merge(temp_rate_df, state_rates_df, on='state_id', how='left')
        # Fill missing rates with previous values
        new_state_unemp = temp_rate_df['unemp_rate'].values
        new_state_emp = temp_rate_df['emp_rate'].values
        nan_mask_state_unemp = np.isnan(new_state_unemp)
        nan_mask_state_emp = np.isnan(new_state_emp)
        new_feature_vectors[:, state_unemp_idx] = np.where(nan_mask_state_unemp, previous_features_np[:, state_unemp_idx], new_state_unemp)
        new_feature_vectors[:, state_emp_idx] = np.where(nan_mask_state_emp, previous_features_np[:, state_emp_idx], new_state_emp)
        temp_rate_df.drop(columns=['unemp_rate', 'emp_rate'], inplace=True, errors='ignore') # Clean up columns

    # Map industry rates
    if ind_col and ind_unemp_idx != -1 and ind_emp_idx != -1 and industry_rates:
        ind_rates_df = pd.DataFrame.from_dict(industry_rates, orient='index').rename_axis('ind_id').reset_index()
        temp_rate_df = pd.merge(temp_rate_df, ind_rates_df, on='ind_id', how='left')
        # Fill missing rates with previous values
        new_ind_unemp = temp_rate_df['unemp_rate'].values
        new_ind_emp = temp_rate_df['emp_rate'].values
        nan_mask_ind_unemp = np.isnan(new_ind_unemp)
        nan_mask_ind_emp = np.isnan(new_ind_emp)
        new_feature_vectors[:, ind_unemp_idx] = np.where(nan_mask_ind_unemp, previous_features_np[:, ind_unemp_idx], new_ind_unemp)
        new_feature_vectors[:, ind_emp_idx] = np.where(nan_mask_ind_emp, previous_features_np[:, ind_emp_idx], new_ind_emp)
        # temp_rate_df.drop(columns=['unemp_rate', 'emp_rate'], inplace=True) # Columns already dropped if state rates were processed

    # Update national rates (same for everyone)
    if nat_unemp_idx != -1: new_feature_vectors[:, nat_unemp_idx] = overall_sample_unemp_rate
    if nat_emp_idx != -1: new_feature_vectors[:, nat_emp_idx] = overall_sample_emp_rate

    # --- Non-Vectorized Part: Update Industry/Occupation/Class Worker Features on Transition ---
    # This part remains a loop due to the complexity of the lookback logic

    # Determine previous state from the one-hot encoding in previous_features
    # Find the index of the maximum value (should be 1.0) within the state feature columns
    previous_state_feature_indices = np.argmax(previous_features_np[:, state_indices_values], axis=1)
    # Map these feature indices back to the integer state representation
    feature_idx_to_state_int = {v: k for k, v in state_int_to_feature_idx.items()}
    previous_states_int = np.array([feature_idx_to_state_int.get(state_indices_values[idx], -1)
                                    for idx in previous_state_feature_indices])

    # Identify individuals transitioning to employment
    transition_to_emp_mask = np.isin(previous_states_int, [unemployed_int, nilf_int]) & (sampled_states_np == employed_int)
    transition_indices = np.where(transition_to_emp_mask)[0]

    if len(transition_indices) > 0:
        for i in tqdm(transition_indices, desc="Updating Transition Features", leave=False, disable=len(transition_indices)<1000): # Add tqdm for long loops
            # Use the full sequence history for this individual *before* the update
            ind_sequence_history = sequence_history_np[i] # Shape (seq_len, n_features)

            # Perform lookback for each group using the helper
            last_ind_idx = _find_last_valid_category_index(ind_sequence_history, ind_group_indices, ind_unknown_indices)
            last_occ_idx = _find_last_valid_category_index(ind_sequence_history, occ_group_indices, occ_unknown_indices)
            last_cls_idx = _find_last_valid_category_index(ind_sequence_history, classwkr_indices, classwkr_unknown_indices)

            # Update new feature vector if a valid historical category was found
            if last_ind_idx != -1:
                new_feature_vectors[i, list(ind_group_indices.values())] = 0.0 # Reset group flags
                new_feature_vectors[i, last_ind_idx] = 1.0 # Set historical flag
            if last_occ_idx != -1:
                new_feature_vectors[i, list(occ_group_indices.values())] = 0.0
                new_feature_vectors[i, last_occ_idx] = 1.0
            if last_cls_idx != -1:
                new_feature_vectors[i, list(classwkr_indices.values())] = 0.0
                new_feature_vectors[i, last_cls_idx] = 1.0
            # Else: Keep the features inherited from the previous step (likely 'Unknown'/'NIU')

    # --- Shift sequences and append ---
    # Shift left (remove oldest time step)
    shifted_sequences_np = sequence_history_np[:, 1:, :]
    # Append the newly calculated feature vectors as the last time step
    # Add a time dimension: (N, F) -> (N, 1, F)
    new_sequences_np = np.concatenate([shifted_sequences_np, new_feature_vectors[:, np.newaxis, :]], axis=1)

    # Convert back to tensor and return on the original device
    return torch.from_numpy(new_sequences_np.astype(np.float32)).to(device) # Ensure float32


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
                                      return_raw_samples: bool = False # Control returning raw sample data
                                      ):
    """
    Runs the multi-period forecast simulation, processing one sample fully at a time
    to conserve memory. Calculates and uses group-specific rates dynamically.
    Returns aggregated forecast statistics (median, CI) and optionally raw sample trajectories.
    """
    print(f"\nStarting multi-period forecast:")
    print(f" - Initial Period: {initial_period}")
    print(f" - Periods to Forecast: {periods_to_forecast}")
    print(f" - Number of Samples: {n_samples}")
    print(f" - Device: {device}")
    print(" - Strategy: Sequential processing of samples, dynamic group-rate calculation.")

    # Store results per sample, per period (for unemployment rate primarily)
    all_samples_period_urs = [] # List of lists: outer=samples, inner=UR per period

    n_individuals, seq_len, n_features = initial_sequences_tensor.shape
    # Keep identifiers constant across samples and time (assuming they don't change)
    current_identifiers_df = initial_identifiers_df.copy() # Use a copy
    # Move initial weights to GPU tensor for calculations
    current_weights_gpu = torch.from_numpy(initial_weights_series.values.astype(np.float32)).to(device)

    # --- Get metadata/params needed within the loop ---
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
        print("Feature indices calculated successfully.")
    except ValueError as e:
        print(f"Fatal error during feature index calculation: {e}")
        raise
    # --- End Pre-calculation ---


    # --- Outer loop: Iterate through each Monte Carlo sample ---
    for s in tqdm(range(n_samples), desc="Processing Samples", disable=(n_samples <= 1)):
        sample_period_urs = [] # Store overall UR for each period within this sample

        # Initialize sequence tensor for this sample by cloning the initial state
        # Ensure it's on the correct device
        current_sample_sequences_gpu = initial_sequences_tensor.clone().to(device)
        current_sim_period = initial_period # Reset start period for each sample

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
            # 1. Predict probability distribution for the next state
            probabilities_gpu = forecast_next_period_pytorch(
                current_sample_sequences_gpu, model, device, metadata
            )

            # 2. Sample states and calculate WEIGHTED OVERALL rates for this sample step
            sampled_states_gpu, weighted_overall_unemp_rate, weighted_overall_emp_rate = sample_states_and_calc_rates(
                 probabilities_gpu, current_weights_gpu, metadata
            )
            # Store the weighted overall UR for this period and sample
            sample_period_urs.append(weighted_overall_unemp_rate)

            # 3. Calculate WEIGHTED Group-Specific Rates (State, Industry) for this sample step
            sampled_states_np = sampled_states_gpu.cpu().numpy() # Needs CPU numpy array
            state_rates_dict = {}
            industry_rates_dict = {}
            # Only calculate if the corresponding identifier column exists
            if state_col and state_col in current_identifiers_df.columns:
                state_rates_dict = calculate_group_rates(
                    sampled_states_np, current_identifiers_df[state_col], initial_weights_series, target_map_inverse
                )
            if ind_col and ind_col in current_identifiers_df.columns:
                 industry_rates_dict = calculate_group_rates(
                    sampled_states_np, current_identifiers_df[ind_col], initial_weights_series, target_map_inverse
                )

            # 4. Update this sample's sequences using its sampled states and calculated rates
            current_sample_sequences_gpu = update_sequences_and_features(
                current_sample_sequences_gpu,   # Pass current GPU tensor
                sampled_states_gpu,             # Pass predicted states on GPU
                current_identifiers_df,         # Pass identifiers DataFrame (CPU)
                feature_names,
                metadata,
                current_sim_period,             # The period *before* the prediction
                state_rates_dict,               # Pass calculated state rates (CPU dict)
                industry_rates_dict,            # Pass calculated industry rates (CPU dict)
                weighted_overall_unemp_rate,    # Pass weighted overall rate for national features
                weighted_overall_emp_rate,      # Pass weighted overall rate for national features
                feature_indices                 # Pass pre-calculated indices
            )
            # updated tensor remains on GPU for the next iteration

            # 5. Advance time for the next period in this sample's simulation
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
        empty_agg_df = pd.DataFrame(columns=['period', 'date', 'unemployment_rate_median', 'unemployment_rate_p10', 'unemployment_rate_p90'])
        empty_raw_df = pd.DataFrame()
        return empty_agg_df, empty_raw_df # Return empty dataframes

    sample_urs_over_time = pd.DataFrame(all_samples_period_urs, columns=forecast_periods_list)

    # Calculate quantiles across samples (axis=0) for each period (column)
    forecast_agg_df = pd.DataFrame({
        'period': forecast_periods_list,
        'unemployment_rate_median': sample_urs_over_time.median(axis=0).values,
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


# --- Plotting Function ---
def plot_unemployment_forecast_py(national_rates_file: Path, # Changed input
                                  forecast_agg_df: pd.DataFrame, # Aggregated (median, p10, p90)
                                  sample_urs_over_time: pd.DataFrame, # Raw sample UR data (Samples x Periods)
                                  output_path: Path,
                                  metadata: dict):
    """Plots historical and forecasted unemployment rates, showing individual sample trajectories, median, and 80% CI."""
    print("Creating forecast visualization...")

    # --- Load Historical National Rates ---
    historical_df = pd.DataFrame() # Initialize empty
    if national_rates_file.exists():
        try:
            historical_df = pd.read_csv(national_rates_file)
            # Ensure correct column names and types
            if 'date' not in historical_df.columns or 'national_unemp_rate' not in historical_df.columns:
                 raise ValueError("National rates file missing required columns: 'date', 'national_unemp_rate'")
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            # Rename column for consistency with plotting code below
            historical_df.rename(columns={'national_unemp_rate': 'unemployment_rate'}, inplace=True)
            historical_df = historical_df.dropna(subset=['date', 'unemployment_rate']).sort_values('date')
            print(f"Loaded historical national rates from: {national_rates_file}")
        except Exception as e:
            print(f"Warning: Failed to load or process historical rates from {national_rates_file}: {e}. Plotting forecast only.")
            historical_df = pd.DataFrame() # Reset on error
    else:
        print(f"Warning: Historical national rates file not found at {national_rates_file}. Plotting forecast only.")

    if forecast_agg_df.empty:
        print("Warning: Aggregated forecast data is empty. Cannot plot forecast.")
        return
    # Allow plotting even if raw samples are empty (e.g., if return_raw_samples=False)
    plot_samples = not sample_urs_over_time.empty

    # Ensure date columns are valid datetime objects
    # historical_df date conversion done above
    forecast_agg_df['date'] = pd.to_datetime(forecast_agg_df['date']) # Already done? Ensure it.
    forecast_agg_df = forecast_agg_df.dropna(subset=['date', 'unemployment_rate_median', 'unemployment_rate_p10', 'unemployment_rate_p90']).sort_values('date')

    # --- Determine Plot Range ---
    forecast_start_date = forecast_agg_df['date'].min() if not forecast_agg_df.empty else pd.Timestamp.now()
    # Show ~1 year of history before forecast starts
    history_display_start_date = forecast_start_date - pd.DateOffset(years=1)

    # Prepare data for plotting
    historical_df_display = historical_df[historical_df['date'] >= history_display_start_date] if not historical_df.empty else pd.DataFrame()
    forecast_dates = forecast_agg_df['date'].values
    forecast_median = forecast_agg_df['unemployment_rate_median'].values
    forecast_p10 = forecast_agg_df['unemployment_rate_p10'].values
    forecast_p90 = forecast_agg_df['unemployment_rate_p90'].values

    # --- Create Plot ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot historical data (limited view)
    last_hist_date = None
    last_hist_rate = None
    if not historical_df_display.empty:
        ax.plot(historical_df_display['date'], historical_df_display['unemployment_rate'],
                label='Historical (Recent)', color='black', linewidth=1.5, zorder=5)
        # Get the last historical point to connect the forecast
        last_hist_point = historical_df_display.iloc[-1]
        last_hist_date = last_hist_point['date']
        last_hist_rate = last_hist_point['unemployment_rate']

    # Prepend last historical point to forecast data for visual connection
    plot_forecast_dates = forecast_dates
    plot_forecast_median = forecast_median
    plot_forecast_p10 = forecast_p10
    plot_forecast_p90 = forecast_p90
    plot_sample_urs_df = sample_urs_over_time # Use original if plotting samples

    if last_hist_date is not None and last_hist_rate is not None and not forecast_agg_df.empty:
        # Ensure the first forecast date is immediately after the last historical date
        # (This assumes monthly data and correct alignment)
        if forecast_dates[0] > last_hist_date:
            plot_forecast_dates = np.insert(forecast_dates, 0, last_hist_date)
            plot_forecast_median = np.insert(forecast_median, 0, last_hist_rate)
            plot_forecast_p10 = np.insert(forecast_p10, 0, last_hist_rate) # Start CI from last known point
            plot_forecast_p90 = np.insert(forecast_p90, 0, last_hist_rate) # Start CI from last known point

            # Also adjust sample trajectories if they exist and are being plotted
            if plot_samples:
                # Create a temporary DataFrame for plotting samples
                plot_sample_urs_df = sample_urs_over_time.copy()
                # Add the last historical rate as the first column, using the last historical date as the column name
                # Convert date to a suitable column name (e.g., string) if necessary
                last_hist_date_col = last_hist_date # pd.Timestamp works as column name
                plot_sample_urs_df.insert(0, last_hist_date_col, last_hist_rate)


    # Plot individual sample trajectories (if available and requested) using potentially adjusted data
    sample_plot_dates = None # Initialize
    if plot_samples:
        n_samples_to_plot = plot_sample_urs_df.shape[0]
        # Use the adjusted dates and sample data
        # Ensure the number of date points matches the number of columns in the sample data
        if plot_forecast_dates.size == plot_sample_urs_df.shape[1]:
            sample_plot_dates = plot_forecast_dates
            for i in range(n_samples_to_plot):
                ax.plot(sample_plot_dates, plot_sample_urs_df.iloc[i].values,
                        color='steelblue', alpha=0.25, linewidth=0.6, zorder=1) # Darker color, slightly higher alpha/linewidth
            # Add a label for the samples collection only once (won't appear in legend by default)
            ax.plot([], [], color='steelblue', alpha=0.5, linewidth=1, label='Sample Trajectories') # Dummy plot for legend
        else:
            print(f"Warning: Mismatch between number of forecast dates ({plot_forecast_dates.size}) and sample data columns ({plot_sample_urs_df.shape[1]}) after adjustment. Skipping sample trajectory plotting.")


    # Plot confidence interval (P10-P90) - light shading using potentially adjusted data
    ax.fill_between(plot_forecast_dates,
                    plot_forecast_p10,
                    plot_forecast_p90,
                    color='skyblue', alpha=0.4, label='Forecast (80% CI)', zorder=2)

    # Plot forecast median (darker line) using potentially adjusted data
    ax.plot(plot_forecast_dates, plot_forecast_median,
            label='Forecast (Median)', color='blue', linewidth=2.0, linestyle='--', zorder=3)

    # Use state mapping for title/labels if needed
    target_map_inverse = metadata.get('target_state_map_inverse', {1: 'Unemployed'})
    unemployed_label = target_map_inverse.get(next((k for k, v in target_map_inverse.items() if v.lower() == 'unemployed'), 1), 'Unemployed')

    # Formatting
    ax.set_title('Unemployment Rate Forecast (Transformer Model)', fontsize=16, fontweight='bold') # Updated Title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Unemployment Rate', fontsize=12) # Updated Label
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}')) # Format as percentage

    # Improve date axis formatting
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    # Filter labels slightly if needed, e.g., remove sample trajectories from explicit legend
    # Example: keep only Historical, Median, CI
    filtered_handles = [h for h, l in zip(handles, labels) if l in ['Historical (Recent)', 'Forecast (Median)', 'Forecast (80% CI)']]
    filtered_labels = [l for l in labels if l in ['Historical (Recent)', 'Forecast (Median)', 'Forecast (80% CI)']]
    # Check if sample trajectories were plotted using the adjusted data check
    if plot_samples and sample_plot_dates is not None:
         # Add the dummy 'Sample Trajectories' label if plotted
         sample_handle = next((h for h, l in zip(handles, labels) if l == 'Sample Trajectories'), None)
         if sample_handle:
              filtered_handles.append(sample_handle)
              filtered_labels.append('Sample Trajectories')

    ax.legend(filtered_handles, filtered_labels, fontsize=11)

    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0) # Start y-axis at 0
    fig.autofmt_xdate() # Rotate date labels if needed

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory

# --- HPT Objective Specific Functions ---

def calculate_hpt_forecast_metrics(model: nn.Module,
                                hpt_val_baked_data: pd.DataFrame, # Still needed to get start periods/sequences
                                national_rates_file: Path, # Path to the truth data
                                metadata: dict,
                                params: dict,
                                device,
                                # num_forecasts: int = 3, # Number of forecasts determined by intervals
                                forecast_horizon: int = 12): # Months to forecast
    """
    Calculates the HPT objective: RMSE and Standard Deviation of aggregate unemployment rate error
    over multiple recursive forecasts, using the pre-calculated national rates file as truth.
    Forecasts start at the beginning of each interval defined in metadata['hpt_validation_intervals'].
    """
    print("\n===== Calculating HPT Metric (Forecast RMSE & Std Dev vs National Rates File) =====")
    if hpt_val_baked_data is None or hpt_val_baked_data.empty:
        print("HPT validation baked data is empty. Cannot select start periods.")
        return {'rmse': float('inf'), 'std_dev': float('inf')} # Changed variance to std_dev
    if not national_rates_file.exists():
        print(f"ERROR: National rates file not found at {national_rates_file}. Cannot calculate HPT forecast metrics.")
        return {'rmse': float('inf'), 'std_dev': float('inf')} # Changed variance to std_dev
    if 'hpt_validation_intervals' not in metadata or not metadata['hpt_validation_intervals']:
        print("ERROR: 'hpt_validation_intervals' not found or empty in metadata. Cannot determine forecast start periods.")
        return {'rmse': float('inf'), 'std_dev': float('inf')} # Changed variance to std_dev

    model.eval() # Ensure model is in eval mode
    all_forecast_errors = [] # Store (predicted_rate - actual_rate) for each month across all forecasts

    # --- Load Actual National Rates ---
    try:
        actual_rates_df = pd.read_csv(national_rates_file)
        if 'date' not in actual_rates_df.columns or 'national_unemp_rate' not in actual_rates_df.columns:
            raise ValueError("National rates file missing required columns: 'date', 'national_unemp_rate'")
        actual_rates_df['date'] = pd.to_datetime(actual_rates_df['date'])
        # Calculate period column for merging
        actual_rates_df['period'] = actual_rates_df['date'].dt.year * 100 + actual_rates_df['date'].dt.month
        actual_rates_df.rename(columns={'national_unemp_rate': 'actual_unemployment_rate'}, inplace=True)
        actual_rates_df = actual_rates_df[['period', 'actual_unemployment_rate']].dropna()
        print(f"Loaded actual national rates from {national_rates_file} for {len(actual_rates_df)} periods.")
        if actual_rates_df.empty:
            print("Warning: Loaded actual rates data is empty.")
            return {'rmse': float('inf'), 'std_dev': float('inf')} # Changed variance to std_dev
    except Exception as e:
        print(f"Error loading or processing actual rates file {national_rates_file}: {e}")
        return {'rmse': float('inf'), 'std_dev': float('inf')} # Changed variance to std_dev

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
        return {'rmse': float('inf'), 'std_dev': float('inf')} # Changed variance to std_dev

    if not start_periods:
        print("ERROR: No valid start periods found based on HPT intervals and available actual rates data.")
        return {'rmse': float('inf'), 'std_dev': float('inf')} # Changed variance to std_dev

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

            # 3. Run the recursive forecast (using n_samples=1 for HPT objective)
            forecast_agg_df, _ = forecast_multiple_periods_pytorch(
                initial_sequences_tensor=initial_sequences_tensor,
                initial_identifiers_df=initial_identifiers_df,
                initial_weights_series=initial_weights_series,
                model=model,
                device=device,
                metadata=metadata,
                params=params,
                initial_period=start_period,
                periods_to_forecast=forecast_horizon,
                n_samples=1, # Only need one sample path (median effectively) for RMSE calculation
                return_raw_samples=False
            )

            if forecast_agg_df.empty:
                print(f"Warning: Forecast from start period {start_period} returned empty results.")
                continue

            # 4. Merge with actual rates (loaded from file) and calculate errors
            comparison_df = pd.merge(
                forecast_agg_df[['period', 'unemployment_rate_median']],
                actual_rates_df, # Use the dataframe loaded from national_rates_file
                on='period',
                how='inner' # Only compare months where both forecast and actual exist
            )

            if len(comparison_df) != forecast_horizon:
                 print(f"Warning: Forecast from {start_period} - Expected {forecast_horizon} comparison months, found {len(comparison_df)}. Check actual data availability in national rates file.")

            if not comparison_df.empty:
                # --- Added Diagnostic Printing ---
                print(f"  Comparison Data (Forecast vs Actual) for start_period {start_period}:")
                print(comparison_df.to_string(index=False, float_format="%.6f"))
                # --- End Added Diagnostic Printing ---

                errors = comparison_df['unemployment_rate_median'] - comparison_df['actual_unemployment_rate']
                all_forecast_errors.extend(errors.tolist())
                run_rmse = np.sqrt(mean_squared_error(comparison_df['actual_unemployment_rate'], comparison_df['unemployment_rate_median']))
                print(f"  -> Forecast run from {start_period} completed. RMSE for this run: {run_rmse:.6f}")
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

    # --- Calculate Final Metrics ---
    if not all_forecast_errors:
        print("ERROR: No forecast errors collected across all runs. Cannot calculate final metrics.")
        return {'rmse': float('inf'), 'std_dev': float('inf')} # Changed variance to std_dev

    final_rmse = np.sqrt(np.mean(np.square(all_forecast_errors)))
    final_std_dev = np.std(all_forecast_errors) # Calculate standard deviation
    print(f"\nCalculated Final HPT Metrics (RMSE and Std Dev over {len(all_forecast_errors)} months across {num_forecasts} forecasts):")
    print(f"  RMSE: {final_rmse:.6f}")
    print(f"  Std Dev: {final_std_dev:.6f}") # Print standard deviation

    return {'rmse': final_rmse, 'std_dev': final_std_dev} # Return std_dev

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
