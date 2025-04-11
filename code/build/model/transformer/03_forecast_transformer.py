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
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import math
import sys
import multiprocessing # Import multiprocessing

# Fix imports - add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent))

# Import Model Definitions with absolute import
from models import PositionalEmbedding, TransformerEncoderBlock, TransformerForecastingModel

# --- Import Config --- # Added
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Make sure it's in the same directory or sys.path is configured correctly.")
    sys.exit(1)

# --- Constants --- # Removed DEFAULT_FORECAST_PERIODS

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

# --- Utility Functions ---
def period_to_date(period):
    """Converts YYYYMM integer to datetime.date object for plotting."""
    if pd.isna(period): return None
    try:
        year = int(period // 100)
        month = int(period % 100)
        if 1 <= month <= 12:
            return datetime(year, month, 1).date()
        else:
            return None # Invalid month
    except:
        return None

# --- Loading Functions ---
# Modified to use config paths
def load_pytorch_model_and_params(model_dir: Path, processed_data_dir: Path, device):
    """Loads the PyTorch model, parameters, and metadata."""
    print(f"Loading model artifacts from: {model_dir}")
    # Use the checkpoint name saved during training
    model_path = model_dir / "best_model_val_loss.pt" # Changed filename
    params_path = model_dir / "model_params.pkl"
    metadata_path = processed_data_dir / config.METADATA_FILENAME # Use config

    if not model_path.exists():
        # Try the old name as a fallback? Or just error. Let's error for clarity.
        # old_model_path = model_dir / "transformer_model.pt"
        # if old_model_path.exists():
        #     print(f"Warning: '{model_path.name}' not found, falling back to '{old_model_path.name}'.")
        #     model_path = old_model_path
        # else:
        raise FileNotFoundError(f"Missing model weights file '{model_path.name}' in {model_dir}")

    if not params_path.exists():
        raise FileNotFoundError(f"Missing model params file '{params_path.name}' in {model_dir}")
    if not metadata_path.exists():
         raise FileNotFoundError(f"Missing metadata file at {metadata_path}")

    # Load parameters used to build the model
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    print("Loaded model parameters.")

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print("Loaded preprocessing metadata.")

    # Validate required metadata keys
    required_meta_keys = ['n_features', 'n_classes', 'target_state_map_inverse', 'feature_names', 'pad_value', 'original_identifier_columns'] # Added original_identifier_columns
    missing_keys = [k for k in required_meta_keys if k not in metadata]
    if missing_keys:
        raise KeyError(f"Metadata is missing required keys: {missing_keys}")

    # Rebuild model architecture using params from training run
    model = TransformerForecastingModel(
        input_dim=params['n_features'],
        seq_len=params['sequence_length'], # Use params['sequence_length']
        embed_dim=params['embed_dim'],
        num_heads=params['num_heads'],
        ff_dim=params['ff_dim'],
        num_transformer_blocks=params['num_transformer_blocks'],
        mlp_units=params['mlp_units'],
        dropout=params['dropout'],
        mlp_dropout=params['mlp_dropout'],
        n_classes=params['n_classes'] # n_classes should also be in params
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode
    print(f"Model loaded successfully from {model_path} and set to eval mode.")

    return model, params, metadata

# Modified to use config paths and params
def load_and_prepare_data(processed_data_dir: Path, raw_data_path: Path, metadata: dict, start_year: int = None, start_month: int = None):
    """Loads the FULL baked data for simulation start and raw data for historical rates.
       Filters data based on specified start_year and start_month if provided, otherwise uses the latest period.
    """
    print("Loading data...")
    # --- Load FULL Baked Data ---
    # Use the full baked data file specified in metadata or config
    full_baked_filename = metadata.get('full_baked_data_path', processed_data_dir / config.FULL_DATA_FILENAME) # Get path from metadata if available
    latest_baked_path = Path(full_baked_filename) # Ensure it's a Path object

    if not latest_baked_path.exists():
        # Fallback to test data if full data path isn't found or specified (optional, depends on desired behavior)
        print(f"Warning: Full baked data not found at {latest_baked_path}. Falling back to test data.")
        latest_baked_path = processed_data_dir / config.TEST_DATA_FILENAME
        if not latest_baked_path.exists():
             raise FileNotFoundError(f"Neither full baked data nor test data found at expected locations.")
        data_source_msg = "TEST"
    else:
        data_source_msg = "FULL"


    latest_baked_df = pd.read_parquet(latest_baked_path)
    print(f"Loaded {data_source_msg} baked data: {latest_baked_df.shape}")
    # --- End Load Baked Data ---

    # Ensure original identifier columns are present
    original_id_cols = metadata.get('original_identifier_columns', [])
    missing_orig_cols = [col for col in original_id_cols if col not in latest_baked_df.columns]
    if missing_orig_cols:
        raise ValueError(f"Baked data is missing required original identifier columns: {missing_orig_cols}")
    print(f"Found original identifier columns in baked data: {original_id_cols}")

    # Calculate period
    latest_baked_df[config.DATE_COL] = pd.to_datetime(latest_baked_df[config.DATE_COL]) # Use config
    latest_baked_df['period'] = latest_baked_df[config.DATE_COL].dt.year * 100 + latest_baked_df[config.DATE_COL].dt.month

    # Determine the start period for the simulation
    if start_year and start_month:
        start_period = start_year * 100 + start_month
        if start_period not in latest_baked_df['period'].unique():
             raise ValueError(f"Specified start period {start_period} ({start_year}-{start_month:02d}) not found in the baked data.")
        print(f"Using specified start period: {start_period}")
    else:
        start_period = latest_baked_df['period'].max()
        print(f"Using latest period in baked data as start period: {start_period}")


    # Get individuals present in the start period
    start_period_ids = latest_baked_df[latest_baked_df['period'] == start_period][config.GROUP_ID_COL].unique() # Use config
    print(f"Found {len(start_period_ids)} individuals in the start period ({start_period}).")

    # Filter the baked data to include only the history *up to* the start period for these individuals
    initial_sim_data = latest_baked_df[
        (latest_baked_df[config.GROUP_ID_COL].isin(start_period_ids)) & # Use config
        (latest_baked_df['period'] <= start_period)
    ].sort_values([config.GROUP_ID_COL, config.DATE_COL]) # Use config

    # --- Delete the large full dataframe ---
    del latest_baked_df
    import gc
    gc.collect()
    print("Deleted full baked dataframe from memory.")
    # ---

    # --- Load Raw Data for Historical Rates ---
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")
    raw_df = pd.read_csv(raw_data_path, dtype={config.GROUP_ID_COL: str}) # Use config
    raw_df[config.DATE_COL] = pd.to_datetime(raw_df[config.DATE_COL]) # Use config
    raw_df['period'] = raw_df[config.DATE_COL].dt.year * 100 + raw_df[config.DATE_COL].dt.month
    # Assuming 'emp_state' is the column name in the raw file
    raw_df['emp_state'] = raw_df['emp_state'].str.lower().replace('not_in_labor_force', 'nilf')

    # Calculate historical rates using ALL available raw data
    # Use the state mapping from metadata for consistency
    target_map_inverse = metadata.get('target_state_map_inverse', {0: 'employed', 1: 'unemployed', 2: 'nilf'}) # Default if missing
    state_map_lower = {k: v.lower().replace('not_in_labor_force', 'nilf') for k, v in target_map_inverse.items()}

    # Group by period and calculate state counts for all periods in raw_df
    historical_rates = raw_df.groupby('period').apply(
        lambda x: pd.Series({
            state_map_lower[0]: (x['emp_state'] == state_map_lower[0]).sum(),
            state_map_lower[1]: (x['emp_state'] == state_map_lower[1]).sum(),
            state_map_lower[2]: (x['emp_state'] == state_map_lower[2]).sum()
        })
    ).reset_index()

    historical_rates['labor_force'] = historical_rates[state_map_lower[0]] + historical_rates[state_map_lower[1]]
    historical_rates['unemployment_rate'] = historical_rates[state_map_lower[1]] / historical_rates['labor_force']
    historical_rates['date'] = historical_rates['period'].apply(period_to_date)
    print(f"Calculated historical rates for {historical_rates.shape[0]} periods (all available).")

    # Return the actual start period used
    return initial_sim_data, historical_rates, start_period, start_period_ids

# --- Forecasting Functions ---

# Modified to use config params and return original identifiers
def get_sequences_for_simulation(sim_data_df: pd.DataFrame, group_col: str, seq_len: int, features: list, original_id_cols: list, pad_val: float, end_period: int):
    """
    Extracts the sequence ending at `end_period` for each individual.
    Also extracts the original identifier columns for the last observation of each individual.
    """
    print(f"Extracting sequences and identifiers ending at period {end_period} for each individual...")
    sequences = {}
    original_identifiers = {} # Store original IDs for the last time step

    # Ensure data is filtered up to the end_period before grouping
    filtered_sim_data = sim_data_df[sim_data_df['period'] <= end_period]

    grouped = filtered_sim_data.groupby(group_col)
    for person_id, person_df in tqdm(grouped, desc="Extracting Sequences"):
        # Ensure we sort by date to get the most recent observations ending at end_period
        person_df = person_df.sort_values(by=config.DATE_COL) # Use config

        # Check if the person exists in the specific end_period
        if person_df['period'].max() != end_period:
            continue # Skip if the person is not present in the exact end period

        # Extract features for sequence
        person_features = person_df[features].values.astype(np.float32)
        # Extract original identifiers for the *last* observation
        last_obs_identifiers = person_df.iloc[-1][original_id_cols].to_dict()

        n_obs = person_features.shape[0]
        if n_obs == 0: continue

        # Extract the last 'seq_len' observations for features
        start_index = max(0, n_obs - seq_len)
        sequence_data = person_features[start_index:, :]
        actual_len = sequence_data.shape[0]
        pad_len = seq_len - actual_len

        if pad_len < 0:
            padded_sequence = sequence_data[-seq_len:, :]
        elif pad_len == 0:
            padded_sequence = sequence_data
        else:
            padding_matrix = np.full((pad_len, len(features)), pad_val, dtype=np.float32)
            padded_sequence = np.vstack((padding_matrix, sequence_data))

        if padded_sequence.shape == (seq_len, len(features)):
            sequences[person_id] = padded_sequence
            original_identifiers[person_id] = last_obs_identifiers # Store the identifiers
        else:
            print(f"Warning: Shape mismatch for {person_id}. Got {padded_sequence.shape}, expected {(seq_len, len(features))}")

    print(f"Extracted sequences for {len(sequences)} individuals ending at period {end_period}.")
    if not sequences:
        raise ValueError(f"No sequences could be extracted for simulation ending at period {end_period}.")

    # Stack sequences into a NumPy array for batch processing
    ids_list = list(sequences.keys())
    sequences_np = np.stack([sequences[pid] for pid in ids_list], axis=0)
    # Create a DataFrame for original identifiers, indexed by person_id
    original_identifiers_df = pd.DataFrame.from_dict(original_identifiers, orient='index').reindex(ids_list)

    return sequences_np, ids_list, original_identifiers_df


# --- Function to Forecast Next Period with PyTorch model ---
def forecast_next_period_pytorch(sequences_tensor: torch.Tensor, model: nn.Module, device, metadata: dict):
    """
    Forecasts the next labor market state for each individual using the transformer model.
    
    Args:
        sequences_tensor: Tensor of shape (n_individuals, seq_len, n_features)
        model: PyTorch model to use for forecasting
        device: PyTorch device
        metadata: Dictionary with metadata including state mapping
        
    Returns:
        sampled_states: Tensor with predicted state indices
        overall_unemp_rate: Unemployment rate across all individuals
        overall_emp_rate: Employment rate across all individuals
    """
    # Ensure model is in evaluation mode
    model.eval()
    n_individuals = sequences_tensor.shape[0]
    n_classes = metadata.get('n_classes', 3)
    target_map_inverse = metadata.get('target_state_map_inverse', {})
    pad_value = metadata.get('pad_value', config.PAD_VALUE) # Get pad value

    # Create padding mask if necessary (assuming model uses it)
    # Check if the model's forward method accepts src_key_padding_mask
    # This check might need refinement based on the actual model implementation
    padding_mask = None
    if 'src_key_padding_mask' in model.forward.__code__.co_varnames:
         padding_mask = torch.all(sequences_tensor == pad_value, dim=-1).to(device)

    # Predict next state probabilities
    with torch.no_grad():
        if padding_mask is not None:
            logits = model(sequences_tensor, src_key_padding_mask=padding_mask)
        else:
            logits = model(sequences_tensor) # Assume model handles padding internally if mask not needed
        probabilities = torch.softmax(logits, dim=1)

    # Sample states based on probabilities
    sampled_states = torch.multinomial(probabilities, num_samples=1).squeeze(1)

    # Calculate unemployment and employment rates
    states_np = sampled_states.cpu().numpy()
    total_individuals = n_individuals

    # Map state indices to state names
    employed_state = 0  # Assuming 0 = employed
    unemployed_state = 1  # Assuming 1 = unemployed

    employed_count = np.sum(states_np == employed_state)
    unemployed_count = np.sum(states_np == unemployed_state)
    labor_force = employed_count + unemployed_count

    if labor_force > 0:
        overall_unemp_rate = unemployed_count / labor_force
        overall_emp_rate = employed_count / labor_force
    else:
        overall_unemp_rate = 0.0
        overall_emp_rate = 0.0

    return sampled_states, overall_unemp_rate, overall_emp_rate

# --- New Function to Calculate Group Rates ---
def calculate_group_rates(sampled_states_np: np.ndarray, # Shape (n_individuals,)
                          grouping_series: pd.Series, # Series with group IDs (statefip or ind_group_cat), index matches sampled_states
                          target_map_inverse: dict):
    """Calculates unemployment and employment rates for each group."""
    # Map integer states to names (assuming 0: employed, 1: unemployed, 2: nilf)
    state_0_name = target_map_inverse.get(0, 'employed')
    state_1_name = target_map_inverse.get(1, 'unemployed')
    state_2_name = target_map_inverse.get(2, 'nilf')

    # Create a DataFrame for calculation
    df = pd.DataFrame({
        'group': grouping_series,
        'state': sampled_states_np
    })

    # Count states within each group
    counts = df.groupby('group')['state'].value_counts().unstack(fill_value=0)

    # Ensure all state columns exist, even if count is 0
    for state_int in target_map_inverse.keys():
        if state_int not in counts.columns:
            counts[state_int] = 0

    # Calculate rates
    employed_count = counts.get(0, 0) # Default to 0 if column doesn't exist
    unemployed_count = counts.get(1, 0)
    labor_force = employed_count + unemployed_count
    rates = pd.DataFrame(index=counts.index)
    rates['unemp_rate'] = np.where(labor_force > 0, unemployed_count / labor_force, 0.0)
    rates['emp_rate'] = np.where(labor_force > 0, employed_count / labor_force, 0.0) # Emp / LF

    return rates.to_dict(orient='index')


# Modified to use group-specific rates and handle ind/occ/class updates
def update_sequences_and_features(current_sequences_tensor: torch.Tensor, # Shape: (n_individuals, seq_len, n_features)
                                  sampled_states: torch.Tensor, # Shape: (n_individuals,)
                                  original_identifiers_df: pd.DataFrame,
                                  feature_names: list,
                                  metadata: dict,
                                  current_period: int,
                                  state_rates: dict,
                                  industry_rates: dict,
                                  overall_sample_unemp_rate: float,
                                  overall_sample_emp_rate: float):
    """
    Updates sequences for the next step for ONE sample run. Shifts sequence, appends new features based on sampled state.
    Handles basic time updates and updates aggregate rate features using the provided GROUP-SPECIFIC rates
    (defaulting to previous value if group rate is missing) and overall sample rates for national features.
    Handles industry/occupation/class updates on transition to employment using historical lookback.
    """
    n_individuals, seq_len, n_features = current_sequences_tensor.shape
    pad_val = config.PAD_VALUE
    n_classes = metadata.get('n_classes', 3)
    original_id_cols = metadata.get('original_identifier_columns', [])
    state_col = 'statefip' if 'statefip' in original_id_cols else None
    ind_col = 'ind_group_cat' if 'ind_group_cat' in original_id_cols else None
    # Add occ_col and class_col if they are part of original_id_cols and needed for lookups (unlikely)
    # occ_col = 'occ_group_cat' if 'occ_group_cat' in original_id_cols else None
    # class_col = 'classwkr_cat' if 'classwkr_cat' in original_id_cols else None

    # --- Identify key feature indices ---
    try:
        target_map_inverse = metadata['target_state_map_inverse']
        # Map integer state to lowercase string representation
        int_to_state_str = {k: v.lower().replace(' ', '_').replace('_in_labor_force', '') for k, v in target_map_inverse.items()}
        employed_int = next((k for k, v in int_to_state_str.items() if v == 'employed'), 0)
        unemployed_int = next((k for k, v in int_to_state_str.items() if v == 'unemployed'), 1)
        nilf_int = next((k for k, v in int_to_state_str.items() if v == 'not'), 2)

        # Indices for one-hot encoded current state
        state_cols = [f for f in feature_names if f.startswith('cat__current_state_')]
        state_indices = {col: feature_names.index(col) for col in state_cols}

        # Indices for other features to update
        age_idx = feature_names.index('num__age') if 'num__age' in feature_names else -1
        durunemp_idx = feature_names.index('num__durunemp') if 'num__durunemp' in feature_names else -1
        months_last_idx = feature_names.index('num__months_since_last') if 'num__months_since_last' in feature_names else -1
        nat_unemp_idx = feature_names.index('num__national_unemp_rate') if 'num__national_unemp_rate' in feature_names else -1
        nat_emp_idx = feature_names.index('num__national_emp_rate') if 'num__national_emp_rate' in feature_names else -1
        state_unemp_idx = feature_names.index('num__state_unemp_rate') if 'num__state_unemp_rate' in feature_names else -1
        state_emp_idx = feature_names.index('num__state_emp_rate') if 'num__state_emp_rate' in feature_names else -1
        ind_unemp_idx = feature_names.index('num__ind_group_unemp_rate') if 'num__ind_group_unemp_rate' in feature_names else -1
        ind_emp_idx = feature_names.index('num__ind_group_emp_rate') if 'num__ind_group_emp_rate' in feature_names else -1

        # Indices for one-hot encoded industry, occupation, class worker features
        ind_group_indices = {f: feature_names.index(f) for f in feature_names if f.startswith('cat__ind_group_cat_')}
        occ_group_indices = {f: feature_names.index(f) for f in feature_names if f.startswith('cat__occ_group_cat_')}
        classwkr_indices = {f: feature_names.index(f) for f in feature_names if f.startswith('cat__classwkr_cat_')}

        # Identify the index for the 'Unknown' or 'NIU' category within each group, if it exists
        # These categories should NOT be carried forward during the lookback
        unknown_suffixes = ['unknown', 'niu', 'other/niu', 'unemployed/niu', '_other_'] # Possible suffixes/names for unknown/NIU categories
        def get_unknown_indices(group_indices, suffixes):
            unknown_idx = set()
            for name, idx in group_indices.items():
                for suffix in suffixes:
                    if name.lower().endswith(suffix):
                        unknown_idx.add(idx)
                        break
            return list(unknown_idx)

        ind_unknown_indices = get_unknown_indices(ind_group_indices, unknown_suffixes)
        occ_unknown_indices = get_unknown_indices(occ_group_indices, unknown_suffixes)
        classwkr_unknown_indices = get_unknown_indices(classwkr_indices, unknown_suffixes)

    except (ValueError, KeyError, StopIteration) as e:
        print(f"ERROR: Could not find expected feature indices or state mapping in metadata. Error: {e}")
        raise

    # --- Calculate next period's time features ---
    # (This part remains the same)
    current_year = current_period // 100
    current_month = current_period % 100
    if current_month == 12: next_month = 1
    else: next_month = current_month + 1

    # --- Create new feature vectors ---
    # Initialize by copying the features from the *last* time step of the input sequence
    new_feature_vectors = np.full((n_individuals, n_features), pad_val, dtype=np.float32)
    last_valid_features_np = current_sequences_tensor.cpu().numpy()[:, -1, :]
    new_feature_vectors[:, :] = last_valid_features_np # Start with previous step's features
    sampled_states_np = sampled_states.cpu().numpy()
    sequence_history_np = current_sequences_tensor.cpu().numpy() # Get full history for lookback

    # --- Update features ---
    for i in range(n_individuals):
        sampled_state_int = sampled_states_np[i]
        ind_identifiers = original_identifiers_df.iloc[i]
        previous_features = last_valid_features_np[i] # Features from the step *before* prediction

        # Determine previous state from the one-hot encoding in previous_features
        previous_state_int = -1
        for state_col_name, state_idx in state_indices.items():
            if previous_features[state_idx] > 0.5: # Check if this state was active
                try:
                    # Extract state integer from column name like 'cat__current_state_employed' -> employed_int
                    state_str = state_col_name.split('cat__current_state_')[-1]
                    previous_state_int = next(k for k, v in int_to_state_str.items() if v == state_str)
                    break
                except (StopIteration, IndexError): continue # Should not happen if names are consistent

        # 1. Update 'current_state' based on sampled_state_int
        for col_name, idx in state_indices.items(): new_feature_vectors[i, idx] = 0.0 # Reset all state flags
        try:
            state_suffix = int_to_state_str[sampled_state_int]
            target_col_name = f'cat__current_state_{state_suffix}'
            target_idx = state_indices[target_col_name]
            new_feature_vectors[i, target_idx] = 1.0
        except KeyError:
             print(f"Warning: Sampled state {sampled_state_int} not found in int_to_state_str map.")

        # 2. Update Time Features (Age, Months Since Last, Duration Unemployed)
        if age_idx != -1 and next_month == 1: new_feature_vectors[i, age_idx] += 1 # Increment age in January
        if months_last_idx != -1: new_feature_vectors[i, months_last_idx] += 1 # Increment months since last observation
        if durunemp_idx != -1:
            is_unemployed_now = (sampled_state_int == unemployed_int)
            # Increment duration if unemployed, reset if not. Use previous duration value.
            new_feature_vectors[i, durunemp_idx] = previous_features[durunemp_idx] + 1 if is_unemployed_now else 0

        # 3. Update Aggregate Rate Features (National, State, Industry)
        # (This part remains the same - uses group rates calculated outside)
        current_state_id = ind_identifiers.get(state_col, None) if state_col else None
        current_ind_id = ind_identifiers.get(ind_col, None) if ind_col else None
        ind_state_rates = state_rates.get(current_state_id, None) if current_state_id else None
        ind_industry_rates = industry_rates.get(current_ind_id, None) if current_ind_id else None

        if state_unemp_idx != -1:
            new_rate = ind_state_rates.get('unemp_rate') if ind_state_rates else None
            new_feature_vectors[i, state_unemp_idx] = new_rate if new_rate is not None else previous_features[state_unemp_idx]
        if state_emp_idx != -1:
            new_rate = ind_state_rates.get('emp_rate') if ind_state_rates else None
            new_feature_vectors[i, state_emp_idx] = new_rate if new_rate is not None else previous_features[state_emp_idx]
        if ind_unemp_idx != -1:
            new_rate = ind_industry_rates.get('unemp_rate') if ind_industry_rates else None
            new_feature_vectors[i, ind_unemp_idx] = new_rate if new_rate is not None else previous_features[ind_unemp_idx]
        if ind_emp_idx != -1:
            new_rate = ind_industry_rates.get('emp_rate') if ind_industry_rates else None
            new_feature_vectors[i, ind_emp_idx] = new_rate if new_rate is not None else previous_features[ind_emp_idx]

        if nat_unemp_idx != -1: new_feature_vectors[i, nat_unemp_idx] = overall_sample_unemp_rate
        if nat_emp_idx != -1: new_feature_vectors[i, nat_emp_idx] = overall_sample_emp_rate

        # 4. Update Industry/Occupation/Class Worker Features on Transition to Employment
        transition_to_emp = (previous_state_int in [unemployed_int, nilf_int]) and (sampled_state_int == employed_int)

        if transition_to_emp:
            # Helper function for lookback
            def find_last_valid_category_index(history, group_indices, unknown_indices):
                last_valid_idx = -1
                # Look backwards from the second-to-last time step (t = seq_len - 2)
                for t in range(seq_len - 2, -1, -1):
                    step_features = history[t, :]
                    # Check if any category in this group was active (value > 0.5)
                    active_indices = [idx for idx in group_indices.values() if step_features[idx] > 0.5]
                    if active_indices:
                        # Found an active category, check if it's NOT an unknown/NIU one
                        found_idx = active_indices[0] # Assume only one is active due to one-hot
                        if found_idx not in unknown_indices:
                            last_valid_idx = found_idx
                            break # Stop lookback once a valid category is found
                return last_valid_idx

            # Perform lookback for each group
            last_ind_idx = find_last_valid_category_index(sequence_history_np[i], ind_group_indices, ind_unknown_indices)
            last_occ_idx = find_last_valid_category_index(sequence_history_np[i], occ_group_indices, occ_unknown_indices)
            last_cls_idx = find_last_valid_category_index(sequence_history_np[i], classwkr_indices, classwkr_unknown_indices)

            # Update new feature vector if a valid historical category was found
            if last_ind_idx != -1:
                for idx in ind_group_indices.values(): new_feature_vectors[i, idx] = 0.0 # Reset group
                new_feature_vectors[i, last_ind_idx] = 1.0 # Set historical category
            # Else: Keep the features inherited from the previous step (likely 'Unknown'/'NIU')

            if last_occ_idx != -1:
                for idx in occ_group_indices.values(): new_feature_vectors[i, idx] = 0.0
                new_feature_vectors[i, last_occ_idx] = 1.0

            if last_cls_idx != -1:
                for idx in classwkr_indices.values(): new_feature_vectors[i, idx] = 0.0
                new_feature_vectors[i, last_cls_idx] = 1.0

    # --- Shift sequences and append ---
    # Shift left (remove oldest time step) from the original tensor
    shifted_sequences = current_sequences_tensor[:, 1:, :].cpu().numpy()
    # Append the newly calculated feature vectors as the last time step
    new_sequences_np = np.concatenate([shifted_sequences, new_feature_vectors[:, np.newaxis, :]], axis=1)
    # Convert back to tensor and return on the original device
    return torch.from_numpy(new_sequences_np).to(current_sequences_tensor.device)


# --- Restructured forecast_multiple_periods_pytorch ---
def forecast_multiple_periods_pytorch(initial_sequences_tensor: torch.Tensor, # Shape: (n_individuals, seq_len, n_features)
                                      initial_identifiers_df: pd.DataFrame,
                                      model: nn.Module,
                                      device,
                                      metadata: dict,
                                      params: dict,
                                      initial_period: int,
                                      periods_to_forecast: int = 12,
                                      n_samples: int = 100):
    """
    Runs the multi-period forecast simulation, processing one sample fully at a time
    to conserve memory. Calculates and uses group-specific rates.
    Returns both aggregated forecast statistics and raw sample trajectories.
    """
    print(f"Starting multi-period forecast for {periods_to_forecast} periods with {n_samples} sequential samples...")
    print("Calculating and using state/industry specific rates.")
    print("Processing one sample simulation across all periods at a time for memory efficiency.")

    # Store results per sample, per period
    # List of lists/arrays: outer list for samples, inner for periods
    all_samples_period_urs = []
    all_samples_period_ers = []

    n_individuals, seq_len, n_features = initial_sequences_tensor.shape
    # Keep identifiers constant across samples and time
    current_identifiers_df = initial_identifiers_df

    # --- Get metadata needed within the loop ---
    pad_value = config.PAD_VALUE
    feature_names = metadata.get('feature_names', [])
    original_id_cols = metadata.get('original_identifier_columns', [])
    state_col = 'statefip' if 'statefip' in original_id_cols else None
    ind_col = 'ind_group_cat' if 'ind_group_cat' in original_id_cols else None
    target_map_inverse = metadata.get('target_state_map_inverse', {})

    if not feature_names: raise ValueError("Feature names not found in metadata.")
    if not target_map_inverse: raise ValueError("Target state map not found in metadata.")

    # --- Outer loop: Iterate through each Monte Carlo sample ---
    for s in tqdm(range(n_samples), desc="Processing Samples"):
        sample_period_urs = [] # Store overall UR for each period within this sample
        sample_period_ers = [] # Store overall ER for each period within this sample

        # Initialize sequence tensor for this sample (copy initial state)
        # Move to GPU for this sample's simulation
        current_sample_sequences_gpu = initial_sequences_tensor.clone().to(device)
        current_sim_period = initial_period

        # --- Inner loop: Iterate through forecast periods for this sample ---
        for i in range(periods_to_forecast):
            # Calculate target period (YYYYMM) for this step
            current_year = current_sim_period // 100
            current_month = current_sim_period % 100
            if current_month == 12:
                target_year = current_year + 1
                target_month = 1
            else:
                target_year = current_year
                target_month = current_month + 1
            target_period = target_year * 100 + target_month

            # 1. Predict next state and OVERALL rates for this sample's current sequences
            sampled_states_gpu, overall_sample_unemp_rate, overall_sample_emp_rate = forecast_next_period_pytorch(
                current_sample_sequences_gpu, model, device, metadata
            )
            # Store the overall rates for this period and sample
            sample_period_urs.append(overall_sample_unemp_rate)
            sample_period_ers.append(overall_sample_emp_rate)

            # --- Calculate Group-Specific Rates for this sample step ---
            sampled_states_np = sampled_states_gpu.cpu().numpy() # Move states to CPU for pandas ops
            state_rates_dict = {}
            industry_rates_dict = {}
            if state_col:
                state_rates_dict = calculate_group_rates(
                    sampled_states_np, current_identifiers_df[state_col], target_map_inverse
                )
            if ind_col:
                 industry_rates_dict = calculate_group_rates(
                    sampled_states_np, current_identifiers_df[ind_col], target_map_inverse
                )
            # --- End Group Rate Calculation ---

            # 2. Update this sample's sequences using its sampled states and calculated rates
            # Pass GPU tensors and CPU data structures as needed by update function
            current_sample_sequences_gpu = update_sequences_and_features(
                current_sample_sequences_gpu, # Pass current GPU tensor
                sampled_states_gpu,           # Pass predicted states on GPU
                current_identifiers_df,       # Pass identifiers DataFrame (CPU)
                feature_names,
                metadata,
                current_sim_period,           # The period *before* the prediction
                state_rates_dict,             # Pass calculated state rates (CPU dict)
                industry_rates_dict,          # Pass calculated industry rates (CPU dict)
                overall_sample_unemp_rate,    # Pass overall rate for national features
                overall_sample_emp_rate       # Pass overall rate for national features
            )
            # updated tensor remains on GPU for the next iteration

            # 3. Advance time for the next period in this sample's simulation
            current_sim_period = target_period

            # Optional: Clear GPU cache periodically if memory pressure is still high
            if (i + 1) % 5 == 0: # Example: clear every 5 periods
                 if device.type == 'mps': torch.mps.empty_cache()
                 elif device.type == 'cuda': torch.cuda.empty_cache()
        # --- End Period Loop for Sample 's' ---

        # Store the list of period rates for this completed sample
        all_samples_period_urs.append(sample_period_urs)
        all_samples_period_ers.append(sample_period_ers)

        # Clean up GPU memory for this sample's tensor before starting next sample
        del current_sample_sequences_gpu
        if device.type == 'mps': torch.mps.empty_cache()
        elif device.type == 'cuda': torch.cuda.empty_cache()
    # --- End Sample Loop ---

    # --- Aggregate results across all samples ---
    print("Aggregating results across periods and samples...")

    # Create forecast period list (YYYYMM)
    forecast_periods = []
    current_period = initial_period
    for i in range(periods_to_forecast):
        current_year = current_period // 100
        current_month = current_period % 100
        if current_month == 12: target_year, target_month = current_year + 1, 1
        else: target_year, target_month = current_year, current_month + 1
        target_period = target_year * 100 + target_month
        forecast_periods.append(target_period)
        current_period = target_period

    # Convert collected lists into DataFrames for easier aggregation
    # Rows = samples, Columns = forecast periods
    sample_urs_over_time = pd.DataFrame(all_samples_period_urs, columns=forecast_periods)
    sample_ers_over_time = pd.DataFrame(all_samples_period_ers, columns=forecast_periods)

    # Calculate quantiles across samples (axis=0) for each period (column)
    forecast_df = pd.DataFrame({
        'period': forecast_periods,
        'unemployment_rate_median': sample_urs_over_time.median(axis=0).values,
        'unemployment_rate_p10': sample_urs_over_time.quantile(0.1, axis=0).values,
        'unemployment_rate_p90': sample_urs_over_time.quantile(0.9, axis=0).values,
        'employment_rate_median': sample_ers_over_time.median(axis=0).values,
        'employment_rate_p10': sample_ers_over_time.quantile(0.1, axis=0).values,
        'employment_rate_p90': sample_ers_over_time.quantile(0.9, axis=0).values,
    })
    forecast_df['date'] = forecast_df['period'].apply(period_to_date)

    print("Multi-period forecast complete.")
    # Return both aggregated results and raw sample data
    return forecast_df, sample_urs_over_time

# --- Plotting Function ---
# Modified to plot individual sample trajectories
def plot_unemployment_forecast_py(historical_df: pd.DataFrame,
                                  forecast_df: pd.DataFrame,
                                  sample_urs_over_time: pd.DataFrame, # Added: Raw sample UR data
                                  output_path: Path,
                                  metadata: dict):
    """Plots historical and forecasted unemployment rates, showing individual sample trajectories, median, and 80% CI."""
    print("Creating forecast visualization with individual trajectories...")

    # Ensure date columns are valid datetime objects
    historical_df[config.DATE_COL] = pd.to_datetime(historical_df[config.DATE_COL])
    forecast_df[config.DATE_COL] = pd.to_datetime(forecast_df[config.DATE_COL])

    # Filter out any rows where date conversion failed
    historical_df = historical_df.dropna(subset=[config.DATE_COL])
    forecast_df = forecast_df.dropna(subset=[config.DATE_COL, 'unemployment_rate_median', 'unemployment_rate_p10', 'unemployment_rate_p90'])

    if historical_df.empty or forecast_df.empty:
        print("Error: No valid dates found in historical or forecast data for plotting.")
        return

    # --- Limit Historical Data Displayed ---
    forecast_start_date = forecast_df[config.DATE_COL].min()
    history_display_start_date = forecast_start_date - pd.DateOffset(years=1)
    historical_df_display = historical_df[historical_df[config.DATE_COL] >= history_display_start_date].copy()
    print(f"Plotting historical data from {history_display_start_date.strftime('%Y-%m-%d')} onwards.")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot historical data (limited view)
    ax.plot(historical_df_display[config.DATE_COL], historical_df_display['unemployment_rate'],
            label='Historical (Last Year)', color='black', linewidth=1.5, zorder=5) # Ensure history is visible

    # Get forecast dates for plotting trajectories
    forecast_dates = forecast_df[config.DATE_COL].values

    # Plot individual sample trajectories (adjust visibility)
    n_samples_to_plot = sample_urs_over_time.shape[0]
    for i in range(n_samples_to_plot):
        # Increase alpha, adjust color/linewidth slightly for better visibility
        ax.plot(forecast_dates, sample_urs_over_time.iloc[i].values,
                color='lightsteelblue', alpha=0.25, linewidth=0.75, zorder=1) # Adjusted alpha, color, linewidth

    # Plot confidence interval (P10-P90) - light shading
    ax.fill_between(forecast_dates,
                    forecast_df['unemployment_rate_p10'],
                    forecast_df['unemployment_rate_p90'],
                    color='skyblue', alpha=0.4, label='Forecast (80% CI)', zorder=2) # Slightly increased alpha for CI too

    # Plot forecast median (darker line)
    ax.plot(forecast_dates, forecast_df['unemployment_rate_median'],
            label='Forecast (Median)', color='blue', linewidth=2.0, linestyle='--', zorder=3) # Prominent median

    # Use state mapping for title/labels if needed
    target_map_inverse = metadata.get('target_state_map_inverse', {0: 'Employed', 1: 'Unemployed', 2: 'NILF'})
    unemployed_label = target_map_inverse.get(1, 'Unemployed')

    # Formatting
    ax.set_title(f'{unemployed_label} Rate Forecast (Transformer Model - Samples, Median & 80% CI)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{unemployed_label} Rate', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Adjust legend to avoid showing all sample lines
    handles, labels = ax.get_legend_handles_labels()
    # Keep only Historical, Median, and CI labels
    filtered_handles = [h for h, l in zip(handles, labels) if l in ['Historical (Last Year)', 'Forecast (Median)', 'Forecast (80% CI)']]
    filtered_labels = [l for l in labels if l in ['Historical (Last Year)', 'Forecast (Median)', 'Forecast (80% CI)']]
    ax.legend(filtered_handles, filtered_labels, fontsize=11)

    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)
    fig.autofmt_xdate()

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)


# --- Main Execution --- # Modified
def main(): # Removed args
    start_time = time.time()
    print("=" * 60)
    print("Starting Transformer Unemployment Forecast (PyTorch)")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Using parameters from config.py")
    print("=" * 60)

    # --- Paths --- Use config paths
    project_root = config.PROJECT_ROOT
    # Point to the specific standard run directory for the model
    model_dir = config.TRAIN_OUTPUT_SUBDIR / "standard_run" # Append standard_run here
    processed_data_dir = config.PREPROCESS_OUTPUT_DIR
    raw_data_path = config.FORECAST_RAW_DATA_FILE
    output_dir = config.FORECAST_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "transformer_unemployment_forecast.png"
    forecast_csv_path = output_dir / "transformer_forecast_results.csv"

    print(f"Project Root: {project_root}")
    print(f"Model Dir: {model_dir}") # Will now print the standard_run path
    print(f"Processed Data Dir: {processed_data_dir}")
    print(f"Raw Data Path: {raw_data_path}")
    print(f"Output Dir: {output_dir}")

    # Print device info once here
    device_name = "Unknown"
    if DEVICE.type == 'cuda':
        device_name = f"CUDA/GPU ({torch.cuda.get_device_name(DEVICE)})"
    elif DEVICE.type == 'mps':
        device_name = "Apple Silicon GPU (MPS)"
    elif DEVICE.type == 'cpu':
        device_name = "CPU"
    print(f"Using device: {device_name}")

    # --- Load Model and Data --- Use config params
    try:
        model, params, metadata = load_pytorch_model_and_params(model_dir, processed_data_dir, DEVICE)
        initial_sim_data, historical_rates, start_period, _ = load_and_prepare_data(
            processed_data_dir, raw_data_path, metadata, config.FORECAST_START_YEAR, config.FORECAST_START_MONTH
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"ERROR loading model or data: {e}")
        raise

    # --- Prepare Initial Sequences --- Use config params
    try:
        initial_sequences_np, sim_ids, initial_identifiers_df = get_sequences_for_simulation(
            initial_sim_data,
            group_col=config.GROUP_ID_COL,
            seq_len=params['sequence_length'],
            features=metadata['feature_names'],
            original_id_cols=metadata['original_identifier_columns'],
            pad_val=config.PAD_VALUE,
            end_period=start_period
        )
        initial_sequences_tensor = torch.from_numpy(initial_sequences_np).to(DEVICE)

        # --- Delete initial_sim_data after use ---
        del initial_sim_data
        del initial_sequences_np # Also delete numpy array version
        import gc
        gc.collect()
        print("Deleted initial simulation data dataframe from memory.")
        # ---

    except ValueError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"ERROR preparing initial sequences: {e}")
        raise

    # --- Run Forecast --- Use config params
    try:
        # Capture both aggregated forecast and raw sample data
        forecast_df, sample_urs_over_time = forecast_multiple_periods_pytorch(
            initial_sequences_tensor=initial_sequences_tensor,
            initial_identifiers_df=initial_identifiers_df,
            model=model,
            device=DEVICE,
            metadata=metadata,
            params=params,
            initial_period=start_period,
            periods_to_forecast=config.FORECAST_PERIODS,
            n_samples=config.MC_SAMPLES
        )
    except Exception as e:
        print(f"ERROR during forecasting: {e}")
        raise

    # --- Save & Plot Results ---
    try:
        forecast_df.to_csv(forecast_csv_path, index=False)
        print(f"Forecast results saved to: {forecast_csv_path}")
    except Exception as e:
        print(f"Error saving forecast results: {e}")

    try:
        # Pass the raw sample data to the plotting function
        plot_unemployment_forecast_py(historical_rates, forecast_df, sample_urs_over_time, plot_path, metadata)
    except Exception as e:
        print(f"Error generating plot: {e}")

    # --- Completion ---
    end_time = time.time()
    elapsed_mins = (end_time - start_time) / 60
    print("-" * 60)
    print(f"Forecast Script Completed | Elapsed: {elapsed_mins:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    if (config.FORECAST_START_YEAR is None and config.FORECAST_START_MONTH is not None) or \
       (config.FORECAST_START_YEAR is not None and config.FORECAST_START_MONTH is None):
        print("ERROR in config.py: FORECAST_START_YEAR and FORECAST_START_MONTH must be provided together or both be None.")
        sys.exit(1)
    if config.FORECAST_START_MONTH is not None and not (1 <= config.FORECAST_START_MONTH <= 12):
        print("ERROR in config.py: FORECAST_START_MONTH must be between 1 and 12.")
        sys.exit(1)

    main()
