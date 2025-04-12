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

# Fix imports - add the parent directory of this script to sys.path
# Assumes 'models.py' and 'config.py' are in the parent directory
# e.g., project_root/scripts/this_script.py and project_root/models.py
script_dir = Path(__file__).resolve().parent

# Import Model Definitions and Config with absolute import path logic
try:
    from models import PositionalEmbedding, TransformerEncoderBlock, TransformerForecastingModel
except ImportError as e:
    print(f"ERROR: Could not import from models.py. Ensure it exists in {script_dir}. Details: {e}") # Updated error message path
    sys.exit(1)

try:
    import config
except ImportError:
    print(f"ERROR: config.py not found. Make sure it's in {script_dir} or sys.path is configured correctly.") # Updated error message path
    sys.exit(1)

# --- Constants --- (Moved relevant constants to config.py)

# --- PyTorch Device Setup ---
def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # MPS for Apple Silicon GPUs
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

def load_and_prepare_data(processed_data_dir: Path, raw_data_path: Path, metadata: dict, start_year: int = None, start_month: int = None):
    """Loads the FULL baked data for simulation start and raw data for historical rates.
       Filters data based on specified start_year and start_month if provided, otherwise uses the latest period.
    """
    print("Loading data...")
    # --- Load FULL Baked Data ---
    full_baked_path = processed_data_dir / config.FULL_DATA_FILENAME
    if not full_baked_path.exists():
         raise FileNotFoundError(f"Required full baked data file not found at {full_baked_path}. Please ensure the preprocessing script ran successfully and saved the file.")

    print(f"Loading FULL baked data from: {full_baked_path} (This might take time/memory)")
    # Use specific columns if possible to reduce memory, though filtering later might require more
    # Consider using pyarrow engine if performance is an issue
    try:
        latest_baked_df = pd.read_parquet(full_baked_path)
    except Exception as e:
        print(f"Error loading Parquet file {full_baked_path}: {e}")
        raise
    print(f"Loaded FULL baked data: {latest_baked_df.shape}")
    # --- End Load Baked Data ---

    # Ensure original identifier columns are present
    original_id_cols = metadata.get('original_identifier_columns', [])
    missing_orig_cols = [col for col in original_id_cols if col not in latest_baked_df.columns]
    if missing_orig_cols:
        raise ValueError(f"Baked data is missing required original identifier columns: {missing_orig_cols}")
    print(f"Found original identifier columns in baked data: {original_id_cols}")

    # Calculate period
    latest_baked_df[config.DATE_COL] = pd.to_datetime(latest_baked_df[config.DATE_COL])
    latest_baked_df['period'] = latest_baked_df[config.DATE_COL].dt.year * 100 + latest_baked_df[config.DATE_COL].dt.month

    # Determine the start period for the simulation
    available_periods = sorted(latest_baked_df['period'].unique())
    if not available_periods:
        raise ValueError("No periods found in the loaded baked data.")

    if start_year and start_month:
        start_period = start_year * 100 + start_month
        if start_period not in available_periods:
             raise ValueError(f"Specified start period {start_period} ({start_year}-{start_month:02d}) not found in the baked data. Available periods range from {min(available_periods)} to {max(available_periods)}.")
        print(f"Using specified start period: {start_period}")
    else:
        start_period = available_periods[-1] # Use the latest period
        start_date = period_to_date(start_period)
        print(f"Using latest period in baked data as start period: {start_period} ({start_date.strftime('%Y-%m') if start_date else 'Invalid Date'})")


    # Get individuals present in the start period
    start_period_ids = latest_baked_df[latest_baked_df['period'] == start_period][config.GROUP_ID_COL].unique() # Use config
    if len(start_period_ids) == 0:
        raise ValueError(f"No individuals found in the specified start period {start_period}. Check data integrity or start period selection.")
    print(f"Found {len(start_period_ids)} unique individuals present in the start period ({start_period}).")

    # Filter the baked data to include only the history *up to* the start period for these individuals
    # This is the crucial step for initial simulation state
    initial_sim_data = latest_baked_df[
        (latest_baked_df[config.GROUP_ID_COL].isin(start_period_ids)) &
        (latest_baked_df['period'] <= start_period)
    ].sort_values([config.GROUP_ID_COL, config.DATE_COL]) # Sort is important for sequence extraction

    # --- Delete the large full dataframe ASAP ---
    del latest_baked_df
    gc.collect()
    print("Released memory from the full baked dataframe.")
    # ---

    # --- Load Raw Data for Historical Rates ---
    print(f"Loading raw data for historical rates calculation from: {raw_data_path}")
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

    # Define dtypes for potentially large ID columns
    dtype_spec = {config.GROUP_ID_COL: str} # Load group ID as string
    # Add other known large or categorical columns if needed

    raw_df = pd.read_csv(raw_data_path, dtype=dtype_spec, low_memory=False) # low_memory=False can help with mixed types
    raw_df[config.DATE_COL] = pd.to_datetime(raw_df[config.DATE_COL])
    raw_df['period'] = raw_df[config.DATE_COL].dt.year * 100 + raw_df[config.DATE_COL].dt.month

    # Ensure 'emp_state' column exists (adjust name if different in raw CSV)
    raw_target_col = 'emp_state' # Adjust if your raw data uses a different name
    if raw_target_col not in raw_df.columns:
        raise KeyError(f"Required target column '{raw_target_col}' not found in the raw data file: {raw_data_path}")

    # Normalize state names from raw data (make lowercase, handle variations)
    # Using a specific replacement map for clarity
    state_replacements = {
        'employed': 'employed',
        'unemployed': 'unemployed',
        'not in labor force': 'nilf',
        'nilf': 'nilf'
        # Add other variations if they exist in raw data
    }
    raw_df[raw_target_col] = raw_df[raw_target_col].str.lower().map(state_replacements).fillna('unknown')
    if 'unknown' in raw_df[raw_target_col].unique():
        print(f"Warning: Some '{raw_target_col}' entries in raw data were not mapped: {raw_df[raw_df[raw_target_col] == 'unknown'].shape[0]} rows")

    # Calculate historical rates using ALL available raw data
    # Use the state mapping from metadata for consistency (map integers back to normalized lowercase strings)
    target_map_inverse = metadata.get('target_state_map_inverse', {0: 'employed', 1: 'unemployed', 2: 'nilf'}) # Default if missing
    # Create the map {0: 'employed', 1: 'unemployed', 2: 'nilf'} using lowercase normalized names
    state_map_int_to_norm_str = {k: v.lower().replace(' ', '_').replace('not_in_labor_force', 'nilf') for k, v in target_map_inverse.items()}
    print(f"DEBUG: Using state map for historical rates: {state_map_int_to_norm_str}")
    employed_str = state_map_int_to_norm_str.get(0, 'employed')
    unemployed_str = state_map_int_to_norm_str.get(1, 'unemployed')
    nilf_str = state_map_int_to_norm_str.get(2, 'nilf')

    # Group by period and calculate state counts for all periods in raw_df
    historical_counts = raw_df.groupby('period')[raw_target_col].value_counts().unstack(fill_value=0)

    # Ensure all expected state columns exist after unstacking
    for state_str in [employed_str, unemployed_str, nilf_str]:
        if state_str not in historical_counts.columns:
            historical_counts[state_str] = 0

    # Calculate rates
    historical_rates = pd.DataFrame(index=historical_counts.index)
    historical_rates[employed_str] = historical_counts[employed_str]
    historical_rates[unemployed_str] = historical_counts[unemployed_str]
    historical_rates[nilf_str] = historical_counts[nilf_str] # Keep NILF count if needed

    historical_rates['labor_force'] = historical_rates[employed_str] + historical_rates[unemployed_str]
    # Use np.where for safe division
    historical_rates['unemployment_rate'] = np.where(
        historical_rates['labor_force'] > 0,
        historical_rates[unemployed_str] / historical_rates['labor_force'],
        0.0
    )
    historical_rates['date'] = historical_rates.index.map(period_to_date) # Use index (period)
    historical_rates.reset_index(inplace=True) # Make 'period' a column

    # Remove rows where date conversion failed
    original_count = historical_rates.shape[0]
    historical_rates.dropna(subset=['date'], inplace=True)
    if historical_rates.shape[0] < original_count:
        print(f"Warning: Dropped {original_count - historical_rates.shape[0]} rows from historical rates due to invalid period/date.")

    print(f"Calculated historical rates for {historical_rates.shape[0]} periods (all available in raw data).")

    # Return the actual start period used and the IDs active in that period
    return initial_sim_data, historical_rates, start_period, start_period_ids

# --- Forecasting Functions ---

def get_sequences_for_simulation(sim_data_df: pd.DataFrame, group_col: str, seq_len: int, features: list, original_id_cols: list, pad_val: float, end_period: int):
    """
    Extracts the sequence ending at `end_period` for each individual active in that period.
    Also extracts the original identifier columns for the last observation of each individual.
    Assumes sim_data_df contains history *up to* end_period for relevant individuals.
    """
    print(f"Extracting sequences and identifiers ending at period {end_period} for each individual...")
    sequences = {}
    original_identifiers = {} # Store original IDs for the last time step

    # Ensure data is sorted correctly before grouping
    # Grouping by the main ID should be sufficient if data is pre-sorted
    grouped = sim_data_df.groupby(group_col)

    valid_ids = 0
    skipped_ids = 0

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

    return sequences_np, ids_list, original_identifiers_df


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


def sample_states_and_calc_rates(probabilities: torch.Tensor, metadata: dict):
    """
    Samples states based on probabilities and calculates overall E/U rates for the sample.

    Args:
        probabilities: Tensor of shape (n_individuals, n_classes) from forecast_next_period_pytorch.
        metadata: Dictionary with metadata including target_state_map_inverse.

    Returns:
        sampled_states: Tensor (CPU) with predicted state indices (shape: n_individuals).
        overall_unemp_rate: Unemployment rate across all individuals in this sample.
        overall_emp_rate: Employment rate (E / LF) across all individuals in this sample.
    """
    n_individuals = probabilities.shape[0]
    target_map_inverse = metadata.get('target_state_map_inverse', {})

    # --- Find integer indices for Employed and Unemployed states ---
    # Handle potential case differences or naming variations
    try:
        employed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'employed')
        unemployed_int = next(k for k, v in target_map_inverse.items() if v.lower() == 'unemployed')
    except StopIteration:
        print("ERROR: Could not find 'employed' or 'unemployed' state in target_state_map_inverse:", target_map_inverse)
        # Fallback or raise error - let's raise for clarity
        raise ValueError("Metadata missing required state definitions ('employed', 'unemployed') in target_state_map_inverse")

    # Sample states based on probabilities (on the device probabilities are on)
    # multinomial expects probabilities, not logits
    sampled_states_gpu = torch.multinomial(probabilities, num_samples=1).squeeze(1)

    # Move to CPU for numpy operations
    sampled_states_np = sampled_states_gpu.cpu().numpy()

    # Calculate overall unemployment and employment rates for this specific sample
    employed_count = np.sum(sampled_states_np == employed_int)
    unemployed_count = np.sum(sampled_states_np == unemployed_int)
    labor_force = employed_count + unemployed_count

    overall_unemp_rate = np.divide(unemployed_count, labor_force, out=np.zeros_like(unemployed_count, dtype=float), where=labor_force>0)
    overall_emp_rate = np.divide(employed_count, labor_force, out=np.zeros_like(employed_count, dtype=float), where=labor_force>0) # E / LF

    return sampled_states_gpu, float(overall_unemp_rate), float(overall_emp_rate)


def calculate_group_rates(sampled_states_np: np.ndarray, # Shape (n_individuals,) on CPU
                          grouping_series: pd.Series, # Series with group IDs (statefip or ind_group_cat), index matching sampled_states
                          target_map_inverse: dict):
    """Calculates unemployment and employment rates for each group."""
    if grouping_series is None or grouping_series.empty:
        print("Warning: Grouping series is empty, cannot calculate group rates.")
        return {}
    if len(sampled_states_np) != len(grouping_series):
        print(f"Warning: Length mismatch between states ({len(sampled_states_np)}) and group IDs ({len(grouping_series)}). Cannot calculate group rates.")
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
        'state': sampled_states_np
    })

    # Count states within each group
    # Use Categorical for potential performance improvement and handling all states
    state_categories = list(target_map_inverse.keys())
    df['state'] = pd.Categorical(df['state'], categories=state_categories)
    counts = df.groupby('group')['state'].value_counts().unstack(fill_value=0) # observed=False includes all categories

    # Ensure all required state columns exist, even if count is 0
    for state_int in [employed_int, unemployed_int]:
        if state_int not in counts.columns:
            counts[state_int] = 0

    # Calculate rates
    employed_count = counts[employed_int]
    unemployed_count = counts[unemployed_int]
    labor_force = employed_count + unemployed_count

    rates_df = pd.DataFrame(index=counts.index)
    rates_df['unemp_rate'] = np.where(labor_force > 0, unemployed_count / labor_force, 0.0)
    rates_df['emp_rate'] = np.where(labor_force > 0, employed_count / labor_force, 0.0) # Emp / LF

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
    Updates sequences for the next step. Shifts sequence, appends new features based on sampled state.
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
    state_indices = feature_indices['state_indices']
    age_idx, months_last_idx, durunemp_idx = feature_indices['age_idx'], feature_indices['months_last_idx'], feature_indices['durunemp_idx']
    nat_unemp_idx, nat_emp_idx = feature_indices['nat_unemp_idx'], feature_indices['nat_emp_idx']
    state_unemp_idx, state_emp_idx = feature_indices['state_unemp_idx'], feature_indices['state_emp_idx']
    ind_unemp_idx, ind_emp_idx = feature_indices['ind_unemp_idx'], feature_indices['ind_emp_idx']
    ind_group_indices, occ_group_indices, classwkr_indices = feature_indices['ind_group_indices'], feature_indices['occ_group_indices'], feature_indices['classwkr_indices']
    ind_unknown_indices, occ_unknown_indices, classwkr_unknown_indices = feature_indices['ind_unknown_indices'], feature_indices['occ_unknown_indices'], feature_indices['classwkr_unknown_indices']

    # --- Determine next period ---
    current_year = current_period // 100
    current_month = current_period % 100
    if current_month == 12:
        next_year, next_month = current_year + 1, 1
    else:
        next_year, next_month = current_year, current_month + 1
    # next_period = next_year * 100 + next_month # Not strictly needed below

    # --- Get data onto CPU for processing ---
    # sequence_history_np: Full history *before* this update step (N, S, F)
    sequence_history_np = current_sequences_tensor.cpu().numpy()
    # previous_features_np: Features from the *last* time step (N, F)
    previous_features_np = sequence_history_np[:, -1, :]
    # sampled_states_np: Predicted states for the *next* step (N,)
    sampled_states_np = sampled_states.cpu().numpy()

    # Create array for the NEW feature vectors (N, F) - initialized from previous step
    new_feature_vectors = np.copy(previous_features_np)

    # --- Get original identifier columns needed ---
    original_id_cols = metadata.get('original_identifier_columns', [])
    state_col = 'statefip' if 'statefip' in original_id_cols else None # Adjust if name differs
    ind_col = 'ind_group_cat' if 'ind_group_cat' in original_id_cols else None # Adjust if name differs

    # --- Update features loop ---
    for i in range(n_individuals):
        sampled_state_int = sampled_states_np[i]
        previous_features = previous_features_np[i] # Features from t-1 (1D array)
        ind_identifiers = original_identifiers_df.iloc[i] # Series with statefip, ind_group_cat etc.

        # Determine previous state from the one-hot encoding in previous_features
        previous_state_int = -1
        for state_col_name, state_idx in state_indices.items():
            if previous_features[state_idx] > 0.5: # Check if this state was active
                # Find the integer key corresponding to this state name in target_map_inverse
                state_str_from_col = state_col_name.split('cat__current_state_')[-1]
                try:
                    previous_state_int = next(k for k, v in target_map_inverse.items() if v == state_str_from_col)
                    break
                except StopIteration:
                    # This should not happen if state_indices and target_map_inverse are consistent
                    print(f"Warning: Could not map previous state feature '{state_col_name}' back to an integer state. Defaulting to -1.")
                    break # Exit inner loop

        # 1. Update 'current_state' based on sampled_state_int
        # Reset all state flags first
        for idx in state_indices.values(): new_feature_vectors[i, idx] = 0.0
        try:
            # Get the ORIGINAL state string (e.g., "Employed", "Not in Labor Force") from the inverse map
            state_name_original = target_map_inverse[int(sampled_state_int)] # Ensure int key
            # Construct the target column name using the ORIGINAL state string
            target_col_name = f'cat__current_state_{state_name_original}'
            # Find the index of this correctly constructed name
            target_idx = state_indices[target_col_name]
            new_feature_vectors[i, target_idx] = 1.0
        except KeyError:
             # This error could now mean the lookup_key is invalid OR the constructed target_col_name is not in state_indices
             print(f"\n--- FATAL ERROR: KeyError during state update ---")
             print(f"Individual Index: {i}")
             print(f"Sampled state (int): {sampled_state_int} (Type: {type(sampled_state_int)})")
             # Check if lookup_key is valid in the original map
             if int(sampled_state_int) not in target_map_inverse:
                 print(f"Failure Cause: Sampled integer key {int(sampled_state_int)} not found in target_map_inverse:")
                 print(target_map_inverse)
             else:
                 # If key is valid, the constructed name must be wrong or missing from state_indices
                 state_name_original = target_map_inverse.get(int(sampled_state_int), "<<Key Error>>")
                 target_col_name = f'cat__current_state_{state_name_original}'
                 print(f"Failure Cause: Constructed feature name '{target_col_name}' not found in available state_indices keys.")
                 print(f"Available state_indices keys: {list(state_indices.keys())}")
             print(f"Previous State Int derived: {previous_state_int}")
             print(f"Previous Feature Vector (partial): {previous_features[:10]}...")
             print(f"--- END FATAL ERROR ---")
             raise # Re-raise the KeyError to halt execution here
        except Exception as e:
             print(f"ERROR updating state for sampled_state_int {sampled_state_int}: {e}")
             state_name_val = locals().get('state_name_original', '<<Not Assigned>>')
             print(f"State name derived: {state_name_val}")
             print(f"Available state indices: {state_indices.keys()}")
             raise # Re-raise other exceptions too

        # 2. Update Time Features (Age, Months Since Last, Duration Unemployed)
        if age_idx != -1 and next_month == 1: new_feature_vectors[i, age_idx] = previous_features[age_idx] + 1
        if months_last_idx != -1: new_feature_vectors[i, months_last_idx] = previous_features[months_last_idx] + 1
        if durunemp_idx != -1:
            is_unemployed_now = (sampled_state_int == unemployed_int)
            # Increment duration if newly unemployed, reset if not. Use previous duration value.
            new_feature_vectors[i, durunemp_idx] = (previous_features[durunemp_idx] + 1) if is_unemployed_now else 0

        # 3. Update Aggregate Rate Features (National, State, Industry)
        # Use group rates calculated for this specific sample step
        current_state_id = ind_identifiers.get(state_col, None) if state_col else None
        current_ind_id = ind_identifiers.get(ind_col, None) if ind_col else None

        # Get group-specific rates for this individual, if available
        ind_state_rates = state_rates.get(current_state_id, None) if current_state_id else None
        ind_industry_rates = industry_rates.get(current_ind_id, None) if current_ind_id else None

        # Update state rates, keeping previous value if group rate is missing
        if state_unemp_idx != -1:
            new_rate = ind_state_rates.get('unemp_rate') if ind_state_rates is not None else None
            new_feature_vectors[i, state_unemp_idx] = new_rate if new_rate is not None else previous_features[state_unemp_idx]
        if state_emp_idx != -1:
            new_rate = ind_state_rates.get('emp_rate') if ind_state_rates is not None else None
            new_feature_vectors[i, state_emp_idx] = new_rate if new_rate is not None else previous_features[state_emp_idx]

        # Update industry rates, keeping previous value if group rate is missing
        if ind_unemp_idx != -1:
            new_rate = ind_industry_rates.get('unemp_rate') if ind_industry_rates is not None else None
            new_feature_vectors[i, ind_unemp_idx] = new_rate if new_rate is not None else previous_features[ind_unemp_idx]
        if ind_emp_idx != -1:
            new_rate = ind_industry_rates.get('emp_rate') if ind_industry_rates is not None else None
            new_feature_vectors[i, ind_emp_idx] = new_rate if new_rate is not None else previous_features[ind_emp_idx]

        # Update national rates (using the overall sample rate calculated for this step)
        if nat_unemp_idx != -1: new_feature_vectors[i, nat_unemp_idx] = overall_sample_unemp_rate
        if nat_emp_idx != -1: new_feature_vectors[i, nat_emp_idx] = overall_sample_emp_rate

        # 4. Update Industry/Occupation/Class Worker Features on Transition to Employment
        if previous_state_int == -1:
             # Cannot determine previous state reliably, skip transition logic
             pass
        else:
            transition_to_emp = (previous_state_int in [unemployed_int, nilf_int]) and (sampled_state_int == employed_int)

            if transition_to_emp:
                # Use the full sequence history for this individual *before* the update
                ind_sequence_history = sequence_history_np[i] # Shape (seq_len, n_features)

                # Perform lookback for each group using the helper
                last_ind_idx = _find_last_valid_category_index(ind_sequence_history, ind_group_indices, ind_unknown_indices)
                last_occ_idx = _find_last_valid_category_index(ind_sequence_history, occ_group_indices, occ_unknown_indices)
                last_cls_idx = _find_last_valid_category_index(ind_sequence_history, classwkr_indices, classwkr_unknown_indices)

                # Update new feature vector if a valid historical category was found
                if last_ind_idx != -1:
                    # Reset all flags in this group first
                    for idx in ind_group_indices.values(): new_feature_vectors[i, idx] = 0.0
                    # Set the historical category flag
                    new_feature_vectors[i, last_ind_idx] = 1.0
                # Else: Keep the features inherited from the previous step (likely 'Unknown'/'NIU' or whatever was carried forward)

                if last_occ_idx != -1:
                    for idx in occ_group_indices.values(): new_feature_vectors[i, idx] = 0.0
                    new_feature_vectors[i, last_occ_idx] = 1.0

                if last_cls_idx != -1:
                    for idx in classwkr_indices.values(): new_feature_vectors[i, idx] = 0.0
                    new_feature_vectors[i, last_cls_idx] = 1.0
            # --- End transition to emp block ---
        # --- End previous state check block ---
    # --- End individual loop ---

    # --- Shift sequences and append ---
    # Shift left (remove oldest time step) from the original tensor's numpy representation
    shifted_sequences_np = sequence_history_np[:, 1:, :]
    # Append the newly calculated feature vectors as the last time step
    # Need to add a time dimension to new_feature_vectors: (N, F) -> (N, 1, F)
    new_sequences_np = np.concatenate([shifted_sequences_np, new_feature_vectors[:, np.newaxis, :]], axis=1)

    # Convert back to tensor and return on the original device
    return torch.from_numpy(new_sequences_np).to(device)


def forecast_multiple_periods_pytorch(initial_sequences_tensor: torch.Tensor, # Shape: (N, S, F), on DEVICE
                                      initial_identifiers_df: pd.DataFrame, # CPU DataFrame
                                      model: nn.Module, # On DEVICE
                                      device,
                                      metadata: dict,
                                      params: dict,
                                      initial_period: int, # Starting period YYYYMM
                                      periods_to_forecast: int = 12,
                                      n_samples: int = 100):
    """
    Runs the multi-period forecast simulation, processing one sample fully at a time
    to conserve memory. Calculates and uses group-specific rates dynamically.
    Returns both aggregated forecast statistics and raw sample trajectories for unemployment rate.
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
    for s in tqdm(range(n_samples), desc="Processing Samples"):
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

            # 2. Sample states and calculate OVERALL rates for this sample step
            sampled_states_gpu, overall_sample_unemp_rate, overall_sample_emp_rate = sample_states_and_calc_rates(
                 probabilities_gpu, metadata
            )
            # Store the overall UR for this period and sample
            sample_period_urs.append(overall_sample_unemp_rate)

            # 3. Calculate Group-Specific Rates (State, Industry) for this sample step
            sampled_states_np = sampled_states_gpu.cpu().numpy() # Needs CPU numpy array
            state_rates_dict = {}
            industry_rates_dict = {}
            # Only calculate if the corresponding identifier column exists
            if state_col and state_col in current_identifiers_df.columns:
                state_rates_dict = calculate_group_rates(
                    sampled_states_np, current_identifiers_df[state_col], target_map_inverse
                )
            if ind_col and ind_col in current_identifiers_df.columns:
                 industry_rates_dict = calculate_group_rates(
                    sampled_states_np, current_identifiers_df[ind_col], target_map_inverse
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
                overall_sample_unemp_rate,      # Pass overall rate for national features
                overall_sample_emp_rate,        # Pass overall rate for national features
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
        empty_df = pd.DataFrame(columns=['period', 'date', 'unemployment_rate_median', 'unemployment_rate_p10', 'unemployment_rate_p90'])
        return empty_df, pd.DataFrame() # Return empty dataframes

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
    # Return both aggregated results and raw sample UR data
    return forecast_agg_df, sample_urs_over_time


# --- Plotting Function ---
def plot_unemployment_forecast_py(historical_df: pd.DataFrame,
                                  forecast_agg_df: pd.DataFrame, # Aggregated (median, p10, p90)
                                  sample_urs_over_time: pd.DataFrame, # Raw sample UR data (Samples x Periods)
                                  output_path: Path,
                                  metadata: dict):
    """Plots historical and forecasted unemployment rates, showing individual sample trajectories, median, and 80% CI."""
    print("Creating forecast visualization...")

    if historical_df.empty:
        print("Warning: Historical data is empty. Plotting forecast only.")
    if forecast_agg_df.empty:
        print("Warning: Aggregated forecast data is empty. Cannot plot forecast.")
        return
    if sample_urs_over_time.empty:
        print("Warning: Raw sample data is empty. Plotting aggregated forecast only.")

    # Ensure date columns are valid datetime objects
    if not historical_df.empty:
        historical_df['date'] = pd.to_datetime(historical_df['date']) # Already done in loading? Ensure it.
        historical_df = historical_df.dropna(subset=['date', 'unemployment_rate']).sort_values('date')
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
    plot_sample_urs = sample_urs_over_time

    if last_hist_date is not None and last_hist_rate is not None and not forecast_agg_df.empty:
        # Ensure the first forecast date is immediately after the last historical date
        # (This assumes monthly data and correct alignment)
        if forecast_dates[0] > last_hist_date:
            plot_forecast_dates = np.insert(forecast_dates, 0, last_hist_date)
            plot_forecast_median = np.insert(forecast_median, 0, last_hist_rate)
            plot_forecast_p10 = np.insert(forecast_p10, 0, last_hist_rate) # Start CI from last known point
            plot_forecast_p90 = np.insert(forecast_p90, 0, last_hist_rate) # Start CI from last known point

            # Also adjust sample trajectories if they exist
            if not sample_urs_over_time.empty:
                # Create a temporary DataFrame for plotting samples
                plot_sample_urs = sample_urs_over_time.copy()
                # Add the last historical rate as the first column, using the last historical date as the column name
                plot_sample_urs.insert(0, last_hist_date, last_hist_rate)


    # Plot individual sample trajectories (if available) using potentially adjusted data
    if not plot_sample_urs.empty:
        n_samples_to_plot = plot_sample_urs.shape[0]
        # Use the adjusted dates and sample data
        sample_plot_dates = plot_forecast_dates if 'plot_forecast_dates' in locals() and plot_forecast_dates.size == plot_sample_urs.shape[1] else None

        if sample_plot_dates is not None:
            for i in range(n_samples_to_plot):
                ax.plot(sample_plot_dates, plot_sample_urs.iloc[i].values,
                        color='lightsteelblue', alpha=0.15, linewidth=0.5, zorder=1) # Low alpha, thin lines
            # Add a label for the samples collection only once (won't appear in legend by default)
            ax.plot([], [], color='lightsteelblue', alpha=0.4, linewidth=1, label='Sample Trajectories') # Dummy plot for legend
        else:
            print("Warning: Mismatch between number of forecast dates and sample data columns after adjustment. Skipping sample trajectory plotting.")


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
    ax.set_title(f'{unemployed_label} Rate Forecast (Transformer Model)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{unemployed_label} Rate', fontsize=12)
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
    if not plot_sample_urs.empty and 'sample_plot_dates' in locals() and sample_plot_dates is not None:
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


# --- Main Execution ---
def main():
    script_start_time = time.time()
    print("=" * 60)
    print(" Starting Transformer Unemployment Forecast Script (PyTorch) ")
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    print(" Using parameters from config.py ")
    print("=" * 60)

    # --- Paths --- Use config paths
    project_root = config.PROJECT_ROOT
    # Point to the specific standard run directory containing the model artifacts
    model_dir = config.TRAIN_OUTPUT_SUBDIR / "standard_run" # Assumes model saved here
    processed_data_dir = config.PREPROCESS_OUTPUT_DIR
    raw_data_path = config.FORECAST_RAW_DATA_FILE
    output_dir = config.FORECAST_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "transformer_unemployment_forecast_py.png"
    forecast_csv_path = output_dir / "transformer_forecast_results_py.csv"
    raw_samples_csv_path = output_dir / "transformer_forecast_raw_samples_ur_py.csv" # Optional save

    print(f"Project Root: {project_root}")
    print(f"Model Directory: {model_dir}")
    print(f"Processed Data Directory: {processed_data_dir}")
    print(f"Raw Data Path (for historical): {raw_data_path}")
    print(f"Output Directory: {output_dir}")

    # Print device info
    device_name = "Unknown"
    if DEVICE.type == 'cuda':
        try:
            device_name = f"CUDA/GPU ({torch.cuda.get_device_name(DEVICE)})"
        except Exception:
            device_name = "CUDA/GPU (Name N/A)"
    elif DEVICE.type == 'mps':
        device_name = "Apple Silicon GPU (MPS)"
    elif DEVICE.type == 'cpu':
        device_name = "CPU"
    print(f"Using device: {device_name}")

    # --- Load Model and Data ---
    print("\n--- Loading Model and Data ---")
    try:
        model, params, metadata = load_pytorch_model_and_params(model_dir, processed_data_dir, DEVICE)
        # Load data, potentially specifying start date from config
        initial_sim_data_df, historical_rates_df, simulation_start_period, _ = load_and_prepare_data(
            processed_data_dir, raw_data_path, metadata, config.FORECAST_START_YEAR, config.FORECAST_START_MONTH
        )
        print(f"Data loaded. Simulation will start from period: {simulation_start_period}")
    except (FileNotFoundError, ValueError, KeyError, Exception) as e:
        print(f"\nERROR: Failed during model or data loading.")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()
        return # Stop execution

    # --- Prepare Initial Sequences ---
    print("\n--- Preparing Initial Sequences for Simulation ---")
    try:
        sequence_length = params['sequence_length']
        initial_sequences_np, sim_ids, initial_identifiers_df = get_sequences_for_simulation(
            sim_data_df=initial_sim_data_df,
            group_col=config.GROUP_ID_COL,
            seq_len=sequence_length,
            features=metadata['feature_names'],
            original_id_cols=metadata['original_identifier_columns'],
            pad_val=metadata['pad_value'],
            end_period=simulation_start_period
        )
        # Convert initial sequences to Tensor and move to device
        initial_sequences_tensor = torch.from_numpy(initial_sequences_np).to(DEVICE)
        print(f"Initial sequences prepared. Shape: {initial_sequences_tensor.shape}")
        print(f"Initial identifiers prepared. Shape: {initial_identifiers_df.shape}")

        # --- Explicitly delete large intermediate dataframes ---
        del initial_sim_data_df
        del initial_sequences_np
        gc.collect()
        print("Cleaned up intermediate dataframes from memory.")

    except (ValueError, KeyError, Exception) as e:
        print(f"\nERROR: Failed preparing initial sequences.")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()
        return # Stop execution

    # --- Run Forecast ---
    print("\n--- Running Multi-Period Forecast Simulation ---")
    try:
        # Capture both aggregated forecast and raw sample data
        forecast_agg_df, sample_urs_over_time = forecast_multiple_periods_pytorch(
            initial_sequences_tensor=initial_sequences_tensor,
            initial_identifiers_df=initial_identifiers_df,
            model=model,
            device=DEVICE,
            metadata=metadata,
            params=params, # Pass params if needed inside, e.g. seq_len (though already used)
            initial_period=simulation_start_period,
            periods_to_forecast=config.FORECAST_PERIODS,
            n_samples=config.MC_SAMPLES
        )
    except (ValueError, KeyError, RuntimeError, Exception) as e:
        print(f"\nERROR: An error occurred during the forecasting simulation.")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()
        # Attempt to save whatever results might exist before exiting
        if 'forecast_agg_df' in locals() and not forecast_agg_df.empty:
            forecast_agg_df.to_csv(output_dir / "transformer_forecast_PARTIAL_RESULTS_py.csv", index=False)
            print("Attempted to save partial aggregated results.")
        if 'sample_urs_over_time' in locals() and not sample_urs_over_time.empty:
            sample_urs_over_time.to_csv(output_dir / "transformer_forecast_PARTIAL_RAW_SAMPLES_py.csv")
            print("Attempted to save partial raw sample results.")
        return # Stop execution

    # --- Save & Plot Results ---
    print("\n--- Saving Forecast Results and Generating Plot ---")
    try:
        if not forecast_agg_df.empty:
            forecast_agg_df.to_csv(forecast_csv_path, index=False)
            print(f"Aggregated forecast results saved to: {forecast_csv_path}")
        else:
            print("Skipping aggregated results saving (dataframe is empty).")

        # Optionally save raw sample UR data
        if config.SAVE_RAW_SAMPLES and not sample_urs_over_time.empty:
            # Use original column names (periods) for saving raw data
            sample_urs_over_time.to_csv(raw_samples_csv_path)
            print(f"Raw sample unemployment rates saved to: {raw_samples_csv_path}")

    except Exception as e:
        print(f"Warning: Error saving forecast results CSV: {e}")

    try:
         # Pass the original raw sample data to the plotting function
         # The plotting function will handle prepending the historical point internally
        plot_unemployment_forecast_py(
            historical_rates_df,
            forecast_agg_df,
            sample_urs_over_time, # Pass the original raw samples
            plot_path,
            metadata
        )
    except Exception as e:
        print(f"Warning: Error generating plot: {e}")
        import traceback
        traceback.print_exc()


    # --- Completion ---
    script_end_time = time.time()
    elapsed_secs = script_end_time - script_start_time
    elapsed_mins = elapsed_secs / 60
    print("-" * 60)
    print(f" Forecast Script Completed ")
    print(f" End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    print(f" Total Elapsed Time: {elapsed_mins:.2f} minutes ({elapsed_secs:.1f} seconds) ")
    print("=" * 60)


if __name__ == "__main__":
    # Validate essential config settings before running main
    if (config.FORECAST_START_YEAR is None and config.FORECAST_START_MONTH is not None) or \
       (config.FORECAST_START_YEAR is not None and config.FORECAST_START_MONTH is None):
        print("\n--- CONFIGURATION ERROR ---")
        print(" In config.py: FORECAST_START_YEAR and FORECAST_START_MONTH must be provided together")
        print(" OR both must be set to None (to use the latest available data).")
        print("---------------------------\n")
        sys.exit(1)
    if config.FORECAST_START_MONTH is not None and not (1 <= config.FORECAST_START_MONTH <= 12):
        print("\n--- CONFIGURATION ERROR ---")
        print(f" In config.py: FORECAST_START_MONTH ({config.FORECAST_START_MONTH}) must be between 1 and 12.")
        print("---------------------------\n")
        sys.exit(1)
    if not config.FORECAST_PERIODS or config.FORECAST_PERIODS <= 0:
        print("\n--- CONFIGURATION ERROR ---")
        print(f" In config.py: FORECAST_PERIODS ({config.FORECAST_PERIODS}) must be a positive integer.")
        print("---------------------------\n")
        sys.exit(1)
    if not config.MC_SAMPLES or config.MC_SAMPLES <= 0:
        print("\n--- CONFIGURATION ERROR ---")
        print(f" In config.py: MC_SAMPLES ({config.MC_SAMPLES}) must be a positive integer.")
        print("---------------------------\n")
        sys.exit(1)

    # Run the main forecasting process
    main()