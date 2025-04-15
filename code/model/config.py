# config.py
import os
from pathlib import Path

# --- Project Root ---
# Assume config.py is located at PROJECT_ROOT/code/model/
# Adjust if the location is different
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Navigate 2 levels up

# --- Common Parameters ---
RANDOM_SEED = 42
GROUP_ID_COL = "cpsidp"
DATE_COL = "date"
TARGET_COL = "target_state"
PAD_VALUE = -99.0
WEIGHT_COL = "wtfinl" # Define the weight column name explicitly

# --- 01_preprocess_cps_data.py Parameters ---
PREPROCESS_INPUT_FILE = PROJECT_ROOT / "data/processed/cps_transitions.csv"
PREPROCESS_OUTPUT_DIR = PROJECT_ROOT / "data/processed/transformer_input"
PREPROCESS_START_DATE = None # "YYYY-MM-DD" or None
PREPROCESS_END_DATE = None   # "YYYY-MM-DD" or None
# Sampling for FINAL training/validation splits (after HPT)
PREPROCESS_NUM_INDIVIDUALS_FULL = 500000 # Integer or None for all individuals before TRAIN_END_DATE (excluding HPT intervals)
# Sampling for HPT training/validation splits (during HPT)
PREPROCESS_NUM_INDIVIDUALS_HPT = 200000 # Integer or None for all individuals before HPT intervals
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# Test split is implicitly 1 - TRAIN_SPLIT - VAL_SPLIT
SPARSITY_THRESHOLD = 0.01 # Threshold for grouping sparse categorical features

# Time-based split for HPT validation
# Define intervals as (start_date, end_date) inclusive.
# Data *within* these intervals will be used for HPT validation.
# Data *before* the earliest start date will be used for fitting/standard splits.
HPT_VALIDATION_INTERVALS = [
    ("2007-07-01", "2008-09-01"),
    ("2013-01-01", "2014-03-01"),
    ("2018-01-01", "2019-03-01"),
]
# Define the end date for the main training set (used to separate test data)
# Must be after the latest HPT_VALIDATION_INTERVALS end date if they overlap
TRAIN_END_DATE_PREPROCESS = "2019-12-31" # Date used during preprocessing to split train/test

# Derived Preprocessing Output Filenames
# HPT-specific splits (smaller, used during tuning)
HPT_TRAIN_DATA_FILENAME = "hpt_train_baked.parquet"
HPT_VAL_DATA_FILENAME = "hpt_val_baked.parquet"
# Full splits (potentially larger, used for final model training)
FULL_TRAIN_DATA_FILENAME = "full_train_baked.parquet"
FULL_VAL_DATA_FILENAME = "full_val_baked.parquet"
# Test split (data after TRAIN_END_DATE_PREPROCESS)
TEST_DATA_FILENAME = "test_baked.parquet"
# HPT Validation split (data within HPT_VALIDATION_INTERVALS + lookback)
HPT_INTERVAL_DATA_FILENAME = "hpt_interval_data_baked.parquet" # Renamed for clarity
METADATA_FILENAME = "preprocessing_metadata.pkl"
RECIPE_FILENAME = "preprocessing_recipe.pkl" # Preprocessing pipeline
# FULL_BAKED_FILENAME = "full_baked.parquet" # REMOVED - Redundant
NATIONAL_RATES_FILE = PROJECT_ROOT / "data/processed/national_unemployment_rate.csv" # Path for national rates

# --- 02_train_transformer.py Parameters ---
# Input directory is PREPROCESS_OUTPUT_DIR
TRAIN_OUTPUT_SUBDIR = PROJECT_ROOT / "models"
SEQUENCE_CACHE_DIR_NAME = "sequence_cache_py" # Relative to PREPROCESS_OUTPUT_DIR

# Model Hyperparameters
SEQUENCE_LENGTH = 28 # Forecast periods + 16 (max possible length of CPS history)
NUM_HEADS = 4
FF_DIM = 64 # Feed-forward inner dim in transformer block
NUM_TRANSFORMER_BLOCKS = 2
MLP_UNITS = [64] # List of units for MLP head layers
DROPOUT = 0.1 # Dropout rate for transformer blocks
MLP_DROPOUT = 0.2 # Dropout rate for MLP head
EMBED_DIM = 64 # Embedding dimension (added)
EPOCHS = 75
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# Training Stability Parameters
MAX_GRAD_NORM = 1.0          # Maximum gradient norm for clipping
EARLY_STOPPING_PATIENCE = 5  # Epochs to wait before early stopping
LR_SCHEDULER_FACTOR = 0.5     # Factor to reduce LR by when plateau detected
LR_SCHEDULER_PATIENCE = 10    # Epochs to wait before reducing LR
HPT_EPOCHS = 3 # Number of epochs to run during hyperparameter tuning trials

# Training Flags
REFRESH_MODEL = False # Force retraining even if model exists
REFRESH_SEQUENCES = False # Force regeneration of sequences even if cache exists
DEBUG_MODE = False # Enable debug mode (minimal usage currently)

# Parallel Workers (Set dynamically in script if <= 0)
# Use -1 in script logic to detect and set dynamically based on cores
PARALLEL_WORKERS = -1 # Set to specific number (e.g., 4) or -1 for auto

# --- Hyperparameter Tuning (HPT) Parameters ---
HPT_N_TRIALS = 50 # Number of trials for Optuna
HPT_TIMEOUT_SECONDS = None # Optional timeout for the entire study (e.g., 3600 * 6 for 6 hours)
HPT_STUDY_NAME = "transformer_hpt_study" # Name for the Optuna study database file
# HPT_EPOCHS defined above
HPT_OBJECTIVE_METRIC = 'std_dev' # Choose 'rmse' or 'std_dev' to minimize
HPT_FORECAST_HORIZON = 12 # Number of months to forecast in HPT objective calculation
HPT_RESULTS_CSV = "hpt_results.csv" # Filename for HPT results log within study dir
BEST_HPARAMS_PKL = "best_hparams.pkl" # Filename for best HPT params within study dir
# HPT Validation Data Path (used by HPT objective function)
# HPT_VALIDATION_DATA_PATH_FOR_METRIC = PREPROCESS_OUTPUT_DIR / HPT_INTERVAL_DATA_FILENAME # REMOVED - Path constructed directly in tuning_helpers

# HPT Search Space Definitions
HPT_EMBED_DIM_OPTIONS = [32, 64, 128]
HPT_NUM_HEADS_OPTIONS = [2, 4, 8] # Must divide embed_dim
HPT_FF_DIM_MIN = 32
HPT_FF_DIM_MAX = 256
HPT_FF_DIM_STEP = 32
HPT_NUM_BLOCKS_MIN = 1
HPT_NUM_BLOCKS_MAX = 6 # Increased from 4
HPT_MLP_UNITS_MIN = 16 # For the single layer MLP head example
HPT_MLP_UNITS_MAX = 128
HPT_MLP_UNITS_STEP = 16
HPT_DROPOUT_MIN = 0.0
HPT_DROPOUT_MAX = 0.3
HPT_MLP_DROPOUT_MIN = 0.0
HPT_MLP_DROPOUT_MAX = 0.6 # Increased from 0.5
HPT_LR_MIN = 1e-5
HPT_LR_MAX = 1e-3
HPT_BATCH_SIZE_OPTIONS = [32, 64, 128, 256] # Added 256
# Add search space for loss weight factor
# Interpolates between unweighted (0.0) and inverse frequency weights (1.0).
# Factor = 0.0 -> No weights (equal weight per class)
# Factor = 1.0 -> Standard inverse frequency weighting for all classes
# Factor = (0.0, 1.0) -> Linear interpolation between equal and inverse frequency weights
HPT_LOSS_WEIGHT_FACTOR_MIN = 0.0 # Start from unweighted
HPT_LOSS_WEIGHT_FACTOR_MAX = 0.1 # End at full inverse frequency weighting

# HPT Pruner Settings
HPT_PRUNER_STARTUP = 5 # Number of trials before pruning starts
HPT_PRUNER_WARMUP = 3 # Number of epochs within a trial before pruning can occur

# --- 03_forecast_transformer.py Parameters ---
# Input directory for baked data is PREPROCESS_OUTPUT_DIR
# Input directory for model is TRAIN_OUTPUT_SUBDIR
FORECAST_OUTPUT_SUBDIR = PROJECT_ROOT / "output/forecast_transformer"
# Input data for forecasting will be TEST_DATA_FILENAME
# FORECAST_INPUT_DATA_FILENAME = FULL_BAKED_FILENAME # REMOVED

# Simulation Parameters
FORECAST_PERIODS = 12 # Number of periods to forecast ahead
MC_SAMPLES = 10 # Number of Monte Carlo samples per period
FORECAST_START_YEAR = 2021 # YYYY or None (defaults to latest in data)
FORECAST_START_MONTH = 12 # MM or None (defaults to latest in data)
SAVE_RAW_SAMPLES = True # Set to True to save the raw unemployment rate from each sample path

# Only print this message when the file is run directly, not when imported
if __name__ == "__main__":
    print(f"Config loaded. Project Root: {PROJECT_ROOT}")
