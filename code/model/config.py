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
PREPROCESS_NUM_INDIVIDUALS = None # Integer or None
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

# Derived Preprocessing Output Filenames (used by training script)
TRAIN_DATA_FILENAME = "train_baked.parquet"
VAL_DATA_FILENAME = "val_baked.parquet"
TEST_DATA_FILENAME = "test_baked.parquet"
HPT_VAL_DATA_FILENAME = "hpt_val_data.parquet" # Filename for HPT validation data
METADATA_FILENAME = "preprocessing_metadata.pkl"
RECIPE_FILENAME = "preprocessing_recipe.pkl" # Although not created by preprocess, it's related
FULL_DATA_FILENAME = "full_baked.parquet" # New filename for all processed data
NATIONAL_RATES_FILE = PROJECT_ROOT / "data/processed/national_unemployment_rate.csv" # Path for national rates

# --- 02_train_transformer.py Parameters ---
# Input directory is PREPROCESS_OUTPUT_DIR
TRAIN_OUTPUT_SUBDIR = PROJECT_ROOT / "models" # Removed /standard_run
SEQUENCE_CACHE_DIR_NAME = "sequence_cache_py" # Relative to PREPROCESS_OUTPUT_DIR
# Add date filtering for training data (applied *after* loading baked data)
TRAIN_START_DATE = None # "YYYY-MM-DD" or None to use all available data before TRAIN_END_DATE
TRAIN_END_DATE = "2019-12-31"   # "YYYY-MM-DD" or None to use all available data after TRAIN_START_DATE

# Model Hyperparameters
SEQUENCE_LENGTH = 28 # Forecast periods + 16 (max possible length of CPS history)
NUM_HEADS = 4
FF_DIM = 64 # Feed-forward inner dim in transformer block
NUM_TRANSFORMER_BLOCKS = 2
MLP_UNITS = [64] # List of units for MLP head layers
DROPOUT = 0.1 # Dropout rate for transformer blocks
MLP_DROPOUT = 0.2 # Dropout rate for MLP head
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
HPT_OBJECTIVE_METRIC = 'variance' # Choose 'rmse' or 'variance' to minimize
HPT_FORECAST_HORIZON = 12 # Number of months to forecast in HPT objective calculation
HPT_RESULTS_CSV = "hpt_results.csv" # Filename for HPT results log within study dir
BEST_HPARAMS_PKL = "best_hparams.pkl" # Filename for best HPT params within study dir

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

# Simulation Parameters
FORECAST_PERIODS = 12 # Number of periods to forecast ahead
MC_SAMPLES = 10 # Number of Monte Carlo samples per period
FORECAST_START_YEAR = 2020 # YYYY or None (defaults to latest in data)
FORECAST_START_MONTH = 12 # MM or None (defaults to latest in data)
SAVE_RAW_SAMPLES = True # Set to True to save the raw unemployment rate from each sample path

# Only print this message when the file is run directly, not when imported
if __name__ == "__main__":
    print(f"Config loaded. Project Root: {PROJECT_ROOT}")
