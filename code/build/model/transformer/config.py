# config.py
import os
from pathlib import Path

# --- Project Root ---
# Assume config.py is located at PROJECT_ROOT/code/build/model/transformer/
# Adjust if the location is different
PROJECT_ROOT = Path(__file__).resolve().parents[4] # Navigate 4 levels up

# --- Common Parameters ---
RANDOM_SEED = 42
GROUP_ID_COL = "cpsidp"
DATE_COL = "date"
TARGET_COL = "target_state"
PAD_VALUE = -99.0

# --- 01_preprocess_cps_data.py Parameters ---
PREPROCESS_INPUT_FILE = PROJECT_ROOT / "data/processed/cps_transitions.csv"
PREPROCESS_OUTPUT_DIR = PROJECT_ROOT / "data/processed/transformer_input"
PREPROCESS_START_DATE = None # "YYYY-MM-DD" or None
PREPROCESS_END_DATE = None   # "YYYY-MM-DD" or None
PREPROCESS_NUM_INDIVIDUALS = 200000 # Integer or None
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# Test split is implicitly 1 - TRAIN_SPLIT - VAL_SPLIT

# Derived Preprocessing Output Filenames (used by training script)
TRAIN_DATA_FILENAME = "train_baked.parquet"
VAL_DATA_FILENAME = "val_baked.parquet"
TEST_DATA_FILENAME = "test_baked.parquet"
METADATA_FILENAME = "preprocessing_metadata.pkl"
RECIPE_FILENAME = "preprocessing_recipe.pkl" # Although not created by preprocess, it's related
FULL_DATA_FILENAME = "full_baked.parquet" # New filename for all processed data

# --- 02_train_transformer.py Parameters ---
# Input directory is PREPROCESS_OUTPUT_DIR
TRAIN_OUTPUT_SUBDIR = PROJECT_ROOT / "models/transformer_pytorch"
SEQUENCE_CACHE_DIR_NAME = "sequence_cache_py" # Relative to PREPROCESS_OUTPUT_DIR
# Add date filtering for training data (applied *after* loading baked data)
TRAIN_START_DATE = None # "YYYY-MM-DD" or None to use all available data before TRAIN_END_DATE
TRAIN_END_DATE = "2022-12-31"   # "YYYY-MM-DD" or None to use all available data after TRAIN_START_DATE

# Model Hyperparameters
SEQUENCE_LENGTH = 12
EMBED_DIM = 64
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

# Training Flags
REFRESH_MODEL = False # Force retraining even if model exists
REFRESH_SEQUENCES = False # Force regeneration of sequences even if cache exists
DEBUG_MODE = False # Enable debug mode (minimal usage currently)
USE_WEIGHTED_SAMPLING = False # Set to False to disable WeightedRandomSampler for training
USE_WEIGHTED_LOSS = False # Set to True to use weighted CrossEntropyLoss

# Parallel Workers (Set dynamically in script if <= 0)
# Use -1 in script logic to detect and set dynamically based on cores
PARALLEL_WORKERS = -1 # Set to specific number (e.g., 4) or -1 for auto

# --- 03_forecast_transformer.py Parameters ---
# Input directory for baked data is PREPROCESS_OUTPUT_DIR
# Input directory for model is TRAIN_OUTPUT_SUBDIR
FORECAST_OUTPUT_SUBDIR = PROJECT_ROOT / "output/forecast_transformer"
FORECAST_RAW_DATA_FILE = PROJECT_ROOT / "data/processed/cps_transitions.csv" # For historical rates

# Simulation Parameters
FORECAST_PERIODS = 24 # Number of periods to forecast ahead
MC_SAMPLES = 10 # Number of Monte Carlo samples per period
FORECAST_START_YEAR = 2020 # YYYY or None (defaults to latest in data)
FORECAST_START_MONTH = 12 # MM or None (defaults to latest in data)

# --- Derived Paths (can be constructed here or in scripts) ---
# Example:
# TRAIN_DATA_PATH = PREPROCESS_OUTPUT_DIR / TRAIN_DATA_FILENAME
# MODEL_PATH = TRAIN_OUTPUT_SUBDIR / "transformer_model.pt"

# Only print this message when the file is run directly, not when imported
if __name__ == "__main__":
    print(f"Config loaded. Project Root: {PROJECT_ROOT}")
