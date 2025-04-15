#!/bin/bash

#SBATCH --job-name=model_pipeline_job       # Job name
#SBATCH --output=model_pipeline_output.txt   # Standard output and error log
#SBATCH --ntasks=1                           # Number of tasks (usually 1 for a single job)
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task
#SBATCH --mem=128G                             # Memory required per node
#SBATCH --time=6:00:00                      # Time limit hrs:min:sec
#SBATCH -p gpu --gres=gpu:1

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Always calculate PROJECT_ROOT relative to the script directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
VENV_DIR="$PROJECT_ROOT/.venv"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
PYTHON_CMD="python3" # Or just "python" depending on your system

echo "============================================="
echo " Starting Model Pipeline Runner "
echo " Project Root: $PROJECT_ROOT"
echo " Script Directory: $SCRIPT_DIR"
echo "============================================="

# --- 1. Setup Virtual Environment ---
echo "\n--- Checking/Setting up Virtual Environment ---"
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found at $VENV_DIR. Creating..."
  "$PYTHON_CMD" -m venv "$VENV_DIR"
  echo "Virtual environment created."
else
  echo "Virtual environment found at $VENV_DIR."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

# --- 2. Install Requirements ---
echo "\n--- Installing Dependencies ---"
if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing packages from $REQUIREMENTS_FILE..."
  pip install -r "$REQUIREMENTS_FILE"
  echo "Dependencies installed."
else
  echo "ERROR: requirements.txt not found at $REQUIREMENTS_FILE"
  exit 1
fi

# --- 4. Run Python Scripts ---
# Navigate to the script directory to ensure relative paths in Python scripts work as expected
cd "$SCRIPT_DIR"

echo "\n--- Running 01_preprocess_cps_data.py ---"
"$PYTHON_CMD" 01_preprocess_cps_data.py
echo "Preprocessing script finished."

echo "\n--- Running 02_main_train_tune.py (Standard Training) ---"
# Add arguments here if needed, e.g., --use_trial TRIAL_NUM or --tune
"$PYTHON_CMD" 02_main_train_tune.py --tune
echo "Training script finished."

echo "\n--- Running 03_main_forecast.py ---"
"$PYTHON_CMD" 03_main_forecast.py
echo "Forecasting script finished."

# Deactivate virtual environment (optional)
# deactivate
# echo "Virtual environment deactivated."

echo "\n============================================="
echo " Model Pipeline Finished Successfully "
echo "============================================="

exit 0
