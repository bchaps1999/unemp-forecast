# Unemployment Rate Forecasting with Microdata

This project implements a pipeline to download and process Current Population Survey (CPS) data from IPUMS, train a Transformer-based deep learning model to predict individual labor market state transitions, and generate aggregate unemployment forecasts.

## Replication Instructions

1.  **IPUMS API Key:**
    *   Obtain an API key from [IPUMS USA](https://usa.ipums.org/usa/).
    *   Create a file named `.env` in the project root directory (`/Users/brendanchapuis/Projects/research/labor-abm`).
    *   Add the following line to the `.env` file, replacing `your_api_key_here` with your actual key:
        ```
        IPUMS_API_KEY=your_api_key_here
        ```
    *   The R scripts will automatically detect and use this key.

2.  **Adjust Parameters (Optional - for resource constraints):**
    *   **Data Download/Processing (R):** Edit `code/data/get_data.R`:
        *   Modify `extract_start_date_param`, `extract_end_date_param`, `subset_start_date_param`, `subset_end_date_param` to use a smaller date range.
        *   Set `refresh_extract_param = FALSE` to use previously downloaded raw IPUMS data if available.
    *   **Model Training/Tuning (Python):** Edit `code/model/config.py`:
        *   Reduce `PREPROCESS_NUM_INDIVIDUALS_FULL` and `PREPROCESS_NUM_INDIVIDUALS_HPT` to use fewer individuals for training/validation.
        *   Reduce `HPT_N_TRIALS` for faster hyperparameter tuning.
        *   Adjust `EPOCHS` or `HPT_EPOCHS` for shorter training times.
        *   Modify `SEQUENCE_LENGTH`, `BATCH_SIZE`, or model dimensions (`EMBED_DIM`, `FF_DIM`, etc.) for lower memory usage (may impact performance).

3.  **Run the Pipeline:**
    *   Ensure you have R and the required R packages installed (the `renv` package is used for dependency management). Run `renv::restore()` in the R console within the project directory if needed.
    *   Ensure you have Python 3 and the required Python packages installed (see `requirements.txt`). The `run_model_pipeline.sh` script attempts to set up a virtual environment and install requirements.
    *   Execute the main R orchestration script from the project root directory:
        ```R
        # In your R console, ensure the working directory is the project root
        # setwd("/Users/brendanchapuis/Projects/research/labor-abm") 
        source("code/main.R")
        ```
    *   This script will:
        *   Load the `renv` environment.
        *   Run the R data preparation script (`code/data/get_data.R`).
        *   Execute the shell script (`code/model/run_model_pipeline.sh`) which handles Python environment setup and runs the Python model pipeline scripts in sequence.

## Script Descriptions

*   **`code/main.R`**: The main R script that orchestrates the entire pipeline, calling the data preparation script and then the model pipeline shell script.
*   **`code/data/get_data.R`**: R script that defines parameters and calls functions to download, clean, process, and save the CPS data suitable for the model.
*   **`code/data/clean_cps.R`**: Contains R functions used by `get_data.R` for interacting with the IPUMS API, cleaning the raw CPS data, deriving features (like employment status, aggregate rates), and preparing the panel dataset.
*   **`code/data/scrape_cps_samples.R`**: An R script to scrape available CPS sample IDs from the IPUMS website, used to determine which samples to request.
*   **`code/model/run_model_pipeline.sh`**: A shell script executed by `main.R`. It sets up the Python virtual environment (if needed), installs dependencies from `requirements.txt`, and runs the Python preprocessing, training/tuning, and forecasting scripts in order.
*   **`code/model/01_preprocess_cps_data.py`**: Python script that takes the R-processed data, performs feature engineering specific to the Transformer model (e.g., scaling, one-hot encoding), handles time-based splitting for training, validation, testing, and HPT validation, and saves the "baked" data and preprocessing metadata/pipeline.
*   **`code/model/02_main_train_tune.py`**: Python script for either running a standard training process using specified hyperparameters (potentially loaded from a previous tuning run) or initiating hyperparameter tuning using Optuna. It handles data loading, sequence generation, model building, training loops, evaluation, and saving results/checkpoints.
*   **`code/model/03_main_forecast.py`**: Python script that loads a trained model and preprocessed test data, runs a multi-period forecasting simulation using Monte Carlo sampling, calculates aggregate unemployment rates, saves the forecast results, and generates a plot comparing the forecast to historical data.
*   **`code/model/config.py`**: Central configuration file for the Python model pipeline, defining file paths, hyperparameters, tuning settings, and simulation parameters.
*   **`code/model/utils.py`**: Python utility functions used across the model scripts (e.g., device selection, sequence generation, dataset class, signal handling).
*   **`code/model/models.py`**: Defines the PyTorch `TransformerForecastingModel` architecture.
*   **`code/model/training_helpers.py`**: Contains Python functions related to the model training loop (e.g., `train_epoch`, `evaluate_epoch`, creating dataloaders, building the model).
*   **`code/model/forecasting_helpers.py`**: Contains Python functions specifically for the forecasting process (e.g., loading models/data, running the multi-period simulation, updating features, calculating aggregate rates, plotting results, HPT objective calculation).
*   **`code/model/tuning_helpers.py`**: Contains Python functions related to hyperparameter tuning using Optuna (e.g., the `objective` function, the HPT runner `run_hyperparameter_tuning`).