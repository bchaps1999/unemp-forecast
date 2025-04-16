# Unemployment Rate Forecasting with Microdata

This project implements a pipeline to download and process Current Population Survey (CPS) data from IPUMS, train a Transformer-based deep learning model to predict individual labor market state transitions, and generate aggregate unemployment forecasts.

## Replication Instructions

1.  **IPUMS API Key:**
    *   Obtain an API key from [IPUMS USA](https://usa.ipums.org/usa/).
    *   Create a file named `.env` in the project root directory.
    *   Add the following line to the `.env` file, replacing `your_api_key_here` with your actual key:
        ```
        IPUMS_API_KEY=your_api_key_here
        ```
    *   The R scripts will automatically detect and use this key.

2.  **Adjust Parameters (Optional - for resource constraints):**
    *   The default parameters were tested on an M3 MacBook Air with 16GB RAM and 8 cores. For machines with less memory or fewer cores, consider these adjustments:
    *   **Data Download/Processing (R):** Edit `code/data/get_data.R`:
        *   Reduce the date range to just a few years (e.g., Jan 2018 to Dec 2022) by changing `extract_start_date_param` and `extract_end_date_param`. This range provides sufficient data before and after Jan 2020 for all scripts to work properly.
        *   Set `refresh_extract_param = FALSE` to use previously downloaded raw IPUMS data if available.
    *   **Model Training/Tuning (Python):** Edit `code/model/config.py`:
        *   Reduce `PREPROCESS_NUM_INDIVIDUALS_FULL` and `PREPROCESS_NUM_INDIVIDUALS_HPT` to approximately 50,000 individuals for training/validation on machines with limited RAM.
        *   Change `EARLY_STOPPING_PATIENCE` to 1 to reduce model training time.

3.  **Run the Pipeline:**
    *   Execute the main R orchestration script from the project root directory, which will take a little while:
        ```R
        # In your R console, ensure the working directory is the project root
        source("code/main.R")
        ```
    *   This script will:
        *   Load the `renv` environment.
        *   Run the R data preparation script (`code/data/get_data.R`).
        *   Execute the shell script (`code/model/run_model_pipeline.sh`) which handles Python environment setup and runs the Python model pipeline scripts in sequence.
        * Run the R plotting script (`code/plot/forecast_plots.R`)

    *   If the pipeline fails, you can also run the scripts in this order (in an R session or Python virtual environment, depending on the script):
        *   `code/data/get_data.R`
        *   `code/model/01_preprocess_cps_data.py`
        *   `code/model/02_main_train_tune.py`
        *   `code/model/03_main_forecast.py`
        *   `code/plot/plot_forecast.R`

    *   Feel free to contact me if you can't get it to run!

## Script Descriptions

*   **`code/main.R`**: The main R script that orchestrates the entire pipeline, calling the data preparation script and then the model pipeline shell script.
*   **`code/data/get_data.R`**: R script that defines parameters and calls functions to download, clean, process, and save the CPS data suitable for the model.
*   **`code/data/clean_cps.R`**: Contains R functions used by `get_data.R` for interacting with the IPUMS API, cleaning the raw CPS data, deriving features (like employment status, aggregate rates), and preparing the panel dataset.
*   **`code/data/scrape_cps_samples.R`**: An R script to scrape available CPS sample IDs from the IPUMS website, used to determine which samples to request.
*   **`code/model/run_model_pipeline.sh`**: A shell script executed by `main.R`. It sets up the Python virtual environment (if needed), installs dependencies from `requirements.txt`, and runs the Python preprocessing, training/tuning, and forecasting scripts in order.
*   **`code/model/01_preprocess_cps_data.py`**: Python script that takes the R-processed data, performs feature engineering specific to the Transformer model (e.g., scaling, one-hot encoding), handles time-based splitting for training, validation, testing, and HPT validation, and saves the "baked" data and preprocessing metadata/pipeline.
*   **`code/model/02_main_train_tune.py`**: Python script for either running a standard training process using specified hyperparameters (potentially loaded from a previous tuning run) or initiating hyperparameter tuning using Optuna. It handles data loading, sequence generation, model building, training loops, evaluation, and saving results/checkpoints. You can run with "--tune" argument to enable hyperparameter tuning.
*   **`code/model/03_main_forecast.py`**: Python script that loads a trained model and preprocessed test data, runs a multi-period forecasting simulation using Monte Carlo sampling, calculates aggregate unemployment rates, saves the forecast results, and generates a plot comparing the forecast to historical data.
*   **`code/plot/plot_forecast.R`**: R script that batch-processes forecast outputs, generating publication-quality plots. It processes multiple forecast periods, adds confidence intervals, plots sample trajectories, and compares forecasts with historical unemployment rates.
*   **`code/model/config.py`**: Central configuration file for the Python model pipeline, defining file paths, hyperparameters, tuning settings, and simulation parameters.
*   **`code/model/utils.py`**: Python utility functions used across the model scripts (e.g., device selection, sequence generation, dataset class, signal handling).
*   **`code/model/models.py`**: Defines the PyTorch `TransformerForecastingModel` architecture.
*   **`code/model/training_helpers.py`**: Contains Python functions related to the model training loop (e.g., `train_epoch`, `evaluate_epoch`, creating dataloaders, building the model).
*   **`code/model/forecasting_helpers.py`**: Contains Python functions specifically for the forecasting process (e.g., loading models/data, running the multi-period simulation, updating features, calculating aggregate rates, plotting results, HPT objective calculation).
*   **`code/model/tuning_helpers.py`**: Contains Python functions related to hyperparameter tuning using Optuna (e.g., the `objective` function, the HPT runner `run_hyperparameter_tuning`).