import torch
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
import sys
import gc
import traceback # Import traceback at the top

# --- Project Imports ---
import config # General configuration
# Ensure utils provides get_device and period_to_date
from utils import get_device, period_to_date
# Ensure models provides TransformerForecastingModel
from models import TransformerForecastingModel
# Ensure forecasting_helpers provides the necessary functions
from forecasting_helpers import (
    load_pytorch_model_and_params, load_forecasting_data,
    get_sequences_for_simulation, forecast_multiple_periods_pytorch
    # Remove plot_unemployment_forecast_py from this import list
)

# --- Main Execution ---
def main():
    overall_start_time = time.time()
    print("=" * 60)
    print(" Starting Transformer Multi-Forecast Script ")
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    print(" Using parameters from config.py ")
    print("=" * 60)

    # --- Paths --- Use config paths
    project_root = config.PROJECT_ROOT
    model_dir = config.TRAIN_OUTPUT_SUBDIR / "standard_run" # Assuming standard run model
    processed_data_dir = config.PREPROCESS_OUTPUT_DIR
    national_rates_path = config.NATIONAL_RATES_FILE
    output_dir = config.FORECAST_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project Root: {project_root}")
    print(f"Model Directory: {model_dir}")
    print(f"Processed Data Directory: {processed_data_dir}")
    print(f"National Rates File (for historical): {national_rates_path}")
    print(f"Output Directory: {output_dir}")

    # --- Get Device ---
    DEVICE = get_device()
    device_name = "Unknown"
    if DEVICE.type == 'cuda':
        try: device_name = f"CUDA/GPU ({torch.cuda.get_device_name(DEVICE)})"
        except Exception: device_name = "CUDA/GPU (Name N/A)"
    elif DEVICE.type == 'mps': device_name = "Apple Silicon GPU (MPS)"
    elif DEVICE.type == 'cpu': device_name = "CPU"
    print(f"Using device: {device_name}")

    # --- Load Model and FULL Test Data (Once) ---
    print("\n--- Loading Model and Full Test Data ---")
    try:
        model, params, metadata = load_pytorch_model_and_params(model_dir, processed_data_dir, DEVICE)
        # Load FULL test data and available periods
        full_test_data_df, available_periods = load_forecasting_data(processed_data_dir, metadata)
        if not available_periods:
            raise ValueError("No available forecast periods found in the test data.")
        latest_available_period = max(available_periods)
        print(f"Model and full test data loaded. Latest available period: {latest_available_period}")
    except (FileNotFoundError, ValueError, KeyError, Exception) as e:
        print(f"\nERROR: Failed during model or full data loading.")
        print(f"Details: {e}")
        traceback.print_exc()
        return # Stop execution

    # --- Determine Simulation Start Periods ---
    target_start_periods = set()
    launch_dates_str = config.FORECAST_LAUNCH_DATES
    print(f"\n--- Determining Simulation Start Periods based on Launch Dates: {launch_dates_str} ---")

    for launch_date_str in launch_dates_str:
        try:
            launch_dt = pd.to_datetime(launch_date_str)
            launch_period = launch_dt.year * 100 + launch_dt.month
            # Find the latest available period STRICTLY BEFORE the launch period
            potential_start = None
            for p in sorted(available_periods, reverse=True):
                if p < launch_period:
                    potential_start = p
                    break
            if potential_start:
                target_start_periods.add(potential_start)
                print(f" - For launch date {launch_date_str} (period {launch_period}), using start period: {potential_start}")
            else:
                print(f" - Warning: No available period found before launch date {launch_date_str} (period {launch_period}). Skipping.")
        except Exception as e:
            print(f" - Warning: Could not process launch date '{launch_date_str}': {e}")

    # Add the absolute latest available period
    if latest_available_period:
        print(f" - Adding latest available period: {latest_available_period}")
        target_start_periods.add(latest_available_period)

    if not target_start_periods:
        print("\nERROR: No valid simulation start periods determined. Exiting.")
        return

    sorted_start_periods = sorted(list(target_start_periods))
    print(f"\nWill run forecasts starting from periods: {sorted_start_periods}")

    # --- Loop Through Each Simulation Start Period ---
    for simulation_start_period in sorted_start_periods:
        run_start_time = time.time()
        print("\n" + "=" * 60)
        print(f" Starting Forecast Run for Start Period: {simulation_start_period} ")
        print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
        print("=" * 60)

        # --- Calculate First Forecast Period (for naming) ---
        start_year = simulation_start_period // 100
        start_month = simulation_start_period % 100
        if start_month == 12:
            first_forecast_year = start_year + 1
            first_forecast_month = 1
        else:
            first_forecast_year = start_year
            first_forecast_month = start_month + 1
        first_forecast_period = first_forecast_year * 100 + first_forecast_month
        first_forecast_period_str = str(first_forecast_period)
        print(f" -> First period to be forecasted: {first_forecast_period_str}") # Add info

        # --- Filter Data for Current Start Period ---
        print(f"\n--- Filtering Data for Start Period {simulation_start_period} ---")
        try:
            start_period_ids = full_test_data_df[full_test_data_df['period'] == simulation_start_period][config.GROUP_ID_COL].unique()
            if len(start_period_ids) == 0:
                print(f"Warning: No individuals found for start period {simulation_start_period}. Skipping this run.")
                continue

            initial_sim_data_df = full_test_data_df[
                (full_test_data_df[config.GROUP_ID_COL].isin(start_period_ids)) &
                (full_test_data_df['period'] <= simulation_start_period)
            ].copy().sort_values([config.GROUP_ID_COL, config.DATE_COL]) # Use copy()
            print(f"Filtered data shape for period {simulation_start_period}: {initial_sim_data_df.shape}")

        except Exception as e:
            print(f"ERROR filtering data for start period {simulation_start_period}: {e}")
            continue # Skip to next start period

        # --- Prepare Initial Sequences ---
        print(f"\n--- Preparing Initial Sequences for Start Period {simulation_start_period} ---")
        try:
            sequence_length = params['sequence_length']
            initial_sequences_np, sim_ids, initial_identifiers_df, initial_weights_series = get_sequences_for_simulation(
                sim_data_df=initial_sim_data_df,
                group_col=config.GROUP_ID_COL,
                seq_len=sequence_length,
                features=metadata['feature_names'],
                original_id_cols=metadata['original_identifier_columns'],
                pad_val=metadata['pad_value'],
                end_period=simulation_start_period
            )
            initial_sequences_tensor = torch.from_numpy(initial_sequences_np).to(DEVICE)
            print(f"Initial sequences prepared. Shape: {initial_sequences_tensor.shape}")
            print(f"Initial identifiers prepared. Shape: {initial_identifiers_df.shape}")
            print(f"Initial weights prepared. Shape: {initial_weights_series.shape}")

            # Explicitly delete large intermediate dataframes for this run
            del initial_sim_data_df, initial_sequences_np
            gc.collect()
            print("Cleaned up intermediate dataframes from memory for this run.")

        except (ValueError, KeyError, Exception) as e:
            print(f"\nERROR: Failed preparing initial sequences for start period {simulation_start_period}.")
            print(f"Details: {e}")
            traceback.print_exc()
            # Clean up tensor if it exists before skipping
            if 'initial_sequences_tensor' in locals(): del initial_sequences_tensor
            gc.collect()
            continue # Skip to next start period

        # --- Run Forecast ---
        print(f"\n--- Running Multi-Period Forecast Simulation for Start Period {simulation_start_period} ---")
        forecast_agg_df = pd.DataFrame() # Initialize empty
        sample_urs_over_time = pd.DataFrame() # Initialize empty
        try:
            forecast_agg_df, sample_urs_over_time = forecast_multiple_periods_pytorch(
                initial_sequences_tensor=initial_sequences_tensor,
                initial_identifiers_df=initial_identifiers_df,
                initial_weights_series=initial_weights_series,
                model=model,
                device=DEVICE,
                metadata=metadata,
                params=params,
                initial_period=simulation_start_period,
                periods_to_forecast=config.FORECAST_PERIODS,
                n_samples=config.MC_SAMPLES,
                return_raw_samples=config.SAVE_RAW_SAMPLES, # Use config flag
                forecast_batch_size=config.FORECAST_BATCH_SIZE
            )
        except (ValueError, KeyError, RuntimeError, Exception) as e:
            print(f"\nERROR: An error occurred during the forecasting simulation for start period {simulation_start_period}.")
            print(f"Details: {e}")
            traceback.print_exc()
            # Attempt to save partial results if they exist
            # (Saving logic moved to next step, just continue)
        finally:
            # Clean up GPU memory for this specific run's forecast
            del initial_sequences_tensor # Delete the main input tensor for this run
            # Model is kept, identifiers/weights might be reused if logic changes, keep for now
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            elif DEVICE.type == 'mps': torch.mps.empty_cache()
            gc.collect()
            print("Cleaned up GPU memory after forecast for this run.")

        # --- Define Output Paths for this Run ---
        # Use the first_forecast_period_str for file naming
        forecast_csv_path = output_dir / f"transformer_forecast_results_{first_forecast_period_str}.csv"
        raw_samples_csv_path = output_dir / f"transformer_forecast_raw_samples_ur_{first_forecast_period_str}.csv"

        # --- Save Results for this Run --- # Updated section title
        print(f"\n--- Saving Forecast Results (Launched: {first_forecast_period_str}) ---") # Updated print statement
        try:
            if not forecast_agg_df.empty:
                forecast_agg_df.to_csv(forecast_csv_path, index=False)
                print(f"Aggregated forecast results saved to: {forecast_csv_path}")
            else:
                print("Skipping aggregated results saving (dataframe is empty).")

            if config.SAVE_RAW_SAMPLES and not sample_urs_over_time.empty:
                sample_urs_over_time.to_csv(raw_samples_csv_path)
                print(f"Raw sample unemployment rates saved to: {raw_samples_csv_path}")
            elif config.SAVE_RAW_SAMPLES:
                 print("Skipping raw samples saving (dataframe is empty).")

        except Exception as e:
            print(f"Warning: Error saving forecast results CSV for start period {simulation_start_period}: {e}")

        run_end_time = time.time()
        elapsed_secs = run_end_time - run_start_time
        print(f"\n--- Forecast Run for Start Period {simulation_start_period} Completed ---")
        print(f" End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
        print(f" Elapsed Time for this run: {elapsed_secs / 60:.2f} minutes ({elapsed_secs:.1f} seconds) ")
        print("-" * 60)

        # Explicit cleanup at end of loop iteration
        del forecast_agg_df, sample_urs_over_time, initial_identifiers_df, initial_weights_series
        gc.collect()

    # --- Overall Completion ---
    overall_end_time = time.time()
    overall_elapsed_secs = overall_end_time - overall_start_time
    overall_elapsed_mins = overall_elapsed_secs / 60
    print("\n" + "=" * 60)
    print(f" All Forecast Runs Completed ")
    print(f" Overall End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    print(f" Total Elapsed Time: {overall_elapsed_mins:.2f} minutes ({overall_elapsed_secs:.1f} seconds) ")
    print("=" * 60)


if __name__ == "__main__":
    # Validate essential config settings before running main
    # Remove validation for FORECAST_START_YEAR/MONTH
    if not config.FORECAST_LAUNCH_DATES:
         print("\n--- CONFIGURATION ERROR ---")
         print(" In config.py: FORECAST_LAUNCH_DATES list cannot be empty.")
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
