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

# --- Project Imports ---
import config # General configuration
# Ensure utils provides get_device and period_to_date
from utils import get_device, period_to_date
# Ensure models provides TransformerForecastingModel
from models import TransformerForecastingModel
# Ensure forecasting_helpers provides the necessary functions
from forecasting_helpers import (
    load_pytorch_model_and_params, load_forecasting_data,
    get_sequences_for_simulation, forecast_multiple_periods_pytorch,
    plot_unemployment_forecast_py
)

# --- Main Execution ---
def main():
    script_start_time = time.time()
    print("=" * 60)
    print(" Starting Transformer Unemployment Forecast Script ")
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    print(" Using parameters from config.py ")
    print("=" * 60)

    # --- Paths --- Use config paths
    project_root = config.PROJECT_ROOT
    model_dir = config.TRAIN_OUTPUT_SUBDIR / "standard_run"
    processed_data_dir = config.PREPROCESS_OUTPUT_DIR
    national_rates_path = config.NATIONAL_RATES_FILE # Use new path
    output_dir = config.FORECAST_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    # Output file paths will be defined *after* simulation_start_period is known

    print(f"Project Root: {project_root}")
    print(f"Model Directory: {model_dir}")
    print(f"Processed Data Directory: {processed_data_dir}")
    print(f"National Rates File (for historical): {national_rates_path}") # Added
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

    # --- Load Model and Data ---
    print("\n--- Loading Model and Data ---")
    try:
        model, params, metadata = load_pytorch_model_and_params(model_dir, processed_data_dir, DEVICE)
        # Load data for forecasting - uses TEST data, requires FULL TRAIN for history
        initial_sim_data_df, simulation_start_period, _ = load_forecasting_data(
            processed_data_dir, metadata, config.FORECAST_START_YEAR, config.FORECAST_START_MONTH
        )
        print(f"Data loaded. Simulation will start from period: {simulation_start_period} (first period in test set or specified)")

        # --- Define output paths now that simulation_start_period is known ---
        start_period_str = str(simulation_start_period) # e.g., "202312"
        plot_path = output_dir / f"transformer_unemployment_forecast_{start_period_str}.png"
        forecast_csv_path = output_dir / f"transformer_forecast_results_{start_period_str}.csv"
        raw_samples_csv_path = output_dir / f"transformer_forecast_raw_samples_ur_{start_period_str}.csv"
        print(f"Plot will be saved to: {plot_path}")
        print(f"Forecast results CSV will be saved to: {forecast_csv_path}")
        if config.SAVE_RAW_SAMPLES:
            print(f"Raw samples CSV will be saved to: {raw_samples_csv_path}")
        # --- End defining output paths ---

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
        # Get sequences, identifiers, and weights
        initial_sequences_np, sim_ids, initial_identifiers_df, initial_weights_series = get_sequences_for_simulation(
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
        print(f"Initial weights prepared. Shape: {initial_weights_series.shape}")

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
            initial_weights_series=initial_weights_series, # Pass weights
            model=model,
            device=DEVICE,
            metadata=metadata,
            params=params,
            initial_period=simulation_start_period,
            periods_to_forecast=config.FORECAST_PERIODS,
            n_samples=config.MC_SAMPLES,
            return_raw_samples=True # Get raw samples for plotting/saving
        )
    except (ValueError, KeyError, RuntimeError, Exception) as e:
        print(f"\nERROR: An error occurred during the forecasting simulation.")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()
        # Attempt to save whatever results might exist before exiting
        if 'forecast_agg_df' in locals() and not forecast_agg_df.empty:
            forecast_agg_df.to_csv(output_dir / "transformer_forecast_PARTIAL_RESULTS.csv", index=False)
            print("Attempted to save partial aggregated results.")
        if 'sample_urs_over_time' in locals() and not sample_urs_over_time.empty:
            sample_urs_over_time.to_csv(output_dir / "transformer_forecast_PARTIAL_RAW_SAMPLES.csv")
            print("Attempted to save partial raw sample results.")
        return # Stop execution
    finally:
        # Clean up GPU memory
        del initial_sequences_tensor, model
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        elif DEVICE.type == 'mps': torch.mps.empty_cache()
        gc.collect()
        print("Cleaned up GPU memory after forecast.")


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
         # Pass the path to the national rates file for historical data
        plot_unemployment_forecast_py(
            national_rates_file=national_rates_path, # Pass the file path
            forecast_agg_df=forecast_agg_df,
            sample_urs_over_time=sample_urs_over_time, # Pass the raw samples
            output_path=plot_path, # Use the updated path
            metadata=metadata
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
