import optuna
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import csv
import traceback
import sys
import gc
from sklearn.utils.class_weight import compute_class_weight

# Assuming config.py contains HPT settings and paths
import config
# Assuming utils.py contains get_device
from utils import get_device # Removed load_and_prepare_data import
# Assuming forecasting_helpers.py contains calculate_hpt_forecast_metrics
from forecasting_helpers import calculate_hpt_forecast_metrics, evaluate_aggregate_unemployment_error
# train_and_evaluate_internal is passed as an argument

# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, train_evaluate_func,
              national_rates_file: Path, # Path to truth data
              metadata, device):
    """
    Optuna objective function to minimize the selected forecast metric (RMSE or Std Dev).
    """
    study_dir = Path(config.TRAIN_OUTPUT_SUBDIR) / trial.study.study_name
    hpt_results_file = study_dir / config.HPT_RESULTS_CSV
    best_hparams_file = study_dir / config.BEST_HPARAMS_PKL
    objective_metric_name = config.HPT_OBJECTIVE_METRIC.lower() # 'rmse' or 'std_dev'
    processed_data_dir = Path(config.PREPROCESS_OUTPUT_DIR) # Get path from config

    # --- Define Hyperparameter Search Space ---
    hparams = {
        # Tunable Parameters
        'embed_dim': trial.suggest_categorical("embed_dim", config.HPT_EMBED_DIM_OPTIONS),
        'num_heads': trial.suggest_categorical("num_heads", config.HPT_NUM_HEADS_OPTIONS),
        'ff_dim': trial.suggest_int("ff_dim", config.HPT_FF_DIM_MIN, config.HPT_FF_DIM_MAX, step=config.HPT_FF_DIM_STEP),
        'num_transformer_blocks': trial.suggest_int("num_transformer_blocks", config.HPT_NUM_BLOCKS_MIN, config.HPT_NUM_BLOCKS_MAX),
        'mlp_units': [trial.suggest_int(f"mlp_units_{i}", config.HPT_MLP_UNITS_MIN, config.HPT_MLP_UNITS_MAX, step=config.HPT_MLP_UNITS_STEP) for i in range(trial.suggest_int("mlp_layers", 1, 2))],
        'dropout': trial.suggest_float("dropout", config.HPT_DROPOUT_MIN, config.HPT_DROPOUT_MAX),
        'mlp_dropout': trial.suggest_float("mlp_dropout", config.HPT_MLP_DROPOUT_MIN, config.HPT_MLP_DROPOUT_MAX),
        'learning_rate': trial.suggest_float("learning_rate", config.HPT_LR_MIN, config.HPT_LR_MAX, log=True),
        'batch_size': trial.suggest_categorical("batch_size", config.HPT_BATCH_SIZE_OPTIONS),
        'loss_weight_factor': trial.suggest_float("loss_weight_factor", config.HPT_LOSS_WEIGHT_FACTOR_MIN, config.HPT_LOSS_WEIGHT_FACTOR_MAX),
        # Fixed Parameters (passed from config)
        'sequence_length': config.SEQUENCE_LENGTH,
        'epochs': config.HPT_EPOCHS,
        'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        'lr_scheduler_factor': config.LR_SCHEDULER_FACTOR,
        'lr_scheduler_patience': config.LR_SCHEDULER_PATIENCE,
        'max_grad_norm': config.MAX_GRAD_NORM,
        'pad_value': config.PAD_VALUE,
        'parallel_workers': config.PARALLEL_WORKERS,
        'refresh_sequences': config.REFRESH_SEQUENCES,
        'random_seed': config.RANDOM_SEED,
        'hpt_forecast_horizon': config.HPT_FORECAST_HORIZON,
    }
    # Store the tunable keys separately for logging
    tunable_keys = [
        'embed_dim', 'num_heads', 'ff_dim', 'num_transformer_blocks', 'mlp_units',
        'dropout', 'mlp_dropout', 'learning_rate', 'batch_size', 'loss_weight_factor'
    ]

    # --- Constraint Check: Embed dim vs Num heads ---
    if hparams['embed_dim'] % hparams['num_heads'] != 0:
        msg = f"embed_dim ({hparams['embed_dim']}) must be divisible by num_heads ({hparams['num_heads']})."
        print(f"Pruning trial {trial.number}: {msg}")
        raise optuna.TrialPruned(msg)

    # --- Run Training and Standard Evaluation ---
    # This function trains the model using HPT data splits and returns results.
    # It handles its own data loading, sequence gen, training loop etc.
    train_results = train_evaluate_func(hparams, trial=trial) # Pass trial for potential pruning

    # Extract results needed for HPT objective calculation and logging
    model = train_results.get('model') # Model object is returned if training completed
    run_status = train_results.get('status', 'Unknown')
    best_val_loss = train_results.get('best_val_loss', float('inf')) # Loss on HPT validation set
    std_val_agg_error = train_results.get('std_val_agg_error', float('inf')) # Agg error on HPT validation set

    objective_value = float('inf') # Initialize objective value
    forecast_rmse = float('inf')
    forecast_std_dev = float('inf') # Changed from variance
    hpt_val_data_loaded = None # Initialize

    # --- Calculate HPT Objective Metrics (Forecast RMSE and Std Dev) ---
    # Only proceed if training completed successfully and returned a model
    if run_status == "Completed" and model is not None:
        try:
            # --- Load HPT Interval Data (Baked) ---
            # This data contains the specific time intervals needed for the HPT forecast metric
            hpt_interval_data_file = processed_data_dir / config.HPT_INTERVAL_DATA_FILENAME # Use the correct file
            if not hpt_interval_data_file.exists():
                raise FileNotFoundError(f"HPT interval data file not found: {hpt_interval_data_file}")

            print(f"Loading HPT interval baked data from: {hpt_interval_data_file}")
            try:
                hpt_interval_data_loaded = pd.read_parquet(hpt_interval_data_file)
                if hpt_interval_data_loaded.empty:
                    raise ValueError("HPT interval data file is empty.")
                # Ensure 'period' column exists (needed by calculate_hpt_forecast_metrics)
                if 'period' not in hpt_interval_data_loaded.columns:
                     date_col = config.DATE_COL # Get date column name from config
                     hpt_interval_data_loaded['period'] = pd.to_datetime(hpt_interval_data_loaded[date_col]).dt.year * 100 + pd.to_datetime(hpt_interval_data_loaded[date_col]).dt.month
                print(f"Loaded HPT interval data ({hpt_interval_data_loaded.shape}) for forecast metric calculation.")
            except Exception as e:
                print(f"Error loading HPT interval data: {e}")
                raise # Re-raise to be caught by the outer try-except

            # Get intervals from metadata
            if 'hpt_validation_intervals' not in metadata:
                 raise ValueError("Metadata is missing 'hpt_validation_intervals'. Cannot calculate HPT objective.")

            # Call the function to calculate forecast metrics against national rates
            forecast_metrics = calculate_hpt_forecast_metrics(
                model=model,
                hpt_val_baked_data=hpt_interval_data_loaded, # Pass the loaded interval data
                national_rates_file=national_rates_file, # Pass the path to actual rates
                metadata=metadata, # Pass full metadata
                params=hparams, # Pass the hyperparameters used for this trial
                device=device,
                forecast_horizon=hparams['hpt_forecast_horizon']
            )
            forecast_rmse = forecast_metrics.get('rmse', float('inf'))
            forecast_std_dev = forecast_metrics.get('std_dev', float('inf')) # Changed from variance

            # Select the metric to optimize based on config
            if objective_metric_name == 'rmse':
                objective_value = forecast_rmse
            elif objective_metric_name == 'std_dev': # Changed from variance
                objective_value = forecast_std_dev
            else:
                raise ValueError(f"Invalid HPT_OBJECTIVE_METRIC: '{config.HPT_OBJECTIVE_METRIC}'. Choose 'rmse' or 'std_dev'.") # Updated error message

            print(f"Trial {trial.number} objective ({objective_metric_name}): {objective_value:.6f} | RMSE: {forecast_rmse:.6f}, Std Dev: {forecast_std_dev:.6f}") # Changed Variance to Std Dev

        except Exception as e:
            print(f"ERROR calculating HPT forecast metrics for trial {trial.number}: {e}")
            traceback.print_exc()
            objective_value = float('inf') # Penalize trial if calculation fails
            forecast_rmse = float('inf')
            forecast_std_dev = float('inf') # Changed from variance
            run_status = "Failed HPT Metric Calc" # Update status
    elif run_status == "Pruned":
        print(f"Trial {trial.number} was pruned during training. Objective set to Inf.")
        objective_value = float('inf')
    else: # Failed, Interrupted, etc.
        print(f"Trial {trial.number} did not complete successfully (Status: {run_status}). Objective set to Inf.")
        objective_value = float('inf')

    # Handle NaN/Inf objective values before returning to Optuna
    if np.isnan(objective_value) or np.isinf(objective_value):
        print(f"Warning: Trial {trial.number} resulted in NaN or Inf objective value ({objective_value}). Returning infinity.")
        objective_value = float('inf') # Ensure Optuna receives a valid float

    # --- Log results to CSV ---
    params_flat = hparams.copy()
    params_flat['mlp_units'] = str(params_flat['mlp_units']) # Convert list to string for CSV
    # Update fieldnames to include both metrics and standard validation error
    fieldnames = ['trial_number', 'status', f'hpt_forecast_{objective_metric_name}',
                  'hpt_forecast_rmse', 'hpt_forecast_std_dev', # Changed variance to std_dev
                  'hpt_val_loss', 'hpt_val_agg_error'] + tunable_keys # Use HPT val metrics here
    log_entry = {
        'trial_number': trial.number,
        'status': run_status,
        f'hpt_forecast_{objective_metric_name}': f"{objective_value:.6f}" if not np.isinf(objective_value) else 'inf',
        'hpt_forecast_rmse': f"{forecast_rmse:.6f}" if not np.isinf(forecast_rmse) else 'inf',
        'hpt_forecast_std_dev': f"{forecast_std_dev:.6f}" if not np.isinf(forecast_std_dev) else 'inf', # Changed variance to std_dev
        'hpt_val_loss': f"{best_val_loss:.6f}" if not np.isinf(best_val_loss) and not np.isnan(best_val_loss) else 'inf',
        'hpt_val_agg_error': f"{std_val_agg_error:.6f}" if not np.isinf(std_val_agg_error) and not np.isnan(std_val_agg_error) else 'inf',
        **{k: params_flat[k] for k in tunable_keys}
    }
    try:
        study_dir.mkdir(parents=True, exist_ok=True)
        write_header = not hpt_results_file.exists() or hpt_results_file.stat().st_size == 0
        with open(hpt_results_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if write_header: writer.writeheader()
            writer.writerow(log_entry)
        # print(f"Logged trial {trial.number} results to {hpt_results_file}") # Less verbose logging

    except Exception as e: print(f"ERROR logging trial {trial.number} results to CSV: {e}")

    # --- Update best hyperparameters file ---
    current_best_value = float('inf')
    try: # Handle case where study has no completed trials yet
        if trial.study.best_trial and trial.study.best_trial.state == optuna.trial.TrialState.COMPLETE:
             current_best_value = trial.study.best_value
    except ValueError: pass # Keep current_best_value as inf

    # Check if current trial completed successfully and improved the objective
    if run_status == "Completed" and not np.isinf(objective_value) and objective_value < current_best_value:
         print(f"Trial {trial.number} improved HPT objective ({objective_metric_name}) to {objective_value:.6f}. Updating best hyperparameters.")
         try:
             best_params_to_save = trial.params # Save only the *tuned* parameters suggested by Optuna
             with open(best_hparams_file, 'wb') as f: pickle.dump(best_params_to_save, f)
             print(f"Best hyperparameters updated in {best_hparams_file}")
         except Exception as e: print(f"ERROR saving best hyperparameters for trial {trial.number}: {e}")

    # --- Clean up model and loaded data from memory ---
    # Check if variables exist before deleting
    if 'model' in locals() and model is not None:
        del model
    if 'hpt_interval_data_loaded' in locals() and hpt_interval_data_loaded is not None:
        del hpt_interval_data_loaded
    if device.type in ['cuda', 'mps']:
        torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache()
    gc.collect()

    # Return the chosen objective value for Optuna to minimize
    return objective_value


# --- HPT Runner Function ---

def run_hyperparameter_tuning(args, base_output_dir, train_evaluate_func):
    """Sets up and runs the Optuna hyperparameter tuning study."""
    print(f"\n--- Starting Hyperparameter Tuning ({args.n_trials} trials) ---")
    study_name = args.study_name
    study_dir = base_output_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    storage_path = study_dir / f"{study_name}.db"
    storage_name = f"sqlite:///{storage_path}"
    hpt_results_file = study_dir / config.HPT_RESULTS_CSV
    best_hparams_file = study_dir / config.BEST_HPARAMS_PKL

    # --- Validate HPT Metric Choice ---
    optimization_metric = config.HPT_OBJECTIVE_METRIC.lower()
    if optimization_metric not in ['rmse', 'std_dev']: # Changed variance to std_dev
        print(f"ERROR: Invalid HPT_OBJECTIVE_METRIC: '{config.HPT_OBJECTIVE_METRIC}'. Choose 'rmse' or 'std_dev'.") # Updated error message
        sys.exit(1)

    print(f"Optuna study name: {study_name}")
    print(f"Optuna storage: {storage_name}")
    print(f"HPT results log: {hpt_results_file}")
    print(f"Best hyperparameters file: {best_hparams_file}")
    print(f"HPT Objective: Minimize '{optimization_metric}' on HPT Validation Intervals")
    print(f"HPT Forecast Horizon: {config.HPT_FORECAST_HORIZON} months")
    print(f"HPT Loss Weight Factor Range: [{config.HPT_LOSS_WEIGHT_FACTOR_MIN}, {config.HPT_LOSS_WEIGHT_FACTOR_MAX}]")

    # --- Load Metadata and Paths needed for Objective Calculation ---
    # The objective function loads the actual HPT interval data itself.
    print("\nLoading paths and metadata needed for HPT objective calculation...")
    national_rates_file = config.NATIONAL_RATES_FILE
    metadata_file = Path(config.PREPROCESS_OUTPUT_DIR) / config.METADATA_FILENAME
    metadata = None
    try:
        if not national_rates_file.exists():
            raise FileNotFoundError(f"National rates file not found: {national_rates_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'rb') as f: metadata = pickle.load(f)

        if 'hpt_validation_intervals' not in metadata:
            raise ValueError("Metadata loaded, but 'hpt_validation_intervals' key is missing.")
        print(f"Loaded metadata. Using HPT intervals: {metadata['hpt_validation_intervals']}")
        print(f"Using national rates file: {national_rates_file}")

    except Exception as e:
        print(f"ERROR: Failed to load data/metadata needed for HPT objective: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Get Device ---
    device = get_device() # Get device once for all trials

    # --- Create Study ---
    pruner = optuna.pruners.MedianPruner(
            n_startup_trials=config.HPT_PRUNER_STARTUP,
            n_warmup_steps=config.HPT_PRUNER_WARMUP, # Epochs before pruning can happen
            interval_steps=1 # Prune check after each epoch (post warmup)
        )
    print(f"Optuna pruner enabled: MedianPruner (startup={config.HPT_PRUNER_STARTUP}, warmup={config.HPT_PRUNER_WARMUP})")

    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True,
        direction="minimize", # Minimize the chosen metric
        pruner=pruner
    )

    # --- Optimize ---
    try:
        if study.best_trial and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Resuming study. Existing best value: {study.best_value:.6f}")
        else:
            print("Starting new study or resuming with no completed trials.")
    except ValueError: print("Starting new study or resuming with no completed trials.")

    try:
        # Pass necessary arguments to the objective function using a lambda
        study.optimize(
            lambda trial: objective(trial, train_evaluate_func, national_rates_file, metadata, device),
            n_trials=args.n_trials,
            timeout=config.HPT_TIMEOUT_SECONDS,
            gc_after_trial=True, # Enable garbage collection after each trial
            n_jobs=1 # Run trials sequentially (GPU resource constraint)
        )
    except KeyboardInterrupt:
        print("\nOptuna optimization interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn critical error occurred during Optuna optimization: {e}")
        traceback.print_exc()

    # --- Print HPT Summary ---
    print("\n--- HPT Finished ---")
    try:
        print(f"Study statistics: ")
        trials = study.get_trials(deepcopy=False)
        n_finished = len(trials)
        print(f"  Number of finished trials (all states): {n_finished}")

        # Count trials by state
        state_counts = {state: 0 for state in optuna.trial.TrialState}
        for t in trials: state_counts[t.state] += 1
        print(f"  States: " + ", ".join([f"{state.name}({count})" for state, count in state_counts.items() if count > 0]))

        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            # Find best trial based on the chosen metric value
            best_trial = min(completed_trials, key=lambda t: t.value if t.value is not None else float('inf'))

            if best_trial.value is not None and not np.isinf(best_trial.value):
                 print(f"\nBest trial overall (among completed):")
                 print(f"  Trial Number: {best_trial.number}")
                 print(f"  Value (Min HPT '{optimization_metric}'): {best_trial.value:.6f}")
                 print(f"  HPT Validation Intervals Used: {metadata.get('hpt_validation_intervals', 'Not Found')}")
                 print("  Best Tuned Params: ")
                 for key, value in best_trial.params.items(): print(f"    {key}: {value}")
            else:
                 print("\nBest completed trial had Inf or None value.")
        else: print("\nNo trials completed successfully.")

        print(f"\nDetailed results logged to: {hpt_results_file}")
        if best_hparams_file.exists(): print(f"Best hyperparameters saved to: {best_hparams_file}")
        else: print(f"Best hyperparameters file not created (no successful trials improved objective or error saving).")
    except Exception as e: print(f"Error retrieving final study results: {e}")

    # Clean up large data
    del metadata
    gc.collect()
