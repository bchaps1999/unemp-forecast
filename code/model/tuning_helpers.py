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
from utils import get_device
# Assuming forecasting_helpers.py contains calculate_hpt_forecast_metrics which returns {'rmse': float, 'variance': float}
from forecasting_helpers import calculate_hpt_forecast_metrics, evaluate_aggregate_unemployment_error # Keep evaluate_aggregate_unemployment_error import if used elsewhere indirectly or for logging
# Assuming main_train_tune.py will contain train_and_evaluate_internal (or similar)
# We need to import it carefully to avoid circular dependencies if possible,
# or pass it as an argument to run_hyperparameter_tuning.
# For now, assume train_and_evaluate_internal is available in the scope where objective is called.

# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, train_evaluate_func,
              processed_data_dir: Path, # ADDED - Need path to load data
              national_rates_file: Path, # Path to truth data
              metadata, device):
    """
    Optuna objective function to minimize the selected forecast metric (RMSE or Variance).

    Args:
        trial (optuna.Trial): Optuna trial object.
        train_evaluate_func (callable): The function that trains the model and returns results.
        processed_data_dir (Path): Path to the directory containing processed data.
        national_rates_file (Path): Path to the CSV with actual national rates.
        metadata (dict): Loaded metadata (must contain 'hpt_validation_intervals').
        device (torch.device): The device to use.

    Returns:
        float: The calculated value of the metric specified by config.HPT_OBJECTIVE_METRIC.
    """
    study_dir = Path(config.TRAIN_OUTPUT_SUBDIR) / trial.study.study_name
    hpt_results_file = study_dir / config.HPT_RESULTS_CSV
    best_hparams_file = study_dir / config.BEST_HPARAMS_PKL
    objective_metric_name = config.HPT_OBJECTIVE_METRIC.lower() # 'rmse' or 'variance'

    # --- Define Hyperparameter Search Space ---
    hparams = {
        # --- Tunable Parameters ---
        'embed_dim': trial.suggest_categorical("embed_dim", config.HPT_EMBED_DIM_OPTIONS),
        'num_heads': trial.suggest_categorical("num_heads", config.HPT_NUM_HEADS_OPTIONS),
        'ff_dim': trial.suggest_int("ff_dim", config.HPT_FF_DIM_MIN, config.HPT_FF_DIM_MAX, step=config.HPT_FF_DIM_STEP),
        'num_transformer_blocks': trial.suggest_int("num_transformer_blocks", config.HPT_NUM_BLOCKS_MIN, config.HPT_NUM_BLOCKS_MAX),
        'mlp_units': [trial.suggest_int(f"mlp_units_{i}", config.HPT_MLP_UNITS_MIN, config.HPT_MLP_UNITS_MAX, step=config.HPT_MLP_UNITS_STEP) for i in range(trial.suggest_int("mlp_layers", 1, 2))], # Example: variable layers
        'dropout': trial.suggest_float("dropout", config.HPT_DROPOUT_MIN, config.HPT_DROPOUT_MAX),
        'mlp_dropout': trial.suggest_float("mlp_dropout", config.HPT_MLP_DROPOUT_MIN, config.HPT_MLP_DROPOUT_MAX),
        'learning_rate': trial.suggest_float("learning_rate", config.HPT_LR_MIN, config.HPT_LR_MAX, log=True),
        'batch_size': trial.suggest_categorical("batch_size", config.HPT_BATCH_SIZE_OPTIONS),
        # Add loss weight factor
        'loss_weight_factor': trial.suggest_float("loss_weight_factor", config.HPT_LOSS_WEIGHT_FACTOR_MIN, config.HPT_LOSS_WEIGHT_FACTOR_MAX),
        # --- Fixed Parameters (during HPT, but need to be passed) ---
        'sequence_length': config.SEQUENCE_LENGTH,
        'epochs': config.HPT_EPOCHS, # Use HPT epochs for tuning runs
        'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        'lr_scheduler_factor': config.LR_SCHEDULER_FACTOR,
        'lr_scheduler_patience': config.LR_SCHEDULER_PATIENCE,
        'max_grad_norm': config.MAX_GRAD_NORM,
        'pad_value': config.PAD_VALUE,
        'parallel_workers': config.PARALLEL_WORKERS,
        'refresh_sequences': config.REFRESH_SEQUENCES, # Usually False during HPT
        # 'train_start_date': config.TRAIN_START_DATE, # REMOVED
        # 'train_end_date': config.TRAIN_END_DATE, # REMOVED
        'random_seed': config.RANDOM_SEED,
        # Add HPT forecast horizon
        'hpt_forecast_horizon': config.HPT_FORECAST_HORIZON,
        # 'hpt_num_forecast_runs' is no longer needed here, derived from metadata intervals
    }
    # Store the tunable keys separately for logging
    tunable_keys = [
        'embed_dim', 'num_heads', 'ff_dim', 'num_transformer_blocks', 'mlp_units',
        'dropout', 'mlp_dropout', 'learning_rate', 'batch_size',
        'loss_weight_factor' # Add to tunable keys
    ]

    # --- Constraint Check: Embed dim vs Num heads ---
    if hparams['embed_dim'] % hparams['num_heads'] != 0:
        print(f"Pruning trial {trial.number}: embed_dim {hparams['embed_dim']} not divisible by num_heads {hparams['num_heads']}")
        raise optuna.TrialPruned(f"embed_dim ({hparams['embed_dim']}) must be divisible by num_heads ({hparams['num_heads']}).")

    # --- Run Training and Standard Evaluation ---
    # This function trains the model based on hparams and returns results, including the model object
    # It handles its own data loading, sequence gen, training loop etc.
    # We pass the trial object for potential pruning based on validation loss during training.
    train_results = train_evaluate_func(hparams, trial=trial)

    # Extract the trained model and other relevant results
    model = train_results.get('model')
    run_status = train_results.get('status', 'Unknown')
    best_val_loss = train_results.get('best_val_loss', float('inf'))
    # Add other metrics if needed for logging
    std_val_agg_error = train_results.get('std_val_agg_error', float('inf'))

    objective_value = float('inf') # Initialize objective value
    forecast_rmse = float('inf')
    forecast_variance = float('inf')

    # --- Calculate HPT Objective Metrics (Forecast RMSE and Variance) ---
    if run_status == "Completed" and model is not None:
        try:
            # Get intervals from metadata
            if 'hpt_validation_intervals' not in metadata:
                 raise ValueError("Metadata is missing 'hpt_validation_intervals'. Cannot calculate HPT objective.")

            # Call the function assumed to return both metrics
            # Assumes calculate_hpt_forecast_metrics returns a dict: {'rmse': value, 'variance': value}
            forecast_metrics = calculate_hpt_forecast_metrics(
                model=model,
                processed_data_dir=processed_data_dir, # ADDED
                national_rates_file=national_rates_file, # Pass the path
                metadata=metadata, # Pass full metadata (contains intervals)
                params=hparams, # Pass the hyperparameters used for this trial
                device=device,
                forecast_horizon=hparams['hpt_forecast_horizon'] # e.g., 12
            )
            forecast_rmse = forecast_metrics.get('rmse', float('inf'))
            forecast_variance = forecast_metrics.get('variance', float('inf'))

            # Select the metric to optimize based on config
            if objective_metric_name == 'rmse':
                objective_value = forecast_rmse
            elif objective_metric_name == 'variance':
                objective_value = forecast_variance
            else:
                raise ValueError(f"Invalid HPT_OBJECTIVE_METRIC in config: '{config.HPT_OBJECTIVE_METRIC}'. Choose 'rmse' or 'variance'.")

            print(f"Trial {trial.number} objective ({objective_metric_name}): {objective_value:.6f} | RMSE: {forecast_rmse:.6f}, Variance: {forecast_variance:.6f}")

        except Exception as e:
            print(f"ERROR calculating HPT forecast metrics for trial {trial.number}: {e}")
            traceback.print_exc()
            objective_value = float('inf') # Penalize trial if calculation fails
            forecast_rmse = float('inf')
            forecast_variance = float('inf')
            run_status = "Failed HPT Metric Calc"
    elif run_status == "Pruned":
        print(f"Trial {trial.number} was pruned during training. Objective set to Inf.")
        objective_value = float('inf') # Optuna handles pruned state, but good to be explicit
    else:
        print(f"Trial {trial.number} did not complete successfully (Status: {run_status}). Objective set to Inf.")
        objective_value = float('inf')


    # Handle NaN/Inf objective values before returning to Optuna
    if np.isnan(objective_value) or np.isinf(objective_value):
        print(f"Warning: Trial {trial.number} resulted in NaN or Inf objective value ({objective_value}). Returning infinity.")
        objective_value = float('inf') # Ensure Optuna receives a valid float

    # --- Log results to CSV ---
    params_flat = hparams.copy()
    params_flat['mlp_units'] = str(params_flat['mlp_units'])
    # Update fieldnames to include both metrics
    fieldnames = ['trial_number', 'status', f'hpt_forecast_{objective_metric_name}', 'hpt_forecast_rmse', 'hpt_forecast_variance', 'best_val_loss', 'std_val_agg_error'] + tunable_keys
    log_entry = {
        'trial_number': trial.number,
        'status': run_status,
        f'hpt_forecast_{objective_metric_name}': f"{objective_value:.6f}" if not np.isinf(objective_value) else 'inf', # Log the primary objective value
        'hpt_forecast_rmse': f"{forecast_rmse:.6f}" if not np.isinf(forecast_rmse) else 'inf', # Log RMSE
        'hpt_forecast_variance': f"{forecast_variance:.6f}" if not np.isinf(forecast_variance) else 'inf', # Log Variance
        'best_val_loss': f"{best_val_loss:.6f}" if not np.isinf(best_val_loss) and not np.isnan(best_val_loss) else 'inf',
        'std_val_agg_error': f"{std_val_agg_error:.6f}" if not np.isinf(std_val_agg_error) and not np.isnan(std_val_agg_error) else 'inf',
        **{k: params_flat[k] for k in tunable_keys}
    }
    try:
        study_dir.mkdir(parents=True, exist_ok=True)
        write_header = not hpt_results_file.exists() or hpt_results_file.stat().st_size == 0
        with open(hpt_results_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if write_header: writer.writeheader()
            writer.writerow(log_entry)
        print(f"Logged trial {trial.number} results to {hpt_results_file}")

    except Exception as e: print(f"ERROR logging trial {trial.number} results to CSV: {e}")

    # --- Update best hyperparameters file ---
    current_best_value = float('inf')
    try: # Handle case where study has no completed trials yet or best trial failed/pruned
        if trial.study.best_trial and trial.study.best_trial.state == optuna.trial.TrialState.COMPLETE:
             current_best_value = trial.study.best_value
    except ValueError: pass # Keep current_best_value as inf if no completed trials

    # Check if current trial completed successfully (including HPT metric calc) and improved the objective (lower is better)
    if run_status == "Completed" and not np.isinf(objective_value) and objective_value < current_best_value:
         print(f"Trial {trial.number} improved HPT objective ({objective_metric_name}) to {objective_value:.6f}. Updating best hyperparameters.")
         try:
             # Save the parameters suggested by Optuna for this trial
             best_params_to_save = trial.params # These are only the *tuned* parameters
             with open(best_hparams_file, 'wb') as f: pickle.dump(best_params_to_save, f)
             print(f"Best hyperparameters updated in {best_hparams_file}")
         except Exception as e: print(f"ERROR saving best hyperparameters for trial {trial.number}: {e}")

    # --- Clean up model from memory ---
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    elif device.type == 'mps': torch.mps.empty_cache()
    gc.collect()

    # Return the chosen objective value for Optuna to minimize
    return objective_value


# --- HPT Runner Function ---

def run_hyperparameter_tuning(args, base_output_dir, train_evaluate_func):
    """
    Sets up and runs the Optuna hyperparameter tuning study.

    Args:
        args (argparse.Namespace): Command-line arguments.
        base_output_dir (Path): Base directory for outputs.
        train_evaluate_func (callable): The function to call for training/evaluating a single trial.
    """
    print(f"\n--- Starting Hyperparameter Tuning ({args.n_trials} trials) ---")
    study_name = args.study_name
    study_dir = base_output_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    storage_path = study_dir / f"{study_name}.db" # Use .db extension for SQLite
    storage_name = f"sqlite:///{storage_path}"
    hpt_results_file = study_dir / config.HPT_RESULTS_CSV
    best_hparams_file = study_dir / config.BEST_HPARAMS_PKL

    # --- Validate HPT Metric Choice ---
    optimization_metric = config.HPT_OBJECTIVE_METRIC.lower()
    if optimization_metric not in ['rmse', 'variance']:
        print(f"ERROR: Invalid HPT_OBJECTIVE_METRIC in config: '{config.HPT_OBJECTIVE_METRIC}'. Choose 'rmse' or 'variance'.")
        sys.exit(1)

    print(f"Optuna study name: {study_name}")
    print(f"Optuna storage: {storage_name}")
    print(f"HPT results log: {hpt_results_file}")
    print(f"Best hyperparameters file: {best_hparams_file}")
    print(f"Fixed Sequence Length: {config.SEQUENCE_LENGTH}") # Indicate fixed length
    print(f"HPT Objective: Minimize '{optimization_metric}' on HPT Validation Set") # Use configured metric
    print(f"HPT Forecast Horizon: {config.HPT_FORECAST_HORIZON} months")
    # Update description of the factor's effect
    print(f"HPT Loss Weight Factor Range: [{config.HPT_LOSS_WEIGHT_FACTOR_MIN}, {config.HPT_LOSS_WEIGHT_FACTOR_MAX}] (0=Unweighted, 1=Inverse Freq)")

    # --- Load Data needed for Objective Calculation ---
    # REMOVED loading of hpt_val_baked_data here. Only need paths and metadata.
    print("\nLoading paths and metadata needed for HPT objective calculation...")
    processed_data_dir = Path(config.PREPROCESS_OUTPUT_DIR)
    national_rates_file = config.NATIONAL_RATES_FILE
    metadata_file = processed_data_dir / config.METADATA_FILENAME
    metadata = None
    try:
        if not national_rates_file.exists():
            raise FileNotFoundError(f"National rates file not found: {national_rates_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        if 'hpt_validation_intervals' not in metadata:
            raise ValueError("Metadata loaded, but 'hpt_validation_intervals' key is missing.")
        print(f"Loaded metadata.") # Simplified print
        print(f"Using HPT intervals from metadata: {metadata['hpt_validation_intervals']}")
        print(f"Using national rates file: {national_rates_file}")

    except Exception as e:
        print(f"ERROR: Failed to load data/metadata needed for HPT objective: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Get Device ---
    device = get_device() # Get device once

    # --- Create Study ---
    pruner = optuna.pruners.MedianPruner(
            n_startup_trials=config.HPT_PRUNER_STARTUP,
            n_warmup_steps=config.HPT_PRUNER_WARMUP, # Number of epochs before pruning can happen
            interval_steps=1 # Prune after each epoch (post warmup)
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
            print(f"Existing best value in study: {study.best_value}")
        else:
            print("No completed trials with valid value yet in study.")
    except ValueError: print("No trials completed yet in study.")

    try:
        # Pass necessary arguments to the objective function using a lambda
        study.optimize(
            lambda trial: objective(trial, train_evaluate_func, processed_data_dir, national_rates_file, metadata, device), # Pass dir instead of data
            n_trials=args.n_trials,
            timeout=config.HPT_TIMEOUT_SECONDS,
            gc_after_trial=True, # Garbage collect after each trial
        )
    except KeyboardInterrupt:
        print("\nOptuna optimization interrupted by user (Ctrl+C).")
        print("Results for completed trials (and the current best parameters) should be saved.")
    except Exception as e:
        print(f"\nAn critical error occurred during Optuna optimization: {e}")
        traceback.print_exc()

    # --- Print HPT Summary ---
    print("\n--- HPT Finished ---")
    try:
        print(f"Study statistics: ")
        trials = study.get_trials(deepcopy=False) # Avoid copying large trial objects
        n_finished = len(trials)
        print(f"  Number of finished trials (all states): {n_finished}")

        states = [t.state for t in trials]
        complete_count = states.count(optuna.trial.TrialState.COMPLETE)
        pruned_count = states.count(optuna.trial.TrialState.PRUNED)
        fail_count = states.count(optuna.trial.TrialState.FAIL)
        running_count = states.count(optuna.trial.TrialState.RUNNING) # Should be 0 if optimize finished
        waiting_count = states.count(optuna.trial.TrialState.WAITING) # Should be 0

        print(f"  States: Complete({complete_count}), Pruned({pruned_count}), Fail({fail_count}), Running({running_count}), Waiting({waiting_count})")

        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            # Find best trial based on the chosen metric
            best_trial = min(completed_trials, key=lambda t: t.value if t.value is not None else float('inf'))

            if best_trial.value is not None:
                 print(f"\nBest trial overall (among completed):")
                 print(f"  Trial Number: {best_trial.number}")
                 print(f"  Value (Min HPT '{optimization_metric}'): {best_trial.value:.6f}") # Updated metric name
                 # Optionally log the other metric for the best trial from the CSV or user_attrs if stored
                 # Example: print(f"  Corresponding RMSE/Variance: ...")
                 print(f"  Fixed Sequence Length: {config.SEQUENCE_LENGTH}") # Remind user it was fixed
                 print(f"  HPT Validation Intervals Used: {metadata.get('hpt_validation_intervals', 'Not Found')}") # Show intervals used
                 print("  Best Tuned Params: ")
                 for key, value in best_trial.params.items(): print(f"    {key}: {value}")
            else:
                 print("\nBest completed trial had None value.")

        else: print("\nNo trials completed successfully.")

        print(f"\nDetailed results logged to: {hpt_results_file}")
        if best_hparams_file.exists(): print(f"Best hyperparameters saved to: {best_hparams_file}")
        else: print(f"Best hyperparameters file not created (no successful trials or error saving).")
    except Exception as e: print(f"Error retrieving final study results: {e}")

    # Clean up large data
    del metadata
    gc.collect()
