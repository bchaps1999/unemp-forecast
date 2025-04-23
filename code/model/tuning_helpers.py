import optuna
from optuna.pruners import NopPruner  # disable pruning entirely
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
# Assuming forecasting_helpers.py contains calculate_hpt_forecast_metrics
from forecasting_helpers import calculate_hpt_forecast_metrics, evaluate_aggregate_unemployment_error
# train_and_evaluate_internal is passed as an argument

# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, train_evaluate_func,
              national_rates_file: Path, # Path to truth data
              metadata, device):
    """
    Optuna objective function to minimize selected metrics based on config settings.
    Always calculates RMSE, Std Dev, and Trend Correlation, then combines them
    according to the HPT_OBJECTIVE_TYPE setting.
    """
    # always write results into the canonical HPT folder
    study_dir = Path(config.TRAIN_OUTPUT_SUBDIR) / config.HPT_STUDY_NAME
    hpt_results_file = study_dir / config.HPT_RESULTS_CSV
    best_hparams_file = study_dir / config.BEST_HPARAMS_PKL
    
    # Get objective configuration
    objective_type = config.HPT_OBJECTIVE_TYPE.lower()
    primary_metric = config.HPT_PRIMARY_METRIC.lower()
    secondary_metric = config.HPT_SECONDARY_METRIC.lower()
    primary_weight = config.HPT_PRIMARY_WEIGHT
    secondary_weight = 1.0 - primary_weight
    
    # Initialize objective values and description with defaults
    objective_value = float('inf')
    objective_desc = f"Default ({objective_type})"  # Default description
    forecast_rmse = float('inf')
    forecast_std_dev = float('inf')
    forecast_slope_error = float('inf') # This will hold the Root Mean Squared Slope Error

    processed_data_dir = Path(config.PREPROCESS_OUTPUT_DIR) # Get path from config

    # --- Define Hyperparameter Search Space ---
    # choose embed_dim first
    embed_dim = trial.suggest_categorical("embed_dim", config.HPT_EMBED_DIM_OPTIONS)
    # restrict heads to divisors of embed_dim
    valid_heads = [h for h in config.HPT_NUM_HEADS_OPTIONS if embed_dim % h == 0]
    num_heads = trial.suggest_categorical("num_heads", valid_heads)

    hparams = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'ff_dim': trial.suggest_int("ff_dim", config.HPT_FF_DIM_MIN, config.HPT_FF_DIM_MAX, step=config.HPT_FF_DIM_STEP),
        'num_transformer_blocks': trial.suggest_int("num_transformer_blocks", config.HPT_NUM_BLOCKS_MIN, config.HPT_NUM_BLOCKS_MAX),
        'mlp_units': [trial.suggest_int(f"mlp_units_{i}", config.HPT_MLP_UNITS_MIN, config.HPT_MLP_UNITS_MAX, step=config.HPT_MLP_UNITS_STEP) for i in range(trial.suggest_int("mlp_layers", 1, 2))],
        'learning_rate': trial.suggest_float("learning_rate", config.HPT_LR_MIN, config.HPT_LR_MAX, log=True),
        'batch_size': trial.suggest_categorical("batch_size", config.HPT_BATCH_SIZE_OPTIONS),
        'focal_loss_gamma': trial.suggest_float("focal_loss_gamma", config.HPT_FOCAL_LOSS_GAMMA_MIN, config.HPT_FOCAL_LOSS_GAMMA_MAX), # Add gamma tuning
        'transition_weight_factor': trial.suggest_float("transition_weight_factor", config.HPT_TRANSITION_WEIGHT_FACTOR_MIN, config.HPT_TRANSITION_WEIGHT_FACTOR_MAX), # Add transition factor tuning
        # Tune weight decay too
        'weight_decay': trial.suggest_float("weight_decay", config.HPT_WEIGHT_DECAY_MIN, config.HPT_WEIGHT_DECAY_MAX, log=True),

        # Fixed Parameters (passed from config)
        'dropout': config.DROPOUT,
        'mlp_dropout': config.MLP_DROPOUT,
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
        'learning_rate', 'batch_size',
        'focal_loss_gamma', 'transition_weight_factor', 'weight_decay'
    ]

    # --- New: log tuned vs fixed ---
    print("\n>>> Tuned hyperparameters for this trial:")
    for k in tunable_keys:
        print(f"    - {k}: {hparams[k]}")
    print(">>> Fixed hyperparameters:")
    for k, v in hparams.items():
        if k not in tunable_keys:
            print(f"    - {k}: {v}")

    # --- Run Training and Standard Evaluation ---
    # This function trains the model using HPT data splits and returns results.
    # It handles its own data loading, sequence gen, training loop etc.
    train_results = train_evaluate_func(hparams, trial=trial) # Pass trial for potential pruning

    # Extract results needed for HPT objective calculation and logging
    model = train_results.get('model') # Model object is returned if training completed
    run_status = train_results.get('status', 'Unknown')
    best_val_loss = train_results.get('best_val_loss', float('inf')) # Loss on HPT validation set
    std_val_agg_error = train_results.get('std_val_agg_error', float('inf')) # Agg error on HPT validation set

    objective_value = float('inf') # Initialize final objective value to be minimized
    primary_objective_value = float('inf') # Value of the primary metric (RMSE or Std Dev)
    forecast_rmse = float('inf')
    forecast_std_dev = float('inf')
    forecast_slope_error = float('inf')
    hpt_val_data_loaded = None # Initialize

    # --- Calculate HPT Objective Metrics (Forecast RMSE, Std Dev, RMS Slope Error) --- # Updated comment
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
            # This function now returns rmse, std_dev, and the ROOT MEAN SQUARED slope_error
            forecast_metrics = calculate_hpt_forecast_metrics(
                model=model,
                hpt_val_baked_data=hpt_interval_data_loaded, # Pass the loaded interval data
                national_rates_file=national_rates_file, # Pass the path to actual rates
                metadata=metadata, # Pass full metadata
                params=hparams, # Pass the hyperparameters used for this trial
                device=device,
                forecast_horizon=hparams['hpt_forecast_horizon'],
                hpt_mc_samples=config.HPT_MC_SAMPLES, # Explicitly pass the value from config
                forecast_batch_size=config.FORECAST_BATCH_SIZE # Pass forecast batch size
            )
            
            # Extract all metrics
            forecast_rmse = forecast_metrics.get('rmse', float('inf'))
            forecast_std_dev = forecast_metrics.get('std_dev', float('inf'))
            forecast_slope_error = forecast_metrics.get('slope_error', float('inf')) # This is now RMS Slope Error
            
            # diagnostic logging
            print(f"Trial {trial.number} Forecast Metrics -> "
                  f"RMSE: {forecast_rmse:.6f}, "
                  f"Std Dev: {forecast_std_dev:.6f}, "
                  f"RMS Slope Error: {forecast_slope_error:.6f}") # Updated print label

            # Calculate the objective value based on the chosen objective type
            if objective_type == 'rmse':
                objective_value = forecast_rmse
                objective_desc = "RMSE"
            elif objective_type == 'std_dev':
                objective_value = forecast_std_dev
                objective_desc = "Std Dev"
            elif objective_type == 'slope_error':
                objective_value = forecast_slope_error # Use the RMS Slope Error directly
                objective_desc = "RMS Slope Error" # Updated description
            elif objective_type == 'combined':
                # Get raw metric values (slope_error is now RMS Slope Error)
                primary_value = forecast_metrics.get(primary_metric, float('inf'))
                secondary_value = forecast_metrics.get(secondary_metric, float('inf'))
                
                # At this point both primary_value and secondary_value are positive values we want to minimize
                
                # Handle potential inf values before combining
                if np.isinf(primary_value) and np.isinf(secondary_value):
                    objective_value = float('inf')
                elif np.isinf(primary_value):
                    objective_value = secondary_value
                elif np.isinf(secondary_value):
                    objective_value = primary_value
                else:
                    # RMSE, std_dev, and RMS slope_error are all positive values
                    # that we want to minimize, so we can safely combine them with weights
                    objective_value = primary_weight * primary_value + secondary_weight * secondary_value
                
                objective_desc = f"Combined ({primary_weight:.2f}*{primary_metric} + {secondary_weight:.2f}*{secondary_metric})"
            else:
                # Default to RMSE if invalid selection
                print(f"Warning: Invalid HPT_OBJECTIVE_TYPE '{objective_type}'. Using 'rmse' instead.")
                objective_value = forecast_rmse
                objective_desc = "RMSE (default)"

            print(f"Trial {trial.number} Metrics -> "
                  f"RMSE: {forecast_rmse:.6f}, "
                  f"Std Dev: {forecast_std_dev:.6f}, "
                  f"RMS Slope Error: {forecast_slope_error:.6f}") # Updated print label
            if objective_type == 'slope_error':
                print(f"  -> Objective ({objective_desc}): {objective_value:.6f}")
            elif objective_type == 'combined':
                print(f"  -> Objective ({objective_desc}): {objective_value:.6f}")
            else:
                print(f"  -> Objective ({objective_desc}): {objective_value:.6f}")

        except Exception as e:
            print(f"ERROR calculating HPT forecast metrics for trial {trial.number}: {e}")
            traceback.print_exc()
            objective_value = float('inf') # Penalize trial if calculation fails
            forecast_rmse = float('inf')
            forecast_std_dev = float('inf')
            forecast_slope_error = float('inf') # Also set RMS slope error to worst value
            objective_desc = f"Failed HPT Metric Calc: {objective_type}"  # Set description for error
            run_status = "Failed HPT Metric Calc" # Update status
    elif run_status == "Pruned":
        print(f"Trial {trial.number} was pruned during training. Objective set to Inf.")
        objective_value = float('inf')
        objective_desc = f"Pruned: {objective_type}"  # Set description for pruned trials
    else: # Failed, Interrupted, etc.
        print(f"Trial {trial.number} did not complete successfully (Status: {run_status}). Objective set to Inf.")
        objective_value = float('inf')
        objective_desc = f"Failed: {objective_type}"  # Set description for failed trials

    # Handle NaN/Inf objective values before returning to Optuna
    if np.isnan(objective_value) or np.isinf(objective_value):
        print(f"Warning: Trial {trial.number} resulted in NaN or Inf final objective value ({objective_value}). Returning infinity.")
        objective_value = float('inf') # Ensure Optuna receives a valid float

    # record everything in the trial for easy UI inspection
    trial.set_user_attr("forecast_rmse",        forecast_rmse)
    trial.set_user_attr("forecast_std_dev",     forecast_std_dev)
    trial.set_user_attr("forecast_slope_error", forecast_slope_error) # Keep key name, value is RMS Slope Error
    trial.set_user_attr("hpt_val_loss",         best_val_loss)
    trial.set_user_attr("hpt_val_agg_error",    std_val_agg_error)

    # --- Log results to CSV ---
    params_flat = hparams.copy()
    params_flat['mlp_units'] = str(params_flat['mlp_units']) # Convert list to string for CSV
    # Update fieldnames to include all metrics and the final objective
    # Keep 'hpt_forecast_slope_error' as the column name for consistency, but it holds RMS Slope Error
    fieldnames = ['trial_number', 'status', 'final_objective', 'objective_desc',
                  'hpt_forecast_rmse', 'hpt_forecast_std_dev', 'hpt_forecast_slope_error',
                  'hpt_val_loss', 'hpt_val_agg_error'] + tunable_keys
    
    log_entry = {
        'trial_number': trial.number,
        'status': run_status,
        'final_objective': f"{objective_value:.6f}" if not np.isinf(objective_value) else 'inf',
        'objective_desc': objective_desc,
        'hpt_forecast_rmse': f"{forecast_rmse:.6f}" if not np.isinf(forecast_rmse) else 'inf',
        'hpt_forecast_std_dev': f"{forecast_std_dev:.6f}" if not np.isinf(forecast_std_dev) else 'inf',
        'hpt_forecast_slope_error': f"{forecast_slope_error:.6f}" if not np.isinf(forecast_slope_error) else 'inf', # Log RMS Slope Error
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
        # print(f"Logged trial {trial.number} results to {hpt_results_file") # Less verbose logging

    except Exception as e: print(f"ERROR logging trial {trial.number} results to CSV: {e}")

    # --- Update best hyperparameters file ---
    current_best_value = float('inf')
    try: # Handle case where study has no completed trials yet
        if trial.study.best_trial and trial.study.best_trial.state == optuna.trial.TrialState.COMPLETE:
             current_best_value = trial.study.best_value
    except ValueError: pass # Keep current_best_value as inf

    # Check if current trial completed successfully and improved the objective
    if run_status == "Completed" and not np.isinf(objective_value) and objective_value < current_best_value:
         print(f"Trial {trial.number} improved {objective_desc} to {objective_value:.6f}. Updating best hyperparameters.")
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
    # ensure we always use the configured study name folder
    study_name = config.HPT_STUDY_NAME
    study_dir = base_output_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    storage_path = study_dir / f"{study_name}.db"
    storage_name = f"sqlite:///{storage_path}"
    hpt_results_file = study_dir / config.HPT_RESULTS_CSV
    best_hparams_file = study_dir / config.BEST_HPARAMS_PKL

    # --- Validate HPT Objective Type ---
    objective_type = config.HPT_OBJECTIVE_TYPE.lower()
    valid_types = ['rmse', 'std_dev', 'slope_error', 'combined']
    
    if objective_type not in valid_types:
        print(f"ERROR: Invalid HPT_OBJECTIVE_TYPE: '{objective_type}'. Valid options are: {valid_types}")
        sys.exit(1)
        
    # Validate combined objective parameters if needed
    if objective_type == 'combined':
        primary_metric = config.HPT_PRIMARY_METRIC.lower()
        secondary_metric = config.HPT_SECONDARY_METRIC.lower()
        valid_metrics = ['rmse', 'std_dev', 'slope_error']
        
        if primary_metric not in valid_metrics:
            print(f"ERROR: Invalid HPT_PRIMARY_METRIC: '{primary_metric}'. Valid options are: {valid_metrics}")
            sys.exit(1)
            
        if secondary_metric not in valid_metrics:
            print(f"ERROR: Invalid HPT_SECONDARY_METRIC: '{secondary_metric}'. Valid options are: {valid_metrics}")
            sys.exit(1)
            
        if primary_metric == secondary_metric:
            print(f"WARNING: HPT_PRIMARY_METRIC and HPT_SECONDARY_METRIC are the same: '{primary_metric}'.")
            print(f"This is allowed but may not be what you intended.")
            
        primary_weight = config.HPT_PRIMARY_WEIGHT
        if not 0.0 <= primary_weight <= 1.0:
            print(f"ERROR: Invalid HPT_PRIMARY_WEIGHT: {primary_weight}. Must be between 0.0 and 1.0.")
            sys.exit(1)

    # --- Print HPT Configuration ---
    print(f"Optuna study name: {study_name}")
    print(f"Optuna storage: {storage_name}")
    print(f"HPT results log: {hpt_results_file}")
    print(f"Best hyperparameters file: {best_hparams_file}")
    
    if objective_type == 'combined':
        print(f"HPT Objective: Minimize Combined")
        print(f"  - Primary: {config.HPT_PRIMARY_METRIC} (weight: {config.HPT_PRIMARY_WEIGHT})")
        print(f"  - Secondary: {config.HPT_SECONDARY_METRIC} (weight: {1.0 - config.HPT_PRIMARY_WEIGHT})")
    elif objective_type == 'slope_error':
        print(f"HPT Objective: Minimize Root Mean Squared Slope Error") # Updated description
    else:
        print(f"HPT Objective: Minimize {objective_type.upper()}")

    print(f"HPT Forecast Horizon: {config.HPT_FORECAST_HORIZON} months")
    print(f"HPT MC Samples per Run: {config.HPT_MC_SAMPLES}")

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

    # --- Create Optuna Study (always minimize) + median pruner ---
    direction = 'minimize'
    pruner = NopPruner()  # no pruning
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction=direction,
        pruner=pruner,
        load_if_exists=True
    )

    try:
        # Check if the study already exists
        if storage_path.exists():
            print(f"Found existing study database at {storage_path}")
            print("Loading existing study to continue from where it left off...")
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            existing_trials = len(study.trials)
            print(f"Loaded study with {existing_trials} existing trials")
            print(f"Best value so far: {study.best_value}")
            remaining_trials = args.n_trials - existing_trials
            if remaining_trials <= 0:
                print(f"Study already has {existing_trials} trials, which is >= the requested {args.n_trials} trials.")
                print("If you want to run more trials, increase --n_trials.")
                print("Loading best parameters and exiting tuning.")
                
                # Still save the best hyperparameters if available
                if study.best_trial:
                    best_params = study.best_trial.params
                    try:
                        with open(best_hparams_file, 'wb') as f:
                            pickle.dump(best_params, f)
                        print(f"Best hyperparameters saved to {best_hparams_file}")
                    except Exception as e:
                        print(f"Error saving best hyperparameters: {e}")
                return
        else:
            print(f"Creating new study database at {storage_path}")
            study = optuna.create_study(
                study_name=study_name, 
                storage=storage_name,
                direction="minimize",
                load_if_exists=True  # This ensures we load if it exists despite our check
            )
            existing_trials = 0
            remaining_trials = args.n_trials
            
    except Exception as e:
        print(f"Error loading/creating study: {e}")
        print("Creating study in memory (results won't persist if interrupted)")
        study = optuna.create_study(direction="minimize")
        existing_trials = 0
        remaining_trials = args.n_trials

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
            # Find best trial based on the final objective value
            best_trial = min(completed_trials, key=lambda t: t.value if t.value is not None else float('inf'))

            if best_trial.value is not None and not np.isinf(best_trial.value):
                # Construct objective description based on settings
                if objective_type == 'combined':
                    objective_desc = f"Combined ({config.HPT_PRIMARY_WEIGHT:.2f}*{config.HPT_PRIMARY_METRIC} + " \
                                    f"{1.0 - config.HPT_PRIMARY_WEIGHT:.2f}*{config.HPT_SECONDARY_METRIC})"
                elif objective_type == 'slope_error':
                     objective_desc = "Min HPT RMS Slope Error" # Updated description
                else:
                    objective_desc = f"Min HPT '{objective_type}'"
                    
                print(f"\nBest trial overall (among completed):")
                print(f"  Trial Number: {best_trial.number}")
                print(f"  Value ({objective_desc}): {best_trial.value:.6f}")
                # Optionally retrieve and print individual metrics for the best trial from logs or user_attrs if stored
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
