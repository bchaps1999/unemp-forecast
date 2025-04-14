import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
import random
import argparse
from pathlib import Path
import sys
import signal
import optuna
import traceback
import ast
import gc
from sklearn.utils.class_weight import compute_class_weight # Import for weight calculation

# --- Project Imports ---
import config # General configuration
from torch.utils.data import DataLoader
from utils import ( # Common utilities
    get_device, signal_handler, worker_init_fn, stop_training_flag,
    load_and_prepare_data, setup_sequence_generation, SequenceDataset
)
from models import TransformerForecastingModel # Model definition
from training_helpers import ( # Functions for the training loop
    create_dataloaders, build_model, run_training_loop, evaluate_epoch
)
from forecasting_helpers import ( # For final evaluation and HPT objective
    evaluate_aggregate_unemployment_error, calculate_hpt_forecast_metrics # Renamed from calculate_hpt_forecast_rmse
)
from tuning_helpers import run_hyperparameter_tuning # Optuna runner

# --- Main Train/Evaluate Function (Internal) ---

def train_and_evaluate_internal(hparams: dict, trial: optuna.Trial = None):
    """
    Internal function to load data, build model, train, and evaluate based on hyperparameters.
    Handles graceful exit and Optuna integration. Called by both standard training and HPT objective.

    Args:
        hparams (dict): Dictionary containing all hyperparameters for the run.
        trial (optuna.Trial, optional): Optuna trial object if called during HPT.

    Returns:
        dict: Dictionary containing evaluation results and the trained model object:
              {'model': model_object_or_None, 'std_val_agg_error': float, 'best_val_loss': float,
               'final_val_acc': float, 'best_epoch': int, 'status': str}
              Returns status='Failed' or 'Interrupted' or 'Pruned' on issues.
              'std_val_agg_error' is based on the standard validation set.
    """
    global stop_training_flag
    stop_training_flag = False # Reset flag at the start of each run/trial
    training_was_interrupted = False # Track if training loop specifically was interrupted

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    start_run_time = time.time()
    std_val_agg_error_final = float('inf') # Error on standard validation set
    best_val_loss_final = float('inf')
    final_val_acc_final = float('nan')
    best_epoch_final = -1
    run_status = "Started"
    model = None # Initialize model to None
    val_dataset_loaded = None # To hold validation dataset for final eval
    val_loader_params_stored = None # To hold loader params

    # --- Get Device ---
    # Ensure device is determined consistently
    # Note: get_device() might print multiple times if called repeatedly across runs.
    # Consider passing device as an argument if running many trials in one process.
    DEVICE = get_device()

    try:
        # --- Setup Run Info & Dirs ---
        print("=" * 60)
        run_type = f"Optuna Trial {trial.number}" if trial else "Standard Run"
        print(f"Starting Transformer Model Run: {run_type}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Hyperparameters:")
        for key, value in hparams.items(): print(f"  - {key}: {value}")
        print("-" * 60)

        processed_data_dir = Path(config.PREPROCESS_OUTPUT_DIR)
        run_id = f"trial_{trial.number}" if trial else "standard_run"
        base_output_dir = Path(config.TRAIN_OUTPUT_SUBDIR)
        model_output_dir = base_output_dir / run_id
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run output directory: {model_output_dir}")
        history_file = model_output_dir / "training_history.pkl"
        checkpoint_path = model_output_dir / "best_model_val_loss.pt" # Based on standard val loss
        params_path = model_output_dir / "model_params.pkl"

        # --- Set Random Seeds ---
        seed_value = hparams['random_seed']
        os.environ['PYTHONHASHSEED'] = str(seed_value); random.seed(seed_value)
        np.random.seed(seed_value); torch.manual_seed(seed_value)
        if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed_value)
        print(f"Random seeds set to: {seed_value}")

        # --- Step 1: Load Data (Baked) ---
        train_file = processed_data_dir / config.TRAIN_DATA_FILENAME
        val_file = processed_data_dir / config.VAL_DATA_FILENAME
        hpt_val_file = processed_data_dir / config.HPT_VAL_DATA_FILENAME # Path only
        metadata_file = processed_data_dir / config.METADATA_FILENAME
        # Load train/val/metadata. HPT val data is loaded separately in run_hyperparameter_tuning if needed.
        train_data_baked, val_data_baked, _, metadata, feature_names, n_features, n_classes = load_and_prepare_data(
            train_file, val_file, hpt_val_file, metadata_file, config.DATE_COL, config.GROUP_ID_COL,
            hparams.get('train_start_date'), hparams.get('train_end_date')
        )

        # --- Add derived params to hparams for saving ---
        hparams['n_features'] = n_features
        hparams['n_classes'] = n_classes
        weight_col = metadata.get('weight_column', config.WEIGHT_COL) # Get weight column name
        target_map_inverse = metadata.get('target_state_map_inverse', {}) # Get target map

        # --- Step 2: Generate Sequences (Train/Val only for training loop) ---
        # HPT val sequences are generated separately if needed by the objective function
        x_train_np, y_train_np, _, weight_train_np, \
        x_val_np, y_val_np, _, weight_val_np, \
        _, _, _, _, \
        parallel_workers = setup_sequence_generation(
            hparams, train_data_baked, val_data_baked, None, # Pass None for HPT data here
            processed_data_dir, config.GROUP_ID_COL, config.DATE_COL,
            feature_names, n_features, weight_col, config.SEQUENCE_CACHE_DIR_NAME
        )

        # --- Calculate Class Weights for Loss Criterion ---
        print("\nCalculating class weights for loss function...")
        class_weights = None # Initialize to None (unweighted)
        try:
            # Calculate standard inverse frequency weights using the training targets
            if y_train_np is not None and len(y_train_np) > 0:
                unique_classes = np.unique(y_train_np)
                # Ensure n_classes matches the number of unique classes found in training data
                if len(unique_classes) != n_classes:
                     print(f"Warning: Number of unique classes in y_train ({len(unique_classes)}) does not match metadata n_classes ({n_classes}). Using unique classes from y_train.")
                     # Adjust n_classes if necessary? Or raise error? Let's proceed with caution.
                     # It's safer to rely on the unique classes found in the actual training data.

                base_weights_np = compute_class_weight('balanced', classes=unique_classes, y=y_train_np)
                W_inv_freq = torch.tensor(base_weights_np, dtype=torch.float).to(DEVICE)
                print(f"Base class weights (inverse frequency): {W_inv_freq.cpu().numpy()}")

                # Define equal weights tensor (ones)
                W_equal = torch.ones_like(W_inv_freq).to(DEVICE)
                print(f"Equal weights baseline: {W_equal.cpu().numpy()}")

                # Get the tunable factor
                loss_weight_factor = hparams.get('loss_weight_factor', 1.0) # Default to 1.0 (full inverse freq)
                print(f"Interpolation factor (loss_weight_factor): {loss_weight_factor}")

                # Clamp factor to [0, 1] just in case
                factor = max(0.0, min(1.0, loss_weight_factor))

                if factor == 0.0:
                    class_weights = None # Use unweighted loss
                    print("Factor is 0. Using unweighted CrossEntropyLoss.")
                else:
                    # Interpolate: W_interp = (1 - factor) * W_equal + factor * W_inv_freq
                    class_weights = (1 - factor) * W_equal + factor * W_inv_freq
                    print(f"Final interpolated class weights for CrossEntropyLoss: {class_weights.cpu().numpy()}")

            else:
                print("Warning: y_train_np is empty or None. Cannot compute class weights. Using unweighted loss.")

        except Exception as e:
            print(f"Warning: Error calculating class weights: {e}. Using unweighted loss.")
            class_weights = None

        # --- Create Loss Criterion ---
        # Pass the calculated class_weights (can be None or a Tensor)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
        print(f"Loss criterion created {'with' if class_weights is not None else 'without'} class weights.")

        del train_data_baked, val_data_baked # Clear raw dataframes
        gc.collect()

        # --- Step 3: Create DataLoaders (Train/Val) ---
        train_loader, val_loader, val_loader_params, val_dataset_loaded = create_dataloaders( # Removed class_weights_tensor
            x_train_np, y_train_np, weight_train_np, x_val_np, y_val_np, weight_val_np,
            hparams, n_classes, DEVICE, parallel_workers
        )
        val_loader_params_stored = val_loader_params.copy() # Store for final eval
        del x_train_np, y_train_np, weight_train_np, x_val_np, y_val_np, weight_val_np # Clear numpy arrays
        gc.collect()

        # --- Step 4: Build Model ---
        model = build_model(hparams, n_features, n_classes, DEVICE)

        # --- Save Model Parameters Immediately ---
        try:
            with open(params_path, 'wb') as f: pickle.dump(hparams, f)
            print(f"Model parameters saved to: {params_path}")
        except Exception as e: print(f"Warning: Could not save model parameters: {e}")

        # --- Step 5: Setup Training Components ---
        # Criterion is now created above with weights
        optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=hparams['lr_scheduler_factor'],
            patience=hparams['lr_scheduler_patience'], verbose=True,
            threshold=0.0001, threshold_mode='rel'
        )

        # --- Step 6: Run Training Loop ---
        history, best_val_loss, training_interrupted_flag, last_epoch, epochs_no_improve = run_training_loop(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            hparams, DEVICE, checkpoint_path, trial # Pass trial for pruning
        )
        best_val_loss_final = best_val_loss # Store best loss from training (on standard val set)
        training_was_interrupted = training_interrupted_flag # Store if training loop was interrupted

        # --- Save Training History ---
        try:
            with open(history_file, 'wb') as f: pickle.dump(history, f)
            print(f"Training history saved to: {history_file}")
        except Exception as e: print(f"Warning: Could not save training history: {e}")

        # --- Reset stop flag BEFORE final evaluation ---
        print("Resetting stop flag before final evaluation.")
        stop_training_flag = False

        # --- Step 7: Final Evaluation (on Standard Validation Set - Aggregate Error) ---
        print("\n===== STEP 5: Evaluating Model (Weighted Aggregate Error on Standard Val Set) =====")

        # Load best weights (based on standard val loss) if checkpoint exists and training wasn't interrupted prematurely
        if checkpoint_path.exists() and not training_was_interrupted:
            print(f"Loading best model weights from {checkpoint_path} for final evaluation.")
            try: model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            except Exception as e: print(f"Warning: Could not load best weights from {checkpoint_path}: {e}. Using last weights.")
        elif training_was_interrupted:
             print("Warning: Training was interrupted. Using last weights for evaluation.")
        else: print("Warning: Best model checkpoint not found. Using last weights from training for evaluation.")

        # Recreate standard validation loader for clean evaluation if it exists
        final_val_loader = None
        if val_dataset_loaded and val_loader_params_stored:
            print("Recreating standard validation DataLoader for final evaluation...")
            try:
                 if isinstance(val_dataset_loaded, SequenceDataset) and len(val_dataset_loaded) > 0:
                     final_val_loader = DataLoader(val_dataset_loaded, **val_loader_params_stored)
                     print("Standard validation DataLoader recreated.")
                 else:
                     print("Standard validation dataset is invalid or empty. Skipping final evaluation on this set.")
            except Exception as e:
                 print(f"Error recreating standard validation DataLoader: {e}. Skipping final evaluation on this set.")
        else:
            print("Standard validation dataset/params not available. Skipping final evaluation on this set.")

        # Run weighted evaluation on standard validation set
        if final_val_loader and model:
            model.eval() # Ensure model is in eval mode
            std_val_agg_error_final = evaluate_aggregate_unemployment_error(model, final_val_loader, DEVICE, metadata)
            print(f"Final Weighted Aggregate Unemployment Rate Error (MSE) on Standard Validation Set: {std_val_agg_error_final:.6f}")
        else:
            std_val_agg_error_final = float('inf') # Mark as infinite if not evaluated
            print("Final Weighted Aggregate Unemployment Rate Error (MSE) on Standard Validation Set: Not Calculated")

        # --- Extract Final Metrics ---
        try:
            final_val_acc_final = next((h for h in reversed(history.get('val_accuracy', [])) if h is not None and not np.isnan(h)), float('nan'))
        except: final_val_acc_final = float('nan')

        try:
            if not np.isnan(best_val_loss_final) and best_val_loss_final != float('inf'):
                 best_epoch_final = history.get('val_loss', []).index(best_val_loss_final) + 1
            else: best_epoch_final = last_epoch + 1 # Fallback if no valid best loss found
        except (ValueError, IndexError, TypeError):
             best_epoch_final = last_epoch + 1 # Fallback

        # Determine final status
        run_successful = not np.isinf(std_val_agg_error_final) and not np.isnan(std_val_agg_error_final)

        if training_was_interrupted:
            run_status = "Interrupted"
        elif run_successful:
            run_status = "Completed" # Completed standard training and evaluation
        else:
            run_status = "Finished with Error" # Could be NaN/Inf metric or other failure

    except optuna.TrialPruned as e:
        print(f"Optuna Trial Pruned: {e}")
        run_status = "Pruned"
        training_was_interrupted = True # Mark as interrupted for result handling
        std_val_agg_error_final = float('inf')
    except KeyboardInterrupt:
        print("\nRun interrupted by user (KeyboardInterrupt caught in train_and_evaluate_internal).")
        run_status = "Interrupted"
        training_was_interrupted = True
        std_val_agg_error_final = float('inf')
    except (FileNotFoundError, ValueError, RuntimeError, TypeError, KeyError) as e: # Catch specific expected errors
        print(f"\nA known error occurred during the run: {type(e).__name__}: {e}")
        traceback.print_exc()
        run_status = "Failed"
        std_val_agg_error_final = float('inf')
    except Exception as e: # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred during the run: {type(e).__name__}: {e}")
        traceback.print_exc()
        run_status = "Failed"
        std_val_agg_error_final = float('inf')

    finally:
        # --- Cleanup ---
        signal.signal(signal.SIGINT, original_sigint_handler) # Restore original handler
        print("Restored original SIGINT handler.")
        # Don't delete the model here if status is Completed, as it's needed by the HPT objective
        if run_status != "Completed" and model is not None:
            del model
            model = None
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            elif DEVICE.type == 'mps': torch.mps.empty_cache()
            print("Cleaned up model and GPU memory due to non-completion.")
        # Clean up datasets and loaders
        if 'train_loader' in locals(): del train_loader
        if 'val_loader' in locals(): del val_loader
        if 'final_val_loader' in locals(): del final_val_loader
        if 'val_dataset_loaded' in locals(): del val_dataset_loaded
        gc.collect()

        end_run_time = time.time()
        elapsed_mins = (end_run_time - start_run_time) / 60
        print("-" * 60)
        print(f"Run Status: {run_status} | Elapsed: {elapsed_mins:.2f} minutes")
        print(f"Final Weighted Agg Error (Standard Val): {std_val_agg_error_final}")
        print("=" * 60)

    # Return results including the model if completed successfully
    results = {
        'model': model if run_status == "Completed" else None, # Return model only if run completed
        'std_val_agg_error': std_val_agg_error_final,
        'best_val_loss': best_val_loss_final,
        'final_val_acc': final_val_acc_final,
        'best_epoch': best_epoch_final,
        'status': run_status
    }
    return results


# --- Standard Training Runner ---

def run_standard_training(args, base_output_dir):
    """Runs a standard training process, potentially using best HPT params or specific trial params."""
    print("\n--- Starting Standard Training Run ---")
    study_name = args.study_name # Use study name from args
    study_dir = base_output_dir / study_name
    best_hparams_file = study_dir / config.BEST_HPARAMS_PKL
    hpt_results_file = study_dir / config.HPT_RESULTS_CSV # Path to HPT results CSV

    # Start with default hyperparameters from config
    standard_hparams = {
        'sequence_length': config.SEQUENCE_LENGTH,
        'embed_dim': config.EMBED_DIM,
        'num_heads': config.NUM_HEADS,
        'ff_dim': config.FF_DIM,
        'num_transformer_blocks': config.NUM_TRANSFORMER_BLOCKS,
        'mlp_units': config.MLP_UNITS,
        'dropout': config.DROPOUT,
        'mlp_dropout': config.MLP_DROPOUT,
        'learning_rate': config.LEARNING_RATE,
        'batch_size': config.BATCH_SIZE,
        'loss_weight_factor': 1.0, # Default to standard inverse frequency weighting (factor=1.0)
        'epochs': config.EPOCHS, # Use standard epochs
        'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        'max_grad_norm': config.MAX_GRAD_NORM,
        'lr_scheduler_factor': config.LR_SCHEDULER_FACTOR,
        'lr_scheduler_patience': config.LR_SCHEDULER_PATIENCE,
        'pad_value': config.PAD_VALUE,
        'parallel_workers': config.PARALLEL_WORKERS,
        'refresh_sequences': config.REFRESH_SEQUENCES, # Use standard setting
        'train_start_date': config.TRAIN_START_DATE,
        'train_end_date': config.TRAIN_END_DATE,
        'random_seed': config.RANDOM_SEED,
        # Add HPT forecast horizon (not used directly in standard run, but good practice)
        'hpt_forecast_horizon': config.HPT_FORECAST_HORIZON,
    }

    # --- Parameter Loading Logic ---
    params_loaded_from = "Defaults from config.py"
    loaded_tuned_params = {} # Store only the parameters loaded from file

    if args.use_trial is not None:
        print(f"\nAttempting to load parameters from Trial {args.use_trial} in study '{study_name}'...")
        if not hpt_results_file.exists():
            print(f"ERROR: HPT results file not found at {hpt_results_file}. Cannot load trial parameters.")
            print("Proceeding with default hyperparameters.")
        else:
            try:
                hpt_df = pd.read_csv(hpt_results_file)
                trial_row = hpt_df[hpt_df['trial_number'] == args.use_trial]

                if trial_row.empty:
                    print(f"ERROR: Trial {args.use_trial} not found in {hpt_results_file}.")
                    print("Proceeding with default hyperparameters.")
                else:
                    print(f"Found Trial {args.use_trial}. Loading its parameters.")
                    trial_params_csv = trial_row.iloc[0].to_dict()
                    num_updated = 0
                    # Load only tunable keys from CSV
                    # Add 'loss_weight_factor' to the list of tunable keys
                    tunable_keys_from_csv = [col for col in trial_params_csv if col in standard_hparams and col not in ['sequence_length', 'epochs', 'refresh_sequences', 'early_stopping_patience', 'max_grad_norm', 'lr_scheduler_factor', 'lr_scheduler_patience', 'pad_value', 'parallel_workers', 'train_start_date', 'train_end_date', 'random_seed', 'hpt_forecast_horizon']]

                    for key in tunable_keys_from_csv:
                        value = trial_params_csv[key]
                        try:
                            # Attempt to convert type (e.g., str -> list, str -> bool/int/float)
                            target_type = type(standard_hparams[key])
                            if target_type == list and isinstance(value, str):
                                parsed_value = ast.literal_eval(value)
                                if isinstance(parsed_value, list):
                                    loaded_tuned_params[key] = parsed_value
                                    num_updated += 1
                                else: print(f"Warning: Parsed value for '{key}' is not a list: {parsed_value}. Skipping.")
                            elif target_type == bool and isinstance(value, (str, int, float)): # Handle various bool representations
                                if isinstance(value, str):
                                    loaded_tuned_params[key] = value.lower() in ['true', '1', 't', 'y', 'yes']
                                else:
                                    loaded_tuned_params[key] = bool(value)
                                    num_updated += 1
                            elif target_type in [int, float]:
                                loaded_tuned_params[key] = target_type(value)
                                num_updated += 1
                            else: # Assume string or already correct type
                                loaded_tuned_params[key] = value
                                num_updated += 1
                        except (ValueError, SyntaxError, TypeError) as parse_err:
                            print(f"Warning: Could not parse value '{value}' for parameter '{key}'. Skipping. Error: {parse_err}")
                        except Exception as e:
                             print(f"Warning: Unexpected error parsing parameter '{key}' with value '{value}'. Skipping. Error: {e}")

                    params_loaded_from = f"Trial {args.use_trial} from {hpt_results_file}"
                    print(f"Loaded {num_updated} tunable parameters from Trial {args.use_trial}.")

            except Exception as e:
                print(f"ERROR loading or parsing HPT results file {hpt_results_file}: {e}")
                traceback.print_exc()
                print("Proceeding with default hyperparameters.")

    # If not using a specific trial, try loading the best params file (contains only tuned params)
    elif best_hparams_file.exists():
        print(f"\nFound best hyperparameters file: {best_hparams_file}")
        try:
            with open(best_hparams_file, 'rb') as f:
                loaded_tuned_params = pickle.load(f) # Load the dictionary of tuned params
            print("Successfully loaded tuned hyperparameters from best_hparams.pkl.")
            params_loaded_from = f"Best parameters from {best_hparams_file}"
            print(f"Loaded {len(loaded_tuned_params)} tuned hyperparameters.")

        except Exception as e:
            print(f"Warning: Failed to load or apply best hyperparameters from {best_hparams_file}: {e}")
            print("Proceeding with default hyperparameters from config.py.")
            loaded_tuned_params = {} # Reset loaded params

    else:
        print(f"\nBest hyperparameters file not found at {best_hparams_file}.")
        print("Proceeding with default hyperparameters from config.py.")

    # Update standard_hparams with the loaded tuned parameters
    num_applied = 0
    for key, value in loaded_tuned_params.items():
        if key in standard_hparams:
            standard_hparams[key] = value
            num_applied += 1
        else:
            print(f"Warning: Loaded parameter '{key}' not found in standard config defaults. Ignoring.")

    if num_applied > 0:
        print(f"Applied {num_applied} parameters from: {params_loaded_from}")
    else:
        print(f"Using parameters loaded from: {params_loaded_from}")


    # Run training using the final standard_hparams
    try:
        # Ensure batch_size is present before calling train_and_evaluate_internal
        if 'batch_size' not in standard_hparams:
             print(f"ERROR: 'batch_size' missing from final standard_hparams. Using default: {config.BATCH_SIZE}")
             standard_hparams['batch_size'] = config.BATCH_SIZE

        # Call the internal training function without a trial object
        results = train_and_evaluate_internal(standard_hparams, trial=None)
        print(f"\nStandard run finished with status: {results['status']}")
        print(f"Final Aggregate Unemployment Error (Standard Validation Set): {results['std_val_agg_error']:.6f}")

        # Clean up the returned model object if it exists
        if results.get('model') is not None:
            del results['model']
            DEVICE = get_device() # Re-get device if needed for cleanup
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            elif DEVICE.type == 'mps': torch.mps.empty_cache()
            gc.collect()
            print("Cleaned up model object after standard run.")

    except KeyboardInterrupt: print("\nStandard training run interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn critical error occurred during the standard training run: {e}")
        traceback.print_exc()


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Model or Run Hyperparameter Tuning")
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning using Optuna')
    parser.add_argument('--n_trials', type=int, default=config.HPT_N_TRIALS, help='Number of trials for Optuna tuning')
    parser.add_argument('--study_name', type=str, default=config.HPT_STUDY_NAME, help='Name for the Optuna study')
    parser.add_argument('--use_trial', type=int, default=None, metavar='TRIAL_NUM',
                        help='Run standard training using parameters from a specific HPT trial number (overrides best_hparams.pkl)')

    args = parser.parse_args()

    print("\n===== Transformer Training/Tuning Script =====")

    base_output_dir = Path(config.TRAIN_OUTPUT_SUBDIR)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base output directory: {base_output_dir}")

    if args.tune:
        # Cannot use --use_trial when tuning
        if args.use_trial is not None:
             print("Warning: --use_trial argument is ignored when running hyperparameter tuning (--tune).")
        # Pass the internal training function to the HPT runner
        run_hyperparameter_tuning(args, base_output_dir, train_and_evaluate_internal)
    else:
        # Pass args to standard training to access --use_trial and --study_name
        run_standard_training(args, base_output_dir)

    print("\n--- Python Script Finished ---")
