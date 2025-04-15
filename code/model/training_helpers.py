from pathlib import Path
import pickle
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import multiprocessing
import traceback
import gc

# Assuming utils.py contains get_device, SequenceDataset, worker_init_fn, stop_training_flag
from utils import get_device, SequenceDataset, worker_init_fn, stop_training_flag
# Assuming models.py contains the model definition
from models import TransformerForecastingModel

# --- Training and Evaluation Functions (during training loop) ---
def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Disable tqdm progress bar if not in the main process
    is_main_process = multiprocessing.current_process().name == 'MainProcess'
    data_iterator = tqdm(dataloader, desc="Training", leave=False, disable=not is_main_process)

    for x_batch, y_batch, _, mask_batch in data_iterator: # Weight (w_batch) is unused here
        if stop_training_flag:
            print("Stop signal detected during training batch iteration.")
            break # Exit the inner loop

        x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch, src_key_padding_mask=mask_batch) # Pass padding mask
        loss = criterion(outputs, y_batch) # Loss calculation uses weights if criterion was initialized with them
        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() # Aggregate base loss value
        _, predicted = torch.max(outputs.data, 1)
        total_samples += batch_size
        correct_predictions += (predicted == y_batch).sum().item()

        # Update tqdm description dynamically in the main process
        if is_main_process and total_samples > 0:
             current_loss = total_loss / total_samples # Average loss per sample so far
             current_acc = correct_predictions / total_samples
             data_iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    if stop_training_flag: return float('nan'), float('nan') # Return NaN if interrupted

    # Avoid division by zero if epoch finishes with no samples
    if total_samples == 0: return 0.0, 0.0

    avg_loss = total_loss / total_samples # Final average loss for the epoch
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluates the model for one epoch (e.g., on validation set)."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    is_main_process = multiprocessing.current_process().name == 'MainProcess'
    data_iterator = tqdm(dataloader, desc="Evaluating", leave=False, disable=not is_main_process)

    with torch.no_grad():
        for x_batch, y_batch, _, mask_batch in data_iterator: # Weight (w_batch) is unused here
             if stop_training_flag:
                 print("Stop signal detected during evaluation iteration.")
                 break

             x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

             outputs = model(x_batch, src_key_padding_mask=mask_batch) # Pass padding mask
             
             # --- DIAGNOSTIC PRINTS (Validation) ---
             if total_samples == 0: # Print only for the first batch
                 print(f"\nDEBUG (Validation, Batch 0):")
                 print(f"  outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                 print(f"  y_batch shape: {y_batch.shape}, dtype: {y_batch.dtype}")
             loss = criterion(outputs, y_batch) # Calculate loss *after* printing shapes
             if total_samples == 0:
                 print(f"  Calculated loss: {loss.item():.6f}")
             # --- END DIAGNOSTIC PRINTS ---

             batch_size = x_batch.size(0)
             total_loss += loss.item() # Aggregate base loss value
             _, predicted = torch.max(outputs.data, 1)
             total_samples += batch_size
             correct_predictions += (predicted == y_batch).sum().item()
             all_preds.extend(predicted.cpu().numpy())
             all_targets.extend(y_batch.cpu().numpy())

             # Update tqdm description dynamically
             if is_main_process and total_samples > 0:
                 current_loss = total_loss / total_samples
                 current_acc = correct_predictions / total_samples
                 data_iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    if stop_training_flag: return float('nan'), float('nan'), np.array([]), np.array([]) # Return NaN if interrupted

    if total_samples == 0:
         print("Warning: Evaluation completed with zero samples.")
         return float('nan'), float('nan'), np.array([]), np.array([])

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy, np.array(all_targets), np.array(all_preds)


def create_dataloaders(x_train_np, y_train_np, weight_train_np, x_val_np, y_val_np, weight_val_np, hparams, device, parallel_workers):
    """Creates Datasets and DataLoaders for training and validation."""
    print("\nCreating PyTorch Datasets and DataLoaders...")
    batch_size = hparams['batch_size']
    pad_value = hparams['pad_value']

    try:
         # Ensure data exists before creating datasets
         if x_train_np.shape[0] == 0: raise ValueError("Training data (x_train_np) is empty.")
         if x_val_np.shape[0] == 0: raise ValueError("Validation data (x_val_np) is empty.")

         train_dataset = SequenceDataset(x_train_np, y_train_np, weight_train_np, pad_value)
         val_dataset = SequenceDataset(x_val_np, y_val_np, weight_val_np, pad_value)
         print(f"Created Train Dataset ({len(train_dataset)} samples), Validation Dataset ({len(val_dataset)} samples)")
    except Exception as e:
         print(f"ERROR creating SequenceDataset objects: {e}")
         traceback.print_exc()
         raise

    # --- DataLoaders ---
    # Determine number of workers, ensuring it's not more than available cores
    max_workers = multiprocessing.cpu_count()
    dataloader_workers = min(max(0, parallel_workers), max_workers, 4) # Use min(config, cpu_count, 4)
    print(f"Using {dataloader_workers} workers for DataLoaders.")
    pin_memory = device.type == 'cuda' # Pin memory only if using CUDA
    # Use persistent workers only if num_workers > 0 to avoid overhead
    persistent_workers = dataloader_workers > 0

    # Common DataLoader arguments
    loader_args = {
        'num_workers': dataloader_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'worker_init_fn': worker_init_fn if dataloader_workers > 0 else None,
        'drop_last': False # Keep last batch even if smaller
    }

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, # Shuffle training data
        **loader_args
    )
    # Use larger batch size for validation if possible, but respect batch_size param
    val_batch_size = max(batch_size, batch_size * 2) # Example: double batch size for validation
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size,
        shuffle=False, # No need to shuffle validation data
        **loader_args
    )

    print(f"DataLoaders created (Train BS={batch_size}, Val BS={val_batch_size}).")
    if dataloader_workers > 0: print("DataLoader workers configured to ignore SIGINT.")

    # Store validation loader parameters for potential reuse
    val_loader_params = {'batch_size': val_batch_size, 'shuffle': False, **loader_args}

    return train_loader, val_loader, val_loader_params, val_dataset


def build_model(hparams, n_features, n_classes, device):
    """Builds the Transformer model based on hyperparameters."""
    print("\n===== STEP 3: Building Transformer Model =====")
    try:
        model = TransformerForecastingModel(
            input_dim=n_features,
            seq_len=hparams['sequence_length'],
            embed_dim=hparams['embed_dim'],
            num_heads=hparams['num_heads'],
            ff_dim=hparams['ff_dim'],
            num_transformer_blocks=hparams['num_transformer_blocks'],
            mlp_units=hparams['mlp_units'],
            dropout=hparams['dropout'],
            mlp_dropout=hparams['mlp_dropout'],
            n_classes=n_classes
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model built successfully on {device}. Trainable parameters: {total_params:,}")
        return model
    except Exception as e:
        print(f"ERROR building model: {e}")
        traceback.print_exc()
        raise


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, hparams, device, checkpoint_path, trial=None):
    """
    Runs the main training loop with validation, checkpointing, early stopping, and Optuna pruning.
    Uses the provided criterion (potentially with class weights).
    """
    print("\n===== STEP 4: Training the Model =====")
    epochs = hparams['epochs']
    early_stopping_patience = hparams['early_stopping_patience']
    max_grad_norm = hparams['max_grad_norm']

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    training_interrupted = False
    last_epoch = 0 # Track the last epoch number completed

    print(f"Starting training loop: Max epochs={epochs}, Batch size={hparams['batch_size']}, LR={hparams['learning_rate']:.6f}")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print(f"Best model checkpoint path: {checkpoint_path}")

    for epoch in range(epochs):
        last_epoch = epoch
        epoch_start_time = time.time()

        if stop_training_flag:
            print(f"\nStop signal detected before starting epoch {epoch+1}. Stopping training.")
            training_interrupted = True
            break

        # --- Training Step ---
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)

        if stop_training_flag or np.isnan(train_loss): # Check for interruption or NaN loss
            print(f"\nStop signal detected or NaN training loss after epoch {epoch+1}. Stopping training.")
            training_interrupted = True
            history['loss'].append(train_loss); history['accuracy'].append(train_acc)
            history['val_loss'].append(float('nan')); history['val_accuracy'].append(float('nan'))
            break

        # --- Validation Step ---
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        epoch_end_time = time.time()

        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_end_time - epoch_start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.7f}")

        history['loss'].append(train_loss); history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_accuracy'].append(val_acc)

        # Handle NaN validation loss - treat as no improvement
        if np.isnan(val_loss):
            print("Warning: Validation loss is NaN. Treating as no improvement.")
            epochs_no_improve += 1
        else:
            # --- Learning Rate Scheduler Step ---
            scheduler.step(val_loss)

            # --- Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                try:
                    # Ensure parent directory exists before saving
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"  -> Val loss improved to {val_loss:.4f}. Saved best model checkpoint.")
                except Exception as e: print(f"  -> ERROR saving best model checkpoint: {e}")
            else:
                epochs_no_improve += 1

        # --- Optuna Pruning Check ---
        if trial:
            # Report the standard validation loss for pruning
            trial.report(val_loss if not np.isnan(val_loss) else float('inf'), epoch) # Report inf if NaN
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch+1} based on intermediate validation loss.")
                # Save history before pruning
                try:
                    history_file = checkpoint_path.parent / "training_history_pruned.pkl"
                    with open(history_file, 'wb') as f: pickle.dump(history, f)
                except Exception as e: print(f"Warning: Could not save history before pruning: {e}")
                raise optuna.TrialPruned() # Let Optuna handle the pruning

        # --- Check Early Stopping Condition ---
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience: {early_stopping_patience}).")
            break

    # --- End of Training Loop ---
    if training_interrupted: print("Training loop was interrupted.")
    elif epochs_no_improve < early_stopping_patience: print(f"Training loop finished after completing all {epochs} epochs.")

    return history, best_val_loss, training_interrupted, last_epoch, epochs_no_improve
