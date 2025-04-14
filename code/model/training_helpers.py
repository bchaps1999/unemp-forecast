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

    # Disable tqdm in worker processes to avoid multiple bars
    is_main_process = not torch.utils.data.get_worker_info() if torch.utils.data.get_worker_info() else True
    data_iterator = tqdm(dataloader, desc="Training", leave=False, disable=not is_main_process)

    for x_batch, y_batch, _, mask_batch in data_iterator:
        # Check for stop signal *during* batch iteration (more responsive)
        if stop_training_flag:
            print("Stop signal detected during training batch iteration.")
            break # Exit the inner loop

        x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()
        # Pass the padding mask to the model
        outputs = model(x_batch, src_key_padding_mask=mask_batch)
        # Note: criterion might use weights internally if configured, but loss calculation is standard
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size # Unweighted loss for reporting consistency
        _, predicted = torch.max(outputs.data, 1)
        total_samples += batch_size
        correct_predictions += (predicted == y_batch).sum().item()

        # Update tqdm description dynamically if it's the main process
        if is_main_process:
             current_loss = total_loss / total_samples if total_samples > 0 else 0
             current_acc = correct_predictions / total_samples if total_samples > 0 else 0
             data_iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    # Avoid division by zero if epoch finishes with no samples or was interrupted early
    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluates the model for one epoch (typically on validation set during training)."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0 # Unweighted sample count
    all_preds = []
    all_targets = []

    # Disable tqdm in worker processes
    is_main_process = not torch.utils.data.get_worker_info() if torch.utils.data.get_worker_info() else True
    data_iterator = tqdm(dataloader, desc="Evaluating", leave=False, disable=not is_main_process)

    with torch.no_grad():
        for x_batch, y_batch, _, mask_batch in data_iterator:
             # Check for stop signal (less critical here, but good practice)
             if stop_training_flag:
                 print("Stop signal detected during evaluation iteration.")
                 break

             x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

             # Pass the padding mask
             outputs = model(x_batch, src_key_padding_mask=mask_batch)
             loss = criterion(outputs, y_batch) # Use the same criterion as training (might be weighted)

             batch_size = x_batch.size(0)
             total_loss += loss.item() * batch_size # Unweighted loss for reporting consistency
             _, predicted = torch.max(outputs.data, 1)
             total_samples += batch_size
             correct_predictions += (predicted == y_batch).sum().item()
             all_preds.extend(predicted.cpu().numpy())
             all_targets.extend(y_batch.cpu().numpy())

             # Update tqdm description dynamically
             if is_main_process:
                 current_loss = total_loss / total_samples if total_samples > 0 else 0
                 current_acc = correct_predictions / total_samples if total_samples > 0 else 0
                 data_iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})


    if total_samples == 0:
         print("Warning: Evaluation completed with zero samples.")
         # Return values indicating failure or empty evaluation
         return float('nan'), float('nan'), np.array([]), np.array([])

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    # Return loss, accuracy, and raw predictions/targets/weights if needed elsewhere
    return avg_loss, accuracy, np.array(all_targets), np.array(all_preds)


def create_dataloaders(x_train_np, y_train_np, weight_train_np, x_val_np, y_val_np, weight_val_np, hparams, n_classes, device, parallel_workers):
    """Creates Datasets and DataLoaders for training and validation."""
    print("\nCreating PyTorch Datasets and DataLoaders...")
    batch_size = hparams['batch_size']
    pad_value = hparams['pad_value']

    try:
         train_dataset = SequenceDataset(x_train_np, y_train_np, weight_train_np, pad_value)
         val_dataset = SequenceDataset(x_val_np, y_val_np, weight_val_np, pad_value)
    except Exception as e:
         print(f"ERROR creating SequenceDataset objects: {e}")
         traceback.print_exc()
         raise

    # --- Class Weights for Loss ---
    # Class weights are now calculated and applied in train_and_evaluate_internal before creating the criterion
    print("Class weights (if any) are applied directly to the loss criterion.")

    # --- DataLoaders ---
    dataloader_workers = min(4, parallel_workers) if parallel_workers > 0 else 0
    print(f"Using {dataloader_workers} workers for DataLoaders.")
    pin_memory = device.type == 'cuda'
    persistent_workers = dataloader_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, # Always shuffle training data
        num_workers=dataloader_workers,
        worker_init_fn=worker_init_fn if dataloader_workers > 0 else None,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader_params = {
        'batch_size': batch_size * 2, # Often use larger batch for validation
        'shuffle': False, 'num_workers': dataloader_workers,
        'pin_memory': pin_memory, 'persistent_workers': persistent_workers,
        'worker_init_fn': worker_init_fn if dataloader_workers > 0 else None
    }
    val_loader = DataLoader(val_dataset, **val_loader_params)

    print(f"DataLoaders created (Training uses standard shuffling).")
    if dataloader_workers > 0: print("DataLoader workers configured to ignore SIGINT.")

    # Return val_dataset and params for potential reuse in final evaluation
    return train_loader, val_loader, val_loader_params, val_dataset


def build_model(hparams, n_features, n_classes, device):
    """Builds the Transformer model."""
    print("\n===== STEP 3: Building Transformer Model =====")
    try:
        model = TransformerForecastingModel(
            input_dim=n_features, seq_len=hparams['sequence_length'],
            embed_dim=hparams['embed_dim'], num_heads=hparams['num_heads'],
            ff_dim=hparams['ff_dim'], num_transformer_blocks=hparams['num_transformer_blocks'],
            mlp_units=hparams['mlp_units'], dropout=hparams['dropout'],
            mlp_dropout=hparams['mlp_dropout'], n_classes=n_classes
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model built successfully.")
        print(f"Total trainable parameters: {total_params:,}")
        return model
    except Exception as e:
        print(f"ERROR building model: {e}")
        traceback.print_exc()
        raise


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, hparams, device, checkpoint_path, trial=None):
    """Runs the main training loop with validation, checkpointing, and early stopping.
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
    last_epoch = 0 # Track the last epoch number run

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

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)

        if stop_training_flag:
            print(f"\nStop signal detected after training epoch {epoch+1}. Stopping training.")
            training_interrupted = True
            history['loss'].append(train_loss); history['accuracy'].append(train_acc)
            history['val_loss'].append(float('nan')); history['val_accuracy'].append(float('nan'))
            break

        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        epoch_end_time = time.time()

        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_end_time - epoch_start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.7f}")

        history['loss'].append(train_loss); history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_accuracy'].append(val_acc)

        # Handle NaN validation loss
        if np.isnan(val_loss):
            print("Warning: Validation loss is NaN. Skipping LR scheduler step and early stopping check.")
            # Optionally trigger early stopping if NaN persists?
            # epochs_no_improve += 1 # Treat NaN as no improvement
        else:
            # --- Learning Rate Scheduler ---
            scheduler.step(val_loss)

            # --- Early Stopping & Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                try:
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"  -> Val loss improved to {val_loss:.4f}. Saved best model checkpoint.")
                except Exception as e: print(f"  -> ERROR saving best model checkpoint: {e}")
            else:
                epochs_no_improve += 1

        # --- Optuna Pruning ---
        if trial:
            # Report the standard validation loss for pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch+1} based on intermediate validation loss.")
                # Save history before pruning?
                try:
                    history_file = Path(checkpoint_path).parent / "training_history_pruned.pkl"
                    with open(history_file, 'wb') as f: pickle.dump(history, f)
                except Exception as e: print(f"Warning: Could not save history before pruning: {e}")
                raise optuna.TrialPruned() # Let Optuna handle the pruning

        # --- Check Early Stopping Condition ---
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    if training_interrupted: print("Training loop was interrupted by user (Ctrl+C).")
    elif epochs_no_improve < early_stopping_patience: print("Training loop finished after completing all epochs.")

    return history, best_val_loss, training_interrupted, last_epoch, epochs_no_improve
