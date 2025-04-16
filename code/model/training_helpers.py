from pathlib import Path
import pickle
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F # Import functional
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import multiprocessing
import traceback
import gc

# --- Project Imports --- # Add this section
import config # Add this line

# Assuming utils.py contains get_device, SequenceDataset, worker_init_fn, stop_training_flag
from utils import get_device, SequenceDataset, worker_init_fn, stop_training_flag
# Assuming models.py contains the model definition
from models import TransformerForecastingModel

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    """
    Focal Loss implementation.
    Handles logits input. Assumes reduction is handled outside or set during init.
    gamma > 0 reduces the relative loss for well-classified examples (pt > 0.5),
    putting more focus on hard, misclassified examples.
    """
    def __init__(self, gamma=2.0, reduction='mean'): 
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are expected to be logits (N, C)
        # targets are expected to be class indices (N,)

        # Calculate Cross Entropy loss without reduction
        # NOTE: Always use reduction='none' here as weighting/reduction is handled outside
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss component
        focal_term = (1 - pt)**self.gamma
        loss = focal_term * ce_loss

        # Apply reduction based on init parameter
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none' or invalid
            return loss # Return per-sample loss if reduction is 'none'

# --- Training and Evaluation Functions (during training loop) ---
def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm):
    """
    Trains the model for one epoch using sample weights.
    Assumes criterion has reduction='none'.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    sum_sample_weights = 0.0 # For potential weighted accuracy/loss normalization

    # Disable tqdm progress bar if not in the main process
    is_main_process = multiprocessing.current_process().name == 'MainProcess'
    data_iterator = tqdm(dataloader, desc="Training", leave=False, disable=not is_main_process)

    # Expect sample_weight_batch as the 3rd element
    for x_batch, y_batch, sample_weight_batch, mask_batch in data_iterator:
        if stop_training_flag:
            print("Stop signal detected during training batch iteration.")
            break # Exit the inner loop

        x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
        sample_weight_batch = sample_weight_batch.to(device) # Move weights to device

        optimizer.zero_grad()
        outputs = model(x_batch, src_key_padding_mask=mask_batch) # Pass padding mask

        # Calculate per-sample loss (criterion should have reduction='none')
        loss_per_sample = criterion(outputs, y_batch)

        # Apply sample weights
        weighted_loss = loss_per_sample * sample_weight_batch

        # Calculate the mean loss for the batch
        loss = weighted_loss.mean()

        # Backpropagate the mean weighted loss
        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = x_batch.size(0)
        # Aggregate the *mean* batch loss, weighted by batch size for overall average
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.data, 1)
        total_samples += batch_size
        correct_predictions += (predicted == y_batch).sum().item()
        sum_sample_weights += sample_weight_batch.sum().item() # Sum weights in batch

        # Update tqdm description dynamically in the main process
        if is_main_process and total_samples > 0:
             # Report average loss per sample so far
             current_loss = total_loss / total_samples
             current_acc = correct_predictions / total_samples
             # Optionally report weighted loss avg: total_loss / sum_sample_weights
             data_iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    if stop_training_flag: return float('nan'), float('nan') # Return NaN if interrupted

    # Avoid division by zero if epoch finishes with no samples
    if total_samples == 0: return 0.0, 0.0

    # Final average loss for the epoch (mean of mean batch losses)
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    # avg_weighted_loss = total_loss / sum_sample_weights if sum_sample_weights > 0 else 0.0
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model for one epoch (e.g., on validation set).
    Calculates standard unweighted loss and accuracy.
    Assumes criterion has reduction='none'.
    """
    model.eval()
    total_loss = 0.0 # Unweighted loss
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    is_main_process = multiprocessing.current_process().name == 'MainProcess'
    data_iterator = tqdm(dataloader, desc="Evaluating", leave=False, disable=not is_main_process)

    with torch.no_grad():
        # Expect sample_weight_batch as the 3rd element, but ignore it here
        for x_batch, y_batch, _, mask_batch in data_iterator:
             if stop_training_flag:
                 print("Stop signal detected during evaluation iteration.")
                 break

             x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)

             outputs = model(x_batch, src_key_padding_mask=mask_batch) # Pass padding mask

             # Calculate per-sample loss (criterion has reduction='none')
             loss_per_sample = criterion(outputs, y_batch)
             # Calculate the unweighted mean loss for the batch
             loss = loss_per_sample.mean()

             # --- Diagnostic Print (Optional) ---
             # if total_samples == 0:
             #     print(f"  Eval Batch Loss (unweighted mean): {loss.item():.6f}")
             # --- END DIAGNOSTIC PRINTS ---

             batch_size = x_batch.size(0)
             # Aggregate unweighted batch loss
             total_loss += loss.item() * batch_size
             _, predicted = torch.max(outputs.data, 1)
             total_samples += batch_size
             correct_predictions += (predicted == y_batch).sum().item()
             all_preds.extend(predicted.cpu().numpy())
             all_targets.extend(y_batch.cpu().numpy())

             # Update tqdm description dynamically
             if is_main_process and total_samples > 0:
                 current_loss = total_loss / total_samples # Avg unweighted loss
                 current_acc = correct_predictions / total_samples
                 data_iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    if stop_training_flag: return float('nan'), float('nan'), np.array([]), np.array([]) # Return NaN if interrupted

    if total_samples == 0:
         print("Warning: Evaluation completed with zero samples.")
         return float('nan'), float('nan'), np.array([]), np.array([])

    avg_loss = total_loss / total_samples # Final avg unweighted loss
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy, np.array(all_targets), np.array(all_preds)


def create_dataloaders(x_train_np, y_train_np, sample_weights_train_np, # Renamed weight_train_np
                       x_val_np, y_val_np, sample_weights_val_np, # Renamed weight_val_np
                       hparams, device, parallel_workers):
    """Creates Datasets and DataLoaders for training and validation using sample weights.""" # Docstring updated
    print("\nCreating PyTorch Datasets and DataLoaders...")
    batch_size = hparams['batch_size']
    pad_value = hparams['pad_value']

    try:
         # Ensure data exists before creating datasets
         if x_train_np.shape[0] == 0: raise ValueError("Training data (x_train_np) is empty.")
         if x_val_np.shape[0] == 0: raise ValueError("Validation data (x_val_np) is empty.")

         # Pass sample weights to SequenceDataset
         train_dataset = SequenceDataset(x_train_np, y_train_np, sample_weights_train_np, pad_value)
         val_dataset = SequenceDataset(x_val_np, y_val_np, sample_weights_val_np, pad_value)
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
    Uses the provided criterion (assumed to have reduction='none' for training).
    Validation loss is calculated as unweighted mean.
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
    print(f"Using sample weights for training loss (Factor: {hparams.get('transition_weight_factor', config.TRANSITION_WEIGHT_FACTOR):.2f}).")
    print(f"Using FocalLoss gamma: {hparams.get('focal_loss_gamma', config.FOCAL_LOSS_GAMMA):.2f}.")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print(f"Best model checkpoint path: {checkpoint_path}")

    # --- Create separate criterion for evaluation with mean reduction ---
    # This avoids issues if the main criterion instance is modified
    eval_criterion = FocalLoss(gamma=criterion.gamma, reduction='mean').to(device)


    for epoch in range(epochs):
        last_epoch = epoch
        epoch_start_time = time.time()

        if stop_training_flag:
            print(f"\nStop signal detected before starting epoch {epoch+1}. Stopping training.")
            training_interrupted = True
            break

        # --- Training Step ---
        # train_epoch now handles sample weighting internally using the main criterion (reduction='none')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)

        if stop_training_flag or np.isnan(train_loss): # Check for interruption or NaN loss
            print(f"\nStop signal detected or NaN training loss after epoch {epoch+1}. Stopping training.")
            training_interrupted = True
            history['loss'].append(train_loss); history['accuracy'].append(train_acc)
            history['val_loss'].append(float('nan')); history['val_accuracy'].append(float('nan'))
            break

        # --- Validation Step ---
        # evaluate_epoch calculates unweighted loss using the eval_criterion (reduction='mean')
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, eval_criterion, device)
        epoch_end_time = time.time()

        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_end_time - epoch_start_time:.2f}s | "
              f"Train Loss (Avg Wtd): {train_loss:.4f}, Train Acc: {train_acc:.4f} | " # Clarify Train Loss is weighted avg
              f"Val Loss (Unwtd): {val_loss:.4f}, Val Acc: {val_acc:.4f} | " # Clarify Val Loss is unweighted avg
              f"LR: {optimizer.param_groups[0]['lr']:.7f}")

        history['loss'].append(train_loss); history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_accuracy'].append(val_acc)

        # Handle NaN validation loss - treat as no improvement
        if np.isnan(val_loss):
            print("Warning: Validation loss is NaN. Treating as no improvement.")
            epochs_no_improve += 1
        else:
            # --- Learning Rate Scheduler Step ---
            scheduler.step(val_loss) # Step based on unweighted validation loss

            # --- Checkpointing ---
            if val_loss < best_val_loss: # Checkpoint based on unweighted validation loss
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
            # Report the unweighted validation loss for pruning
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
