# Copyright (c) 2025 Ibad Desmukh
#
# SPDX-License-Identifier: MIT
#
"""Implement and test basic checkpointing.

Creates a basic layer, checkpoints it and loads it back.
"""

import os
import torch
import torch.nn as nn
import torch.optim
import time

# Make a test model.
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)

def save_checkpoint(model, optimizer, epoch, step, loss, is_best=False, checkpoint_dir='./checkpoints', timestamp=None, batch_idx=0):
    """Save model, optimizer and resumable training state."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    if timestamp is None:
        timestamp = time.time()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': timestamp,
        'batch_idx': batch_idx,
        'epoch': epoch,
        'step': step,
        'loss': loss,
    }

    filepath = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')

    # Remove orphaned temp file from previous interrupted save.
    temp_filepath = filepath + '.tmp'
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)

    # Atomic save for file integrity.
    torch.save(checkpoint, temp_filepath)
    os.replace(temp_filepath, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_filepath)

def load_checkpoint(checkpoint_type='latest', model=None, optimizer=None, checkpoint_dir='./checkpoints'):
    """Load model, optimizer, training state, and return position.

    Args:
        checkpoint_type: 'latest' or 'best'
        model: Model to load
        optimizer: Optimizer to load
        checkpoint_dir: Directory having checkpoints
    """
    if checkpoint_type == 'best':
        filepath = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
    else:
        filepath = os.path.join(checkpoint_dir,'checkpoint_latest.pt')

    # Load the checkpoint from disk.
    checkpoint = torch.load(filepath, map_location='cpu')

    # Validate required keys.
    required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'step', 'timestamp', 'batch_idx']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key: '{key}'")

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Resume from last save state.
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', 0.0)
    timestamp = checkpoint.get('timestamp', None)
    batch_idx = checkpoint.get('batch_idx', 0)

    return epoch, step, loss, timestamp, batch_idx

if __name__ == "__main__":
    # Create model and optimizer.
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters())

    # Generate state for optimizer.
    x = torch.randn(8, 10)
    y = model(x)
    loss = y.mean()
    loss.backward()
    optimizer.step()

    # Test save checkpoint.
    epoch = 1
    step = 1
    loss = 0.5
    save_checkpoint(model, optimizer, epoch, step, loss, is_best=False)
    print("Successfully saved latest checkpoint")

    # Test best checkpoint save.
    lower_loss = 0.2
    save_checkpoint(model, optimizer, epoch=2, step=2, loss=lower_loss, is_best=True)
    print("Successfully saved best checkpoint")

    # Create second model and optimizer.
    model2 = TestModel()
    optimizer2 = torch.optim.Adam(model2.parameters())

    # Test load checkpoint function from latest path.    
    latest_path = './checkpoints/checkpoint_latest.pt'
    print("\nLoading from latest checkpoint")
    try:
        epoch, step, loss, timestamp, batch_idx = load_checkpoint('latest', model2, optimizer2)
        print(f"Successfully loaded latest checkpoint from epoch {epoch}, step {step} and loss {loss}")
    except FileNotFoundError:
        print("Latest checkpoint not found, restarting")
    except KeyError as e:
        print(f"Latest checkpoint file integrity error: {e}")

    # Test load checkpoint function from best path.
    best_path = './checkpoints/checkpoint_best.pt'
    print("\nLoading from best checkpoint")
    try:
        epoch_best, step_best, loss_best, timestamp_best, batch_idx_best = load_checkpoint('best', model2, optimizer2)
        print(f"Best checkpoint also exists with epoch {epoch_best}, step {step_best} and loss {loss_best}")
    except FileNotFoundError:
        print("Best checkpoint not found, restarting")
    except KeyError as e:
        print(f"Best checkpoint file integrity error: {e}")

    # Verify if optimizer state exists.
    print(f"Optimizer state is: {optimizer2.state}")