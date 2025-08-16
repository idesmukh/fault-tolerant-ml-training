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
        super().__init__() # Initialize the parent class.
        self.layer = nn.Linear(10, 2) # Create a fully connected layer.

    def forward(self, x):
        return self.layer(x) # Pass input tensor through linear layer.

def save_checkpoint(model, optimizer, epoch, step, loss, is_best=False, checkpoint_dir='./checkpoints'):
    """Save model, optimizer and resumable training state."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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

def is_checkpoint_valid(filepath):
    """Check the data integrity of the checkpoint file."""
    if not os.path.exists(filepath):
        return False

    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'step']
        return all(key in checkpoint for key in required_keys)
    except Exception as e:
        print(f"Checkpoint validation failed: {e}")
        return False

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

    # Validate before loading.
    if not is_checkpoint_valid(filepath):
        raise ValueError(f"Checkpoint {filepath} is invalid")

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Resume from last save state.
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', 0)
    return epoch, step, loss

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

    # Testing data integrity check.
    if os.path.exists('checkpoint.pt.tmp'):
        print("Temp file found from previous interrupted checkpoint save")
        if is_checkpoint_valid('checkpoint.pt.tmp'):
            print("Temp file is valid")
        else:
            print("Temp file is invalid")

    # Test load checkpoint function from latest path.
    latest_path = './checkpoints/checkpoint_latest.pt'
    if is_checkpoint_valid(latest_path):
        print("Checkpoint exists, resuming from last checkpoint")
        epoch, step, loss = load_checkpoint('latest', model2, optimizer2)
        print(f"Successfully loaded checkpoint from epoch {epoch}, step {step} and loss {loss}")

    # Test load checkpoint function from best path.
    best_path = './checkpoints/checkpoint_best.pt'
    if is_checkpoint_valid(best_path):
        epoch_best, step_best, loss_best = load_checkpoint('best', model2, optimizer2)
        print(f"Best checkpoint also exists with epoch {epoch_best}, step {step_best} and loss {loss_best}")

    # Verify if optimizer state exists.
    print(f"Optimizer state is: {optimizer2.state}")