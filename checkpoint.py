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

# Create model and optimizer.
model = TestModel()
optimizer = torch.optim.Adam(model.parameters())

# Generate state for optimizer.
x = torch.randn(8, 10)
y = model(x)
loss = y.mean()
loss.backward()
optimizer.step()

def checkpoint_exists(filepath):
    """Verify if checkpoint file exists on path."""
    return os.path.exists(filepath)

def save_checkpoint(model, optimizer, epoch, step, filepath):
    """Save model, optimizer and resumable training state."""
    checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'step': step,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    """Load model, optimizer, training state, and return position."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Resume from last save state.
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    return epoch, step

# Test save checkpoint function.
epoch = 1
step = 1
save_checkpoint(model, optimizer, epoch, step, 'checkpoint.pt')
print("Successfully saved checkpoint")

# Create second model and optimizer.
model2 = TestModel()
optimizer2 = torch.optim.Adam(model2.parameters())

# Test load checkpoint function.
if checkpoint_exists('checkpoint.pt'):
    print("Checkpoint exists, resuming from last checkpoint")
    epoch, step = load_checkpoint('checkpoint.pt', model2, optimizer2)
    print(f"Successfully loaded checkpoint from epoch {epoch} and step {step}")
else:
    print("No checkpoint exists, starting from beginning")
    epoch, step = 0, 0

# Verify if optimizer state exists.
print(f"Optimizer state is: {optimizer2.state}")