# Copyright (c) 2025 Ibad Desmukh
#
# SPDX-License-Identifier: MIT
#
"""Implement and test basic checkpointing.

Creates a basic layer, checkpoints it and loads it back.
"""

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

def save_checkpoint(model, optimizer, filepath):
    """Save model and optimizer."""
    checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    """Load model and optimizer."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Test save checkpoint function.
save_checkpoint(model, optimizer, 'checkpoint.pt')
print("Successfully saved checkpoint")

# Create second model and optimizer.
model2 = TestModel()
optimizer2 = torch.optim.Adam(model2.parameters())

# Test load checkpoint function.
load_checkpoint('checkpoint.pt', model2, optimizer2)
print("Successfully loaded checkpoint")

# Verify if optimizer state exists.
print(f"Optimizer state is: {optimizer2.state}")