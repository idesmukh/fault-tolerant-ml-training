# Copyright (c) 2025 Ibad Desmukh
#
# SPDX-License-Identifier: MIT
#
"""Implement and test basic checkpointing.

Creates a basic layer, checkpoints it and loads it back.
"""

import torch
import torch.nn as nn
import time

# Make a test model.
class TestModel(nn.Module):
    def __init__(self):
        super().__init__() # Initialize the parent class.
        self.layer = nn.Linear(10, 2) # Create a fully connected layer.

    def forward(self, x): 
        return self.layer(x) # Pass input tensor through linear layer.

# Create model and save it.
model = TestModel()
torch.save(model.state_dict(), 'checkpoint.pt')
print("✓ Saved model to checkpoint.pt")

# Load model.
model2 = TestModel()
model2.load_state_dict(torch.load('checkpoint.pt'))
print("✓ Loaded model from checkpoint.pt")