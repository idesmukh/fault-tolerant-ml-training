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

# Save model and optimizer.
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')
print("Saved model and optimizer to checkpoint.pt")

# Load model and optimizer.
model2 = TestModel()
optimizer2 = torch.optim.Adam(model2.parameters())

checkpoint = torch.load('checkpoint.pt')
model2.load_state_dict(checkpoint['model_state_dict'])
optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
print("Loaded model and optimizer from checkpoint.pt")

print(f"Optimizer state is: {optimizer2.state}")