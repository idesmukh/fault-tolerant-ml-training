import torch
import torch.nn as nn

def train_one_epoch(model, train_loader, optimizer, epoch):
    """Train model for one epoch and return average loss."""
    model.train()
    criterion = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        
        # Do forward pass.
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # Do backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Determine total loss.
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss