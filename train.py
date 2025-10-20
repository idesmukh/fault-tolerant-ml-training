import torch
import torch.nn as nn
import os
from model import SolarPowerPredictionLSTM, create_dataloaders
from checkpoint import save_checkpoint, load_checkpoint

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

def train_with_checkpointing(num_epochs=10, checkpoint_dir='./checkpoints', batch_size=32):
    """Train model with automatic checkpointing and recovery."""

    model = SolarPowerPredictionLSTM()
    optimizer = torch.optim.Adam(model.parameters())
    train_loader, val_loader = create_dataloaders(batch_size=batch_size)

    # Resume from checkpoint.
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, step, loss, timestamp, batch_idx = load_checkpoint(
            'latest', model, optimizer, checkpoint_dir
        )
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from beginnning")

    # Setup training loop.
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train over one epoch.
        avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        print(f"Average loss: {avg_loss: .4f}")

        # After every epoch, save checkpoint.
        save_checkpoint(
            model,
            optimizer,
            epoch=epoch + 1,
            step=0,
            loss=avg_loss,
            checkpoint_dir=checkpoint_dir
        )
        print(f"Checkpoint saved at epoch {epoch + 1}")

    print(f"\nTraining completed with final epoch: {num_epochs}")