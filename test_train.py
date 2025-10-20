import pytest
import torch
import os
import shutil

def test_train_one_epoch():
    """Test that training can complete one epoch and save checkpoint"""
    from train import train_one_epoch
    from model import SolarPowerPredictionLSTM, create_dataloaders

    model = SolarPowerPredictionLSTM()
    optimizer = torch.optim.Adam(model.parameters())
    train_loader, _ = create_dataloaders(batch_size=32)

    checkpoint_dir = './test_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        avg_loss = train_one_epoch(model, train_loader, optimizer, epoch=0)

        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    finally:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

def test_checkpoint_integration():
    """Test to see if checkpointing works"""
    from train import train_with_checkpointing
    from model import SolarPowerPredictionLSTM, create_dataloaders
    from checkpoint import load_checkpoint
    import os
    import shutil

    checkpoint_dir = './test_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        train_with_checkpointing(num_epochs=2, checkpoint_dir=checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        assert os.path.exists(checkpoint_path)

        model = SolarPowerPredictionLSTM()
        optimizer = torch.optim.Adam(model.parameters())
        epoch, step, loss, timestamp, batch_idx = load_checkpoint('latest', model, optimizer, checkpoint_dir)

        assert epoch == 2
    
    finally:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

def test_training_fault_tolerance():
    """Test if training can recover after fault."""
    from train import train_with_checkpointing
    from checkpoint import load_checkpoint
    import os
    import shutil

    checkpoint_dir = './test_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        train_with_checkpointing(num_epochs=2, checkpoint_dir=checkpoint_dir)

        train_with_checkpointing(num_epochs=5, checkpoint_dir=checkpoint_dir)

        from model import SolarPowerPredictionLSTM
        model = SolarPowerPredictionLSTM()
        optimizer = torch.optim.Adam(model.parameters())
        epoch, step, loss, timestamp, batch_idx = load_checkpoint(
            'latest', model, optimizer, checkpoint_dir
        )

        assert epoch == 5
    
    finally:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)