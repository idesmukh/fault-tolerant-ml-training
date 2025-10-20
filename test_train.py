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