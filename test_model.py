import pytest
import torch

from model import SolarPowerPredictionLSTM

def test_model_exists():
    """Test if SolarPowerPredictionLSTM can be run"""
    
    model = SolarPowerPredictionLSTM()

    test_input = torch.randn(2, 12, 4)

    output = model(test_input)

    assert output.shape == (2, 1)

def test_synthetic_dataset():
    """Test if SyntheticSolarDataset generates correct data"""
    from model import SyntheticSolarDataset

    dataset = SyntheticSolarDataset(num_samples=100)

    assert len(dataset) == 100
    x, y = dataset[0]
    assert x.shape == (24, 8)
    assert y.shape == (1,)

def test_create_dataloaders():
    """Test if create_dataloaders returns train and val loaders"""
    from model import create_dataloaders

    train_loader, val_loader = create_dataloaders(batch_size=32)

    assert train_loader is not None
    assert val_loader is not None

    train_batch_x, train_batch_y = next(iter(train_loader))
    assert train_batch_x.shape == (32, 24, 8)
    assert train_batch_y.shape == (32, 1)