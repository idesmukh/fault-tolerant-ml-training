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