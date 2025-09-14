import pytest
import torch

from model import SolarPowerPredictionLSTM

def test_model_exists():
    """Test if SolarPowerPredictionLSTM can be run"""
    
    model = SolarPowerPredictionLSTM()

    test_input = torch.randn(2, 12, 4)

    output = model(test_input)

    assert output.shape == (2, 1)