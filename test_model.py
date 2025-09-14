import pytest
import torch

def test_model_exists():
    from model import SolarPowerPredictionLSTM
    assert SolarPowerPredictionLSTM is not None