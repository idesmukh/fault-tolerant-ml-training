import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SolarPowerPredictionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class SyntheticSolarDataset(Dataset):
    """Generate synthetic data for testing."""

    def __init__(self, num_samples=10000):
        self.X = torch.randn(num_samples, 24, 8)
        self.y = torch.randn(num_samples, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]