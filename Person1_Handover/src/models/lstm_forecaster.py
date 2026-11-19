import torch
from torch import nn

class LSTMForecaster(nn.Module):
    """Maps [batch, input_window, features] to [batch, horizon, features]."""
    def __init__(self, input_size, hidden_size=80, num_layers=2, dropout=0.3, horizon=10):
        super().__init__()
        self.horizon = horizon
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )
        self.head = nn.Linear(hidden_size, horizon * input_size)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        last = sequence[:, -1, :]
        forecast = self.head(last)
        return forecast.view(x.size(0), self.horizon, self.input_size)
