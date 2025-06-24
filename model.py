# model.py

import torch.nn as nn
import torch

class xLSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_sizes=[32, 64, 128], dropout=0.2):
        super().__init__()
        self.lstm_blocks = nn.ModuleList([
            nn.LSTM(input_size, h, batch_first=True) for h in hidden_sizes
        ])
        self.fc = nn.Sequential(
            nn.Linear(sum(hidden_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        outputs = []
        for lstm in self.lstm_blocks:
            _, (h_n, _) = lstm(x)
            outputs.append(h_n[-1])
        x = torch.cat(outputs, dim=1)
        return self.fc(x)
