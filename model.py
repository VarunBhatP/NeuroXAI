import torch
import torch.nn as nn
import torch.nn.functional as F

class SeizureModel(nn.Module):
    def __init__(self,dropout_rate=0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(22, 64, 25)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = SeizureModel()
