import torch.nn as nn
from collections import OrderedDict


class model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size = 1000,
            hidden_size = 500,
            batch_first = True,
            num_layers = 3
        )
    def forward(self, x):
        hidden = None
        out, hidden = self.lstm(x, hidden)
        return out, hidden
