import torch
import torch.nn as nn
from model.cnn import model as cnn_encoder
from model.lstm import model as lstm_decoder
from collections import OrderedDict
import torch.nn.functional as F

class model(nn.Module):

    def __init__(self):
        # in_channels : input image features
        # out_channels : output features, equals number of classes
        super().__init__()

        self.encoder = cnn_encoder()
        self.decoder = lstm_decoder()

        self.fc1 = nn.Linear(500, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,6)

        self.relu = F.relu

        self.softmax = F.softmax

    def forward(self, video):
        for frames in video:
            features = self.encoder(frames)
            output, hidden = self.decoder(features)                
            
        x = self.relu(self.fc1(output))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))

        return x
        


        




