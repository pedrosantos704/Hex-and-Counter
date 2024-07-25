import math
import os
import time

import hexagdly
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from Games.SCS.SCS_Game import SCS_Game

from .blocks import *


class LSTMNet(nn.Module):

    def __init__(self, in_channels, policy_channels, lstm_hidden_dim = 256, lstm_layers = 1, hex=True):

        super().__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)



        # Policy Head
        self.policy_head = Conv_PolicyHead(self.embed_dim, policy_channels, hex=hex)

        # Value Head
        self.value_head = Reduce_ValueHead(self.embed_dim, hex=hex)


    



    def forward(self, x):


        lstm_out, (h_n, c_n) = self.lstm(x) # lstm_out shape: (batch_size, sequence_length, lstm_hidden_dim)
        
        # last output of the LSTM
        lstm_out = lstm_out[:, -1, :]  # shape: (batch_size, lstm_hidden_dim)


        policy = self.policy_head(x)
        value = self.value_head(x)


        return policy, value
    




