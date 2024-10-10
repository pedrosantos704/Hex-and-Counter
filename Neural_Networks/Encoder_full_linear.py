import math
import os
import time

import hexagdly
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch_geometric.data import Data, DataLoader

from Games.SCS.SCS_Game import SCS_Game


from .blocks import *


class EncoderFullLinear(nn.Module):

    def __init__(self, in_channels, policy_channels, embed_dim=68, num_trans_heads=2, num_trans_layers=1, posicional_encoding=None, hex=True):

        super().__init__(),
        self.recurrent = False


        self.in_channels = in_channels
        self.policy_channels = policy_channels

        self.embed_dim = embed_dim


        self.posicional_encoding = posicional_encoding


        # First Linear Layer
        self.linear_1 = nn.Linear(self.in_channels, embed_dim)

        # Second Linear Layer
        if posicional_encoding == "xyconcat":
            self.linear_2 = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.linear_2 = nn.Linear(embed_dim, embed_dim)

        # tranformer encoder 
        if posicional_encoding == "xyconcat":
            transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim * 2, nhead=num_trans_heads, dim_feedforward=512, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_trans_layers)
        else:
            transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_trans_heads, dim_feedforward=512, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_trans_layers)
        


        # Policy Head
        self.policy_1 = nn.Linear(self.embed_dim, 128)

        policy_filters = int(math.pow(2, math.ceil(math.log(self.policy_channels, 2))))
        self.policy_2 = nn.Linear(128, policy_filters)

        self.policy_3 = nn.Linear(policy_filters, self.policy_channels)


        # Value Head
        [256, 64, 8 , 1]
        self.value_1 = nn.Linear(self.embed_dim, 256)
        self.value_2 = nn.Linear(256, 64)
        self.value_3 = nn.Linear(64, 8)
        self.value_4 = nn.Linear(8, 1)




    def forward(self, x):
        channels, heigth, width, batch = x.size()[-3], x.size()[-2], x.size()[-1], x.size()[0]



        x = torch.reshape(x,(-1, channels, heigth*width))


        x = torch.permute(x, (0, 2, 1))

        edge_index = hexToEdges(length=width, hight=heigth)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        edge_index = edge_index.to(device)

        #print(edge_index)

        relu = nn.ReLU()
        tanh = nn.Tanh()
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()

        # Process each graph in the batch separately
        out_value = []
        out_policy = []
        for i in range(batch):

            x_i = x[i]
            x_i = self.linear_1(x_i)
            x_i = F.relu(x_i)

            if not (self.posicional_encoding is None):
                if self.posicional_encoding == "neighbours":
                    pos_enc = hexPositionalEncoding(edge_index, self.embed_dim)[:heigth*width,:]
                    #print("pos_enc:",pos_enc.size())
                    x_i = x_i + pos_enc
                elif self.posicional_encoding == "xy":
                    #print("xypos")
                    pos_enc = xyPositionalEncoding(higth=heigth, length=width, d_model=self.embed_dim)[:heigth*width,:]
                    #print("pos_enc:",pos_enc.size())
                    x_i = x_i + pos_enc
                elif self.posicional_encoding == "xyconcat":
                    #print("x_i:",x_i.size())
                    pos_enc = xyPositionalEncoding(higth=heigth, length=width, d_model=self.embed_dim)[:heigth*width,:]
                    #print("pos_enc:",pos_enc.size())
                    x_i = torch.cat((x_i, pos_enc), 1)
                    #print("after cat:", x_i.size())

                else:
                    print("WARNING: In EncoderGNN, unknown posicional encoding.")

            

            x_i = self.transformer_encoder(x_i)

            #print("after encoder:",x_i.size())



            x_i = self.linear_2(x_i)
            x_i = F.relu(x_i)

            policy = self.policy_1(x_i)
            policy = relu(policy)
            policy = self.policy_2(policy)
            policy = relu(policy)
            policy = self.policy_3(policy)

            out_policy.append(policy.unsqueeze(0))

            value = self.value_1(x_i)
            value = relu(value)
            value = self.value_2(value)
            value = relu(value)
            value = self.value_3(value)
            value = relu(value)
            value = self.value_4(value)
            value = pool(value)
            value = flatten(value)
            value = tanh(value)

            out_value.append(value.unsqueeze(0))

        out_policy = torch.cat(out_policy, dim=0)  
        out_policy = torch.permute(out_policy, (0, 2, 1))
        out_policy = torch.reshape(out_policy, (batch, self.policy_channels, heigth, width))


        out_value = torch.cat(out_value, dim=0) 
        torch.permute(out_value, (0, 2, 1))
        out_value = torch.reshape(out_value, (batch, 1))
       
        return out_policy, out_value
    


def xyPositionalEncoding(higth, length, d_model, max_len=12*12):

    posicional_encodings_x = positional_encoding(max_len, math.ceil(d_model/2))
    posicional_encodings_y = positional_encoding(max_len, math.floor(d_model/2))
    #print(posicional_encodings_x, posicional_encodings_y)

    r = torch.zeros(max_len, d_model)
    
    for y in range(higth):
        for x in range(length):       
            i = torch.cat((posicional_encodings_x[x,:], posicional_encodings_y[y,:]),0)
            cell = x + y*length
            r[cell] = i

    return r


def hexToEdges(length, hight):
    edges = torch.empty((2, 0), dtype=torch.long)

    for j in range(hight):
        for i in range(length):
            
            # up and down
            if j > 0:
                edges = torch.cat([edges, torch.tensor([[i+length*(j-1)], [i+length*j]], dtype=torch.long)], dim=1)
            if j < hight - 1:
                edges = torch.cat([edges, torch.tensor([[i+length*(j+1)], [i+length*j]], dtype=torch.long)], dim=1)


            # sides
            if i%2 == 1:
                if i > 0:
                    edges = torch.cat([edges, torch.tensor([[i-1+length*j], [i+length*j]], dtype=torch.long)], dim=1)
                    if j < length-1:
                        edges = torch.cat([edges, torch.tensor([[i-1+length*(j+1)], [i+length*j]], dtype=torch.long)], dim=1)

                if i < length-1:                    
                    edges = torch.cat([edges, torch.tensor([[i+1+length*j], [i+length*j]], dtype=torch.long)], dim=1)
                    if  j < hight - 1:
                        edges = torch.cat([edges, torch.tensor([[i+1+length*(j+1)], [i+length*j]], dtype=torch.long)], dim=1)
            else:
                if i > 0:
                    edges = torch.cat([edges, torch.tensor([[i-1+length*j], [i+length*j]], dtype=torch.long)], dim=1)
                    if j > 0:
                        edges = torch.cat([edges, torch.tensor([[i-1+length*(j-1)], [i+length*j]], dtype=torch.long)], dim=1)

                if i < length-1:        
                    edges = torch.cat([edges, torch.tensor([[i+1+length*j], [i+length*j]], dtype=torch.long)], dim=1)
                    if  j > 0:
                        edges = torch.cat([edges, torch.tensor([[i+1+length*(j-1)], [i+length*j]], dtype=torch.long)], dim=1)


    return edges

def hexPositionalEncoding(edges, d_model, max_len=12*12):
    posicional_encodings = positional_encoding(max_len, d_model)

    r = torch.zeros(max_len, d_model)


    for i in range(len(edges[0])):
        r[edges[0][i]] += posicional_encodings[edges[1][i]]

    return r

def positional_encoding(max_len, d_model):
    """
    Generates a positional encoding matrix for a given maximum length and model dimension.
    
    Parameters:
    max_len (int): The maximum length of the sequences to be encoded.
    d_model (int): The dimension of the model (number of features).

    Returns:
    torch.Tensor: A tensor of shape (max_len, d_model) containing positional encodings.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe