import math
import os
import time

import hexagdly
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader

from Games.SCS.SCS_Game import SCS_Game


from .blocks import *


class Gnn(nn.Module):

    def __init__(self, in_channels, policy_channels, hex=True):

        super().__init__()
        self.recurrent = False


        self.in_channels = in_channels
        self.policy_channels = policy_channels

        self.embed_dim = in_channels

        # GAT Layer
        self.gat_conv1 = GATConv(self.in_channels, self.embed_dim, heads=8, concat=True)
        self.gat_conv2 = GATConv(self.embed_dim * 8, self.embed_dim, heads=1, concat=True)

        # fully connected layer
        self.fc = nn.Linear(self.embed_dim * 8, self.embed_dim * 8)


        # Policy Head
        self.policy_head = Conv_PolicyHead(self.embed_dim, policy_channels, hex=hex)


        # Value Head
        self.value_head = Reduce_ValueHead(self.embed_dim, hex=hex)



    def forward(self, x):
        heigth, width, batch = x.size()[-2], x.size()[-1], x.size()[0]


        x = torch.reshape(x,(-1,self.in_channels, heigth*width))


        x = torch.permute(x, (0, 2, 1))



        edge_index = hexToEdges(length=width, hight=heigth)

        #print(edge_index)

        # Process each graph in the batch separately
        out = []
        for i in range(batch):

            x_i = x[i]
            print("before gat", x_i.size())
            x_i = self.gat_conv1(x_i, edge_index)
            x_i = F.relu(x_i)

            print("after gat:", x_i.size())

            x_i = self.fc(x_i)
            x_i = F.relu(x_i)

            print("after linear:",x.size())

            x_i = self.gat_conv2(x_i, edge_index)
            x_i = F.relu(x_i)
            out.append(x_i)

            print("after 2nd gat:",x.size())
        
        x = torch.stack(out)

        #print("x after GAT:", x.size())
        
        x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x, (batch, self.in_channels, heigth, width))
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value



def hexToEdges(length, hight):
    edges = torch.empty((2, 0), dtype=torch.long)

    for j in range(hight):
        for i in range(length):
            
            # up and down
            if j > 0:
                edges = torch.cat([edges, torch.tensor([[i], [length*(j-1)]], dtype=torch.long)], dim=1)
                edges = torch.cat([edges, torch.tensor([[length*(j-1)], [i]], dtype=torch.long)], dim=1)
            if j < hight - 1:
                edges = torch.cat([edges, torch.tensor([[i], [length*(j+1)]], dtype=torch.long)], dim=1)
                edges = torch.cat([edges, torch.tensor([[length*(j+1)], [i]], dtype=torch.long)], dim=1)


            # sides
            if i%2 == 1:
                if i > 0:
                    edges = torch.cat([edges, torch.tensor([[i-1], [length*j]], dtype=torch.long)], dim=1)
                    edges = torch.cat([edges, torch.tensor([[length*j], [i-1]], dtype=torch.long)], dim=1)
                    if j < length-1:
                        edges = torch.cat([edges, torch.tensor([[i-1], [length*(j+1)]], dtype=torch.long)], dim=1)
                        edges = torch.cat([edges, torch.tensor([[length*(j+1)], [i-1]], dtype=torch.long)], dim=1)

                if i < length-1:                    
                    edges = torch.cat([edges, torch.tensor([[i+1], [length*j]], dtype=torch.long)], dim=1)
                    edges = torch.cat([edges, torch.tensor([[length*j], [i+1]], dtype=torch.long)], dim=1)
                    if  j < hight - 1:
                        edges = torch.cat([edges, torch.tensor([[i+1], [length*(j+1)]], dtype=torch.long)], dim=1)
                        edges = torch.cat([edges, torch.tensor([[length*(j+1)], [i+1]], dtype=torch.long)], dim=1)
            else:
                if i > 0:
                    edges = torch.cat([edges, torch.tensor([[i-1], [length*j]], dtype=torch.long)], dim=1)
                    edges = torch.cat([edges, torch.tensor([[length*j], [i-1]], dtype=torch.long)], dim=1)
                    if j > 0:
                        edges = torch.cat([edges, torch.tensor([[i-1], [length*(j-1)]], dtype=torch.long)], dim=1)
                        edges = torch.cat([edges, torch.tensor([[length*(j-1)], [i-1]], dtype=torch.long)], dim=1)

                if i < length-1:        
                    edges = torch.cat([edges, torch.tensor([[i+1], [length*j]], dtype=torch.long)], dim=1)
                    edges = torch.cat([edges, torch.tensor([[length*j], [i+1]], dtype=torch.long)], dim=1)            
                    if  j > 0:
                        edges = torch.cat([edges, torch.tensor([[i+1], [length*(j-1)]], dtype=torch.long)], dim=1)
                        edges = torch.cat([edges, torch.tensor([[length*(j-1)], [i+1]], dtype=torch.long)], dim=1)


    return edges