import math
import os
import time

import hexagdly
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from transformer_encoder import TransformerEncoder

from Games.SCS.SCS_Game import SCS_Game

from .blocks import *


class TransParallel(nn.Module):

    def __init__(self, in_channels, policy_channels, kernel_size=1, num_filters=256, num_heads=1, num_layers=1, hex=True, convolution=True):

        super().__init__()
        self.recurrent = False

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        self.in_channels = in_channels
        self.policy_channels = policy_channels

        self.convolution = convolution

        if self.convolution:
            self.embed_dim = num_filters + in_channels
        else:
            self.embed_dim = in_channels


        self.conv_input_dim = in_channels


        # !! in_channels must be divisible by num_heads
        transformer_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)


        if self.convolution:
            num_layers = 1
            # General Module
            general_layer_list = []
            if hex:
                general_layer_list.append(hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.conv_input_dim, out_channels=self.num_filters, bias=False))
            else:
                general_layer_list.append(nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.conv_input_dim, out_channels=self.num_filters, bias=False))
            general_layer_list.append(nn.ELU())
            
            for i in range(num_layers):
                if hex:
                    general_layer_list.append(hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False))
                else:
                    general_layer_list.append(nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False))
                general_layer_list.append(nn.ELU())
            
            
            self.general_module = nn.Sequential(*general_layer_list)
            


        
        # Policy Head
        self.policy_head = Conv_PolicyHead(self.embed_dim, policy_channels, hex=hex)


        # Value Head
        self.value_head = Reduce_ValueHead(self.embed_dim, hex=hex)    



    def forward(self, x):
        heigth, width = x.size()[-2], x.size()[-1]
    
        print("h,w:",heigth, width)
        print("input:",x.size())

        input = x

        
        # each word corresponds to a board location
        lstm_x = torch.reshape(x,(-1,self.in_channels, heigth*width))
        lstm_x = torch.permute(lstm_x, (0, 2, 1))
        
        input_data = x.detach().clone()

        print("before encoder:", lstm_x.size())

        lstm_x = self.transformer_encoder(lstm_x)
        print("after encoder:", lstm_x.size())
        
        #print_attention_info(self.transformer_encoder, input_data)
        lstm_x = torch.permute(lstm_x, (0, 2, 1)) 
        lstm_x = torch.reshape(lstm_x,(-1,self.in_channels,heigth,width))
        
        #print(torch.eq(test, x))


        if self.convolution:
            print("before convolution:", x.size())
            
            conv_x = self.general_module(x)

            print("after convolution:", conv_x.size())
            x =  torch.cat((conv_x, lstm_x), 1)
            print("after concat:",x.size())
        else:
            x = lstm_x


        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    




def extract_self_attention_maps(transformer_encoder, input_data):
    self_attention_maps = []
    for layer in transformer_encoder.layers:

        self_attention_maps += [layer.self_attn(input_data, input_data, input_data,average_attn_weights=False, need_weights=True)[1]]      

    return self_attention_maps


def print_attention_info(transformer_encoder, input_data):
    self_attention_maps = extract_self_attention_maps(transformer_encoder, input_data)

    print("number of layers:",transformer_encoder.num_layers," number of heads:",transformer_encoder.layers[0].self_attn.num_heads)
    
    stamp = str(int(time.time()))

    with open("Attention_matrixes/input_"+stamp, 'w+') as file:
        #file.write(str(input_data.tolist()))
        #file.write(str(torch.permute(input_data,(0,2,1)).tolist()))
        file.write(pretty_state(torch.permute(input_data,(0,2,1)).tolist(), reiforcements_limit=3, stack_limit=1))


    print(len(self_attention_maps))
    for i in range(len(self_attention_maps)):
        print("Map of layer",i,":")
        for j in range(self_attention_maps[i].size()[1]):
            #print("Head",j,":")
            #print(self_attention_maps[i][0][j])            
            visualize_matrix(self_attention_maps[i][0][j], save_name="layer_"+str(i)+"head_"+str(j)+"_"+stamp)
            with open("Attention_matrixes/matr_layer_"+str(i)+"head_"+str(j)+"_"+stamp, 'w') as file:
                file.write(str(self_attention_maps[i][0][j].numpy()))
            print("Matrix saved")

            #print("sum",sum_rows(self_attention_maps[i][0][j]))


def visualize_matrix(matrix, save_folder="Attention_matrixes", save_name = "att_matr"):

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='gray', vmin=1, vmax=0)
    plt.colorbar()
    plt.savefig(save_folder + "/" + save_name)
    plt.close()


def sum_rows(matrix):
    row_sums = torch.sum(matrix, dim=1)
    return row_sums


def print_board_layer(layer, side):
    string = ""
    for i in range(len(layer)):
        string += str(layer[i]) + " "
        if (i+1) % side == 0:
            string += "\n"
    string += "----------------\n"

    return string
    

def pretty_state(state, reiforcements_limit=3, stack_limit=1):
    length = len(state[0][0])
    side_length = int(math.sqrt(length))

    string = ""

    string += str(side_length) + " by " + str(side_length) +" board\n"

    string += "\nterrain:\n"
    string += print_board_layer(state[0][0], side_length)
    string += print_board_layer(state[0][1], side_length)
    string += print_board_layer(state[0][2], side_length)

    string += "\nvictory points:\n"
    string += print_board_layer(state[0][3], side_length)
    string += print_board_layer(state[0][4], side_length)

    index = 5

    string += "\nreiforcments p1:\n"
    for i in range(reiforcements_limit):
        string += "stack "+str(i) + "\n"
        for j in range(6):
            string += print_board_layer(state[0][index + j + i*6], side_length)
            if j == 2:
                string += "duration:\n"
                
    index = index + reiforcements_limit*6

    
    string += "\nreiforcments p2:\n"
    for i in range(reiforcements_limit):
        string += "reiforcment "+str(i)+"\n"
        for j in range(6):
            string += print_board_layer(state[0][index + j + i*6], side_length)
            if j == 2:
                string += "duration:\n"

    index = index + reiforcements_limit*6
    
    string += "\nunits p1:\n"
    for i in range(3*3*stack_limit):
        string += print_board_layer(state[0][index + i], side_length)

    index = index + 3*3*stack_limit

    string += "\nunits p2:\n"
    for i in range(3*3*stack_limit):
        string += print_board_layer(state[0][index + i], side_length)

    index = index + 3*3*stack_limit

    string += "\ntarget tiles:\n"
    string += print_board_layer(state[0][index], side_length)

    index = index + 1
    
    string += "\nattackers:\n"
    for i in range(stack_limit):
        string += print_board_layer(state[0][index + i], side_length)

    index = index + stack_limit

    string += "\nsub phases (reinforcement, movement, choosing target, choosing attackers):\n"
    for i in range(4):
        string += print_board_layer(state[0][index + i], side_length)

    index = index + 4
    
    string += "\nturn plane:\n"
    string += print_board_layer(state[0][index], side_length)

    string += "\nplayer plane:\n"
    string += print_board_layer(state[0][index + 1], side_length)

    string += str(index+2)+" planes shown. Should be " + str(len(state[0]))

    return string




    




    