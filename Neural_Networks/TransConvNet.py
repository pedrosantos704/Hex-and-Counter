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


class TransConvNet(nn.Module):

    def __init__(self, in_channels, policy_channels, kernel_size=1, num_filters=256, num_heads=1, num_layers=1, hex=True, convolution=True, recall=True, positional_encoding=None):

        super().__init__()
        self.recurrent = False

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        self.in_channels = in_channels
        self.policy_channels = policy_channels

        self.recall = recall
        self.convolution = convolution

        self.positional_encoding = positional_encoding

        self.d_model = in_channels

        if self.convolution:
            self.embed_dim = num_filters
        else:
            self.embed_dim = in_channels

        if self.recall:
            self.conv_input_dim = in_channels*2
        else:
            self.conv_input_dim = in_channels

        if self.positional_encoding == "xyconcat":
            self.conv_input_dim = in_channels*3
            self.d_model = in_channels*2


        # !! in_channels must be divisible by num_heads
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dim_feedforward=512, batch_first=True)
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
        heigth, width, batch = x.size()[-2], x.size()[-1], x.size()[0]
    
        print("h,w:",heigth, width)
        print("input:",x.size())

        input = x

        #test = x.detach().clone()

        
        # each word corresponds to a board location
        x_enc = torch.reshape(x,(-1,self.in_channels, heigth*width))
        x_enc = torch.permute(x_enc, (0, 2, 1))
        
        #input_data = x.detach().clone()

        pos_enc = xyPositionalEncoding(higth=heigth, length=width, d_model=self.in_channels)[:heigth*width,:]
        for i in range(batch):
            if self.posicional_encoding == "xy":
                x_enc[i] = x_enc[i] + pos_enc

            elif self.positional_encoding == "xyconcat":                
                x_enc[i] = torch.cat((x_enc[i], pos_enc), 1)

        print("before encoder:", x_enc.size())

        x_enc = self.transformer_encoder(x_enc)
        print("after encoder:", x_enc.size())
        
        #print_attention_info(self.transformer_encoder, input_data)
        x_enc = torch.permute(x_enc, (0, 2, 1)) 
        x_enc = torch.reshape(x_enc,(-1,self.in_channels,heigth,width))
        
        #print(torch.eq(test, x))
        print("before recall", x_enc.size())

        if self.recall:
            x = torch.cat((input, x_enc), 1)

        if self.convolution:
            print("before convolution:", x.size())
            
            x = self.general_module(x)

            print("after convolution:", x.size())

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    


def xyPositionalEncoding(higth, length, d_model, max_len=12*12):
    #print("xy pos enc:", math.ceil(d_model/2), math.floor(d_model/2) )

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
    
    # Calculate the divisor term for even indices.
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    if d_model % 2 == 1:  # If d_model is odd
        pe[:, 1:d_model:2] = torch.cos(position * div_term[:-1])  # Drop the last element of div_term
    else:
        pe[:, 1::2] = torch.cos(position * div_term)


    return pe

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




    




    