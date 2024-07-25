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


class FullTransformer(nn.Module):

    def __init__(self, in_channels, policy_channels, num_heads=1, num_enc_layers=1, num_dec_layers=1, pre_trans_layer=False, hex=True):

        super().__init__()
        self.recurrent = False


        self.in_channels = in_channels
        self.policy_channels = policy_channels

        self.embed_dim = in_channels

        self.pre_trans_layer = pre_trans_layer

        if pre_trans_layer:
            self.num_filters = 128
            self.kernel_size = 1
            general_layer_list = []
            general_layer_list.append(nn.Conv1d(kernel_size=self.kernel_size, in_channels=self.in_channels, out_channels=self.num_filters, bias=False))
            general_layer_list.append(nn.ELU())
            general_layer_list.append(nn.Conv1d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.in_channels, bias=False))
            general_layer_list.append(nn.ELU())


            self.general_module = nn.Sequential(*general_layer_list)



        self.transformer = nn.Transformer(d_model=in_channels, nhead=num_heads, num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers, batch_first=True)



        # Policy Head
        self.policy_head = Conv_PolicyHead(self.embed_dim, policy_channels, hex=hex)


        # Value Head
        self.value_head = Reduce_ValueHead(self.embed_dim, hex=hex)



    def forward(self, x):
        heigth, width, batch = x.size()[-2], x.size()[-1], x.size()[0]

        #print("h,w:",heigth, width)
        #print("input:",x.size())

        # each word corresponds to a board location
        x = torch.reshape(x,(-1,self.in_channels, heigth*width))


        #print("before conv:", x.size())

        if self.pre_trans_layer:
            x = self.general_module(x)

        x = torch.permute(x, (0, 2, 1))

        #input_data = x.detach().clone()

        #print("before encoder:", x.size())

        tgt_seq = torch.zeros(batch, heigth*width, self.in_channels)
        #print("tgt:",tgt_seq.size())
        x = self.transformer(x, tgt_seq)
        #print("after encoder:", x.size())

        #print_attention_info(self.transformer_encoder, input_data)

        x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x,(-1,self.in_channels,heigth,width))


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
