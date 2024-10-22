import argparse
import copy
import glob
import math
import os
import pickle
import random
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import ray
import torch
from progress.bar import ChargingBar
from ray.runtime_env import RuntimeEnv
from scipy.special import softmax

import Interactive
from Agents.Generic.MctsAgent import MctsAgent
from Agents.Generic.PolicyAgent import PolicyAgent
from Agents.Generic.RandomAgent import RandomAgent
from Agents.SCS.GoalRushAgent import GoalRushAgent
from AlphaZero import AlphaZero
from Gamer import Gamer
from Games.SCS.SCS_Game import SCS_Game
from Games.SCS.SCS_Renderer import SCS_Renderer
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe
from Neural_Networks.ConvNet import ConvNet
from Neural_Networks.FullTransformer import FullTransformer
from Neural_Networks.JustConv import JustConv
from Neural_Networks.LSTMNet import LSTMNet
from Neural_Networks.MLP_Network import MLP_Network as MLP_Net
from Neural_Networks.RecurrentNet import RecurrentNet
from Neural_Networks.ResNet import ResNet
from Neural_Networks.Torch_NN import Torch_NN
from Neural_Networks.TransConvNet import TransConvNet
from Neural_Networks.TransformerNet import TransformerNet
from Neural_Networks.TransLSTMNet import TransLSTMNet
from Neural_Networks.TransParallel import TransParallel
from Neural_Networks.TransPosPar import TransPosPar
from Neural_Networks.TransGNN.Model import TransGNN
from Neural_Networks.Gnn import Gnn
from Neural_Networks.Gnn_encoder import EncoderGnn
from Neural_Networks.Gcn_encoder import EncoderGcn
from Neural_Networks.Encoder_linear import EncoderLinear
from Neural_Networks.Encoder_full_linear import EncoderFullLinear
from RemoteStorage import RemoteStorage
from RemoteTester import RemoteTester
from ReplayBuffer import ReplayBuffer
from Tester import Tester
from TestManager import TestManager
from Utils.Caches.DictCache import DictCache
from Utils.Caches.KeylessCache import KeylessCache
from Utils.general_utils import *
from Utils.Progress_Bars.PrintBar import PrintBar

from ruamel.yaml import YAML


def main():
    pid = os.getpid()
    process = psutil.Process(pid)

    parser = argparse.ArgumentParser()
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "--interactive", action='store_true',
        help="Create a simple training setup interactivly"
    )
    exclusive_group.add_argument(
        "--training-preset", type=int,
        help="Choose one of the preset training setups"
    )
    exclusive_group.add_argument(
        "--testing-preset", type=int,
        help="Choose one of the preset testing setups"
    )
    exclusive_group.add_argument(
        "--debug", type=int,
        help="Choose one of the debug modes"
    )
    parser.add_argument(
        "--log-driver", action='store_true',
        help="log_to_driver=True"
    )
    

    args = parser.parse_args()

    print("\n\nCUDA Available: " + str(torch.cuda.is_available()))

    log_to_driver = False
    if args.log_driver:
        log_to_driver = True

    if args.training_preset is not None:
        
        ##############################################################################################################
        # ---------------------------------------------------------------------------------------------------------- #
        # ------------------------------------------   TRAINING-PRESETS   ------------------------------------------ #
        # ---------------------------------------------------------------------------------------------------------- #
        ##############################################################################################################

        match args.training_preset:

            case 0: # Tic_tac_toe example
                game_class = tic_tac_toe
                game_args_list = [[]]
                game = game_class()

                alpha_config_path="Configs/Config_Files/Training/small_test_training_config.yaml"
                search_config_path="Configs/Config_Files/Search/test_search_config.yaml"

                state_set = create_r_unbalanced_state_set(game)

                model = MLP_Net(out_features=9)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model, state_set=state_set)
                alpha_zero.run()
                
            case 1: # SCS_Example
                
                game_class = SCS_Game
                game_args_list = [["Games/SCS/Game_configs/solo_soldier_config_4.yml"]]#, ["Games/SCS/Game_configs/solo_soldier_config_5.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t4.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t5.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t7.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t9.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/change_turns/solo_soldier_config_5_t6.yml"],["Games/SCS/Game_configs/change_turns/solo_soldier_config_5_t8.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/unbalanced_config_5.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/mirrored_config_5.yml"]]
                game = game_class(*game_args_list[0])

                alpha_config_path="Configs/Training/Examples/SCS_example_training_config.yaml"
                search_config_path="Configs/Search/Examples/SCS_example_search_config.yaml"

                #state_set = create_solo_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = ConvNet(in_channels, policy_channels, kernel_size=3, num_filters=256, num_layers=2, hex=True)
                #model = RecurrentNet(in_channels, policy_channels, in_channels, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="tanh", hex=True)
                #model = ResNet(in_channels, policy_channels, num_filters=128, num_blocks=2, kernel_size=1, policy_head="conv", value_head="reduce", hex=True)
                #model = MLP_Net(out_features=192)
                #model = TransformerNet(in_channels, policy_channels, num_filters=128, hex=True, num_heads=1, num_layers=1, kernel_size=1, recall=False)
                #model = TransConvNet(in_channels, policy_channels, num_filters=256, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1, recall=True)
                #model = TransLSTMNet(in_channels, policy_channels, num_heads=1, num_trans_layers=1, num_LSTM_layers=4,hex=True, recall=False)
                #model = JustConv(in_channels, policy_channels, num_filters=128, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1, recall=False)
                #model = TransParallel(in_channels, policy_channels, num_filters=128, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1)
                #model = TransPosPar(in_channels, policy_channels, num_filters=256, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1)
                #model = FullTransformer(in_channels, policy_channels, num_heads=1, num_enc_layers=1, num_dec_layers=1, pre_trans_layer=False, hex=True)
                #model = TransGNN()
                #model = Gnn(in_channels, policy_channels)
                #model = EncoderGnn(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xy")
                #model = EncoderGcn(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xy")
                #model = EncoderLinear(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding=None)
                #model = EncoderFullLinear(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xy")

                #alpha_config_path="Configs/Training/mirror_map_training_config.yaml"
                #search_config_path="Configs/Search/mirror_map_search_config.yaml"
                #model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu", hex=True)


                #initialize_parameters(model)
                #initialze_transformer(model)
                #initialize_xavier(model)
                initialize_all(model)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model) #, state_set=state_set)
                alpha_zero.run()

                
            ###############  CUSTOM PRESETS  ###################
                
            case 2:
                game_class = SCS_Game
                game_args_list = [["Games/SCS/Game_configs/solo_soldier_config_4.yml"]]#, ["Games/SCS/Game_configs/solo_soldier_config_5.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t4.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t5.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t7.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t9.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/change_turns/solo_soldier_config_5_t6.yml"],["Games/SCS/Game_configs/change_turns/solo_soldier_config_5_t8.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/unbalanced_config_5.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/mirrored_config_5.yml"]]
                game = game_class(*game_args_list[0])

                alpha_config_path="Configs/Training/Examples/SCS_example_training_config.yaml"
                search_config_path="Configs/Search/Examples/SCS_example_search_config.yaml"

                #state_set = create_solo_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                #model = ConvNet(in_channels, policy_channels, kernel_size=1, num_filters=256, hex=True)
                #model = TransformerNet(in_channels, policy_channels, num_filters=128, hex=True, num_heads=1, num_layers=1, kernel_size=1, recall=False)
                model = TransConvNet(in_channels, policy_channels, num_filters=256, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1, recall=True, positional_encoding="xyconcat")
                #model = TransLSTMNet(in_channels, policy_channels, num_heads=1, num_trans_layers=1, num_LSTM_layers=4,hex=True, recall=False)
                #model = TransParallel(in_channels, policy_channels, num_filters=128, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1)
                #model = TransPosPar(in_channels, policy_channels, num_filters=256, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1, positional_encoding="xyconcat")
                #model = EncoderGnn(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xy")
                #model = EncoderLinear(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xyconcat")
                #model = EncoderFullLinear(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xy")

                initialize_all(model)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model) #, state_set=state_set)
                alpha_zero.run()


            case 3:
                game_class = SCS_Game
                game_args_list = [["Games/SCS/Game_configs/solo_soldier_config_4.yml"]]#, ["Games/SCS/Game_configs/solo_soldier_config_5.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t4.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t5.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t7.yml"], ["Games/SCS/Game_configs/change_turns/solo_soldier_config_4_t9.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"], ["Games/SCS/Game_configs/solo_soldier_config_4.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/change_turns/solo_soldier_config_5_t6.yml"],["Games/SCS/Game_configs/change_turns/solo_soldier_config_5_t8.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/unbalanced_config_5.yml"]]
                #game_args_list = [["Games/SCS/Game_configs/mirrored_config_5.yml"]]
                game = game_class(*game_args_list[0])

                alpha_config_path="Configs/Training/Examples/SCS_example_training_config.yaml"
                search_config_path="Configs/Search/Examples/SCS_example_search_config.yaml"

                #state_set = create_solo_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = ConvNet(in_channels, policy_channels, kernel_size=1, num_filters=256, hex=True)
                #model = TransformerNet(in_channels, policy_channels, num_filters=128, hex=True, num_heads=1, num_layers=1, kernel_size=1, recall=False)
                #model = TransConvNet(in_channels, policy_channels, num_filters=256, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1, recall=True)
                #model = TransLSTMNet(in_channels, policy_channels, num_heads=1, num_trans_layers=1, num_LSTM_layers=4,hex=True, recall=False)
                #model = TransParallel(in_channels, policy_channels, num_filters=128, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1)
                #model = TransPosPar(in_channels, policy_channels, num_filters=256, hex=True, num_heads=1, num_layers=1, convolution=True, kernel_size=1)
                #model = EncoderGnn(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xy")
                #model = EncoderLinear(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xyconcat")
                #model = EncoderFullLinear(in_channels, policy_channels, num_trans_heads=1, embed_dim=68, posicional_encoding="xy")

                initialize_all(model)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model) #, state_set=state_set)
                alpha_zero.run()


            case 4:
                # Define your setup here
                exit()

            case _:
                print("\n\nUnknown training preset")
                exit()

    
    elif args.testing_preset is not None:

        ##############################################################################################################
        # ---------------------------------------------------------------------------------------------------------- #
        # ------------------------------------------   TESTING-PRESETS   ------------------------------------------- #
        # ---------------------------------------------------------------------------------------------------------- #
        ##############################################################################################################

        match args.testing_preset:
            case 0: # Render Example
                game_class = SCS_Game
                game_args = ["Games/SCS/Game_configs/solo_soldier_config_5.yml"]
                #game_args = ["Games/SCS/Game_configs/unbalanced_config_8.yml"]
                #game_args = ["Games/SCS/Game_configs/mirrored_config_8.yml"]

                test_config = "Configs/Testing/Examples/render_config.yaml"
                test_manager = TestManager(game_class, game_args)
                results = test_manager.test_from_config(test_config_path=test_config)
                
                #print(results)
                yaml_parser = YAML()
                test_config = load_yaml_config(yaml_parser, test_config)
                filename = test_config["Agents"]["p2_agent"]["Network"]["Checkpoint"]["cp_network_name"] + "_test5x5"
                print("plot saved:",filename) 
                plot_variable(results, filename=filename, path="/home/users/clarapais/Desktop/Tese/Results_graphs", show=False, plot_p2=False)



            case 1: # Extrapolation Test Example
                game_class = SCS_Game
                game_args = ["Games/SCS/Game_configs/solo_soldier_config_4.yml"]

                test_config = "Configs/Testing/Examples/extrapolation_config.yaml"
                test_manager = TestManager(game_class, game_args)
                results = test_manager.test_from_config(test_config_path=test_config)
                
                #'''
                x, y = zip(*results)
                plt.plot(x, y)
                plt.show()
                #'''


            ###############  CUSTOM PRESETS  ###################

            case 2:
                pass

            case _:
                raise Exception("Unknown testing preset.")

        
        
            
    elif args.debug is not None:
        match args.debug:
            
            case 0: # Test initialization
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/randomized_config_5.yml"]
                game = game_class(*game_args)


                #nn, search_config = load_trained_network(game, "adam_se_mse_mirror", 130)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")
                #model = ResNet(in_channels, policy_channels, num_filters=256, num_blocks=20, policy_head="conv", value_head="dense")

                print(model)
                #'''
                for name, param in model.named_parameters():
                    #print(name)
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param, gain=0.75)
                    
                #'''
                nn = Torch_NN(model)

                
                play_actions = 9
                for _ in range(play_actions):
                    valid_actions_mask = game.possible_actions()
                    valid_actions_mask = valid_actions_mask.flatten()
                    n_valids = np.sum(valid_actions_mask)
                    probs = valid_actions_mask/n_valids
                    action_i = np.random.choice(game.num_actions, p=probs)
                    action_coords = np.unravel_index(action_i, game.action_space_shape)
                    game.step_function(action_coords)

                
                state = game.generate_state_image()

                policy, value = nn.inference(state, False, 30)
                

                print("\n\n")
                #print(policy)
                print("\n\n")
                print(torch.sum(policy))
                print(value)
                print("\n\n----------------\n\n")
                
                all_weights = torch.Tensor().cpu()
                for param in nn.get_model().parameters():
                    #print(param)
                    all_weights = torch.cat((all_weights, param.clone().detach().flatten().cpu()), 0) 

                print(all_weights)

                
                

            case 1:
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/randomized_config_5.yml"]
                game = game_class(*game_args)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")

                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,20,100], gamma=0.1)

                file_name = "optimizer"
                torch.save(optimizer.state_dict(), file_name)
                optimizer_sd = torch.load(file_name)
                

                print("\n\n\n\n\n")

                file_name = "scheduler"
                torch.save(scheduler.state_dict(), file_name)
                loaded_data = torch.load(file_name)
                print(loaded_data)
                os.remove(file_name)



    elif args.interactive:
        Interactive.start_interative()


    
    return

def initialize_all(model):
    for name, param in model.named_parameters():
        print(name)
        if "bias" in name:
            torch.nn.init.zeros_(param.data)
        elif param.data.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            #torch.nn.init.normal_(param)
            torch.nn.init.uniform_(param.data)


def initialize_parameters(model):
    for name, param in model.named_parameters():
        if ".weight" not in name:
            #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
            torch.nn.init.xavier_uniform_(param)

def initialze_transformer(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            torch.nn.init.kaiming_uniform_(param)

def initialize_xavier(model):
    for m in model.modules():
        for name, param in m.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name and param.data.dim() == 2:
                torch.nn.init.zeros_(param.data)



##########################################################################
# ----------------------------               --------------------------- #
# -----------------------       STATE SETS       ----------------------- #
# ----------------------------               --------------------------- #
##########################################################################

def create_mirrored_state_set(game):
    renderer = SCS_Renderer()

    state_set = []
    game.reset_env()
    game.set_simple_game_state(9, [1], [(0,1)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(9, [1,1,1], [(0,1),(1,1),(0,0)], [2,2,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(9, [1], [(4,4)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    
    game.reset_env()
    game.set_simple_game_state(9, [1,1,1,1], [(0,1),(0,1),(0,0),(0,0)], [2,2,1,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(9, [1,1,1], [(4,3),(3,3),(4,4)], [1,1,2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(9, [1], [(4,4)], [1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    

    game.reset_env()
    return state_set

def create_unbalanced_state_set(game):
    renderer = SCS_Renderer()

    state_set = []

    game.reset_env()
    game.set_simple_game_state(7, [1], [(0,1)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1,1,1], [(0,1),(1,1),(0,0)], [2,2,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(4,4)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    
    game.reset_env()
    game.set_simple_game_state(7, [1,1], [(2,2),(2,1)], [2,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(3,0)], [1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(4,4)], [1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    

    game.reset_env()
    return state_set

def create_r_unbalanced_state_set(game):
    renderer = SCS_Renderer()

    state_set = []

    game.reset_env()
    game.set_simple_game_state(7, [1], [(1,2)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1,1], [(0,1),(4,3)], [2,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    
    game.reset_env()
    game.set_simple_game_state(7, [1,1], [(2,3),(3,3)], [1,2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1,1,1], [(1,4),(2,2),(2,3)], [1,1,2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(1,4)], [1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1,1], [(4,3),(4,3)], [1,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    

    game.reset_env()
    return state_set

def create_solo_state_set(game):
    renderer = SCS_Renderer()

    state_set = []
    game.reset_env()
    game.set_simple_game_state(7, [1], [(3,3)], [2], [[(3,3)],[(1,1)]])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    
    game.reset_env()
    game.set_simple_game_state(7, [1], [(0,0)], [2], [[(0,0)],[(1,1)]])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(0,0)], [2], [[(3,3)],[(1,1)]])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(3,3)], [2], [[(0,0)],[(1,1)]])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    return state_set

def plot_variable(data, filename="plot", path="", show=True, plot_p2=True):
    x = [item[0] for item in data]  
    y1 = [item[1][0] for item in data]  
    y2 = [item[1][1] for item in data] 
    y3 = [item[1][2] for item in data]  

   
    plt.figure(figsize=(10, 6)) 
    
    plt.plot(x, y2, color='g', label='P2 win ratio') 
    if plot_p2:
        plt.plot(x, y1, color='r', label='P1 win ratio') 
        plt.plot(x, y3, color='y', label='Draw')
    plt.ylim(0, 1)

    plt.legend()
    


    if show:
        plt.show() 
    else:
        full_path = path + "/" + filename
        plt.savefig(full_path)
        plt.close()




##########################################################################
# ----------------------------               --------------------------- #
# ---------------------------       RAY       -------------------------- #
# ----------------------------               --------------------------- #
##########################################################################

def start_ray_local(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    context = ray.init(log_to_driver=log_to_driver)
    return context

def start_ray_local_cluster(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    runtime_env=RuntimeEnv \
					(
					working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
					pip="./requirements.txt"
					)
		
    context = ray.init(address='auto', runtime_env=runtime_env, log_to_driver=log_to_driver)
    return context

def start_ray_rnl(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    '''
    env_vars={"CUDA_VISIBLE_DEVICES": "-1",
            "LD_LIBRARY_PATH": "$NIX_LD_LIBRARY_PATH"
            }
    '''

    runtime_env=RuntimeEnv \
					(
					working_dir="/mnt/cirrus/users/5/2/ist189452/TESE/NuZero",
					pip="./requirements.txt",
					)
		
    context = ray.init(address='auto', runtime_env=runtime_env, log_to_driver=log_to_driver)
    return context


# ------------------------

if __name__ == "__main__":
    main()
