import torch
import numpy as np
from time import time

from hex import Hex
from game.neural_net import ConvolutionalNeuralNetOld, LinearResidualNetOld
from agent import PolicyAgent
from game_interface import GameInterface


if __name__ == '__main__':
    np.set_printoptions(suppress=True, edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3f" % x))

    board_size = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    model = LinearResidualNetOld(board_size)
    model.load_state_dict(torch.load('holy_grail/7by7_490iter_145_model.pt', map_location=torch.device(device)))
    # model = ConvolutionalNeuralNetOld(board_size)
    # model.load_state_dict(torch.load('checkpoints_conv_bad/7by7_735iter_50_model.pt', map_location=torch.device(device)))


    model.to(device)
    model.eval()

    # Test the agent
    game: GameInterface = Hex(board_size, current_black_player=True)
    test_agent = PolicyAgent(board_size, model, device, 0.0)

    game_length = 0
    while not game.is_final_state():
        game_length += 1
        board, black_to_play = game.get_state(False)
        action = test_agent.select_action(board, black_to_play, game.get_legal_actions(), verbose = True)
        print(action)
        game.display_current_state()
        game.perform_action(action)
    
    print(game_length)
    print(game.get_final_state_reward())
