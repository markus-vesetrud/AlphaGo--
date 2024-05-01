import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import matplotlib.pyplot as plt
from time import time

from hex import Hex
from neural_net import ConvolutionalNeuralNet, DeepConvolutionalNeuralNet, LinearNeuralNet, LinearResidualNet
from reinforcement_learning import ReinforcementLearning
from agent import PolicyAgent
from game_interface import GameInterface


def save_datset(game_states, target_values, filename: str):
    game_states_np = np.zeros(shape=(0,game_states[0].shape[0],game_states[0].shape[1],game_states[0].shape[2]))
    target_values_np = np.zeros(shape=(0,game_states[0].shape[0]**2))
    for i in range(len(game_states)):
        game_states_np = np.append(game_states_np, game_states[i][np.newaxis,:,:,:], axis=0)
        target_values_np = np.append(target_values_np, target_values[i][np.newaxis,:], axis=0)
    with open(filename, 'wb') as f:
        np.save(f, game_states_np)
        np.save(f, target_values_np)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3f" % x))

    # -------------- Hyperparameters -------------
    # Search parameters
    board_size = 7
    exploration_weight = 1.0
    epsilon = 1.0
    epsilon_decay = 1.0
    search_iterations = 30*board_size**2
    num_games = 15
    replay_buffer_max_length = 10000

    start_epoch = 0
    total_search_count = 100

    # Policy network parameters
    learning_rate = 1e-3
    l2_regularization = 1e-5 # Set to 0 for no regularization
    batch_size = 2048
    num_epochs = 10
    log_interval = 15
    save_interval = 5
    # --------------------------------------------


    # ------------- Other Variables --------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = LinearResidualNet(board_size)
    # model = ConvolutionalNeuralNet(board_size)
    # model = DeepConvolutionalNeuralNet(board_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)



    # reinforcement_learning = ReinforcementLearning(board_size, exploration_weight, epsilon, epsilon_decay,
    #                                             search_iterations, num_games, total_search_count, 
    #                                             batch_size, num_epochs, log_interval, save_interval, loss_fn=criterion, 
    #                                             optimizer=optimizer, model=model, verbose=True, 
    #                                             start_epoch=start_epoch, replay_buffer_max_length=replay_buffer_max_length, 
    #                                             initial_replay_buffer_state=None, initial_replay_buffer_target=None)

    # for i in range(10):
    #     game_states, target_values = reinforcement_learning.simulate_games()
    #     save_datset(game_states, target_values, f'test_{i}.npy')

    # total_game_states = np.zeros(shape=(0,7,7,3))
    # total_target_values = np.zeros(shape=(0,7**2))
    # for i in range(10):
    #     with open(f'test_{i}.npy', 'rb') as f:
    #         game_states: np.ndarray = np.load(f)
    #         target_values: np.ndarray = np.load(f)
    #         total_game_states = np.append(total_game_states, game_states, axis=0)
    #         total_target_values = np.append(total_target_values, target_values, axis=0)
    # print(total_game_states.shape)
    # print(total_target_values.shape)
    # with open('7by7_1470_iter_150_games.npy', 'wb') as f:
    #     np.save(f, total_game_states)
    #     np.save(f, total_target_values)
    with open('checkpoints_residual/7by7_490iter_145_replay_buffer.npy', 'rb') as f:
        game_states: np.ndarray = np.load(f)
        target_values: np.ndarray = np.load(f)
    # for i in range(0, game_states.shape[0], 2):
    #     print(target_values[i,:].reshape((board_size, board_size)))
    #     Hex(board_size, game_states[i,:,:,:2]).display_current_state()
    
    
    print(game_states.shape)

    model.load_state_dict(torch.load('checkpoints/7by7_980iter_160_model.pt', map_location=torch.device(device)))

    model.to(device)
    # model.train()
    # dataset = TensorDataset(torch.from_numpy(game_states).float(), torch.from_numpy(target_values))
    # random_minibatch_sampler = RandomSampler(dataset, num_samples=batch_size)
    # data_loader = DataLoader(dataset, sampler=random_minibatch_sampler, batch_size=batch_size)
    # loss_history = []

    # for epoch in range(1, num_epochs+1):
    #     for i, (data, target) in enumerate(data_loader):
    #         data = data.permute(0, 3, 1, 2).to(device)
    #         target = target.to(device)

    #         output = model(data)
    #         loss = criterion(output, target)

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         loss_history.append(float(loss))
    #         optimizer.step()

    #         if i % log_interval == 0:
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 epoch, i * batch_size, len(data_loader.dataset),
    #                 100. * i / len(data_loader), loss.item()))
    
    # plt.plot(list(range(len(loss_history))), loss_history)
    # plt.title('Loss during training')
    # plt.xlabel('Batch')
    # plt.ylabel('Loss')
    # plt.show()

    # torch.save(model.state_dict(), 'test_model.pt')
    
                
    # game: GameInterface = Hex(board_size, current_black_player=False)
    # test_agent = PolicyAgent(board_size, model, device, 0.0)
    # board, black_to_play = game.get_state(False)
    # print(test_agent.select_action(board, black_to_play, game.get_legal_actions(), verbose = True))

    # Test the agent
    game: GameInterface = Hex(board_size, current_black_player=True)
    test_agent = PolicyAgent(board_size, model, device, 0.0)
    model.eval()

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