import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from game_interface import GameInterface
from hex import Hex
from copy import deepcopy
from mcts import MCTreeSearch, MCTreeNode
from neural_net import ConvolutionalNeuralNet, DeepConvolutionalNeuralNet


# Hyperparameters
board_size = 5

exploration_weight = 1.0
search_iterations = 50

game_states: list[np.ndarray] = []
target_values: list[np.ndarray] = []

flip_target_values = np.ones(board_size*board_size)

input_size = 28*28
hidden_size = 500
output_size = 10
learning_rate = 0.001
batch_size = 100
num_epochs = 5
num_games = 5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
# model = LinearNeuralNet(input_size, hidden_size, output_size).to(device) # not made for hex
# model = ConvolutionalNeuralNet(board_size).to(device)
model = DeepConvolutionalNeuralNet(board_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ------------------- Game -------------------
# this part runs num_games games and collects the game states and target values (probabilities of actions)
# the opening move is only saved in one configuration, the rest of the game is saved in two configurations (original and 180 degree rotated)
# both moves are saved in the perspective of the player to move (black and red)

for _ in range(num_games):
    # game: GameInterface = Nim([10,10], 4)
    game: GameInterface = Hex(board_size)
    board, black_to_play = game.get_state(False)

    root_node = MCTreeNode(None, -1, black_to_play, game.get_legal_acions(True), game.get_action_count())

    # make first move without copies
    mcts = MCTreeSearch(root_node, game, exploration_weight, random_policy=True)

    mcts.UCTSearch(search_iterations)
    best_action = mcts.root.best_action()

    current_game_state, current_black_to_play = game.get_state(False)
    if(current_black_to_play): # should always be opening move by black
        current_target_values = root_node.action_values / np.sum(root_node.action_values)

        # adding original board
        game_states.append(deepcopy(current_game_state))
        target_values.append(current_target_values)
    
    root_node = mcts.root.get_child(best_action)
    root_node.make_root()

    game.perform_action(best_action)

    # play the rest of the game
    while not game.is_final_state():
        
        mcts = MCTreeSearch(root_node, game, exploration_weight, random_policy=True)

        mcts.UCTSearch(search_iterations)
        best_action = mcts.root.best_action()

        node_action_values = root_node.action_values

        current_game_state, current_black_to_play = game.get_state(False)
        if(current_black_to_play):
            current_target_values = node_action_values / np.sum(node_action_values)

            # adding original board
            game_states.append(deepcopy(current_game_state))
            target_values.append(current_target_values)

            # adding 180 degree rotated board
            game_states.append(deepcopy(game.get_fully_rotated_state(False)[0]))
            fully_rotated_current_target_values = np.rot90(current_target_values.reshape((board_size, board_size)), k=2).flatten()
            target_values.append(fully_rotated_current_target_values)

        else:
            mask = np.isin(np.arange(board_size**2), mcts.root.legal_actions) # replace with root_node ? 
            current_target_values = (flip_target_values - node_action_values) * mask
            current_target_values = current_target_values / np.sum(current_target_values)

            current_target_values_copy = current_target_values.reshape((board_size, board_size)).copy()
            current_target_values_copy2 = current_target_values_copy.copy()
            
            # flipping the board to red's perspective
            board_np = np.array(current_game_state)
            empty_mask = np.all(board_np == [False, False], axis=-1)
            board_np = np.logical_not(board_np)
            board_np[empty_mask] = [False, False]

            board_copy = board_np.copy()
            
            for i in range(board_size):
                for j in range(board_size):
                    board_np[board_size - j - 1][board_size - i - 1] = board_copy[i][j]
                    current_target_values_copy[board_size - j - 1][board_size - i - 1] = current_target_values_copy2[i][j]
            
            current_target_values = current_target_values_copy.flatten()

            # adding "original" board
            game_states.append(deepcopy(board_np))
            target_values.append(current_target_values)

            # adding 180 degree rotated board
            game_states.append(deepcopy(np.rot90(board_np, k=2)))
            target_values.append(np.rot90(current_target_values_copy, k=2).flatten())
        
        root_node = mcts.root.get_child(best_action)
        root_node.make_root()

        game.perform_action(best_action)
        # game_states.append(deepcopy(game))
        # game.display_current_state()


# ------------------- Convolutional neural network -------------------
# this part trains the convolutional neural network on the collected game states and target values
# the model is trained to predict the target values from the game states

# Train the model
total_step = len(game_states)
for epoch in range(num_epochs):
    for i, (game_state, target_value) in enumerate(zip(game_states, target_values)):
        tensor_input_board = torch.tensor(game_state, dtype=torch.float32).to(device)
        input_board = tensor_input_board.permute(2, 0, 1).unsqueeze(0).to(device)

        tensor_target_value = torch.tensor(target_value, dtype=torch.float32).to(device)

        outputs = model(input_board).to(device)
        loss = criterion(outputs[0], tensor_target_value).to(device)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
""" with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total)) """

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')