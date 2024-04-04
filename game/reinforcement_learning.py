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

input_size = 28*28
hidden_size = 500
output_size = 10
learning_rate = 0.001
batch_size = 100
num_epochs = 5



# ------------------- Game -------------------

# game: GameInterface = Nim([10,10], 4)
game: GameInterface = Hex(board_size)
board, black_to_play = game.get_state(False)

# game_states: list[GameInterface] = [deepcopy(game)]

root_node = MCTreeNode(None, -1, black_to_play, game.get_legal_acions(True), game.get_action_count())

# print(board)
input_board = board.copy()

while not game.is_final_state():
    
    mcts = MCTreeSearch(root_node, game, exploration_weight, random_policy=True)

    mcts.UCTSearch(search_iterations)
    best_action = mcts.root.best_action()

    # if mcts.root.black_to_play:
    #     mcts.UCTSearch(search_iterations)
    #     best_action = mcts.root.best_action()
    # else:
    #     best_action = int(input("Action: "))
    
    # with np.printoptions(precision=2, suppress=True):
    #     print('best', best_action)     
    #     print('total node searches:', mcts.root.total_visit_count)     
    #     print(root_node.action_values)

    # train_instance1 = root_node.action_values

    current_game_state, current_black_to_play = game.get_state(False)
    if(current_black_to_play):
        game_states.append(deepcopy(current_game_state))
        target_values.append(root_node.action_values)     
    
    root_node = mcts.root.get_child(best_action)
    root_node.make_root()

    game.perform_action(best_action)
    # game_states.append(deepcopy(game))
    # game.display_current_state()

    # break

print("len board: ", len(game_states))
print("len target: ", len(target_values))

# exit()

""" print(board)
inverted_board = game.get_inverted_state(False)[0]
rotated_board = game.get_rotated_state(False)[0]
fully_rotated_board = game.get_fully_rotated_state(False)[0]

game2: GameInterface = Hex(board_size, inverted_board, False)
game3: GameInterface = Hex(board_size, rotated_board, False)
game4: GameInterface = Hex(board_size, fully_rotated_board, False)

game.display_current_state()
game2.display_current_state()
game3.display_current_state()
game4.display_current_state() """

# stop the file from running
# exit()

# ------------------- Neural network -------------------

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
# model = LinearNeuralNet(input_size, hidden_size, output_size).to(device)
# model = ConvolutionalNeuralNet(board_size).to(device)
model = DeepConvolutionalNeuralNet(board_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
""" total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item())) """


for i in range(len(game_states)):
    tensor_input_board = torch.tensor(game_states[i], dtype=torch.float32).to(device)
    input_board = tensor_input_board.permute(2, 0, 1).unsqueeze(0).to(device)

    tensor_target_value = torch.tensor(target_values[i], dtype=torch.float32).to(device)

    outputs = model(input_board).to(device)
    loss = criterion(outputs[0], tensor_target_value).to(device)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# print("outputs after: ")
# print(model(input_board.to(device)))


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