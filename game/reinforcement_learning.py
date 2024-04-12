import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from game_interface import GameInterface
from hex import Hex
from mcts import MCTreeSearch, MCTreeNode
from neural_net import ConvolutionalNeuralNet, DeepConvolutionalNeuralNet, LinearNeuralNet
from typing import Callable


def softmax(x: np.ndarray):
    # Subtracting the max value for numerical stability, it does not influence the answer
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def flip_board_perspective(board_size, game_board: np.ndarray):
    """
    Flips the board between the perspective of red and black
    """

    # Flips the red tiles to black tiles and vice versa
    empty_mask = np.all(game_board == [False, False], axis=-1)
    game_board = np.logical_not(game_board)
    game_board[empty_mask] = [False, False]

    # Inverts the board so the lines from black start to black end now goes from red start to red end
    # Depending on how you see it, this also rotates the board 180 degrees, but that leaves a board in the same state
    board_copy = game_board.copy()
    for i in range(board_size):
        for j in range(board_size):
            game_board[board_size - j - 1][board_size - i - 1] = board_copy[i][j]
    

    return game_board

def flip_target_values_position(board_size, current_target_values: np.ndarray) -> np.ndarray:
    """
    Switches around the positions of the numbers in the given array, 
    in the same way flip_board_perspective switches the position,
    but does NOT switch high target values to low target values and vice versa.
    """
    # Reshape to 2D
    current_target_values = current_target_values.reshape((board_size, board_size))
    current_target_values_copy = current_target_values.copy()

    # Inverts the target values corresponding to a board where the lines from 
    # black start to black end now goes from red start to red end
    # See function flip_board_perspective
    for i in range(board_size):
        for j in range(board_size):
            current_target_values[board_size - j - 1][board_size - i - 1] = current_target_values_copy[i][j]
        
    return current_target_values.flatten()


def create_cases(board_size: int, game_board: np.ndarray, black_to_play: bool, node_action_values: list[float], legal_actions: list[int]) -> tuple[list[np.ndarray]]:
    """
    Takes in a board, which players turn it is, and information about the best move

    Returns a list of 2 game states and 2 target values, if it is red's turn, the player perspective is flipped
    The 2 game states and target values returned are 180 degree rotations about each other
    """
    game_states = []
    target_values = []
    node_action_values = np.array(node_action_values, dtype=np.float32)

    # The mask is a 1D array of booleans signifying whether the action is a legal action
    legal_mask = np.isin(np.arange(board_size**2), legal_actions)
    # The inf mask is 0 of the action is legal and -inf if not
    legal_inf_mask = np.array([(0.0 if legal else -np.inf) for legal in legal_mask], dtype=np.float32)


    if(black_to_play):
        
        current_target_values = softmax(node_action_values + legal_inf_mask)
        # adding original board
        game_states.append(deepcopy(game_board))
        target_values.append(current_target_values)

        # adding 180 degree rotated board
        game_states.append(deepcopy(np.rot90(game_board, k=2)))
        fully_rotated_current_target_values = np.rot90(current_target_values.reshape((board_size, board_size)), k=2).flatten()
        target_values.append(fully_rotated_current_target_values)

    else:
        # Flip the board from player2's perspective back to player1's perspective

        # node_action_values is between 0 and 1, taking 1 minus that value flips it
        current_target_values = (np.ones(board_size*board_size) - node_action_values)

        # In previous implementation:
        # Sometimes np.sum(current_target_values) == 0
        # This happens if the search is 100% confident black will win no matter what.
        # In this case the node_action_values are 1.0 (exactly) for legal actions and 0.0 for illegal actions
        # That would result in current_target_values all being 0.0 and equally bad

        current_target_values = softmax(current_target_values + legal_inf_mask)

        current_target_values = flip_target_values_position(board_size, current_target_values)
        game_board = flip_board_perspective(board_size, game_board)

        # adding perspective flipped board
        game_states.append(deepcopy(game_board))
        target_values.append(current_target_values)

        # adding perspective flipped and 180 degree rotated board
        game_states.append(deepcopy(np.rot90(game_board, k=2)))
        target_values.append(np.rot90(current_target_values.reshape((board_size, board_size)), k=2).flatten())
    
    return game_states, target_values


def simulate_games(model: nn.Module, board_size: int, exploration_weight: float, 
                   search_iterations: int, num_games: int, verbose: bool,) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a seires of games and returns the game states and target values (probabilities of actions)

    All moves are saved in two configurations (original and 180 degree rotated)
    If a red move is simulated, the saved board and target value is flippes to the perspective of the black player
    """

    game_states: list[np.ndarray] = []
    target_values: list[np.ndarray] = []


    for game_number in range(1, num_games+1):
        
        if verbose:
            print(f'Simulating game number {game_number} of {num_games}')

        def action_selection_policy(game_board: np.ndarray, play_as_black: bool, legal_actions: list[int]) -> int:
            return calculate_action(model, board_size, game_board, play_as_black, legal_actions, best=False)


        game: GameInterface = Hex(board_size)

        # Create the root node with no parent, starting with black to play
        _, black_to_play = game.get_state(False) # black_to_play is always True in this implementation
        root_node = MCTreeNode(None, -1, black_to_play, game.get_legal_acions(True), game.get_action_count())

        # Play a game
        while not game.is_final_state():
            
            mcts = MCTreeSearch(root_node, game, exploration_weight, action_selection_policy)

            # This is what takes all the time
            mcts.UCTSearch(search_iterations)

            current_game_board, current_black_to_play = game.get_state(False)

            new_game_states, new_target_values = create_cases(board_size, current_game_board, current_black_to_play, 
                                                              mcts.root.action_values, mcts.root.legal_actions)

            # Add to the training cases
            game_states.extend(new_game_states)
            target_values.extend(new_target_values)
            
            # Find the best action and the child in the tree corresponding to that action
            # Make that child the new root and perform the action
            best_action = mcts.root.best_action()
            root_node = mcts.root.get_child(best_action)

            # Deletes the link to the parent, and thus python will 
            # delete the whole tree except the subtree of this node
            root_node.make_root() 

            game.perform_action(best_action)
            # game.display_current_state()

    return convert_to_ndarray(board_size, game_states, target_values)



def convert_to_ndarray(board_size: int, game_states: list[np.ndarray], target_values: list[np.ndarray]) -> tuple[np.ndarray]:
    new_game_states = np.zeros(shape=(len(game_states), board_size, board_size, 2), dtype=np.float32)
    new_target_values = np.zeros(shape=(len(target_values), board_size**2), dtype=np.float32)

    for i in range(len(game_states)):
        new_game_states[i,:,:,:] = game_states[i]
        new_target_values[i,:] = target_values[i]
        
    return new_game_states, new_target_values


def rescale_prediction(prediction: np.ndarray, legal_actions: list[int]) -> np.ndarray:
    """
    Rescales the prediction so that the entries with indices not specified by legal_actions are 0,
    but the sum of prediction are still 1.
    """
    legal_mask = np.isin(np.arange(board_size**2), legal_actions)

    prediction *= legal_mask

    return prediction / np.sum(prediction)



def calculate_action(model: nn.Module, board_size: int, game_board: np.ndarray, play_as_black: bool, legal_actions: list[int], best: bool, verbose: bool = False) -> int:
    """
    Runs the given model on the game_board, and selects an action from the output of the model
    With best=True the best action will always be selected, if False a random action will be selected
    with probability depending on the output of the model
    """

    # If it should play as red, then flip the prespective
    # This makes a new board where black is in the same situation as red was
    # Then the model finds (hopefully) good moves in this position
    if not play_as_black:
        game_board = flip_board_perspective(board_size, game_board)
    
    # Convert the board to the format expected by the model
    # (A torch float32 Tensor with an added dimension for the batch size and the last dimension moved to the front)
    game_board = torch.from_numpy(game_board).float()
    game_board = game_board.unsqueeze(0).permute(0, 3, 1, 2)

    model.eval()
    with torch.no_grad():
        prediction = model(game_board)
    
    # Extract the prediction
    prediction = np.array(prediction[0,:])

    # If it played as red, then the prediction needs to be flipped back again
    # After this, the prediction will correspond to the given board, and not 
    # the board the model saw
    if not play_as_black:
        prediction = flip_target_values_position(board_size, prediction)

    # Set illegal moves to be zero, important that this is done after the flipping above
    prediction = rescale_prediction(prediction, legal_actions)

    if verbose:
        print(prediction.reshape((board_size, board_size)))

    # Just return the best prediction
    if best:
        return prediction.argmax()
    # Return a prediction proportional to the probability of that prediction
    else:
        return np.random.choice(np.arange(prediction.shape[0]), p=prediction)


def starter_win_ratio(model: nn.Module, board_size: int, best: bool, num_games: int = 10000):
    """
    Plays the model against itself num_games times
    Returns how often the model won as black
    A random baseline is around 66%+-0.5% while perfect play is 100%

    This may not be a good way to evaluate our models, a model may get 100% if allowed to do the best move, 
    but still almost get random results if using the probability distribution it gives, 
    because a single mistake will lose you the game. And if there is only a 50% chance to pick the correct move 
    then that is far from enough, and the game will not look very intelligent.
    """
    total_starting_wins = 0

    for i in range(num_games):
        # Create a new game alternating between red and black starts
        black_start = i%2==0
        game: GameInterface = Hex(board_size, current_black_player=black_start)

        while not game.is_final_state():
            board, black_to_play = game.get_state(False)
            action = calculate_action(model, board_size, board, play_as_black=black_to_play, 
                                      legal_actions=game.get_legal_acions(), best=best)
            game.perform_action(action)
        
        # The final_state_reward is 1 if black won, and 0 if red won
        # This code adds 1 to the total_starting_wins count if the starting player won
        if black_start:
            total_starting_wins += game.get_final_state_reward()
        else:
            total_starting_wins += 1-game.get_final_state_reward()

    
    return total_starting_wins/num_games


# -------------- Hyperparameters -------------
# Search parameters
board_size = 3
exploration_weight = 1.0
search_iterations = 20*board_size**2
num_games = 50

total_search_count = 10

# Policy network parameters
learning_rate = 1e-3
l2_regularization = 1e-4 # Set to 0 for no regularization
batch_size = 32
num_epochs = 10
log_interval = 1
# --------------------------------------------


# ------------- Other Variables --------------

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = LinearNeuralNet(board_size)
# model = ConvolutionalNeuralNet(board_size)
# model = DeepConvolutionalNeuralNet(board_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)


# with open('3by3_500iter_100_games.npy', 'rb') as f:
#     game_states: np.ndarray = np.load(f)
#     target_values: np.ndarray = np.load(f)

# for i in range(game_states.shape[0]):
#     print(target_values[i,:].reshape((board_size, board_size)))
#     Hex(board_size, game_states[i,:,:,:]).display_current_state()


# --------------- Main RL loop ---------------

for search_number in range(1, total_search_count+1):

    print(f'starting episode number {search_number}')

    game_states, target_values = simulate_games(model, board_size, exploration_weight, search_iterations, num_games, verbose=True)

    dataset_path = f'datasets/{board_size}by{board_size}_{search_iterations}iter_{search_number}.npy'
    
    with open(dataset_path, 'wb') as f:
        np.save(f, game_states[0,:,:,:])
        np.save(f, target_values[0,:])
    
    # ------------------- Convolutional neural network -------------------
    # this part trains the convolutional neural network on the collected game states and target values
    # the model is trained to predict the target values from the game states

    # Split the data into batches
    dataset = TensorDataset(torch.from_numpy(game_states), torch.from_numpy(target_values))
    data_loader = DataLoader(dataset, batch_size=batch_size)

    loss_history = []

    # Train the model
    for epoch in range(1, num_epochs+1):
        for i, (data, target) in enumerate(data_loader):
            data = data.permute(0, 3, 1, 2).to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            loss_history.append(float(loss))
            optimizer.step()

            if i % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * batch_size, len(data_loader.dataset),
                    100. * i / len(data_loader), loss.item()))

    plt.plot(list(range(len(loss_history))), loss_history)
    plt.title('Loss during training')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

    # Test the agent
    game: GameInterface = Hex(board_size, current_black_player=False)

    while not game.is_final_state():
        board, black_to_play = game.get_state(False)
        action = calculate_action(model, board_size, board, play_as_black=black_to_play, 
                                legal_actions=game.get_legal_acions(), best=True, verbose=True)
        game.display_current_state()
        game.perform_action(action)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.pt')


model.load_state_dict(torch.load('model.pt'))

game: GameInterface = Hex(board_size, current_black_player=False)

while not game.is_final_state():
    board, black_to_play = game.get_state(False)
    action = calculate_action(model, board_size, board, play_as_black=black_to_play, 
                            legal_actions=game.get_legal_acions(), best=True, verbose=True)
    game.display_current_state()
    game.perform_action(action)

