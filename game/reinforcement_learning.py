import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from game_interface import GameInterface
from hex import Hex
from mcts import MCTreeSearch, MCTreeNode
from neural_net import ConvolutionalNeuralNet, DeepConvolutionalNeuralNet, LinearNeuralNet
from board_flipping_util import create_training_cases
from agent import PolicyAgent


class ReinforcementLearning():
    def __init__(self, board_size: int, exploration_weight: float, epsilon: float, epsilon_decay: float,
                 search_iterations: int, num_games: int, total_search_count: int,
                 batch_size: int, num_epochs: int, log_interval: int, save_interval: int,
                 loss_fn, optimizer, model: nn.Module,
                 verbose: bool, replay_buffer_max_length: int, 
                 initial_replay_buffer_state: np.ndarray = None, initial_replay_buffer_target: np.ndarray = None) -> None:
        
        self.board_size = board_size
        self.exploration_weight = exploration_weight
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.search_iterations = search_iterations
        self.num_games = num_games
        self.total_search_count = total_search_count

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.model.to(self.device)
        
        self.verbose = verbose
        
        self.replay_buffer_max_length = replay_buffer_max_length
        if initial_replay_buffer_state is None:
            assert initial_replay_buffer_target is None
            self.replay_buffer_state = np.zeros(shape=(0, self.board_size, self.board_size, 2), dtype=np.float32)
            self.replay_buffer_target = np.zeros(shape=(0, self.board_size**2), dtype=np.float32)
        else:
            assert initial_replay_buffer_state.shape[0] == initial_replay_buffer_target.shape[0]
            assert initial_replay_buffer_state.shape[1] == initial_replay_buffer_state.shape[2] == self.board_size
            assert initial_replay_buffer_state.shape[3] == 2
            assert initial_replay_buffer_target.shape[1] == self.board_size**2
            self.replay_buffer_state = initial_replay_buffer_state
            self.replay_buffer_target = initial_replay_buffer_target

    
    def update_replay_buffer(self, game_states: list[np.ndarray], target_values: list[np.ndarray]) -> tuple[np.ndarray]:
        available_spots = self.replay_buffer_max_length - self.replay_buffer_state.shape[0]
        cases_to_remove = max(0, len(game_states) - available_spots)
        cases_to_add = len(game_states) - cases_to_remove


        # Randomly remove as many spots as needed from the array
        indices_to_remove = np.arange(self.replay_buffer_state.shape[0])
        indices_to_remove = np.random.choice(a=indices_to_remove, size=cases_to_remove)

        # Add new cases without replacing
        for i in range(cases_to_add):
            self.replay_buffer_state = np.append(self.replay_buffer_state, game_states[i][np.newaxis,:,:,:], axis=0)
            self.replay_buffer_target = np.append(self.replay_buffer_target, target_values[i][np.newaxis,:], axis=0)

        # Replace states
        for i in range(cases_to_remove):
            remove_index = indices_to_remove[i]
            self.replay_buffer_state[remove_index,:,:,:] = game_states[i+cases_to_add]
            self.replay_buffer_target[remove_index,:] = target_values[i+cases_to_add]


    def simulate_game(self, send_connection) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates a single game and returns the game states and target values (probabilities of actions)

        All moves are saved in two configurations (original and 180 degree rotated)
        If a red move is simulated, the saved board and target value is flippes to the perspective of the black player
        """
            
        agent = PolicyAgent(self.board_size, self.model, self.epsilon)
        game: GameInterface = Hex(self.board_size)

        # Create the root node with no parent, starting with black to play
        _, black_to_play = game.get_state(False) # black_to_play is always True in this implementation
        root_node = MCTreeNode(None, -1, black_to_play, game.get_legal_actions(True), game.get_action_count())

        # Play a game
        while not game.is_final_state():
            
            mcts = MCTreeSearch(root_node, game, self.exploration_weight, agent)

            # This is what takes all the time
            mcts.UCTSearch(self.search_iterations)

            current_game_board, current_black_to_play = game.get_state(False)

            new_game_states, new_target_values = create_training_cases(self.board_size, current_game_board, current_black_to_play, 
                                                            mcts.root.action_values, mcts.root.legal_actions)

            # Add to the training cases
            send_connection.send((new_game_states, new_target_values))
            # game_states.extend(new_game_states)
            # target_values.extend(new_target_values)
            
            # Find the best action and the child in the tree corresponding to that action
            # Make that child the new root and perform the action
            best_action = mcts.root.best_action()
            root_node = mcts.root.get_child(best_action)

            # Deletes the link to the parent, and thus python will 
            # delete the whole tree except the subtree of this node
            root_node.make_root() 

            game.perform_action(best_action)
            # game.display_current_state()
    

    def simulate_games(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates a seires of games and returns the game states and target values (probabilities of actions)

        All moves are saved in two configurations (original and 180 degree rotated)
        If a red move is simulated, the saved board and target value is flippes to the perspective of the black player
        """

        game_states = []
        target_values = []

        pipes = [multiprocessing.Pipe() for i in range(self.num_games)]
        processes = [multiprocessing.Process(target=self.simulate_game, args=(pipes[i][1], )) for i in range(self.num_games)]

        for game_number in range(self.num_games):
            processes[game_number].start()
            
        for game_number in range(self.num_games):
            if self.verbose:
                print(f'Simulating game number {game_number+1} of {self.num_games}')
            processes[game_number].join()
            while pipes[game_number][0].poll():
                game_state, target_value = pipes[game_number][0].recv()
                game_states.extend(game_state)
                target_values.extend(target_value)


        print(len(game_states))

        # for i in range(len(game_states)):
        #     print(target_values[i])
        #     Hex(self.board_size, game_states[i]).display_current_state()

        return (game_states, target_values)

    def save(self, search_number):
        dataset_path = f'checkpoints/{self.board_size}by{self.board_size}_{self.search_iterations}iter_{search_number}_replay_buffer.npy'
        model_path =   f'checkpoints/{self.board_size}by{self.board_size}_{self.search_iterations}iter_{search_number}_model.pt'
        
        with open(dataset_path, 'wb') as f:
            np.save(f, self.replay_buffer_state)
            np.save(f, self.replay_buffer_target)
        torch.save(self.model.state_dict(), model_path)

    def main_loop(self):
        for search_number in range(self.total_search_count):

            print(f'starting episode number {search_number}')

            game_states, target_values = self.simulate_games()

            self.update_replay_buffer(game_states, target_values)

            if search_number % self.save_interval == 0:
                self.save(search_number)
            
            # ------------------- Convolutional neural network -------------------
            # this part trains the convolutional neural network on the collected game states and target values
            # the model is trained to predict the target values from the game states

            # Split the data into batches
            dataset = TensorDataset(torch.from_numpy(self.replay_buffer_state), torch.from_numpy(self.replay_buffer_target))
            data_loader = DataLoader(dataset, batch_size=self.batch_size)

            # loss_history = []

            # Train the model
            for epoch in range(1, self.num_epochs+1):
                for i, (data, target) in enumerate(data_loader):
                    data = data.permute(0, 3, 1, 2).to(self.device)
                    target = target.to(self.device)

                    output = self.model(data)
                    loss = self.loss_fn(output, target)

                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    # loss_history.append(float(loss))
                    self.optimizer.step()

                    if i % self.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, i * self.batch_size, len(data_loader.dataset),
                            100. * i / len(data_loader), loss.item()))
            
            self.epsilon *= self.epsilon_decay

            # plt.plot(list(range(len(loss_history))), loss_history)
            # plt.title('Loss during training')
            # plt.xlabel('Batch')
            # plt.ylabel('Loss')
            # plt.show()

            # Test the agent
            # game: GameInterface = Hex(self.board_size, current_black_player=False)
            # test_agent = PolicyAgent(self.board_size, self.model, 0.0)

            # while not game.is_final_state():
            #     board, black_to_play = game.get_state(False)
            #     action = test_agent.select_action(board, black_to_play, game.get_legal_actions(), verbose = self.verbose)
            #     game.display_current_state()
            #     game.perform_action(action)
            
        self.save(self.total_search_count)

def starter_win_ratio(model: nn.Module, board_size: int, epsilon: float, num_games: int = 10000):
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
            action = PolicyAgent(board_size, model, epsilon).select_action(board, black_to_play, game.get_legal_actions())
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
board_size = 7
exploration_weight = 1.0
epsilon = 1.0
epsilon_decay = 0.95
search_iterations = 4*board_size**2
num_games = 5
replay_buffer_max_length = 2500

total_search_count = 50

# Policy network parameters
learning_rate = 1e-3
l2_regularization = 1e-4 # Set to 0 for no regularization
batch_size = 32
num_epochs = 20
log_interval = 1
save_interval = 5
# --------------------------------------------


# ------------- Other Variables --------------

# Model
model = LinearNeuralNet(board_size)
# model = ConvolutionalNeuralNet(board_size)
# model = DeepConvolutionalNeuralNet(board_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)


# with open('datasets/3by3_2000iter_1.npy', 'rb') as f:
#     game_states: np.ndarray = np.load(f)
#     target_values: np.ndarray = np.load(f)

# for i in range(game_states.shape[0]):
#     print(target_values[i,:].reshape((board_size, board_size)))
#     Hex(board_size, game_states[i,:,:,:]).display_current_state()

# dataset = TensorDataset(torch.from_numpy(game_states), torch.from_numpy(target_values))
# data_loader = DataLoader(dataset, batch_size=batch_size)

# loss_history = []

# # Train the model
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
#         # with torch.no_grad():
#         #     for i in range(data.shape[0]):
#         #         print('actual', np.array(target[i,:].cpu()).reshape((board_size, board_size)))
#         #         print('predic', np.array(output[i,:].cpu()).reshape((board_size, board_size)))
#         #         Hex(board_size, np.array(data[i,:,:,:].permute(1,2,0).cpu(), dtype=np.bool_)).display_current_state()

#         if i % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, i * batch_size, len(data_loader.dataset),
#                 100. * i / len(data_loader), loss.item()))

# plt.plot(list(range(len(loss_history))), loss_history)
# plt.title('Loss during training')
# plt.xlabel('Batch')
# plt.ylabel('Loss')
# plt.show()

# # Test the agent
# game: GameInterface = Hex(board_size, current_black_player=True)
# test_agent = PolicyAgent(board_size, model, 0.0)

# while not game.is_final_state():
#     board, black_to_play = game.get_state(False)
#     action = test_agent.select_action(board, black_to_play, game.get_legal_actions(), verbose = True)
#     game.display_current_state()
#     game.perform_action(action)



# --------------- Main RL loop ---------------


reinforcement_learning = ReinforcementLearning(board_size, exploration_weight, epsilon, epsilon_decay,
                                               search_iterations, num_games, total_search_count, 
                                               batch_size, num_epochs, log_interval, save_interval, loss_fn=criterion, 
                                               optimizer=optimizer, model=model, verbose=True, 
                                               replay_buffer_max_length=replay_buffer_max_length)

reinforcement_learning.main_loop()

model = reinforcement_learning.model

# Save the model checkpoint
torch.save(model.state_dict(), 'model.pt')


model.load_state_dict(torch.load('model.pt'))

game: GameInterface = Hex(board_size, current_black_player=False)

while not game.is_final_state():
    board, black_to_play = game.get_state(False)
    action = PolicyAgent(board_size, model, 0.0).select_action(board, black_to_play, game.get_legal_actions())
    game.display_current_state()
    game.perform_action(action)

