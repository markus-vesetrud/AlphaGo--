import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing 
try:
     torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


from game_interface import GameInterface
from hex import Hex
from mcts import MCTreeSearch, MCTreeNode
from neural_net import ConvolutionalNeuralNetOld, LinearNeuralNet, LinearResidualNet, LinearResidualNetOld
from board_flipping_util import create_training_cases
from agent import PolicyAgent

from parameters import *

class ReinforcementLearning():
    def __init__(self, models_name: str, board_size: int, exploration_weight: float, epsilon: float, epsilon_decay: float,
                 search_iterations: int, num_games: int, total_search_count: int,
                 batch_size: int, num_epochs: int, save_interval: int,
                 loss_fn, optimizer, model: nn.Module,
                 visualize_games: bool, verbose: bool, start_epoch: int, replay_buffer_max_length: int, 
                 initial_replay_buffer_state: np.ndarray = None, initial_replay_buffer_target: np.ndarray = None) -> None:
        
        self.board_size = board_size
        self.exploration_weight = exploration_weight
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.search_iterations = search_iterations
        self.num_games = num_games
        self.total_search_count = total_search_count
        self.models_name = models_name
        self.model_paths = []

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.model.to(self.device)
        
        self.visualize_games = visualize_games
        self.verbose = verbose
        self.start_epoch=start_epoch
        self.replay_buffer_max_length = replay_buffer_max_length
        if initial_replay_buffer_state is None:
            assert initial_replay_buffer_target is None
            self.replay_buffer_state = np.zeros(shape=(0, self.board_size, self.board_size, 3), dtype=np.float32)
            self.replay_buffer_target = np.zeros(shape=(0, self.board_size**2), dtype=np.float32)
        else:
            assert initial_replay_buffer_state.shape[0] == initial_replay_buffer_target.shape[0]
            assert initial_replay_buffer_state.shape[1] == initial_replay_buffer_state.shape[2] == self.board_size
            assert initial_replay_buffer_state.shape[3] == 3
            assert initial_replay_buffer_target.shape[1] == self.board_size**2
            self.replay_buffer_state = initial_replay_buffer_state
            self.replay_buffer_target = initial_replay_buffer_target
        np.set_printoptions(suppress=True, edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3f" % x))

    
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
            
        agent = PolicyAgent(self.board_size, self.model, self.device, self.epsilon)
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

        pipes = [torch.multiprocessing.Pipe() for i in range(self.num_games)]
        processes = [torch.multiprocessing.Process(target=self.simulate_game, args=(pipes[i][1], )) for i in range(self.num_games)]

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
    
    def simulate_games_single_process(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates a seires of games and returns the game states and target values (probabilities of actions)

        All moves are saved in two configurations (original and 180 degree rotated)
        If a red move is simulated, the saved board and target value is flippes to the perspective of the black player
        """

        game_states = []
        target_values = []

        for game_number in range(self.num_games):
            print(f'Simulating game number {game_number+1} of {self.num_games}')

            agent = PolicyAgent(self.board_size, self.model, self.device, self.epsilon)
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


        print(len(game_states))

        # for i in range(len(game_states)):
        #     print(target_values[i])
        #     Hex(self.board_size, game_states[i]).display_current_state()

        return (game_states, target_values)

    def save(self, search_number):
        # dataset_path = f'checkpoints/{self.board_size}by{self.board_size}_{self.search_iterations}iter_{search_number}_replay_buffer.npy'
        # model_path =   f'checkpoints/{self.board_size}by{self.board_size}_{self.search_iterations}iter_{search_number}_model.pt'
        dataset_path = f'checkpoints/{self.models_name}:{self.board_size}by{self.board_size}_{self.search_iterations}iter_{search_number}_replay_buffer.npy'
        model_path = f'checkpoints/{self.models_name}:{self.board_size}by{self.board_size}_{self.search_iterations}iter_{search_number}_model.pt'
        self.model_paths.append(model_path)

        with open(dataset_path, 'wb') as f:
            np.save(f, self.replay_buffer_state)
            np.save(f, self.replay_buffer_target)
        torch.save(self.model.state_dict(), model_path)

    def main_loop(self):
        for search_number in range(self.start_epoch, self.total_search_count + self.start_epoch):

            print(f'starting episode number {search_number}')

            if SIMULATE_GAMES == 'single':
                game_states, target_values = self.simulate_games_single_process()
            elif SIMULATE_GAMES == 'multi':
                game_states, target_values = self.simulate_games()

            if self.visualize_game:
                for i in range(game_states.shape[0]):
                    print(target_values[i,:].reshape((board_size, board_size)))
                    Hex(board_size, game_states[i,:,:,:2]).display_current_state()

            self.update_replay_buffer(game_states, target_values)

            if search_number % self.save_interval == 0:
                self.save(search_number)
            

            # Split the data into batches
            dataset = TensorDataset(torch.from_numpy(self.replay_buffer_state).float(), torch.from_numpy(self.replay_buffer_target).float())
            random_minibatch_sampler = RandomSampler(dataset, num_samples=self.batch_size)
            data_loader = DataLoader(dataset, sampler=random_minibatch_sampler, batch_size=self.batch_size)

            loss_history = []

            self.model.train()

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
                    loss_history.append(float(loss))
                    self.optimizer.step()

                    
                print(f'Train Epoch: {epoch}\tLoss: {float(loss):.6f}')
            
            self.epsilon *= self.epsilon_decay

            game: GameInterface = Hex(self.board_size, current_black_player=True)
            test_agent = PolicyAgent(self.board_size, self.model, self.device, 0.0)
            board, black_to_play = game.get_state(False)
            print(test_agent.select_action(board, black_to_play, game.get_legal_actions(), verbose = self.verbose))

            plt.plot(list(range(len(loss_history))), loss_history)
            plt.title('Loss during training')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.savefig(f'checkpoints/{self.board_size}by{self.board_size}_{self.search_iterations}iter_{search_number}_loss.png')
            plt.close()

            # Test the agent
            # game: GameInterface = Hex(self.board_size, current_black_player=True)
            # test_agent = PolicyAgent(self.board_size, self.model, self.device, 0.0)

            # while not game.is_final_state():
            #     board, black_to_play = game.get_state(False)
            #     action = test_agent.select_action(board, black_to_play, game.get_legal_actions(), verbose = self.verbose)
            #     game.display_current_state()
            #     game.perform_action(action)
        
        print('#####################')
        self.save(self.total_search_count)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    # -------------- Hyperparameters -------------
    # Search parameters
    board_size = BOARD_SIZE
    exploration_weight = EXPLORATION_WEIGHT
    epsilon = EPSILON
    epsilon_decay = EPSILON_DECAY
    search_iterations = NUM_SEARCH
    num_games = NUM_GAMES
    replay_buffer_max_length = REPLAY_BUFFER_MAX_LENGTH
    # Set to None to start from scratch
    dataset_path = INITIAL_REPLAY_BUFFER
    model_path   = MODEL_START

    start_epoch = 0
    total_search_count = NUM_EPISODES

    # Policy network parameters
    learning_rate = LEARNING_RATE
    l2_regularization = L2_REGULARIZATION
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    save_interval = NUM_EPISODES / (NUM_CACHED_ANETS - 1)

    # --------------------------------------------


    # ------------- Other Variables --------------

    # Model
    model = LinearNeuralNet(board_size)
    # model = ConvolutionalNeuralNetOld(board_size)
    # model = LinearResidualNetOld(board_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    elif OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    elif OPTIMIZER == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    elif OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)


    if model_path is not None and dataset_path is not None:
        model.load_state_dict(torch.load(model_path))

        with open(dataset_path, 'rb') as f:
            replay_buffer_state: np.ndarray = np.load(f)
            replay_buffer_target: np.ndarray = np.load(f)

        # for i in range(replay_buffer_state.shape[0]):
        #     print(replay_buffer_target[i,:].reshape((board_size, board_size)))
        #     Hex(board_size, replay_buffer_state[i,:,:,:]).display_current_state()
    else:
        replay_buffer_state = None
        replay_buffer_target = None




    # --------------- Main RL loop ---------------


    reinforcement_learning = ReinforcementLearning('Test', board_size, exploration_weight, epsilon, epsilon_decay,
                                                search_iterations, num_games, total_search_count, 
                                                batch_size, num_epochs, save_interval, loss_fn=criterion, 
                                                optimizer=optimizer, model=model, verbose=True, visualize_game=VISUALIZE_GAMES, 
                                                start_epoch=start_epoch, replay_buffer_max_length=replay_buffer_max_length, 
                                                initial_replay_buffer_state=replay_buffer_state, initial_replay_buffer_target=replay_buffer_target)

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

