import numpy as np
import torch
import torch.nn as nn
from board_flipping_util import flip_board_perspective, flip_target_values_position

class Agent():
    def select_action(self, game_board: np.ndarray, play_as_black: bool, legal_actions: list[int]) -> int:
        """
        Returns one of the integers in legal_actions
        """


class RandomAgent(Agent):
    def select_action(self, game_board: np.ndarray, play_as_black: bool, legal_actions: list[int]) -> int:
        """
        Returns one of the integers in legal_actions
        """
        return np.random.choice(legal_actions)


class PolicyAgent(Agent):
    def __init__(self, board_size, model: nn.Module, device, epsilon: float) -> None:
        super().__init__()

        self.board_size = board_size
        self.model = model
        self.device = device
        self.epsilon = epsilon


    def __rescale_prediction(self, prediction: np.ndarray, legal_actions: list[int]) -> np.ndarray:
        """
        Rescales the prediction so that the entries with indices not specified by legal_actions are 0,
        but the sum of prediction are still 1.
        """
        legal_mask = np.isin(np.arange(self.board_size**2), legal_actions)

        prediction *= legal_mask

        return prediction / np.sum(prediction)


    def select_action(self, game_board: np.ndarray, play_as_black: bool, legal_actions: list[int], verbose = False) -> int:
        """
        Runs the agent model on the game_board, and with probability epsilon selects a random action, and 
        with probability 1-epsilon returns the best action.
        """

        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)

        # If it should play as red, then flip the prespective
        # This makes a new board where black is in the same situation as red was
        # Then the model finds (hopefully) good moves in this position
        if not play_as_black:
            game_board = flip_board_perspective(self.board_size, game_board)
        
        # Convert the board to the format expected by the model
        # (A torch float32 Tensor with an added dimension for the batch size and the last dimension moved to the front)
        game_board = torch.from_numpy(game_board).to(self.device).float()
        game_board = game_board.unsqueeze(0).permute(0, 3, 1, 2)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(game_board).cpu()
        
        # Extract the prediction
        prediction = np.array(prediction[0,:])

        # If it played as red, then the prediction needs to be flipped back again
        # After this, the prediction will correspond to the given board, and not 
        # the board the model saw
        if not play_as_black:
            prediction = flip_target_values_position(self.board_size, prediction)

        if verbose:
            print(prediction.reshape((self.board_size, self.board_size)))

        # Set illegal moves to be zero, important that this is done after the flipping above
        prediction = self.__rescale_prediction(prediction, legal_actions)

        if verbose:
            print(prediction.reshape((self.board_size, self.board_size)))
            print()

        # Return the best prediction
        return prediction.argmax()
    
        # Return a prediction proportional to the probability of that prediction
        # return np.random.choice(np.arange(prediction.shape[0]), p=prediction)