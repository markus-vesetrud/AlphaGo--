import numpy as np
from copy import deepcopy


def flip_board_perspective(board_size, game_board: np.ndarray):
    """
    Flips the board between the perspective of red and black
    (╯°□°)╯︵ ┻━┻
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


def __softmax(x: np.ndarray):
    # Subtracting the max value for numerical stability, it does not influence the answer
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def add_empty_marker(board: np.ndarray):
    # Assumes board is 3 dimensional with shape (board_size, board_size, 2)
    # Returns a board with shape (board_size, board_size, 3)
    extended_board = np.zeros(shape=(board.shape[0], board.shape[1], board.shape[2]+1), dtype=board.dtype)
    extended_board[:,:,:2] = board
    empty_piece = np.zeros(shape=(2), dtype=board.dtype)[np.newaxis, np.newaxis, :]
    empty_locations = np.equal(board, empty_piece).all(axis=2)
    extended_board[:,:,2] = empty_locations
    return extended_board


def create_training_cases(board_size: int, game_board: np.ndarray, black_to_play: bool, node_action_values: list[float], legal_actions: list[int]) -> tuple[list[np.ndarray]]:
    """
    Takes in a board, which players turn it is, and information about the best move

    Returns a list of 2 game states and 2 target values, if it is red's turn, the player perspective is flipped
    The 2 game states and target values returned are 180 degree rotations about each other

    Also extends the board to have 3 channels, the last channel indicating the board is empty
    """
    game_states = []
    target_values = []
    node_action_values = np.array(node_action_values, dtype=np.float32)

    # The mask is a 1D array of booleans signifying whether the action is a legal action
    legal_mask = np.isin(np.arange(board_size**2), legal_actions)
    # The inf mask is 0 of the action is legal and -inf if not
    legal_inf_mask = np.array([(0.0 if legal else -np.inf) for legal in legal_mask], dtype=np.float32)


    if(black_to_play):
        
        current_target_values = __softmax(node_action_values + legal_inf_mask)
        # adding original board
        game_states.append(add_empty_marker(deepcopy(game_board)))
        target_values.append(current_target_values)

        # adding 180 degree rotated board
        game_states.append(add_empty_marker(deepcopy(np.rot90(game_board, k=2))))
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

        current_target_values = __softmax(current_target_values + legal_inf_mask)

        current_target_values = flip_target_values_position(board_size, current_target_values)
        game_board = flip_board_perspective(board_size, game_board)

        # adding perspective flipped board
        game_states.append(add_empty_marker(deepcopy(game_board)))
        target_values.append(current_target_values)

        # adding perspective flipped and 180 degree rotated board
        game_states.append(add_empty_marker(deepcopy(np.rot90(game_board, k=2))))
        target_values.append(np.rot90(current_target_values.reshape((board_size, board_size)), k=2).flatten())
    
    return game_states, target_values