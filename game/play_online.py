import numpy as np
try:
    from game.game_interface import GameInterface
    from game.hex import Hex
    from game.nim import Nim
    from hex_client_23.ActorClient import ActorClient
except ModuleNotFoundError:
    from game_interface import GameInterface
    from hex import Hex
    from agent import Agent, RandomAgent
    from hex_client_23.ActorClient import ActorClient

"""
This script is used to play online against staff agents.
This will expect a folder named `hex_client_23` in the same directory as this script.
The `hex_client_23` folder should contain the `ActorClient.py` file, etc.
"""

actor: Agent = RandomAgent() # will be replaced by the actual trained agent

# Import and override the `handle_get_action` hook in ActorClient
class MyClient(ActorClient):
    def handle_get_action(self, state):
        """Called whenever it's your turn to pick an action

        Args:
            state (list): board configuration as a list of board_size^2 + 1 ints

        Returns:
            tuple: action with board coordinates (row, col) (a list is ok too)

        Note:
            > Given the following state for a 5x5 Hex game
                state = [
                    1,              # Current player (you) is 1
                    0, 0, 0, 0, 0,  # First row
                    0, 2, 1, 0, 0,  # Second row
                    0, 0, 1, 0, 0,  # ...
                    2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ]
            > Player 1 goes "top-down" and player 2 goes "left-right"
            > Returning (3, 2) would put a "1" at the free (0) position
              below the two vertically aligned ones.
            > The neighborhood around a cell is connected like
                  |/
                --0--
                 /|
        """
        board_size = int(np.sqrt(len(state) - 1))
        if state[0] == 2:
            black_to_play = True
        else:
            black_to_play = False

        # Convert the state to a 2D array
        state = np.array(state[1:]).reshape(board_size, board_size)

        # Define replacement arrays
        replace_0 = np.array([False, False])
        replace_1 = np.array([True, False])
        replace_2 = np.array([False, True])

        # Create a new array with the same shape as the original array, but with an extra dimension for the replacement arrays
        new_arr = np.empty(state.shape + (2,), dtype=bool)

        # Replace the elements
        new_arr[state == 0] = replace_0
        new_arr[state == 1] = replace_1
        new_arr[state == 2] = replace_2

        game = Hex(board_size, new_arr, black_to_play)

        board, _ = game.get_state(False)
        legal_actions = game.get_legal_actions()
        
        action = actor.select_action(board, black_to_play, legal_actions) # Your logic
        print("Action: ", action)
        
        row = action // board_size
        col = action % board_size

        return int (row), int (col)

# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient()
    client.run()