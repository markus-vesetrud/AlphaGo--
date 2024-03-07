import numpy as np


class GameInterface:

    def is_final_state(self) -> int:
        """
        Only works for games that cannot be drawn

        returns: -1 for player 2 win (not starting), 0 for not final state, and 1 for player 1 win (starting player)
        """
        pass


    def get_legal_acions(self, flatten: bool = True) -> list[int | tuple[int]]:
        """
        returns a list of indices representing the action corresponding to that board position, 
        if flatten is true, instead each action is represented as a tuple of integers
        """
        pass


    def get_state(self, flatten: bool = True) -> np.ndarray | tuple[np.ndarray, bool]:
        """
        returns a representation of the game as an ndarray, and a boolean signifying whether it is the starting players turn
        """
        pass
    

    def perform_action(self, action: int | tuple[int], flattend_input: bool = True) -> None:
        """
        performs the given action on the game state
        action: an ndarray containing exactly one True, and the rest False
        """
        pass


    def display_current_state(self) -> None:
        """
        displays the current state of the game
        """
        pass


    def get_action_count(self) -> int:
        """
        returns the max number of actions
        """
        pass
