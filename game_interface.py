import numpy as np


class GameInterface:

    # def __init__(self) -> None:
    #     pass
    

    def is_final_state(self) -> int:
        """
        Only works for games that cannot be drawn

        returns: -1 for player 2 win (not starting), 0 for not final state, and 1 for player 1 win (starting player)
        """
        pass


    def get_legal_acions(self) -> np.ndarray:
        """
        returns a list of booleans, where each possible action is marked as legal or not
        """
        pass


    def get_state(self) -> tuple[np.ndarray, bool]:
        """
        returns a representation of the game as an ndarray, and a boolean signifying whether it is the starting players turn
        """
        pass
    

    def display_current_state(self) -> None:
        """
        displays the current state of the game
        """
        pass


    def perform_action(self, action: np.ndarray) -> None:
        """
        performs the given action on the game state
        action: an ndarray containing exactly one True, and the rest False
        """
        pass
