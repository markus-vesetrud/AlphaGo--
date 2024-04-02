import random
import numpy as np

try:
    from game.game_interface import GameInterface
except ModuleNotFoundError:
    from game_interface import GameInterface


class Nim(GameInterface):
    def __init__(self, nbr_of_sticks: list[int], range_of_pick: int) -> None:
        # super().__init__()

        self.nbr_of_sticks = random.randint(nbr_of_sticks[0], nbr_of_sticks[1])
        self.player_1_turn = True # True for player 1, False for player 2
        self.range_of_pick = range_of_pick

    def get_final_state_reward(self, flatten: bool = True) -> int:
        """
        raises an AssertionError if not a final state

        returns 0 for player 2 win (not starting), and 1 for player 1 win (starting player)
        """
        if self.is_final_state():
            # Opposite of what you would expect since player_1_turn has flipped after the match was decided
            return 0 if self.player_1_turn else 1
        else:
            raise AssertionError("Not a final state")

    def is_final_state(self) -> bool:
        """
        returns true if the game is in a final state
        """
        return self.nbr_of_sticks == 0

    def get_legal_acions(self, flatten: bool = True) -> np.ndarray:
        return [i for i in range(min(self.nbr_of_sticks, self.range_of_pick))]
        # return [i <= self.nbr_of_sticks for i in range(1, self.range_of_pick + 1)]

    def get_state(self, flatten: bool = True) -> tuple[np.ndarray, bool]:
        return (self.nbr_of_sticks, self.player_1_turn)
    
    def is_starting_player_turn(self) -> bool:
        # Not neccecarily the starting player, but signifies which turn it is
        return self.player_1_turn

    def display_current_state(self) -> None:
        print("There are", self.nbr_of_sticks, "sticks left. It is player", 1 if self.player_1_turn else 2, "'s turn.")

    def perform_action(self, action: int, flattend_input: bool = True) -> None:
        # if not (np.sum(action) == 1 and np.sum(np.logical_not(action)) == self.range_of_pick - 1):
        #     raise ValueError("The action must be a single True and the rest False.")
        # if np.nonzero(action)[0][0] > self.nbr_of_sticks - 1:
        #     raise ValueError("Cannot pick more sticks than there are left.")

        # action = np.nonzero(action)[0][0] + 1

        if action not in self.get_legal_acions():
            raise ValueError(f"Action not in {self.get_legal_acions()}")

        self.nbr_of_sticks -= action + 1
        self.player_1_turn = not self.player_1_turn
    
    def get_action_count(self) -> int:
        """
        returns the max number of actions
        """
        return self.range_of_pick




if __name__ == "__main__":
    nim: GameInterface = Nim([10,10], 4)

    while(nim.is_final_state() == 0):
        nim.display_current_state()
        print(nim.get_legal_acions())
        action = int(input("Choose a number of sticks to pick: "))
        # nim.perform_action(np.array([i == action for i in range(1, nim.range_of_pick + 1)]))
        nim.perform_action(action)

    print(nim.get_legal_acions())
    print("Player", 1 if nim.is_final_state() == 1 else 2, "wins!")