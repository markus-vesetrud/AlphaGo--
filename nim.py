import random
import numpy as np
from game_interface import GameInterface

class Nim(GameInterface):
    nbr_of_sticks = None
    next_player = None # True for player 1, False for player 2
    range_of_pick = None

    def __init__(self, nbr_of_sticks: list[int], range_of_pick: int) -> None:
        super().__init__()

        self.nbr_of_sticks = random.randint(nbr_of_sticks[0], nbr_of_sticks[1])
        self.next_player = random.choice([True, False])
        self.range_of_pick = range_of_pick

    def is_final_state(self) -> int:
        if self.nbr_of_sticks == 0:
            return_value = -1 if self.next_player else 1
        else:
            return_value = 0
        return return_value

    def get_legal_acions(self) -> np.ndarray:
        return [i <= self.nbr_of_sticks for i in range(1, self.range_of_pick + 1)]

    def get_state(self) -> tuple[np.ndarray, bool]:
        return (self.nbr_of_sticks, self.next_player)

    def display_current_state(self) -> None:
        print("There are", self.nbr_of_sticks, "sticks left. It is player", 1 if self.next_player else 2, "'s turn.")

    def perform_action(self, action: np.ndarray) -> None:
        if not (np.sum(action) == 1 and np.sum(np.logical_not(action)) == self.range_of_pick - 1):
            raise ValueError("The action must be a single True and the rest False.")
        if np.nonzero(action)[0][0] > self.nbr_of_sticks - 1:
            raise ValueError("Cannot pick more sticks than there are left.")

        action = np.nonzero(action)[0][0] + 1
        self.nbr_of_sticks -= action
        self.next_player = not self.next_player



if __name__ == "__main__":
    nim = Nim([1, 2], 4)

    while(nim.is_final_state() == 0):
        nim.display_current_state()
        action = int(input("Choose a number of sticks to pick: "))
        nim.perform_action(np.array([i == action for i in range(1, nim.range_of_pick + 1)]))

    print("Player", 1 if nim.is_final_state() == 1 else 2, "wins!")