import numpy as np
from agent import Agent
from game_interface import GameInterface
from hex import Hex

class TOPP():
    """
    A class to represent the Tournament of Progressive Policies (TOPP).
    Each agent plays against each other agent for a number of games.
    """
    def __init__(self, number_of_games: int, board_size: int, agents: list[Agent]) -> None:
        self.number_of_games = number_of_games
        self.board_size = board_size
        self.agents = agents
        self.scores = np.zeros((len(agents), len(agents)))

    def play_tournament(self) -> np.ndarray:
        """
        Play the tournament of progressive policies.
        Each agent plays against each other agent for a number of games.
        The number of wins is recorded in a matrix in percentage.
        The upper triangular part of the matrix is filled with the win percentage
        of the row agent playing as black against the column agent playing as red.
        """
        for i in range(len(self.agents)):
            for j in range(len(self.agents)):
                if i != j:
                    agent1 = self.agents[i]
                    agent2 = self.agents[j]

                    for _ in range(self.number_of_games):
                        game: GameInterface = Hex(self.board_size)

                        # play the game
                        while not game.is_final_state():
                            board, black_to_play = game.get_state(False)
                            legal_actions = game.get_legal_actions()

                            if black_to_play:
                                action = agent1.select_action(board, black_to_play, legal_actions)
                            else:
                                action = agent2.select_action(board, black_to_play, legal_actions)

                            game.perform_action(action)
                        
                        # update scores
                        self.scores[i, j] += game.get_final_state_reward()
                        
        return self.scores / self.number_of_games

    def visualize_results(self):
        """
        Visualize the results of the tournament.
        Visualization is on the form black agent vs red agent: win percentage of black agent.
        """
        print("Black agent\t| Red agent\t| Win percentage of black")
        for i in range(len(self.agents)):
            print("-------------------------------------------------")
            for j in range(len(self.agents)):
                if i != j:
                    print(f"Agent {i}\t\t| Agent {j}\t| {(self.scores[i, j] / self.number_of_games):.2f}%")

# class TestTOPP():
#     def __init__(self) -> None:
#         self.scores = np.zeros((3, 3))
#         self.number = 0

#     def test_scores(self, k):
#         self.scores[0, 1] += k
#         self.number += 1

#     def visualize_results(self):
#         print("Black agent\t| Red agent\t| Win percentage of black")
#         for i in range(3):
#             print("-------------------------------------------------")
#             for j in range(3):
#                 if i != j:
#                     print(f"Agent {i}\t\t| Agent {j}\t| {(self.scores[i, j] / self.number):.2f}%")

# if __name__ == "__main__":
#     test = TestTOPP()
#     test.test_scores(1)
#     test.test_scores(0)
#     test.test_scores(1)

#     test.visualize_results()
        