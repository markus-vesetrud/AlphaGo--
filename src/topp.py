import numpy as np
import torch
from agent import Agent, RandomAgent, PolicyAgent
from neural_net import LinearNeuralNet, LinearResidualNet, ConvolutionalNeuralNetOld, LinearResidualNetOld
from game_interface import GameInterface
from hex import Hex
from tqdm import tqdm

from parameters import *

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
                
        with tqdm(total=self.number_of_games*len(self.agents)*(len(self.agents)-1), desc='Playing TOPP') as pbar:
            for i in range(len(self.agents)):
                for j in range(len(self.agents)):
                    if i != j:
                        agent1 = self.agents[i]
                        agent2 = self.agents[j]

                        for _ in range(self.number_of_games):
                            pbar.update(1)
                            game: GameInterface = Hex(self.board_size)

                            # play the game
                            while not game.is_final_state():
                                board, black_to_play = game.get_state(False)
                                legal_actions = game.get_legal_actions()

                                if black_to_play:
                                    action = agent1.select_action(board, black_to_play, legal_actions)
                                else:
                                    action = agent2.select_action(board, black_to_play, legal_actions)

                                if VISUALIZE_GAMES:
                                    game.display_current_state()
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
                    print(f"Agent {i}\t\t| Agent {j}\t| {(self.scores[i, j] / self.number_of_games)*100:.1f}%")



if __name__ == '__main__':
    board_size = BOARD_SIZE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    agents = [RandomAgent()]




    # Worst:
    # model = ConvolutionalNeuralNet(board_size)
    # model.load_state_dict(torch.load('checkpoints/7by7_735iter_30_model.pt', map_location=torch.device(device)))
    # model.to(device)
    # model.eval()
    # agent = PolicyAgent(board_size, model, device, 0.0, random_proportional=True)
    # agents.append(agent)

    # Middle:
    model = LinearResidualNetOld(board_size)
    model.load_state_dict(torch.load('checkpoints_residual/7by7_490iter_145_model.pt', map_location=torch.device(device)))
    model.to(device)
    model.eval()
    agent = PolicyAgent(board_size, model, device, 0.1, random_proportional=False)
    agents.append(agent)

    model = LinearResidualNet(board_size)
    model.load_state_dict(torch.load('checkpoints/7by7_980iter_180_model.pt', map_location=torch.device(device)))
    model.to(device)
    model.eval()
    agent = PolicyAgent(board_size, model, device, 0.1, random_proportional=False)
    agents.append(agent)

    # Best:
    # model = LinearResidualNet(board_size)
    # model.load_state_dict(torch.load('checkpoints/7by7_980iter_340_model.pt', map_location=torch.device(device)))
    # model.to(device)
    # model.eval()
    # agent = PolicyAgent(board_size, model, device, 0.1, random_proportional=False)
    # agents.append(agent)


    topp = TOPP(TOPP_NUM_GAMES, board_size, agents)

    topp.play_tournament()
    topp.visualize_results()