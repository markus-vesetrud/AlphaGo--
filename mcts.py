import numpy as np
from game.game_interface import GameInterface
from game.hex import Hex
from game.nim import Nim
from copy import deepcopy
import random

# Overall view
# 1. Tree Search - Traversing the tree from the root to a leaf node by using the tree policy.

# 2. Node Expansion - Generating some or all child states of a parent state, and then connecting the tree node housing
# the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).

# 3. Leaf Evaluation - Estimating the value of a leaf node in the tree by doing a rollout simulation using the default
# policy from the leaf node’s state to a final state.

# 4. Backpropagation - Passing the evaluation of a final state back up the tree, updating relevant data (see course
# lecture notes) at all nodes and edges on the path from the final state to the tree root

class MCTreeNode:
    def __init__(self, parent, parent_action: int,
                 black_to_play: bool,  # board: np.ndarray, black_to_play: bool, 
                 legal_actions: list[int], total_action_count: int) -> None:
        
        # TODO: Reformat the children to be linked to by a dictionary with all the legal moves as possible values


        # Pointers to the parent node and children nodes
        self.parent: MCTreeNode = parent
        self.children: list[MCTreeNode] = [None for _ in range(total_action_count)]
        # The action the parent took to get to this node
        self.parent_action = parent_action

        # Total visits to this node
        self.total_visit_count = 0
        # The number of times each action from this node has been taken
        self.children_visit_count: list[int] = [0 for _ in range(total_action_count)]
        # The Q-value for each action in this node
        self.action_values: list[float] = [0.0 for _ in range(total_action_count)]

        # No need to save a copy of the board for each node, 
        # the position can be deduced from the starting position and the path you take in the tree
        # self.board = board
        self.black_to_play = black_to_play
        self.legal_actions = legal_actions
    

    def backup(self, action: int, sim_result: float) -> None:
        self.total_visit_count += 1
        self.children_visit_count[action] += 1
        self.action_values[action] += (sim_result - self.action_values[action]) / self.children_visit_count[action]

        if self.parent is not None:
            self.parent.backup(self.parent_action, sim_result)
    

    def is_action_taken(self, action: int) -> bool:
        # Returns whether or not this node has a child for the given action
        return self.children[action] != None
    

    def get_child(self, action: int):
        # Kidnapping
        return self.children[action]
    

    def make_root(self):
        self.parent = None

    
    def attach_child(self, child, action: int):
        self.children[action] = child
    

    def best_action(self) -> int:
        best_action = -1
        if self.black_to_play:
            best_action_value = -np.inf
            for action in self.legal_actions:
                
                new_action_value = self.action_values[action]
                if new_action_value > best_action_value:
                    best_action_value = new_action_value
                    best_action = action
        else:
            worst_action_value = np.inf
            for action in self.legal_actions:
                new_action_value = self.action_values[action]
                if new_action_value < worst_action_value:
                    worst_action_value = new_action_value
                    best_action = action
        
        return best_action


    def select_move(self, exploration_weight: float):
        # Division by zero issues here. 
        # The paper does not state how to deal with problems when you have not yet visited the state
        best_action = -1
        if self.black_to_play:
            best_action_value = -np.inf
            for action in self.legal_actions:
                if self.children_visit_count[action] == 0:
                    # Ensure all actions are tried at least once, which is neccecary for MCTS
                    return action
                else:
                    new_action_value = self.action_values[action] + exploration_weight*np.sqrt(np.log(self.total_visit_count)/self.children_visit_count[action])
                if new_action_value > best_action_value:
                    best_action_value = new_action_value
                    best_action = action
        else:
            worst_action_value = np.inf
            for action in self.legal_actions:
                if self.children_visit_count[action] == 0:
                    # Ensure all actions are tried at least once, which is neccecary for MCTS
                    return action
                else:
                    new_action_value = self.action_values[action] - exploration_weight*np.sqrt(np.log(self.total_visit_count)/self.children_visit_count[action])
                if new_action_value < worst_action_value:
                    worst_action_value = new_action_value
                    best_action = action
        
        # if best_action == -1:
        #     raise AssertionError('No action selected, this should never happen')

        return best_action


class MCTreeSearch:
    def __init__(self, root: MCTreeNode, game: GameInterface, exploration_weight: float, random_policy: bool) -> None:
        self.root = root
        self.random_policy = random_policy

        # One to modify, and one to save
        self.game = game
        self.starting_game = deepcopy(game)
        
        self.exploration_weight = exploration_weight


    def sim_default(self):
        while not self.game.is_final_state():
            if self.random_policy:
                action = random.choice(self.game.get_legal_acions(True))
            else:
                raise NotImplementedError("Use policy network")
            self.game.perform_action(action)

        return self.game.get_final_state_reward()


    def sim_tree(self, starting_node: MCTreeNode) -> MCTreeNode:

        current_node = starting_node

        while not self.game.is_final_state():
            action = current_node.select_move(self.exploration_weight)
            self.game.perform_action(action)
            
            if not current_node.is_action_taken(action):
                new_leaf_node = MCTreeNode(current_node, action, self.game.is_starting_player_turn(), self.game.get_legal_acions(True), self.game.get_action_count())
                current_node.attach_child(new_leaf_node, action)
                return new_leaf_node
            current_node = current_node.get_child(action)

        return current_node


    def simulate(self):
        self.game = deepcopy(self.starting_game)

        leaf_node = self.sim_tree(root_node)

        game_result = self.sim_default()

        # Backup from the end of the tree, discarding the playout above, only using the result
        leaf_node.parent.backup(leaf_node.parent_action, game_result)
    

    def UCTSearch(self, iterations: int):
        for _ in range(iterations):
            self.simulate()

        # The board is not reset here to save time, 
        # so make sure that the starting_game is used instead of the game 
        # if the board is to be inspected between simulations
            
        # self.game = deepcopy(self.starting_game)
        return self.root.select_move(self.exploration_weight)



if __name__ == '__main__':

    # game: GameInterface = Nim([10,10], 4)
    game: GameInterface = Hex(5)
    board, black_to_play = game.get_state(False)

    exploration_weight = 1.0
    search_iterations = 4000

    game_states: list[GameInterface] = [deepcopy(game)]
    root_node = MCTreeNode(None, -1, black_to_play, game.get_legal_acions(True), game.get_action_count())

    while not game.is_final_state():
        
        mcts = MCTreeSearch(root_node, game, exploration_weight, random_policy=True)


        mcts.UCTSearch(search_iterations)
        best_action = mcts.root.best_action()

        # if mcts.root.black_to_play:
        #     mcts.UCTSearch(search_iterations)
        #     best_action = mcts.root.best_action()
        # else:
        #     best_action = int(input("Action: "))
        
        with np.printoptions(precision=2, suppress=True):
            print('best', best_action)     
            print('total node searches:', mcts.root.total_visit_count)     
            print(root_node.action_values)
        
        root_node = mcts.root.get_child(best_action)
        root_node.make_root()

        game.perform_action(best_action)
        game_states.append(deepcopy(game))
        game.display_current_state()


        


