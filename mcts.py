import numpy as np
from game.game_interface import GameInterface
from game.hex import Hex
from copy import deepcopy

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

        self.total_action_count = total_action_count
        self.parent: MCTreeNode = parent
        self.parent_action = parent_action
        self.children: list[MCTreeNode] = [None for _ in range(total_action_count)]

        self.total_visit_count = 0
        self.children_visit_count: list[int] = [0 for _ in range(total_action_count)]
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
        return self.children[action] != None
    

    def get_child(self, action: int):
        # Kidnapping
        return self.children[action]


    def select_move(self, exploration_weight: float):
        # Division by zero...
        best_action = -1
        if self.black_to_play:
            best_action_value = -1
            for action in self.legal_actions:
                new_action_value = self.action_values[action] + exploration_weight*np.sqrt(np.log(self.total_visit_count)/self.children_visit_count[action])
                if new_action_value > best_action_value:
                    best_action_value = new_action_value
                    best_action = action
        else:
            worst_action_value = np.inf
            for action in self.legal_actions:
                new_action_value = self.action_values[action] - exploration_weight*np.sqrt(np.log(self.total_visit_count)/self.children_visit_count[action])
                if new_action_value < worst_action_value:
                    worst_action_value = new_action_value
                    best_action = action
        
        if best_action == -1:
            raise AssertionError('No action selected')

        return best_action


class MCTreeSearch:
    def __init__(self, root: MCTreeNode, game: GameInterface, exploration_weight: float, default_policy: callable) -> None:
        self.root = root
        self.default_policy = default_policy

        # One to modify, and one to save
        self.game = game
        self.starting_game = deepcopy(game)
        
        self.exploration_weight = exploration_weight


    def sim_default(self):
        result = self.game.is_final_state()
        while result == 0:
            action = self.default_policy(self.game.get_state())
            self.game.perform_action(action)
            result = self.game.is_final_state()
        return result # +1 or -1


    def sim_tree(self, starting_node: MCTreeNode) -> MCTreeNode:

        current_node = starting_node

        while self.game.is_final_state() == 0:
            action = current_node.select_move()
            self.game.perform_action(action)
            
            if not current_node.is_action_taken(action):
                return MCTreeNode(current_node, action, self.game.get_state(False)[1], self.game.get_legal_acions(True), self.game.get_action_count())

            current_node = current_node.get_child(action)

        return current_node
    

    def simulate(self):
        self.game = deepcopy(self.starting_game)

        leaf_node = self.sim_tree(root_node)

        game_result = self.sim_default()

        leaf_node.backup(leaf_node.parent_action(), game_result)
    

    def UCTSearch(self):
        pass




if __name__ == '__main__':
    game: GameInterface = Hex(5)
    board, black_to_play = game.get_state(False)

    exploration_weight = 1.0

    root_node = MCTreeNode(None, -1, black_to_play, game.get_legal_acions(True), game.get_action_count())
    mcts = MCTreeSearch(root_node, game, exploration_weight, lambda x: 0)