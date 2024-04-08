import numpy as np
import matplotlib.pyplot as plt

try:
    from game.game_interface import GameInterface
except ModuleNotFoundError:
    from game_interface import GameInterface


class Hex(GameInterface):


    def __init__(self, board_size: int, board: np.ndarray = None, current_black_player = True) -> None:
        # Must allow 3-10 range
        self.__board_size = board_size

        # Black player is starting
        # self.board[i,j,:] = [False, False] if cell is empty
        # self.board[i,j,:] = [True, False] if black player has a piece there
        # self.board[i,j,:] = [False, True] if red player has a piece there
        self.__current_black_player = current_black_player
        self.__empty_piece = np.array([False, False])
        self.__black_player_piece = np.array([True, False])
        self.__red_player_piece = np.array([False, True])

        # initialized to empty
        # the board is a diamond shape where 
        # board[0,0,:] -> top, board[-1,0,:] -> left, board[0,-1,:] -> right, board[-1,-1,:] -> bottom
        # black player want to connect top left to bottom right -> along axis 0
        # red player want to connect top right to bottom left -> along axis 1
        # The board is also connected horizontally, which in this representation 
        # means board[i,j,:] is connected to board[i-1,j+1,:] and board[i-1,j+1,:]
        # (In addition to its 4 neighbours)
        if board is None:
            self.__board = np.zeros((board_size, board_size, 2), dtype=np.bool_)
        else:
            self.__board = board

    
    def print_board(self):
        text = ' '
        for i in range(self.__board_size):
            for j in range(self.__board_size):
                if np.equal(self.__board[i,j,:], self.__black_player_piece).all():
                    text += 'x '
                elif np.equal(self.__board[i,j,:], self.__red_player_piece).all():
                    text += 'o '
                else:
                    text += '. '
            text += '\n '
        print(text)
        return text

    
    def __get_neighbours(self, id: tuple[int, int]) -> list[tuple[int, int]]:
        max_index = self.__board_size - 1
        x, y = id

        if x == 0:
            if y == 0:
                return [          (x+1, y),
                        (x, y+1)]
            elif y == max_index:
                return [(x, y-1), (x+1, y-1), 
                                  (x+1, y)]
            else:
                return [(x, y-1), (x+1, y-1), 
                                  (x+1, y),
                        (x, y+1)]
        elif x == max_index:
            if y == 0:
                return [(x-1, y),
                        (x-1, y+1), (x, y+1)]
            elif y == max_index:
                return [            (x, y-1),
                        (x-1, y)]
            else:
                return [            (x, y-1),
                        (x-1, y),
                        (x-1, y+1), (x, y+1)]
        else:
            if y == 0:
                return [(x-1, y),             (x+1, y),
                        (x-1, y+1), (x, y+1)]
            elif y == max_index:
                return [            (x, y-1), (x+1, y-1), 
                        (x-1, y),             (x+1, y)]
            else:
                return [            (x, y-1), (x+1, y-1), 
                        (x-1, y),             (x+1, y),
                        (x-1, y+1), (x, y+1)]


    def __player_win(self, black_player: bool) -> bool:
        visited = np.zeros(shape=(self.__board_size, self.__board_size), dtype=np.bool_)

        if black_player:
            ## Start with the entire left column
            queue = [(i,0) for i in range(self.__board_size)]
            axis = 1 # Call off the search on the last column
            filled_check = self.__black_player_piece

        else: # Red player
            # Start with the entire top row 
            queue = [(0,i) for i in range(self.__board_size)]
            axis = 0 # Call off the search on the last row 
            filled_check = self.__red_player_piece


        i = len(queue) - 1
        while i >= 0:
            current_id = queue.pop(i)
            # Check that the current_id is actually filled by the correct player
            # If it is the id is added back to the queue
            if np.equal(self.__board[current_id[0],current_id[1],:], filled_check).all():
                queue.append(current_id)

            i -= 1

        while len(queue) > 0:
            # Remove the last element of the queue and mark it as visited
            current_id = queue.pop()


            visited[current_id] = True

            neighbours = self.__get_neighbours(current_id)


            for neighbour in neighbours:
                # If the neighbor is not filled by the player currently being checked, then do not consider it
                if not np.equal(self.__board[neighbour[0],neighbour[1],:], filled_check).all():
                    continue
                
                # If the neighbor is filled by the player, and it is on the last row/column the board must be filled
                # since a board position is only added to the queue after a path from that position is found 
                # back to the starting row/column
                if neighbour[axis] == self.__board_size - 1:
                    return True
                
                # If the neighbour has not been visited yet, and we have not planned to visit it in the future
                # (and it is filled by the correct player), then plan to visit the position
                if neighbour not in queue and not visited[neighbour]:
                    queue.append(neighbour)

        # If there was no path for the player from the starting row/column to the ending row/column
        return False


    def is_final_state(self) -> bool:
        """
        returns true if the game is in a final state
        """
        return self.__player_win(black_player=True) or self.__player_win(black_player=False)

    def get_final_state_reward(self) -> int:
        """
        raises an AssertionError if not a final state

        returns 0 for player 2 win (not starting), and 1 for player 1 win (starting player)
        """

        # Board can be in a winning position without being filled
        if self.__player_win(black_player=True):
            return 1
        
        if self.__player_win(black_player=False):
            # Player 2 won
            return 0

        # Since no player won, the board must not be filled yet
        # No need to check it        
        raise AssertionError("Not a final state")



    def get_legal_acions(self, flatten: bool = True) -> np.ndarray:
        empty_x_indices, empty_y_indices = np.where((self.__board == np.zeros(shape=(1,1,2), dtype=np.bool_)).all(axis=2))
        result = []
        if flatten:
            for i in range(empty_x_indices.shape[0]):
                result.append(empty_x_indices[i]*self.__board_size + empty_y_indices[i])
            return result
        else:
            for i in range(empty_x_indices.shape[0]):
                result.append((empty_x_indices[i], empty_y_indices[i]))
            return result 


    def get_state(self, flatten: bool = True) -> np.ndarray | tuple[np.ndarray, bool]:
        if flatten:
            return np.append(self.__board.flatten(), self.__current_black_player, not self.__current_black_player)
        else:
            return (self.__board, self.__current_black_player)
        
        
    def get_fully_rotated_state(self, flatten: bool = True) -> np.ndarray | tuple[np.ndarray, bool]:
        # __board is of size (self.__board_size, self.__board_size, 2)
        # The rotation is done by rotating the board 180 degrees clockwise

        if flatten:
            return np.append(np.rot90(self.__board, k=2).flatten(), self.__current_black_player, not self.__current_black_player)
        else:
            return (np.rot90(self.__board, k=2), self.__current_black_player)
    

    def is_starting_player_turn(self) -> bool:
        """
        returns a boolean signifying whether it is the starting players turn
        """
        return self.__current_black_player


    def perform_action(self, action: int | tuple[int], flattend_input: bool = True) -> None:
        if flattend_input:
            # The following definition means we first loop over all columns in a row, and then start again at the next row
            x, y = (action // self.__board_size, action % self.__board_size)
        else:
            x, y = action
            
        # Extract the locationfirst true element
        # id = np.where(action == True)
        # if id[0].shape[0] != 1:
        #     raise ValueError('Given action contains more than one element')
        # x, y = id[0][0], id[1][0]

        if not np.equal(self.__board[x, y, :], self.__empty_piece).all():
            raise ValueError('Trying to place a piece on a non empty location')

        # Update the board and switch the players
        if self.__current_black_player:
            self.__board[x, y, :] = self.__black_player_piece
        else:
            self.__board[x, y, :] = self.__red_player_piece

        self.__current_black_player = not self.__current_black_player

    def get_action_count(self) -> int:
        """
        returns the max number of actions
        """
        return self.__board_size**2


    def display_current_state(self) -> None:
        # Create lists of indicies of where the different pieces are (In total the lists will contain self.__board_size**2 x and y points)
        # The indices act as a location of where the piece is

        # self.__board == self.__empty_piece checks that the piece at each location is [False False], which results in [True True]
        # (self.__board == self.__empty_piece).all(axis=2) converts [True, True] to True, and all other
        # values, like when the location is filled, becomes False
        # Now the array is 2D and has shape=(self.__board_size, self.__board_size)
        # np.where of that returns a tuple of the indices in the x and y direction that fulfil that condition
        empty_indices = np.where((self.__board == self.__empty_piece).all(axis=2))
        black_player_indices = np.where((self.__board == self.__black_player_piece).all(axis=2))
        red_player_indices = np.where((self.__board == self.__red_player_piece).all(axis=2))

        # For rotating 45 degrees clockwise
        rotation_matrix = np.array([
            [np.cos(-135*np.pi/180), -np.sin(-135*np.pi/180)],
            [np.sin(-135*np.pi/180), np.cos(-135*np.pi/180)]
        ], dtype=np.float32)


        # Applying the rotation to the indices
        def transform_indices(indices: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
            n = indices[0].shape[0]

            result = np.zeros(shape=(n,2) , dtype=np.float32)

            for i in range(n):
                scaled_location = np.array([indices[0][i], indices[1][i]], dtype=np.float32)
                result[i,:] = rotation_matrix.dot(scaled_location)
            
            return result
        
        # Calculating the endpoints for the horizontal, vertical and diagonal lines, and rotating them 
        line_endpoints = []
        for i in range(self.__board_size):
            n = self.__board_size-1
            j = n-i

            start_x = rotation_matrix.dot(np.array((0, i), dtype=np.float32))
            end_x = rotation_matrix.dot(np.array((n, i), dtype=np.float32))
            start_y = rotation_matrix.dot(np.array((i, 0), dtype=np.float32))
            end_y= rotation_matrix.dot(np.array((i, n), dtype=np.float32))

            start_diag1 = rotation_matrix.dot(np.array((0,j), dtype=np.float32))
            end_diag1 = rotation_matrix.dot(np.array((j,0), dtype=np.float32))
            start_diag2 = rotation_matrix.dot(np.array((j, n), dtype=np.float32))
            end_diag2 = rotation_matrix.dot(np.array((n,j), dtype=np.float32))

            line_endpoints.extend(((start_x, end_x), (start_y, end_y), (start_diag1, end_diag1), (start_diag2, end_diag2)))

        # Transform the indicies to locations
        empty_locations = transform_indices(empty_indices)
        black_player_locations = transform_indices(black_player_indices)
        red_player_locations = transform_indices(red_player_indices)


        # Displaying the lines using their endpoints,
        for start, end in line_endpoints:
            plt.plot([start[0], end[0]], [start[1], end[1]], color='grey', zorder=0)

        # Add the red, black or empty pieces
        plt.scatter(empty_locations[:,0], empty_locations[:,1], s=100, facecolors='none', edgecolors='grey', zorder=10)
        plt.scatter(black_player_locations[:,0], black_player_locations[:,1], s=100, facecolors='black', zorder=10)
        plt.scatter(red_player_locations[:,0], red_player_locations[:,1], s=100, facecolors='red', zorder=10)

        # Some helpful text
        centre_edge_dist = np.sqrt(2)*(self.__board_size - 1)/2
        plt.text(-centre_edge_dist-(0.2*centre_edge_dist), -centre_edge_dist/2, "Black start", color='black')
        plt.text(centre_edge_dist-(0.2*centre_edge_dist), -centre_edge_dist/2, "Red start", color='red')
        plt.text(centre_edge_dist-(0.2*centre_edge_dist), -centre_edge_dist*3/2, "Black end", color='black')
        plt.text(-centre_edge_dist-(0.2*centre_edge_dist), -centre_edge_dist*3/2, "Red end", color='red')

        # Make it nice
        plt.gca().set_aspect(1.25)
        plt.gca().set_axis_off()
        
        plt.show()


if __name__ == '__main__':
    # hex: GameInterface = Hex(5)
    
    # hex.perform_action((1,2), flattend_input=False)
    # hex.perform_action((3,2), flattend_input=False)
    # hex.perform_action(10)
    # hex.perform_action((3,3), flattend_input=False)
    # hex.perform_action((1,1), flattend_input=False)
    # hex.perform_action((4,4), flattend_input=False)
    # hex.perform_action((1,3), flattend_input=False)

    # print(hex.get_final_state_reward())
    # print(hex.is_final_state())

    # hex.perform_action((2,4), flattend_input=False)
    # hex.perform_action((1,4), flattend_input=False)


    # print(hex.get_final_state_reward())
    # print(hex.is_final_state())
    # print(hex.get_legal_acions())
    # hex.display_current_state()

    hex: GameInterface = Hex(3)
    
    # hex.perform_action((2,2), flattend_input=False)
    # print(hex.is_final_state())
    # hex.perform_action((0,0), flattend_input=False)
    # print(hex.is_final_state())
    # hex.perform_action((1,1), flattend_input=False)
    # print(hex.is_final_state())
    # hex.perform_action((0,1), flattend_input=False)
    # print(hex.is_final_state())
    # hex.perform_action((1,0), flattend_input=False)
    # print(hex.is_final_state())
    # hex.perform_action((1,2), flattend_input=False)
    # print(hex.is_final_state())
    # hex.perform_action((2,1), flattend_input=False)
    # print(hex.is_final_state())

    # print(hex.get_final_state_reward())


    hex.perform_action(7)
    hex.display_current_state()
