from game_interface import GameInterface
import numpy as np

class Hex(GameInterface):


    def __init__(self, board_size: int, current_player1 = True) -> None:
        # Must allow 3-10 range
        self.__board_size = board_size
        self.current_player1 = current_player1

        # Player 1 is starting
        # self.board[i,j,:] = [False, False] if cell is empty
        # self.board[i,j,:] = [True, False] if player 1 has a piece there
        # self.board[i,j,:] = [False, True] if player 2 has a piece there
        # initialized to empty
        # the board is a diamond shape where 
        # board[0,0,:] -> top, board[-1,0,:] -> left, board[0,-1,:] -> right, board[-1,-1,:] -> bottom
        # player 1 want to connect top left to bottom right -> along axis 1
        # player 2 want to connect top right to bottom left -> along axis 0
        # The board is also connected horizontally, which in this representation 
        # means board[i,j,:] is connected to board[i-1,j+1,:] and board[i-1,j+1,:]
        # (In addition to its 4 neighbours)
        self.__board = np.zeros((board_size, board_size, 2), dtype=np.bool_)
        
    
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


    def __player_win(self, player1: bool) -> bool:
        visited = np.zeros(shape=(self.__board_size, self.__board_size), dtype=np.bool_)

        if player1:
            # Start with the entire top row 
            queue = [(0,i) for i in range(self.__board_size)]
            axis = 0 # Call off the search on the last row 
            filled_check = np.array([True, False])

        else: # Player 2
            # Start with the entire left column
            queue = [(i,0) for i in range(self.__board_size)]
            axis = 1 # Call off the search on the last column
            filled_check = np.array([False, True])



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


    def is_final_state(self) -> int:
        """
        Only works for games that cannot be drawn

        returns: -1 for player 2 win (not starting), 0 for not final state, and 1 for player 1 win (starting player)
        """

        # Board can be in a winning position without being filled
        if self.__player_win(player1=True):
            return 1
        
        if self.__player_win(player1=False):
            # Player 2 won
            return -1

        # Since no player won, the board must not be filled yet
        # No need to check it        
        return 0



    def get_legal_acions(self) -> list[bool]:
        """
        returns
        """
        pass


    def get_state(self) -> tuple[list[int], bool]:
        """
        returns
        """
        pass
    

    def display_current_state(self) -> None:
        pass


if __name__ == '__main__':
    hex: GameInterface = Hex(5)

    print(hex.is_final_state())




