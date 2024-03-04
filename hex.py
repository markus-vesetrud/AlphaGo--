from game_interface import GameInterface
import numpy as np

class Hex(GameInterface):


    def __init__(self, board_size: int) -> None:
        # Must allow 3-10 range
        self.__board_size = board_size

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


    def player_win(self, player1: bool) -> bool:
        queue = []
        visited = []

        if player1:
            # Start with the entire top row 
            queue = [(0,i) for i in range(self.__board_size)]
            axis = 0
        else:
            # Start with the entire left column
            queue = [(i,0) for i in range(self.__board_size)]
            axis = 0


        while len(queue) > 0:
            current_id = queue.pop()
            visited.append(current_id)
            neighbours = self.__get_neighbours(current_id)
            for neighbour in neighbours:
                if neighbour[axis] == self.__board_size - 1:
                    return True
                if neighbour not in queue and neighbour not in visited:
                    queue.append(neighbour)
        return False


    def is_final_state(self) -> int:
        """
        returns: -1 for loss, 0 for not final state, and 1 for win
        """

        # If any of the board positions equals [False, False] the board is not complete
        # if np.any(np.equal(self.__board, np.zeros(shape=(1,1,2), dtype=np.bool_))):
        #     return 0



        pass


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

    test = np.zeros(shape=(1,1,2), dtype=np.bool_)
    print(test)


