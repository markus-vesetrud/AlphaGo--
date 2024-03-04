
class GameInterface:

    # def __init__(self) -> None:
    #     pass

    def is_final_state(self) -> int:
        """
        returns: -1 for loss, 0 for not final state, and 1 for win
        """
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
