from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, apply_player_action
import numpy as np
from typing import Tuple, Optional

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    The function that generates a random move on the board depending 
    on the availability of the columns.

    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player who is currently making a move.
    saved_state : Optional[SavedState]
        The saved state of the game.

    Returns
    ----------
    Tuple[PlayerAction, Optional[SavedState]]
        The optimal action for the player and the saved state.
    """
    # Choose a valid, non-full column randomly and return it as `action`
    while True:
        action = np.random.randint(7)
        if len(np.argwhere(board[:,action] == NO_PLAYER)) != 0:
            break

    return action, saved_state
