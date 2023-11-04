from agents.game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, apply_player_action
import numpy as np
from typing import Tuple, Optional

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    while True:
        action = np.random.randint(7)
        if len(np.argwhere(board[:,action] == NO_PLAYER)) != 0:
            break

    return action, saved_state
