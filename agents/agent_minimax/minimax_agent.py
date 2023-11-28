from agents.game_utils import *
import numpy as np
from typing import Optional, Tuple

# https://realpython.com/documenting-python-code/

def check_terminal_conditions(board: np.ndarray, our_player: BoardPiece, another_player: BoardPiece, valid_pos: np.array, current_depth: int, target_depth: int) -> Tuple[bool, int, int]: 
    """
    The function checks for the terminal condition of recursion:

    1. whether our player won or lost (in which case another player looses)
    2. whether it is a draw (in which case there are no more valid moves to play)
    3. whether we reached a target_depth of a search tree 

    Parameters:
    board (np.ndarray): The game board.
    our_player (BoardPiece): The player who is currently making a move.
    saved_state (Optional[SavedState]): The saved state of the game.

    Returns:
    Tuple[PlayerAction, Optional[SavedState]]: The optimal action for the player and the saved state.
    """
    
    # won
    if check_end_state(board, our_player) == GameState.IS_WIN:
        return (True, None, 10000)

    # lost
    if check_end_state(board, another_player) == GameState.IS_WIN:
        return (True, None, -10000)  

    # No more valid moves to take and we didn't win/loose the game
    if (len(valid_pos) == 0) or (current_depth == target_depth):
        return (True, None, 0)  

    return (False, None, None)       


def agent_move(board: np.ndarray, current_depth: int, target_depth: int, is_our_player: bool, our_player: BoardPiece, alpha: float, beta: float) -> Tuple[PlayerAction, int]:
    """
    The function recursively computes the optimal move and its associated score using the Minimax algorithm with alpha-beta pruning.

    Parameters:
    - board (np.ndarray): The current game board.
    - current_depth (int): The current depth in the search tree.
    - target_depth (int): The target depth for the search tree.
    - is_our_player (bool): Flag indicating if the current player is our player.
    - our_player (BoardPiece): The player for whom the move is being considered.
    - alpha (float): The alpha value for alpha-beta pruning.
    - beta (float): The beta value for alpha-beta pruning.

    Returns:
    Tuple[PlayerAction, int]: The recommended move and its associated score.
    """
    
    valid_pos = get_valid_positions(board)
    # cheap hack: 1 + 2 = 3 :)
    another_player = 3 - our_player
    
    terminal = check_terminal_conditions(board, our_player, another_player, valid_pos, current_depth, target_depth)
    if terminal[0]:
        return terminal[1], terminal[2]
 
    if is_our_player:
        value = -np.inf
        # Just a placeholder
        column = 0

        for col in valid_pos:
            # get the score for the action we play
            mod_board = apply_player_action(board, col, our_player)
            new_score = agent_move(mod_board, current_depth + 1, target_depth, False, another_player, alpha, beta)[1]
            
            if new_score > value:
                value = new_score
                column = col
            
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return column, value

    else:
        value = np.inf
        # Just a placeholder
        column = 0

        for col in valid_pos:
            # get the score for the action we play
            mod_board = apply_player_action(board, col, another_player)
            new_score = agent_move(mod_board, current_depth + 1, target_depth, True, our_player, alpha, beta)[1]
            
            if new_score < value:
                value = new_score
                column = col

            beta = min(beta, value)
            if alpha >= beta:
                break

        return column, value


def minimax_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    A wrapper which signature will be consistent with other agents classes.
    It determines the optimal move for a player in a game using the minimax algorithm with alpha-beta pruning.

    The function uses the minimax algorithm with alpha-beta peuning to explore the game tree up to a certain depth and evaluate the game states. 
    It returns the optimal move for the player and saved state.

    Parameters:
    board (np.ndarray): The game board.
    player (BoardPiece): The player who is currently making a move.
    saved_state (Optional[SavedState]): The saved state of the game.

    Returns:
    Tuple[PlayerAction, Optional[SavedState]]: The optimal action for the player and the saved state.
    """
    current_depth = 0
    target_depth = 4
    is_our_player = True
    alpha = -np.inf
    beta = np.inf

    return agent_move(board, current_depth, target_depth, is_our_player, player, alpha, beta)[0], saved_state