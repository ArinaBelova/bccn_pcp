from game_utils import *
import numpy as np
from typing import Optional, Tuple

# https://realpython.com/documenting-python-code/

def check_terminal_conditions(board: np.ndarray, our_player: BoardPiece, another_player: BoardPiece, valid_pos: np.array, current_depth: int, target_depth: int) -> Tuple[bool, int, int]: 
    """
    The function checks for the terminal condition of recursion:

    1. whether our player won or lost (in which case another player looses)
    2. whether it is a draw (in which case there are no more valid moves to play)
    3. whether we reached a target_depth of a search tree 

    Parameters
    ----------
    board : np.ndarray
        The game board.
    our_player : BoardPiece
        The player who is currently making a move.
    saved_state : Optional[SavedState]
        The saved state of the game.

    Returns
    ----------
    Tuple[PlayerAction, Optional[SavedState]]
        The optimal action for the player and the saved state.
    """
    
    # won
    if check_end_state(board, our_player) == GameState.IS_WIN:
        return (True, None, np.inf)

    # lost
    if check_end_state(board, another_player) == GameState.IS_WIN:
        return (True, None, -np.inf)  

    # No more valid moves to take and we didn't win/loose the game
    if (len(valid_pos) == 0) or (current_depth == target_depth):
        return (True, None, 0)  

    return (False, None, None)       


def update_alpha_value(alpha, beta, score):
    alpha = max(alpha, score)
    if alpha >= beta:
        return True
    else:
        return False   


def update_beta_value(alpha, beta, score):
    beta = min(beta, score)
    if alpha >= beta:
        return True
    else:
        return False 


def agent_move(board: np.ndarray, current_depth: int, target_depth: int, is_our_player: bool, player: BoardPiece, alpha: float, beta: float) -> Tuple[PlayerAction, int]:
    """
    The function recursively computes the optimal move and its associated score using the Minimax algorithm with alpha-beta pruning.

    Parameters
    ----------
    board : np.ndarray
        The current game board.
    current_depth : int
        The current depth in the search tree.
    target_depth : int
        The target depth for the search tree.
    is_our_player : bool
        Flag indicating if the current player is our player.
    our_player : BoardPiece
        The player for whom the move is being considered.
    alpha : float
        The alpha value for alpha-beta pruning.
    beta : float
        The beta value for alpha-beta pruning.

    Returns
    ----------
    Tuple[PlayerAction, int]
        The recommended move and its associated score.
    """
    
    valid_pos = get_valid_positions(board)
    
    # Always assume that "computer"/minimax player is player1 and non-minimax player is player1
    terminal = check_terminal_conditions(board, PLAYER1, PLAYER2, valid_pos, \
                                        current_depth, target_depth)
    if terminal[0]:
        return terminal[1], terminal[2]
 
    if is_our_player:
        score = -np.inf
        # Just a placeholder for a running best columns we can put the stone into
        best_position = 0

        for pos in valid_pos:
            copy_board = copy.deepcopy(board)

            # get the score for the action we play
            apply_player_action(copy_board, pos, PLAYER1)
            new_score = agent_move(copy_board, current_depth + 1, target_depth, False, \
                                 PLAYER2, alpha, beta)[1]
            
            if new_score > score:
                score = new_score
                best_position = pos
            
            if update_alpha_value(alpha, beta, score):
                break

        return best_position, score

    else:
        score = np.inf
        # Just a placeholder for a running best columns we can put the stone into
        best_position = 0

        for pos in valid_pos:
            copy_board = copy.deepcopy(board)

            # get the score for the action we play
            apply_player_action(copy_board, pos, PLAYER2)
            new_score = agent_move(copy_board, current_depth + 1, target_depth, True, \
                                 PLAYER1, alpha, beta)[1]
            
            if new_score < score:
                score = new_score
                best_position = pos

            if update_beta_value(alpha, beta, score):
                break

        return best_position, score


def minimax_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    A wrapper which signature will be consistent with other agents classes.
    It determines the optimal move for a player in a game using the minimax algorithm with alpha-beta pruning.

    The function uses the minimax algorithm with alpha-beta peuning to explore the game tree up to a certain depth and evaluate the game states. 
    It returns the optimal move for the player and saved state.

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
    current_depth = 0
    target_depth = 4
    is_our_player = True
    alpha = -np.inf
    beta = np.inf

    return agent_move(board, current_depth, target_depth, is_our_player, player, alpha, beta)[0], saved_state