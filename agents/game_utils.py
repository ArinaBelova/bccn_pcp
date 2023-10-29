from enum import Enum
import numpy as np
import copy
from collections import Counter
import torch

BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.full(shape=BOARD_SHAPE, fill_value=NO_PLAYER, dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. Here's an example output, note that we use
    PLAYER1_Print (X) to represent PLAYER1 and PLAYER2_Print (O) to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    board_format = lambda x: NO_PLAYER_PRINT if x == 0 else (PLAYER1_PRINT if x == 1 else PLAYER2_PRINT)
    output = ""
    output += "|==============|" + "\n"
    for row in board:
        output += "|"
        for el in row:
            output += board_format(el) + " "
        output += "|" + "\n"
    output += "|==============|" + "\n"
    output += "|0 1 2 3 4 5 6 |"

    return output


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    what_to_remove = lambda x: 0 if ((x == "|") or (x == "=") or x.isdigit()) else 1
    upd = ''.join(i for i in pp_board if what_to_remove(i)) 
    board_format = lambda x: 0 if x == NO_PLAYER_PRINT else (1 if x == PLAYER1_PRINT else 2)

    output = []

    for idx, row in enumerate(upd.split("\n")):
        if (idx == 0) or (idx == len(upd.split("\n")) - 2) or (idx == len(upd.split("\n")) - 1):
            continue 
        
        for idx, el in enumerate(row):
            # We need to delete meaningless spaces
            if idx % 2 == 0:
                output.append(board_format(el))
            else:
                continue 

    output = np.array(output).reshape(BOARD_SHAPE)

    return output


# Action function ?
def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).
    """
    # action is fucking chosen column
    board_copy = copy.deepcopy(board)
    lowest_row = np.argwhere(board[:,action] == 0)
    # We filled the column
    if not len(lowest_row):
        raise ValueError

    lowest_row_position = lowest_row[-1]

    board_copy[lowest_row_position, action] = player

    return board_copy


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    win_status = False

    # For rows
    for row in range(0, BOARD_ROWS):
        for column in range(3, BOARD_COLS):
            if (board[row][column] == board[row][column - 1] ==\
                board[row][column - 2] == board[row][column - 3] == player):
                    win_status = True
            else:
                continue   

    # For columns
    for column in range(0, BOARD_COLS):
        for row in range(3, BOARD_ROWS):
            if (board[row][column] == board[row - 1][column] ==\
                board[row - 2][column] == board[row - 3][column] == player):
                    win_status = True
            else:
                continue

    # For diags (both from left to right and from right to left)
    for row in range(0, 3):
        for column in range(0, 4):        
            if ((board[row][column] == board[row + 1][column + 1] ==\
                board[row + 2][column + 2] == board[row + 3][column + 3] == player) or
                (board[row + 3][column] == board[row + 2][column + 1] ==\
                board[row + 1][column + 2] == board[row][column + 3] == player)):
                    win_status = True
            else:
                continue

    return win_status            

def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN
    # The best draw case I imagine now is when the board is full of pieces    
    elif board.all():
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING         
