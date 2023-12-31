import numpy as np
from game_utils import *
import pytest

# Nice check if the test does nto compile: pytest --collect-only

def test_initialize_game_state():
    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

def test_pretty_board_string_match():
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 2, 1, 1, 0, 0],
                         [0, 2, 1, 2, 2, 0, 0],
                         [0, 2, 2, 1, 1, 0, 0]])

    our_out = pretty_print_board(board[::-1])
    # Don't make fucking tabs for formatting as the test will fail
    str_to_compare = """|==============|
|              |
|              |
|    X X       |
|    O X X     |
|  O X O O     |
|  O O X X     |
|==============|
|0 1 2 3 4 5 6 |"""

    assert our_out == str_to_compare

def test_is_it_identity():
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 2, 1, 1, 0, 0],
                         [0, 2, 1, 2, 2, 0, 0],
                         [0, 2, 2, 1, 1, 0, 0]])  
    
    pp_board = pretty_print_board(board[::-1])
    back_to_array = string_to_board(pp_board)

    assert (board == back_to_array).all()

def test_make_four_row_actions_and_win():
    board = initialize_game_state()
    apply_player_action(board, 0, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

    apply_player_action(board, 1, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

    apply_player_action(board, 2, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

    apply_player_action(board, 3, PLAYER1)
    assert check_end_state(board, PLAYER2) == GameState.STILL_PLAYING
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN 


def test_make_four_column_actions_and_win():
    board = initialize_game_state()
    apply_player_action(board, 0, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

    apply_player_action(board, 0, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

    apply_player_action(board, 0, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

    apply_player_action(board, 0, PLAYER1)
    assert check_end_state(board, PLAYER2) == GameState.STILL_PLAYING
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN 


def test_make_four_diag_actions_and_win():
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 2, 2, 1, 0],
                        [0, 0, 0, 2, 1, 1, 1]])

    # because we have a mess with pretty board                        
    board = board[::-1]
    
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING 
    assert check_end_state(board, PLAYER2) == GameState.STILL_PLAYING

    apply_player_action(board, 3, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN 

def test_draw_condition():
    board = np.array([[2, 1, 1, 2, 1, 2, 1],
                      [2, 1, 2, 1, 2, 1, 2],
                      [1, 2, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 2, 2, 1],
                      [2, 1, 2, 1, 2, 1, 2],
                      [1, 2, 1, 2, 1, 2, 1]])

    assert check_end_state(board, PLAYER2) == GameState.IS_DRAW
    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW

def test_cannot_insert_anything_in_a_full_column():
    board = np.array([[0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1]])

    assert check_move_status(board, 6) == MoveStatus.FULL_COLUMN  
  
def test_get_valid_positions():
    board = np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 1, 1]])

    valid_pos = get_valid_positions(board)                                       
                   
    assert (valid_pos == np.array([0,1,2])).all()