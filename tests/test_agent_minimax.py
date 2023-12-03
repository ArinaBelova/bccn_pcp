import numpy as np
from agents.agent_minimax.minimax_agent import *
from game_utils import *
    

def test_the_fact_that_max_agent_moves():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    player = PLAYER1
    current_depth = 0
    target_depth = 2
    is_our_player = True
    alpha = -np.inf
    beta = np.inf

    action, score = agent_move(board, current_depth, target_depth, is_our_player, player, alpha, beta)
    
    assert action in [0, 1, 2, 3, 4, 5, 6]
    assert isinstance(score, int)

 
 
def test_forced_vertical_win_move():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 0, 0, 0]
    ])  

    board = board[::-1]
    player = PLAYER1
    current_depth = 0
    target_depth = 2
    is_our_player = True
    alpha = -np.inf
    beta = np.inf

    action, _ = agent_move(board, current_depth, target_depth, is_our_player, player, alpha, beta)
    
    assert action == 0


def test_forced_horizontal_win_move():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0]
    ])  
    board = board[::-1]

    player = PLAYER1
    current_depth = 0
    target_depth = 1
    is_our_player = True
    alpha = -np.inf
    beta = np.inf

    action, score = agent_move(board, current_depth, target_depth, is_our_player, player, alpha, beta)
    
    assert action == 3   
    assert score == 10000


def test_forced_diagonal_win_move():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0, 0],
        [0, 1, 2, 2, 0, 0, 0],
        [1, 2, 1, 1, 0, 0, 0]
    ])  

    board = board[::-1]
    player = PLAYER1
    current_depth = 0
    target_depth = 1
    is_our_player = True
    alpha = -np.inf
    beta = np.inf

    action, score = agent_move(board, current_depth, target_depth, is_our_player, player, alpha, beta)
    
    assert action == 3   
    assert score == 10000   

def test_forced_diagonal_win_move_second_player():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0, 0],
        [0, 1, 2, 2, 2, 0, 0],
        [1, 2, 1, 1, 2, 2, 2]
    ])  

    board = board[::-1]
    player = PLAYER2
    current_depth = 0
    target_depth = 1
    is_our_player = False
    alpha = -np.inf
    beta = np.inf

    action, score = agent_move(board, current_depth, target_depth, is_our_player, player, alpha, beta)
    
    assert action == 2  
    assert score == -10000     


def test_the_fact_that_min_agent_moves():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]
    ], dtype=int)

    player = PLAYER1
    current_depth = 0
    target_depth = 2
    is_our_player = False
    alpha = -np.inf
    beta = np.inf

    action, score = agent_move(board, current_depth, target_depth, is_our_player, player, alpha, beta)

    assert action in [0, 1, 2, 3, 4, 5, 6]
    assert isinstance(score, int)


def test_check_terminal_condition_win():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]
    ])

    valid_pos = get_valid_positions(board)

    assert check_terminal_conditions(board, our_player = PLAYER1, another_player = PLAYER2, \
        valid_pos = valid_pos, current_depth = 0, target_depth = 4) == (True, None, 10000)


def test_check_terminal_condition_lost():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0]
    ])

    valid_pos = get_valid_positions(board)

    assert check_terminal_conditions(board, our_player = PLAYER1, another_player = PLAYER2, \
        valid_pos = valid_pos, current_depth = 0, target_depth = 4) == (True, None, -10000)


def test_check_terminal_condition_no_more_valid_moves():
    board = np.array([[2, 1, 1, 2, 1, 2, 1],
                      [2, 1, 2, 1, 2, 1, 2],
                      [1, 2, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 2, 2, 1],
                      [2, 1, 2, 1, 2, 1, 2],
                      [1, 2, 1, 2, 1, 2, 1]])

    valid_pos = get_valid_positions(board)

    assert len(valid_pos) == 0 

    assert check_terminal_conditions(board, our_player = PLAYER1, another_player = PLAYER2, \
        valid_pos = valid_pos, current_depth = 0, target_depth = 4) == (True, None, 0)


def test_check_terminal_condition_reached_max_depth():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0]
    ])

    valid_pos = get_valid_positions(board)

    assert check_terminal_conditions(board, our_player = PLAYER1, another_player = PLAYER2, \
        valid_pos = valid_pos, current_depth = 4, target_depth = 4) == (True, None, 0)


def test_check_terminal_condition_no_termination():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0]
    ])

    valid_pos = get_valid_positions(board)

    assert check_terminal_conditions(board, our_player = PLAYER1, another_player = PLAYER2, \
        valid_pos = valid_pos, current_depth = 0, target_depth = 4) == (False, None, None)