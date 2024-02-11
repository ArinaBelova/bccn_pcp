from copy import deepcopy
import random
from math import *
import numpy as np

import sys
sys.path.append('/home/galan/Desktop/programming_project_bccn/')

from game_utils import *


class Node:
    """ A node object in the game tree. 
    """
    def __init__(self, board_state: np.ndarray, player: BoardPiece, move: int = None, parent = None):
        self.board_state = deepcopy(board_state)
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        # Note wins is always from the viewpoint of playerJustMoved.
        self.wins = 0
        self.visits = 0 #1e-10
        self.untriedMoves = get_valid_positions(self.board_state) # future child nodes
        self.player = player # next player to tha player that just moved is our node's player, no bullshit with alternations anymore

    def UCT_select_child(self, c = sqrt(2)):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        def UCT_value(node):
            if node.visits == 0:
                return float('inf') 
            else:
                return (node.wins / node.visits) + c * sqrt(log(node.parentNode.visits / node.visits)) + 1e-6

        s = sorted(self.childNodes, key = lambda node: UCT_value(node))[-1]
        # print("VALUE UCT: ",UCT_value(s))
        return s

    def add_child(self, child, move):
        self.childNodes.append(child)
        self.untriedMoves = np.delete(self.untriedMoves, np.argwhere(self.untriedMoves == move))     

    def update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result    

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def tree_to_string(self, indent):
        s = self.indent_string(indent) + str(self)
        for c in self.childNodes:
             s += c.tree_to_string(indent+1)
        return s

    def indent_string(self,indent):
        s = "\n"
        for _ in range (1,indent+1):
            s += "| "
        return s

    def children_to_string(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s    


def give_results(winning_player: BoardPiece, our_player: BoardPiece) -> int: 
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
    if winning_player == our_player:
        return 1

    elif winning_player == 3 - our_player:
        return -1 


def switch_player(player):
    return 3 - player


def UCT(rootstate, player, saved_state, num_iterations):
    # as don't want to change the rootstate board on the next iteration
    rootstate = Node(rootstate, player)
    node = rootstate

    # do a first step of expansion from root here

    for move in rootstate.untriedMoves:
        parent_board_copy = deepcopy(rootstate.board_state)
        apply_player_action(parent_board_copy, move, player)
        child = Node(parent_board_copy, switch_player(rootstate.player), move, rootstate)
        #print(parent_board_copy)
        rootstate.add_child(child, move)

    for _ in range(num_iterations):
        node = rootstate
    
        # we want to start with something thus we check the moves from current board state  
        while (not len(node.untriedMoves) and len(node.childNodes)): # node is fully expanded and non-terminal
            #print("Selection")
            # Need to go through the nodes that we have after expand step 
            # print(len(node.childNodes))
            # print(node.parentNode)
            node = node.UCT_select_child()
            print(f"The best move is {node.move}")

        #print("Expansion")
        # expansion criterion -> go to expansion before selecting the next moves 
        while (len(node.untriedMoves) and node.visits != 0):
            #print("I'm in the expansion!!!!")
            move = random.choice(node.untriedMoves)
            parent_board_copy = deepcopy(node.board_state)
            #print(parent_board_copy)
            apply_player_action(parent_board_copy, move, node.player)
            #print(parent_board_copy)
            child = Node(parent_board_copy, switch_player(node.player), move, node)
            node.add_child(child, move) # add a child and remove the move that produced this child

        #print("Rollout")
        # rollout criterion
        # Need the copies again as the board will be re-used in the expansion from a given node later
        current_player = node.player # switch_player(node.player)
        current_board_copy = deepcopy(node.board_state)
        while check_end_state(current_board_copy, current_player) is GameState.STILL_PLAYING:
            # current_player = switch_player(current_player)
            move = random.choice(get_valid_positions(current_board_copy))
            apply_player_action(current_board_copy, move, current_player)
            current_player = switch_player(current_player)

        # check what end state we have and for which player
        reward = give_results(current_player, node.player)     

        #print("Backpropagation")
        # backpropagation criterion
        while node is not None: 
            #print(f"I'm doing update of node with move {node.move} and player {node.player} with visits {node.visits} and wins {node.wins}")
            node.update(reward)
            #print(f"I did update of node with move {node.move} and player {node.player} with visits {node.visits} and wins {node.wins}")
            reward = reward * (-1) # change reward as we will change the prespective player-wise
            node = node.parentNode

        # print([node.visits for node in rootstate.childNodes])

    # print(rootstate.tree_to_string(0))

    return sorted(rootstate.childNodes, key = lambda node: node.wins)[-1].move  # node.wins / node.visits

def mcts_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]):
    # TODO: where to create a network so it will train and will not be re-intiialised every time?
    # SOmehow gve it a trained newotk -> load here saved weights from outside
    # Don't change the network while playing with human opponent, all the trianing is not the responsbilty of this funciton
    return UCT(rootstate = board, player = player, saved_state = saved_state, num_iterations = 3), saved_state

