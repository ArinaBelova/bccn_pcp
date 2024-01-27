import random
from math import *
import numpy as np
from game_utils import *
from typing import Type


class Node:
    """ A node object in the game tree. 
    """
    def __init__(self, board_state: np.ndarray, player: BoardPiece, move: int, parent):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        # Note wins is always from the viewpoint of playerJustMoved.
        self.wins = 0
        self.visits = 0
        self.untriedMoves = get_valid_positions(board_state) # future child nodes
        # self.playerJustMoved = board_state.playerJustMoved # the only part of the board_state that the Node needs later
        self.playerJustMoved = player # player that just moved is our current player

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, move, state, player):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        new_node = Node(board_state = state, player = player, move = move, parent = self)
        #self.untriedMoves.remove(move)
        np.delete(self.untriedMoves, move)
        self.childNodes.append(new_node)
        return new_node
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    # Printing:
    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


def give_results(board: np.ndarray, our_player: BoardPiece) -> int: 
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
        return 1 #np.inf

    # lost
    if check_end_state(board, 3 - our_player) == GameState.IS_WIN:
        return -1 #-np.inf 

    # No more valid moves to take and we didn't win/loose the game
    if check_end_state(board, our_player) == GameState.IS_DRAW or check_end_state(board, 3 - our_player) == GameState.IS_DRAW:
        return 0 

    return 0         


def node_selection():
    pass

def node_expansion():
    pass

def node_simulation():
    pass

def backpropagation():
    pass


def UCT(rootstate, player, num_iterations, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in ther ange [0.0, 1.0]."""

    # TODO: playing from the current player
    rootnode = Node(board_state = rootstate, player = player, move = None, parent = None)

    for _ in range(num_iterations):
        node = rootnode
        # TODO: do we need that?
        board_state = rootstate.copy()

        # Select
        while (not len(node.untriedMoves)) and len(node.childNodes): # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            apply_player_action(board = board_state, action = node.move, player = player)
            #board_state.DoMove(node.move)

        # Expand
        if len(node.untriedMoves): # if we can expand (i.e. board_state/node is non-terminal)
            move = random.choice(node.untriedMoves) 
            apply_player_action(board = board_state, action = move, player = player)
            #board_state.DoMove(m)
            node = node.AddChild(move, board_state, player) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a board_state.GetRandomMove() function
        while len(get_valid_positions(board_state)): # while board_state is non-terminal
            # TODO: Instead of a random do a most probable action from a policy vector from NN?
            apply_player_action(board = board_state, action = random.choice(get_valid_positions(board_state)), player = player)

        # Backpropagate from the expanded node and work back to the root node
        while node is not None: 
            result = give_results(board = board_state, our_player = player)
            node.Update(result = result)
            #node.Update(board_state.GetResult(node.playerJustMoved)) # board_state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): 
        print(rootnode.TreeToString(0))
    else: 
        print(rootnode.ChildrenToString())

    #return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited
    return sorted(rootnode.childNodes, key = lambda node: node.wins / node.visits)[-1].move

def mcts_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]):
    return UCT(rootstate = board, player = player, num_iterations = 1000, verbose = True), saved_state
