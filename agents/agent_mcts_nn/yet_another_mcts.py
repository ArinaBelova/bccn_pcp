from copy import deepcopy
import random
from math import *
import numpy as np

import torch

import sys
sys.path.append('/home/galan/Desktop/programming_project_bccn/')

from game_utils import *

sys.path.append('/home/galan/Desktop/programming_project_bccn/agents/agent_mcts_nn/')
from mlp import MLP

from cnn import CNN



class Node:
    """ A node object in the game tree. 
    """
    def __init__(self, board_state: np.ndarray, player: BoardPiece, move: int = None, parent = None, prior: float = 0):
        self.board_state = deepcopy(board_state)
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        # Note wins is always from the viewpoint of playerJustMoved.
        self.wins = 0
        self.visits = 0
        self.prior = prior
        self.untriedMoves = get_valid_positions(self.board_state) # future child nodes
        self.player = player # next player to tha player that just moved is our node's player, no bullshit with alternations anymore


    def UCT_select_child(self, c = sqrt(2)):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        def UCT_value(node):
            #print(node.parentNode.visits)
            if node.visits == 0:# was float()""inf
                return float("inf")#0 + node.prior * c * sqrt(node.parentNode.visits) / (node.visits+1)
            else:
                return (node.wins / node.visits) + node.prior * c * sqrt(log(node.parentNode.visits) / node.visits)  # + 1e-6

        s = sorted(self.childNodes, key = lambda node: UCT_value(node))[-1]
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
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        # {} self.state.__str__(), 
        return "Prior: {} Count: {} Value: {} \n".format(prior, self.visits, self.wins/self.visits)



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
    else:
        return 0


def UCT(rootstate, player, num_iterations, model):
    # as don't want to change the rootstate board on the next iteration
    rootstate = Node(rootstate, player)
    node = rootstate

    # do a first step of expansion from root here

        # action_probs, value = model.predict(state)
        # valid_moves = self.game.get_valid_moves(state)
        # action_probs = action_probs * valid_moves  # mask invalid moves
        # action_probs /= np.sum(action_probs)
        # root.expand(state, to_play, action_probs)

    action_probs, value = model.predict(node.board_state)
    action_probs = action_probs * vectorise_possible_moves(node.untriedMoves)
    action_probs /= np.sum(action_probs)

    for move in node.untriedMoves:
        parent_board_copy = deepcopy(node.board_state)
        apply_player_action(parent_board_copy, move, player)
        child = Node(board_state = parent_board_copy, player = switch_player(node.player), move = move, parent = rootstate, prior = action_probs[move])
        node.add_child(child, move)

    #print([n.prior for n in node.childNodes])

    for _ in range(num_iterations):
        node = rootstate

        #print("Selection")
        # we want to start with something thus we check the moves from current board state  
        while (not len(node.untriedMoves) and len(node.childNodes)): # node is fully expanded and non-terminal
            # Need to go through the nodes that we have after expand step 
            # print(len(node.childNodes))
            # print(node.parentNode)
            node = node.UCT_select_child()
                
        #print("Expansion")
    
        # expansion criterion -> go to expansion before selecting the next moves 
        parent_board_copy = deepcopy(node.board_state)
        current_player = node.player
        if check_end_state(parent_board_copy, current_player) is GameState.STILL_PLAYING:
            action_probs, value = model.predict(node.board_state)
            valid_moves = vectorise_possible_moves(node.untriedMoves)
            action_probs = action_probs * valid_moves  # mask invalid moves
            # print((action_probs))
            if np.sum(action_probs) != 0.:
                action_probs /= np.sum(action_probs)
            
            for move, prob in enumerate(action_probs):
                # print(prob!=0.)
                if prob != 0.:
                    parent_board_copy = deepcopy(node.board_state)
                    apply_player_action(parent_board_copy, move, current_player)
                    child = Node(parent_board_copy, switch_player(current_player), move, node, prob)
                    node.add_child(child, move)
        elif check_end_state(parent_board_copy, current_player) is GameState.IS_WIN:
            value = 1
            # _, value = model.predict(node.board_state)


        # while (len(node.untriedMoves) and node.visits != 0):
            # move = random.choice(node.untriedMoves)
            # parent_board_copy = deepcopy(node.board_state)
            # apply_player_action(parent_board_copy, move, node.player)
            # child = Node(parent_board_copy, switch_player(node.player), move, node)
            # node.add_child(child, move) # add a child and remove the move that produced this child
        
        
        #print("Rollout") <- don't need it when have NN
        # Need the copies again as the board will be re-used in the expansion from a given node later
        # current_player = node.player
        # current_board_copy = deepcopy(node.board_state)
        # while check_end_state(current_board_copy, current_player) is GameState.STILL_PLAYING:
        #     move = random.choice(get_valid_positions(current_board_copy))
        #     apply_player_action(current_board_copy, move, current_player)
        #     current_player = switch_player(current_player)

        # # check what end state we have and for which player
        # reward = give_results(current_player, node.player)     


        #print("Backpropagation")
        # backpropagation criterion
        while node is not None: 
            node.update(value) # reward
            value = value * (-1)
            # reward = reward # * (-1) # change reward as we will change the prespective player-wise
            node = node.parentNode

    # print(rootstate.childNodes)
    return sorted(rootstate.childNodes, key = lambda node: node.wins / (node.visits ))[-1].move, rootstate

def mcts_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]):
    # TODO: where to create a network so it will train and will not be re-intiialised every time?
    # SOmehow gve it a trained newotk -> load here saved weights from outside
    # Don't change the network while playing with human opponent, all the trianing is not the responsbilty of this funciton
    in_dim = np.prod(board.shape)
    out_dim = 7

    model = MLP(in_dim, out_dim)
    #model.load_state_dict(torch.load("/home/galan/Desktop/programming_project_bccn/agents/agent_mcts_nn/models/2024-02-11_18-28-59.pt"))
    checkpoint = torch.load("/home/galan/Desktop/programming_project_bccn/agents/agent_mcts_nn/models/2024-02-12_07-31-55.pt", map_location='cpu')
    model.load_state_dict(checkpoint) # checkpoint["state_dict"]

    out, _ = UCT(rootstate = board, player = player, num_iterations = 1000, model = model)

    return out, saved_state

