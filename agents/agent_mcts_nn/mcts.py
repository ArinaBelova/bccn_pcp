import random
from math import *
import numpy as np
from game_utils import *
# from typing import Type
from agents.agent_minimax import minimax_move

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191
# https://www.andrew.cmu.edu/course/10-403/slides/S19_evolutionarymethods.pdf
# https://towardsdatascience.com/alphazero-chess-how-it-works-what-sets-it-apart-and-what-it-can-tell-us-4ab3d2d08867

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
        self.playerJustMoved = player # player that just moved is our current player

    def UCT_select_child(self, c = sqrt(10)):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda node: node.wins/node.visits + c * sqrt(log(self.visits)/node.visits))[-1]
        return s
    
    def add_child(self, move, board, player):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        new_node = Node(board_state = board, player = player, move = move, parent = self)
        self.untriedMoves = np.delete(self.untriedMoves, np.argwhere(self.untriedMoves == move)) # moves are 1-7 when np.delete() wants an index
        self.childNodes.append(new_node)
        return new_node
    
    def update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    # Printing:
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


# TODO: make it trainable and define DataLoader object for the dataset (state -> policy, value) so we can batch it well 
class MLP(nn.Module):
    def __init__(self, input_size, num_actions):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.policy_head = nn.Linear(64, num_actions)

        self.value_head = nn.Linear(64, 1)

        # Loss fn - for values MSE loss and for policies NLL loss?
        self.criterion_value = torch.nn.MSELoss()
        self.criterion_policy = torch.nn.NLLLoss()


    def forward(self, input, targets = None):
        input = torch.Tensor(input)

        input = F.relu(self.fc1(input))
        #input = F.relu(self.fc2(input))
        input = self.fc2(input)

        # Make policy vector a vector of probabilities
        policy_output = F.softmax(self.policy_head(input), dim=0)

        # Value is just a single value
        value_output = self.value_head(input)

        # TODO: assume for now that targets will be given in tuples: (policy_vector, value)
        if targets is not None:
            loss_policy = self.criterion_policy(policy_output, targets[0])
            loss_value = self.criterion_value(value_output, targets[1])
            
            # TODO: Additive loss ?
            return loss_policy + loss_value

        return policy_output, value_output


# def nn_training_loop(model, dataloader, optimizer, device, is_training):
#     model.train()

#     # Initialize the total loss for this epoch
#     epoch_loss = 0  

#     # Loop over the data
#     for board, policy_value in enumerate(dataloader):

#         images = images.to(device)
#         masks = masks.to(device)

#         # Pass the batch through the model
#         if is_training:
#             loss = model(images, masks)
#         else:
#             with torch.no_grad():
#                 loss = model(images, masks)

#         # If in training mode, backpropagate the error and update the weights
#         if is_training:
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#         # update the total loss of the epoch
#         loss_item = loss.item()
#         epoch_loss += loss_item

#     # Return the average loss for this epoch
#     return epoch_loss / (batch_id + 1)    


def UCT(rootstate, player, saved_state, num_iterations, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in ther ange [0.0, 1.0]."""

    # TODO: playing from the current player
    rootnode = Node(board_state = rootstate, player = player, move = None, parent = None)

    network = MLP(len(rootstate.flatten()), 7)
    for _ in range(num_iterations):
        node = rootnode
        # TODO: do we need that?
        board_state = rootstate.copy()
        other_player = player

        # Select
        while (not len(node.untriedMoves)) and len(node.childNodes): # node is fully expanded and non-terminal
            node = node.UCT_select_child()
            # print("WE ARE IN SELECTION PHASE")
            # print(node.move)
            # print("\n")
            apply_player_action(board = board_state, action = node.move, player = player)

        # Expand - until we hit a leaf node
        if len(node.untriedMoves): # if we can expand (i.e. board_state/node is non-terminal)
            # apply move which is argmax of policy of NN 
            # Vanilla mcts:
            move = random.choice(node.untriedMoves) 
            
            # NN mcts:
            #policy, value = network(board_state.flatten())
            # TODO: with this it is illegal to make move sometimes:
            #move = torch.argmax(policy)

            # TODO: check of legality of the move!!
            # TODO: keep the probabilities that are only legal
            # legal_policy = policy[node.untriedMoves]
            # move_idx = torch.argmax(legal_policy)
            # move = node.untriedMoves[move_idx]

            apply_player_action(board = board_state, action = move, player = player)
            #board_state.DoMove(m)
            node = node.add_child(move, board_state, player) # add child and descend tree

        # Rollout would no longer existis with a neural network, we choose the best value based on NN pred
        # value is self.visits/self.wins 
        # value, policy = nn(board_state.flatten())
        # apply_player_action(board = board_state, action = np.argmax(policy), player = other_player) 

        # Rollout
        while len(get_valid_positions(board_state)): # while board_state is non-terminal
            # TODO: Instead of a random do a most probable action from a policy vector from NN?
            
            # Tried to be "clever" and use for Rollout a move generated by minimax, didn't work
            # copy_board_state = board_state.copy()
            # clever_move, _ = minimax_move(copy_board_state, other_player, saved_state)
            # print("MINIMAX MOVE")
            # print(clever_move)
            # print("\n")
            # apply_player_action(board = board_state, action = clever_move, player = other_player)

            apply_player_action(board = board_state, action = random.choice(get_valid_positions(board_state)), player = other_player) 
            other_player = 3 - other_player


        # Backpropagate from the expanded node and work back to the root node
        while node is not None: 
            # results are given the the NN value not this handmade policy
            # TODO: HOW THE FUCK TO DO THIS???
            result = give_results(board = board_state, our_player = player) 
            # TODO: do a loss here?  nn(board_state, result)

            # result = value # from NN
            node.update(result = result)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): 
        print(rootnode.tree_to_string(0))
    else: 
        print(rootnode.children_to_string())

    # return the move that was most visited
    return sorted(rootnode.childNodes, key = lambda node: node.visits)[-1].move # node.wins / node.visits

def mcts_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]):
    return UCT(rootstate = board, player = player, saved_state = saved_state, num_iterations = 1000, verbose = False), saved_state
