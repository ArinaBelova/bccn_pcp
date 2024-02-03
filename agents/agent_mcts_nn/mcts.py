import random
from math import *
import numpy as np

import sys
sys.path.append('/home/galan/Desktop/programming_project_bccn/')

from game_utils import *
# from typing import Type
from agents.agent_minimax import minimax_move

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import datetime

# https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191
# https://www.andrew.cmu.edu/course/10-403/slides/S19_evolutionarymethods.pdf
# https://towardsdatascience.com/alphazero-chess-how-it-works-what-sets-it-apart-and-what-it-can-tell-us-4ab3d2d08867

class Node:
    """ A node object in the game tree. 
    """
    def __init__(self, board_state: np.ndarray, player: BoardPiece, move: int, parent):
        self.board_state = board_state
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        # Note wins is always from the viewpoint of playerJustMoved.
        self.wins = 0
        self.visits = 0
        self.untriedMoves = get_valid_positions(self.board_state) # future child nodes
        self.playerJustMoved = player # player that just moved is our current player

    def UCT_select_child(self, c = sqrt(0)):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda node: node.wins/node.visits + c * sqrt(log(node.parentNode.visits)/node.visits) + 1e-6)[-1]
        return s
    
    def add_child(self, move):
        """ Remove move from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        apply_player_action(board = self.board_state, action = move, player = self.playerJustMoved)
        # TODO: Need to change the player here otherwise only one player keeps acting ????
        new_node = Node(board_state = self.board_state.copy(), player = 3 - self.playerJustMoved, move = move, parent = self)
        self.untriedMoves = np.delete(self.untriedMoves, np.argwhere(self.untriedMoves == move)) # moves are 1-7 when np.delete() wants an index
        self.childNodes.append(new_node)
        return new_node
    
    def give_results(self, winning_player: BoardPiece, our_player) -> int: 
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
        if winning_player == 0:
            return 0

        if winning_player == our_player:
            return 1

        elif winning_player == 3 - our_player:
            return -1    


    def get_result_nn(self, value_prediction):
        """
        Calculate the rollout result based on the predicted value from the neural network.

        Parameters:
        - board: The current game board state.
        - value_prediction: The predicted value from the neural network.

        Returns:
        - result: The calculated rollout result.
        """
        # Example: Convert the value prediction to a result (win/loss/draw)
        if value_prediction > 0.5:
            result = 1  # Win
        elif value_prediction < -0.5:
            result = -1  # Loss
        else:
            result = 0  # Draw

        return result


    # Update for a particular player
    # Need to propagate this result up the tree 
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


def nn_training_loop(model, dataloader, optimizer, device, is_training):
    model.train()

    # Initialize the total loss for this epoch
    epoch_loss = 0  

    # Loop over the data
    for board, policy_value in enumerate(dataloader):

        images = images.to(device)
        masks = masks.to(device)

        # Pass the batch through the model
        if is_training:
            loss = model(images, masks)
        else:
            with torch.no_grad():
                loss = model(images, masks)

        # If in training mode, backpropagate the error and update the weights
        if is_training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # update the total loss of the epoch
        loss_item = loss.item()
        epoch_loss += loss_item

    # Return the average loss for this epoch
    return epoch_loss 



def UCT(rootstate, player, num_iterations, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in ther ange [0.0, 1.0]."""

    # TODO: playing from the current player
    rootnode = Node(board_state = rootstate, player = player, move = None, parent = None)

    # TODO: all the methods should act on the nodes and not boards/players
    for _ in range(num_iterations):
        node = rootnode
        # TODO: do we need that?
        # board_state = rootstate.copy()
        our_player = player

        # Select
        while (not len(node.untriedMoves)) and len(node.childNodes): # node is fully expanded and non-terminal
            # Need to go through the nodes that we have after expand step 
            node = node.UCT_select_child()


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

            # TODO:Do a player on the node!
            # Delegated to add_child: apply_player_action(board = board_state, action = move, player = player)
            # Apply player action on the level of the node, only need to know the move to make!
            node = node.add_child(move) # add child and descend the tree

        # Rollout would no longer existis with a neural network, we choose the best value based on NN pred
        # value is self.visits/self.wins 
        # value, policy = nn(board_state.flatten())
        # apply_player_action(board = board_state, action = np.argmax(policy), player = other_player) 

        
        # Rollout: start with a particular node and particular player, last node that we are at
        # Doing these copies as we don't want to screw the boards of the nodes
        copy_board = node.board_state.copy()
        sim_player = node.playerJustMoved
        while len(get_valid_positions(copy_board)): # while board_state is non-terminal
            # TODO: Instead of a random do a most probable action from a policy vector from NN?
            
            winning_player = 0
            if connected_four(copy_board, sim_player):
                winning_player = sim_player 
                break

            if connected_four(copy_board, 3 - sim_player):
                winning_player = 3 - sim_player
                break
            
            policy_vector, _ = mlp(copy_board.flatten())
            move = torch.argmax(policy_vector).item()

            # Still need to have it to change the board, check the reference to the player
            # random.choice(get_valid_positions(copy_board))
            apply_player_action(board = copy_board, action = move, player = sim_player)
            sim_player = 3 - sim_player

            # TODO: There may be a case that there is no connect 4      
            # TODO: check for connect 3 and connect 2 to update heruistics
            # TODO: check if there is a win or not, it may be multiple connect4 here in the current state of the code
            # break out of the loop, other player is a winner

        # Backpropagate from the expanded node and work back to the root node
        while node is not None: 
            # results are given the the NN value not this handmade policy
            # TODO: HOW THE FUCK TO DO THIS???
            #result = node.give_results(winning_player = winning_player, our_player=our_player) 
            
            _, value_prediction = mlp(node.board_state.flatten())
            result = node.give_results_nn(node.board_state, value_prediction)

            # result = value # from NN
            node.update(result = result)
            node = node.parentNode # TODO: that's how it was previously! 


    # Output some information about the tree - can be omitted
    if (verbose): 
        print(rootnode.tree_to_string(0))
    else: 
        print(rootnode.children_to_string())

    # return the move that was most visited
    return sorted(rootnode.childNodes, key = lambda node: node.visits)[-1].move # node.wins / node.visits

def mcts_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], trained_network: Optional[bool] = False):
    # TODO: where to create a network so it will train and will not be re-intiialised every time?
    # SOmehow gve it a trained newotk -> load here saved weights from outside
    # Don't change the network while playing with human opponent, all the trianing is not the responsbilty of this funciton

    return UCT(rootstate = board, player = player, mlp = mlp, saved_state = saved_state, num_iterations = 5000, verbose = False), saved_state




#=================================== Training the network ================================================
# Example training data generation
def generate_training_data(mcts_iterations, mlp, player):
    training_data = []

    initial_state = np.zeros((7, 6))  # Example initial game state
    root_state = initial_state.copy()
    for _ in range(mcts_iterations):
        # Simulate one MCTS iteration starting from the initial state
        selected_move = UCT(root_state, player, mcts_iterations, mlp)

        # TODO: problem with board overload
        # if selected-move is not in get_valid_valid_moves()
        # break the loop and re-init the root_state

        # Apply the selected move to update the board
        apply_player_action(board=root_state, action=selected_move, player=player)

        player = 3 - player

        # Get policy and value from the neural network
        policy_vector, value_prediction = mlp(root_state.flatten())

        # Generate training example
        training_data.append((root_state.flatten(), policy_vector, value_prediction))

    return training_data

# Example training loop
def train_neural_network(mlp, optimizer, training_data, num_epochs, batch_size):
    for epoch in range(num_epochs):
        random.shuffle(training_data)

        for batch_start in range(0, len(training_data), batch_size):
            batch = training_data[batch_start:batch_start + batch_size]
            inputs, targets_policy, targets_value = process_batch(batch)

            optimizer.zero_grad()
            policy_output, value_output = mlp(inputs)

            loss_policy = mlp.criterion_policy(policy_output, targets_policy)
            loss_value = mlp.criterion_value(value_output, targets_value)

            total_loss = loss_policy + loss_value
            total_loss.backward()
            optimizer.step()

# Example batch processing
def process_batch(batch):
    inputs = torch.stack([torch.Tensor(example[0]) for example in batch])
    targets_policy = torch.stack([torch.argmax(example[1]) for example in batch])
    targets_value = torch.stack([example[2] for example in batch])

    return inputs, targets_policy, targets_value



if __name__ == "__main__":
    # Example usage
    input_size = 42  # Example input size for a 7x6 Connect4 board
    num_actions = 7  # Example number of actions (columns to drop pieces into)
    mlp = MLP(input_size, num_actions)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)

    # Example MCTS and training parameters
    mcts_iterations = 1000
    num_epochs = 100
    batch_size = 8 # 32

    # Generate training data using MCTS
    # TODO: how do we cater for different players? 
    training_data = generate_training_data(mcts_iterations, mlp, player=1)

    # Train the neural network
    train_neural_network(mlp, optimizer, training_data, num_epochs, batch_size)

    # save the network
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pt"
    torch.save(mlp.state_dict(), f"./models/{name}")
