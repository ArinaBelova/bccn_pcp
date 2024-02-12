import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim

from yet_another_mcts import UCT
from mlp import MLP
from cnn import CNN

import datetime
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/galan/Desktop/programming_project_bccn/')

from game_utils import *

class Trainer:

    def __init__(self, model, args):
        # TODO: maybe logic to re-load an existing model and train it more
        self.model = model
        self.args = args


    def execute_episode(self):
        train_examples = []
        current_player = 1
        state = initialize_game_state()
        num_iterations = self.args["num_simulations"]

        while True: # until the end of the game -> end of the episode
            action, rootnode = UCT(state, current_player, num_iterations, self.model)

            action_probs = np.zeros(7)

            for m, v in [(c.move, c.visits) for c in rootnode.childNodes]:
                action_probs[m] = v

            action_probs = action_probs * vectorise_possible_moves(get_valid_positions(state))
            if np.sum(action_probs):
                action_probs = action_probs / np.sum(action_probs)
            train_examples.append((state, current_player, action_probs))

            # check validity of the move
            apply_player_action(state, action, current_player)
            current_player = switch_player(current_player)
            end_state = check_end_state(state, current_player)
        
            if end_state is not GameState.STILL_PLAYING:
                if end_state is GameState.IS_WIN:
                    value = 1
                else:
                    value = 0 # ony IS_DRAW case, no support for loosing case

                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, value * ((-1) ** (hist_current_player != current_player))))

                return ret

            # current_player = switch_player(current_player)    

    def learn(self):
        print("Started collecting the data for training")

        for i in range(self.args['numIters']):
            train_examples = []
           
            iteration_train_examples = self.execute_episode()
            train_examples.extend(iteration_train_examples)

        shuffle(train_examples)
        self.train(train_examples)
        ckp_dir = self.args['checkpoint_path']
        self.save_checkpoint(folder = ckp_dir)


    def train(self, examples):
        # optimizer = optim.NAdam(self.model.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        pi_losses = []
        v_losses = []

        avg_policy_losses = []
        avg_value_losses = []
        for epoch in range(self.args['epochs']):
            print(f"Epoch # {epoch}")
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.FloatTensor(np.array(boards).astype(np.float32))
                target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

                # predict
                boards = boards.to(self.model.device)
                target_pis = target_pis.to(self.model.device)
                target_vs = target_vs.to(self.model.device)
                
                # print(boards.shape)
                # print(target_pis.shape)
                # print(target_vs.shape)

                # compute output
                l_pi, l_v = self.model(boards, target_pis, target_vs)
                total_loss = 0.5 * l_pi + 0.5 * l_v     # both losses have equal weights towards the total loss     

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))

            avg_policy_losses.append(np.mean(pi_losses))
            avg_value_losses.append(np.mean(v_losses))  

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  

        axs[0].plot(avg_policy_losses, label='Training Avg Policy Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Policy')
        axs[0].legend()

        axs[1].plot(avg_value_losses, label='Training Avg Value Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Value')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pt"
        filepath = os.path.join(folder, filename)
        torch.save(self.model.state_dict(), filepath)


if __name__ == '__main__':
    args = {
    'batch_size': 32,
    'numIters': 50, # 500,                                # Total number of training iterations
    'num_simulations': 1000,#1000,                         # Total number of MCTS simulations to run when deciding on a move to play
    #'numEps': 100,                                  # Number of full games (episodes) to run during each iteration
    'epochs': 1000,                                    # Number of epochs of training per iteration
    'checkpoint_path': "/home/galan/Desktop/programming_project_bccn/agents/agent_mcts_nn/models"   # location to save latest set of weights
    }


    model = MLP(6*7, 7)
    trainer = Trainer(model, args)
    trainer.learn()


