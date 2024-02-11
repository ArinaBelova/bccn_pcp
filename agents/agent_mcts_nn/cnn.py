import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class CNN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(CNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc_policy = nn.Linear(64 * 6 * 7, num_actions)
        self.fc_value = nn.Linear(64 * 6 * 7, 1)

        # Loss fn - for values MSE loss and for policies NLL loss?
        self.criterion_value = torch.nn.MSELoss()
        self.criterion_policy = torch.nn.CrossEntropyLoss()# torch.nn.NLLLoss()


    def forward(self, input, target_pi = None, targets_val = None):
        input = torch.FloatTensor(input).unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        input = input.view(-1, 1, 6, 7)  # Reshape to (batch_size, input_channels, height, width)

        # Convolutional layers with batch normalization and ReLU activation
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers for policy and value prediction
        policy_output = F.softmax(self.fc_policy(x), dim=1)
        value_output = self.fc_value(x)

        ## Make policy vector a vector of probabilities
        # policy_output = F.softmax(self.policy_head(input), dim=1)
        

        # # Value is just a single value
        # value_output = self.value_head(input)

        # TODO: assume for now that targets will be given in tuples: (policy_vector, value)
        if (targets_val is not None) and (target_pi is not None):
            loss_policy = self.criterion_policy(policy_output, target_pi)
            loss_value = self.criterion_value(value_output, targets_val)
            
            # TODO: Additive loss ?
            return loss_policy, loss_value

        return policy_output, value_output

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(-1, 1, 6, 7)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]            


