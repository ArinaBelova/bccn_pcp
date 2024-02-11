import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, num_actions):
        super(MLP, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.policy_head = nn.Linear(64, num_actions)

        self.value_head = nn.Linear(64, 1)

        # Loss fn - for values MSE loss and for policies NLL loss?
        self.criterion_value = torch.nn.MSELoss()
        self.criterion_policy = torch.nn.CrossEntropyLoss()# torch.nn.NLLLoss()


    def forward(self, input, target_pi = None, targets_val = None):
        input = torch.FloatTensor(input)
        input =  input.view(input.shape[0],-1)

        input = F.relu(self.fc1(input))
        input = self.fc2(input)

        # Make policy vector a vector of probabilities
        policy_output = F.softmax(self.policy_head(input), dim=1)
        

        # Value is just a single value
        value_output = self.value_head(input)

        # TODO: assume for now that targets will be given in tuples: (policy_vector, value)
        if (targets_val is not None) and (target_pi is not None):
            loss_policy = self.criterion_policy(policy_output, target_pi)
            loss_value = self.criterion_value(value_output, targets_val)
            
            # TODO: Additive loss ?
            return loss_policy, loss_value

        return policy_output, value_output

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, -1)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]            


# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Connect2Model(nn.Module):

#     def __init__(self, board_size, action_size, device):

#         super(Connect2Model, self).__init__()

#         self.device = device
#         self.size = board_size
#         self.action_size = action_size

#         self.fc1 = nn.Linear(in_features=self.size, out_features=16)
#         self.fc2 = nn.Linear(in_features=16, out_features=16)

#         # Two heads on our network
#         self.action_head = nn.Linear(in_features=16, out_features=self.action_size)
#         self.value_head = nn.Linear(in_features=16, out_features=1)

#         self.to(device)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))

#         action_logits = self.action_head(x)
#         value_logit = self.value_head(x)

#         return F.softmax(action_logits, dim=1), torch.tanh(value_logit)

    