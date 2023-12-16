import torch
from game_utils import *
from agents.agent_dqn.training.mlp import MLP

def dqn_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> tuple[PlayerAction, Optional[SavedState]]:
    states_dim = 6 * 7
    action_dim = 7
    # load the model
    model = MLP(states_dim, action_dim)
    model.eval()

    action = torch.argmax(model(torch.Tensor(board.flatten()))).item()
    print(action)

    return action, saved_state




