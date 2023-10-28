import numpy as np
from agents.game_utils import BoardPiece, NO_PLAYER

def test_initialize_game_state():
    from agents.game_utils import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)