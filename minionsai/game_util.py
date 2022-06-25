from .engine import adjacent_hexes, Board
from .unit_type import ZOMBIE
import numpy as np
import random
import torch as th

def adjacent_zombies(board: Board, loc, color):
        result = []
        for i, j in adjacent_hexes(*loc):
            if board.board[i][j].unit is not None and board.board[i][j].unit.type == ZOMBIE and board.board[i][j].unit.color == color:
                result.append((i, j))
        return result

def stack_dicts(dicts):
    """
    Turns a list of dicts of numpy arrays into a single dict of arrays stacked along the first axis
    Assumes all entries have the same keys
    """
    keys = list(dicts[0].keys())
    result = {}
    for key in keys:
        result[key] = np.concatenate([d[key] for d in dicts])
    return result

def equal_np_dicts(x, y):
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y)
    elif isinstance(x, dict):
        return all(k in y for k in x) and all(equal_np_dicts(x[k], y[k]) for k in x)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.use_deterministic_algorithms(True)