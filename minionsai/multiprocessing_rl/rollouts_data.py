from dataclasses import dataclass


import numpy as np
from typing import List, Dict

@dataclass
class RolloutBatch:
    """
    Data class for storing rollout data.
    arrays all have batch size as first dimension.
    Output of a RolloutSource.
    """
    obs: Dict[str, np.array]
    labels: np.array
    num_games: int

@dataclass
class RolloutEpisode:
    """
    Data class for storing one episode of data.
    """
    obs: List[Dict[str, np.array]]
    labels: List[float]
    metrics: List[Dict]  # len 2, one per player