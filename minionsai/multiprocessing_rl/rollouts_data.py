from dataclasses import dataclass
from minionsai.game_util import stack_dicts


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
    next_obs: Dict[str, np.array]
    actions: np.array
    next_maxq: np.array
    terminal_action: np.array
    reward: np.array

    def __post_init__(self):
        # Check that all the shapes are consistent.
        num_transitions = self.obs['board'].shape[0]
        for k, v in self.obs.items():
            assert v.shape[0] == num_transitions, (k, v.shape, num_transitions)
        # TODO uncomment
        # for k, v in self.next_obs.items():
        #     assert v.shape[0] == num_transitions, (k, v.shape, num_transitions)
        # TODO handle short trajectories in rollouts
        # assert self.next_maxq.shape == (num_transitions,), (self.next_maxq.shape, num_transitions)
        assert self.actions is None or self.actions.shape == (num_transitions, 2), (self.actions.shape, num_transitions)
        assert self.terminal_action is None or self.terminal_action.shape == (num_transitions,), (self.terminal_action.shape, num_transitions)
        assert self.reward is None or self.reward.shape == (num_transitions,), (self.reward.shape, num_transitions)

    def __add__(self, other):
        return RolloutBatch(
            obs=stack_dicts([self.obs, other.obs]),
            next_obs=stack_dicts([self.next_obs, other.next_obs]),
            actions=np.concatenate([self.actions, other.actions]),
            next_maxq=np.concatenate([self.next_maxq, other.next_maxq]),
            terminal_action=np.concatenate([self.terminal_action, other.terminal_action]),
            reward=np.concatenate([self.reward, other.reward])
        )

@dataclass
class RolloutTrajectory:
    obs: List[Dict[str, np.array]]
    maxq: List[float]
    actions: List[np.array]

    def assemble(self, final_reward: float) -> RolloutBatch:
        """
        Assembles a RolloutBatch from this trajectory.
        """
        null_obs = {k: np.zeros_like(v) for k, v in self.obs[0].items()}
        return RolloutBatch(
            obs=stack_dicts(self.obs),
            next_obs=stack_dicts(self.obs[1:] + [null_obs]),
            actions=np.array(self.actions),
            next_maxq=np.array(self.next_maxq[1:] + [final_reward]),
            terminal_action=np.array([0] * (len(self.obs) - 1) + [1], dtype=np.bool),
            reward=np.array([0.0] * (len(self.obs) - 1) + [final_reward]),
        )

@dataclass
class RolloutEpisode:
    """
    Data class for storing one episode of data.
    """
    disc_obs: List[Dict[str, np.array]]
    disc_labels: List[float]
    gen_obs: List[Dict[str, np.array]]
    gen_actions: np.array  # shape [N, 2]
    gen_labels: np.array  # Shape [N,]
    player_metrics: List[Dict]  # len 2, one per player
    global_metrics: Dict
