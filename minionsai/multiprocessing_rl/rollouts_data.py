from dataclasses import dataclass
from minionsai.discriminator_only.translator import Translator
from minionsai.game_util import stack_dicts
from minionsai.multiprocessing_rl.td_lambda import smooth_labels


import numpy as np
from typing import List, Dict, Optional

def _optional_concatenate(arrays):
    if all(a is None for a in arrays):
        return None
    else:
        return np.concatenate(arrays)

@dataclass
class RolloutBatch:
    """
    Data class for storing rollout data.
    arrays all have batch size as first dimension.
    Output of a RolloutSource.
    """
    obs: Dict[str, np.array]
    next_obs: Dict[str, np.array]
    actions: Optional[np.array]
    next_maxq: Optional[np.array]
    terminal_action: Optional[np.array]
    reward: Optional[np.array]

    def __post_init__(self):
        # Check that all the shapes are consistent.
        num_transitions = len(self)
        for k, v in self.obs.items():
            assert v.shape[0] == num_transitions, (k, v.shape, num_transitions)
        # TODO remove all these escape hatches; -1, None, etc.
        if self.next_obs is not None:
            for k, v in self.next_obs.items():
                assert v.shape[0] == num_transitions, (k, v.shape, num_transitions)
        assert self.next_maxq.shape == (num_transitions - 1,) or self.next_maxq.shape == (num_transitions,), (self.next_maxq.shape, num_transitions)
        assert self.actions is None or self.actions.shape == (num_transitions, 2), (self.actions.shape, num_transitions)
        assert self.terminal_action is None or self.terminal_action.shape == (num_transitions,), (self.terminal_action.shape, num_transitions)
        assert self.reward is None or self.reward.shape == (num_transitions,), (self.reward.shape, num_transitions)

    def __len__(self):
        return len(self.obs['board'])

    def __add__(self, other):
        return RolloutBatch(
            obs=stack_dicts([self.obs, other.obs]),
            next_obs=stack_dicts([self.next_obs, other.next_obs]),
            actions=_optional_concatenate([self.actions, other.actions]),
            next_maxq=_optional_concatenate([self.next_maxq, other.next_maxq]),
            terminal_action=_optional_concatenate([self.terminal_action, other.terminal_action]),
            reward=_optional_concatenate([self.reward, other.reward])
        )

    def add_symmetries(self):
        """
        Add symmetries to the observations and labels.
        """
        symmetrized_obs, symmetrized_actions = Translator.symmetries(self.obs, self.actions)
        separate_batches = [RolloutBatch(obs=obs, actions=act, next_obs=self.next_obs, next_maxq=self.next_maxq, terminal_action=self.terminal_action, reward=self.reward)
        for obs, act in zip(symmetrized_obs, symmetrized_actions)]
        assert len(separate_batches) == 4
        return separate_batches[0] + separate_batches[1] + separate_batches[2] + separate_batches[3]

@dataclass
class RolloutTrajectory:
    obs: List[Dict[str, np.array]]
    maxq: List[float]
    actions: List[np.array]
    # The obs to use as "next_obs" for the previous state
    # By default this will just be obs
    previous_next_obs: Optional[List[Dict[str, np.array]]]  

    @property
    def _resolved_previous_next_obs(self):
        if self.previous_next_obs is None:
            return self.obs
        else:
            return self.previous_next_obs

    def assemble(self, final_reward: float, lam=None) -> RolloutBatch:
        """
        Assembles a RolloutBatch from this trajectory.
        """
        null_obs = {k: np.zeros_like(v) for k, v in self.obs[0].items()}
        next_maxq=np.array(self.maxq[1:] + [final_reward])
        if lam is not None:
            next_maxq = smooth_labels(next_maxq, lam)

        actions = np.array(self.actions) if self.actions is not None else None

        return RolloutBatch(
            obs=stack_dicts(self.obs),
            next_obs=stack_dicts(self._resolved_previous_next_obs[1:] + [null_obs]),
            actions=actions,
            next_maxq=next_maxq,
            terminal_action=np.array([0] * (len(self.obs) - 1) + [1], dtype=np.bool),
            reward=np.array([0.0] * (len(self.obs) - 1) + [final_reward], dtype=np.float32),
        )

@dataclass
class RolloutEpisode:
    """
    Data class for storing one episode of data.
    """
    disc_rollout_batch: RolloutBatch
    gen_obs: List[Dict[str, np.array]]
    gen_actions: np.array  # shape [N, 2]
    gen_labels: np.array  # Shape [N,]
    player_metrics: List[Dict]  # len 2, one per player
    global_metrics: Dict
