import abc
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import torch as th

from ..multiprocessing_rl.rollouts_data import RolloutBatch, RolloutTrajectory
from ..gen_disc.tree_search import DepthFirstTreeSearch, NodePointer
from ..engine import Game
from ..action import ActionList
from ..agent import Agent

class BaseGenerator(abc.ABC):
    @abc.abstractmethod
    def propose_n_actions(self, game: Game, n: int) -> Tuple[List[ActionList], List[Game], Optional[Dict]]:
        """
        Propose a list of actions for the next turn.

        Returns a list of actionlist candidates, the corresponding game states at the end of the turn, and a dictionary of extra info (e.g. for rollouts).
        """
        raise NotImplementedError()

class AgentGenerator(BaseGenerator):
    def __init__(self, agent: Agent):
        self.agent = agent

    def propose_n_actions(self, game: Game, n: int) -> Tuple[List[ActionList], Optional[Dict]]:
        games = [game.copy() for _ in range(n)]
        actions = [self.agent.act(game) for game in games]
        for game, action in zip(games, actions):
            game.full_turn(action)
        return actions, games, {}

def argmax_last_two_indices(arr):
    """
    Given an array of shape [..., n, m], return the index of the maximum value in the last two dimensions.
    Returns an int array of shape [..., 2]
    """
    flattened = arr.reshape(-1, arr.shape[-1] * arr.shape[-2])
    flat_idx = np.argmax(flattened, axis=-1)
    return np.stack([flat_idx // arr.shape[-1], flat_idx % arr.shape[-1]], axis=-1)

class QGenerator(BaseGenerator):
    def __init__(self, translator, model, epsilon_greedy) -> None:
        super().__init__()
        self.model = model
        self.translator = translator
        self.epsilon_greedy = epsilon_greedy

    def propose_n_actions(self, game, n):
        action_lists, final_game_states, trajectory_training_datas = self.tree_search(game, n)
        return action_lists, final_game_states, trajectory_training_datas

    def tree_search(self, game, num_trajectories):
        action_lists = []
        final_game_states = []
        extra_rollout_batches: List[Optional[RolloutBatch]] = []
        trajectories: List[RolloutTrajectory] = []
        tree_search = DepthFirstTreeSearch(lambda: QGeneratorTreeSearchNode(game.copy(), self.translator, self.model))
        for _ in range(num_trajectories):
            numpy_actions, final_node, this_extra_rollout_batch, trajectory = tree_search.run_trajectory(epsilon_greedy=self.epsilon_greedy)
            if final_node is None:
                assert trajectory is None
                break
            action_lists.append(ActionList.from_single_list(
                [self.translator.untranslate_action(action) for action in numpy_actions]
            ))
            final_game_states.append(final_node.game)
            trajectories.append(trajectory)
            extra_rollout_batches.append(this_extra_rollout_batch)
        # print(sum([len(d.next_maxq) for d in training_datas]))
        return action_lists, final_game_states, {"extra_rollout_batches": extra_rollout_batches, 'trajectories': trajectories}

class QGeneratorTreeSearchNode(NodePointer):
    def __init__(self, game, translator, model):
        self.game = game
        self.model = model
        self.translator = translator

    def hash_node(self):
        return self.game.checksum()

    def evaluate_node(self) -> Tuple[Any, List[Any], List[float]]:
        obs = self.translator.translate(self.game)
        logits = self.model(obs)
        winprobs = th.sigmoid(logits).detach().cpu().numpy()  # [1, 30, 30]
        winprobs = np.squeeze(winprobs, axis=0)
        valid_actions = self.translator.valid_actions(self.game)
        # Convert that shape [N, N] bool array into a [K, 2] shape array of where it's True
        # e.g. [[True, False], [False, True]] -> [[0, 0], [1, 1]]
        valid_action_idxs = np.array(np.nonzero(valid_actions)).T
        assert valid_action_idxs.shape[1] == 2
        q_values = winprobs[valid_actions]
        return obs, valid_action_idxs, q_values

    def take_action(self, action) -> None:
        untranslated_action = self.translator.untranslate_action(action)
        self.game.process_single_action(untranslated_action)

