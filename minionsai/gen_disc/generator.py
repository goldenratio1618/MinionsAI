import abc
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch as th

from ..game_util import sigmoid, stack_dicts
from ..engine import Game, print_n_games
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
        return actions, games, None

def argmax_last_two_indices(arr):
    """
    Given an array of shape [..., n, m], return the index of the maximum value in the last two dimensions.
    Returns an int array of shape [..., 2]
    """
    flattened = arr.reshape(-1, arr.shape[-1] * arr.shape[-2])
    flat_idx = np.argmax(flattened, axis=-1)
    return np.stack([flat_idx // arr.shape[-1], flat_idx % arr.shape[-1]], axis=-1)

def gumbel_sample(logits: np.ndarray, temperature) -> np.ndarray:
    """
    Sample from a Gumbel distribution.
    This is equivalent to sampling from softmax(logits / temperature), but is more numerically stable.

    Assumes the last two axes are the distribution, and any before that are batch dimensions.
    """
    argmax_from = logits - temperature * np.log(-np.log(np.random.uniform(size=logits.shape)))
    return argmax_last_two_indices(argmax_from)

class QGenerator(BaseGenerator):
    def __init__(self, translator, model, sampling_temperature, epsilon_greedy, actions_per_turn=10, epsilon_greedy_min=0.1, epsilon_greedy_update=0.99) -> None:
        super().__init__()
        self.model = model
        self.translator = translator
        self.sampling_temperature = sampling_temperature
        self.epsilon_greedy = epsilon_greedy
        self.actions_per_turn = actions_per_turn
        self.epsilon_greedy_min = epsilon_greedy_min
        self.epsilon_greedy_update = epsilon_greedy_update
        self.iter = 0

    def translate_many(self, games):
        obs = []
        valid_actions = []
        for game in games:
            this_obs = self.translator.translate(game)
            obs.append(this_obs)
            this_valid = self.translator.valid_actions(game)  # TODO implement this
            valid_actions.append(this_valid)
        obs = stack_dicts(obs)
        valid_actions = np.array(valid_actions)
        return obs, valid_actions

    def propose_n_actions(self, game, n):
        games = [game.copy() for _ in range(n)]
        recorded_obs = []
        recorded_next_maxq = []
        recorded_numpy_actions = []
        recorded_actions = [[] for _ in range(n)]
        for i in range(self.actions_per_turn):
            obs, valid_actions = self.translate_many(games)
            logits = self.model(obs)
            winprobs = th.sigmoid(logits).detach().cpu().numpy()  # shape = [n, num_things, num_things]
            assert winprobs.ndim == 3 and winprobs.shape[0] == n and winprobs.shape[1] == winprobs.shape[2], (n, winprobs.shape)
            masked_winprobs = np.where(valid_actions, winprobs, -np.inf)  # shape = [n, num_things, num_things]
            assert masked_winprobs.shape == winprobs.shape, (masked_winprobs.shape, winprobs.shape)
            max_winprob = np.max(np.max(masked_winprobs, axis=1), axis=1)  # shape = [n]
            assert max_winprob.shape == (n,), max_winprob.shape
            sampled_numpy_action = self.sample(masked_winprobs) # shape = [n, 2]
            assert sampled_numpy_action.shape == (n, 2), sampled_numpy_action.shape
            sampled_action = [self.translator.untranslate_action(action) for action in sampled_numpy_action]
            
            recorded_obs.append(obs)
            recorded_next_maxq.append(max_winprob)
            recorded_numpy_actions.append(sampled_numpy_action)
            for recorded_list, new_action, game in zip(recorded_actions, sampled_action, games):
                recorded_list.append(new_action)
                game.process_single_action(new_action)

        recorded_obs = stack_dicts(recorded_obs, add_new_axis=True)  # dict of shapes [actions_per_turn, n, ...]
        recorded_next_maxq = np.array(recorded_next_maxq)  # shape = [actions_per_turn, n]
        recorded_numpy_actions = np.array(recorded_numpy_actions)  # shape = [actions_per_turn, n, 2]
        submit_actions = [ActionList.from_single_list(recorded) for recorded in recorded_actions]

        # Shift next_maxq by 1 relative to obs, so it's properly aligned.
        recorded_next_maxq = recorded_next_maxq[1:]

        for k, v in recorded_obs.items():
            assert v.shape[:2] == (self.actions_per_turn, n), (k, v.shape)
        assert recorded_next_maxq.shape == (self.actions_per_turn - 1, n), recorded_next_maxq.shape
        assert recorded_numpy_actions.shape == (self.actions_per_turn, n, 2), recorded_numpy_actions.shape

        return submit_actions, games, {
            "obs": recorded_obs,
            "next_maxq": recorded_next_maxq,
            "numpy_actions": recorded_numpy_actions,
        }

    def increment_iter(self):
        self.iter += 1
    
    def sample(self, logits: np.ndarray) -> int:
        """
        Samples an action from an array ([batch, things, things]) of q values

        Returns an array of index-pairs; shape = [batch, 2]
        """
        # Make a list of whether or not to be greedy for each entry inthe batch
        greedy = np.random.rand(*logits.shape[:-2]) < self.epsilon_greedy_min + (self.epsilon_greedy - self.epsilon_greedy_min) * (self.epsilon_greedy_update ** self.iter)

        # Implement greedy sampling by setting the logits to zero.
        # Can't multiply by literally zero, because many entries are masked (set to -inf)
        tiny_multiplier = self.sampling_temperature * 1e-6
        greedy_multiplier = np.where(greedy, tiny_multiplier, 1)
        greedy_multiplier = np.expand_dims(greedy_multiplier, axis=-1)
        greedy_multiplier = np.expand_dims(greedy_multiplier, axis=-1)
        greedified_logits = logits * greedy_multiplier
        return gumbel_sample(greedified_logits, self.sampling_temperature)



