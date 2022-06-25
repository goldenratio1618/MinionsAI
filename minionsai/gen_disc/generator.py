import abc
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch as th

from ..game_util import stack_dicts
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

def gumbel_sample(logits: np.ndarray, temperature) -> np.ndarray:
    """
    Sample from a Gumbel distribution.
    This is equivalent to sampling from softmax(logits / temperature), but is more numerically stable.

    Assumes the last two axes are the distribution, and any before that are batch dimensions.
    """
    argmax_from = logits - temperature * np.log(-np.log(np.random.uniform(size=logits.shape)))
    # Now take the argmax over the last two axes
    flattened = argmax_from.reshape(-1, logits.shape[-1] * logits.shape[-2])
    flat_idx = np.argmax(flattened, axis=-1)
    # now unravel to get the original idx
    unraveled = np.unravel_index(flat_idx, logits.shape[:-2])
    return unraveled

class QGenerator(BaseGenerator):
    def __init__(self, translator, model, sampling_temperature, epsilon_greedy, actions_per_turn=10) -> None:
        super().__init__()
        self.model = model
        self.translator = translator
        self.sampling_temperature = sampling_temperature
        self.epsilon_greedy = epsilon_greedy
        self.actions_per_turn = actions_per_turn

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
            logits = self.model(obs)  # shape = [n, num_things, num_things]
            assert logits.rank == 3 and logits.shape[0] == n and logits.shape[-2] == logits.shape[-1], logits.shape
            masked_logits = logits + (1 - valid_actions) * -np.inf  # shape = [n, num_things, num_things]
            assert masked_logits.shape == logits.shape, (masked_logits.shape, logits.shape)
            max_winprob = th.sigmoid(th.max(th.max(masked_logits, axis=1), axis=1))  # shape = [n]
            assert max_winprob.shape == (n,), max_winprob.shape
            sampled_action_idx = self.sample(masked_logits) # shape = [n]
            assert sampled_action_idx.shape == (n,), sampled_action_idx.shape
            sampled_numpy_action = valid_actions[sampled_action_idx]  # shape = [n, 2]
            assert sampled_numpy_action.shape == (n, 2), sampled_numpy_action.shape
            sampled_action = [self.translator.untranslate_action(action) for action in sampled_numpy_action]
            
            recorded_obs.append(obs)
            recorded_next_maxq.append(max_winprob)
            recorded_numpy_actions.append(sampled_numpy_action)
            for recorded_list, new_action, game in zip(recorded_actions, sampled_action, games):
                recorded_list.append(new_action)
                game.process_single_action(new_action)

        recorded_obs = stack_dicts(recorded_obs)  # dict of shapes [actions_per_turn, n, ...]
        recorded_next_maxq = np.array(recorded_next_maxq)  # shape = [actions_per_turn, n]
        recorded_numpy_actions = np.array(recorded_numpy_actions)  # shape = [actions_per_turn, n, 2]
        submit_actions = [ActionList(recorded) for recorded in recorded_actions]

        # Shift next_maxq by 1 relative to obs, so it's properly aligned.
        recorded_next_maxq = recorded_next_maxq[1:]

        assert recorded_next_maxq.shape == (self.actions_per_turn - 1, n), recorded_next_maxq.shape
        assert recorded_numpy_actions.shape == (self.actions_per_turn, n, 2), recorded_numpy_actions.shape

        return submit_actions, games, {
            "obs": recorded_obs,
            "next_maxq": recorded_next_maxq,
            "numpy_actions": recorded_numpy_actions,
        }

    def sample(self, logits: np.ndarray) -> int:
        """
        Samples an action from an array ([batch, things, things]) of q values

        Returns an array of index-pairs; shape = [batch, 2]
        """
        greedy = np.random.rand(logits.shape[:-2]) < self.epsilon_greedy
        np.expand_dims(greedy, axis=-1)
        np.expand_dims(greedy, axis=-1)
        greedified_logits = logits + greedy * 1000
        return gumbel_sample(greedified_logits, self.sampling_temperature)



