import abc
from collections import defaultdict
from typing import Dict
from minionsai.game_util import stack_dicts
import numpy as np
import tqdm

from .rollouts_data import RolloutBatch, RolloutEpisode
from .rollout_runner import RolloutRunner
from ..metrics_logger import metrics_logger
from ..discriminator_only.translator import Translator

class OptimizerRolloutSource(abc.ABC):
    """
    A source of rollout data from the optimizer's point of view
    """
    def __init__(self, episodes_per_iteration, game_kwargs, lambda_until_episodes=5000) -> None:
        self.episodes_per_iteration = episodes_per_iteration
        self.game_kwargs = game_kwargs
        self.lambda_until_episodes = lambda_until_episodes

    def get_rollouts(self, iteration: int) -> Dict[str, RolloutBatch]:
        disc_obs = []
        disc_next_obs = []
        disc_labels = []

        gen_obs = []
        gen_labels = []
        gen_actions = []

        games = 0
        player_metrics_accumulated = (defaultdict(list), defaultdict(list))
        global_metrics_accumulated = defaultdict(list)
        # Early in training use td-lambda to reduce variance of gradients
        # Late in training, turn this off since it introduces a bias when epsilon-greedy actions are taken.
        lam = None if iteration * self.episodes_per_iteration >= self.lambda_until_episodes else 1 - iteration * self.episodes_per_iteration / self.lambda_until_episodes
        hparams = {"lambda": lam}
        metrics_logger.log_metrics(hparams)

        self.launch_rollouts(iteration, hparams)
        print(f"Launching rollouts iteration {iteration}.")
        for idx in tqdm.tqdm(range(self.episodes_per_iteration)):
            with metrics_logger.timing('single_episode'):

                rollout_episode = self.next_rollout(iteration, idx)
                disc_obs.extend(rollout_episode.disc_obs)
                disc_null_obs = {k: np.zeros_like(v) for k, v in rollout_episode.disc_obs[0].items()}
                disc_next_obs.extend(rollout_episode.disc_obs[1:] + [disc_null_obs])
                disc_labels.extend(rollout_episode.disc_labels)
                gen_obs.append(rollout_episode.gen_obs)
                assert rollout_episode.gen_actions == [] or rollout_episode.gen_actions.shape == (rollout_episode.gen_labels.shape[0], 2)
                gen_labels.append(rollout_episode.gen_labels)
                gen_actions.append(rollout_episode.gen_actions)
                games += 1
                for color, this_color_metrics in enumerate(rollout_episode.player_metrics):
                    for key in set(player_metrics_accumulated[color].keys()).union(set(this_color_metrics.keys())):
                        player_metrics_accumulated[color][key].append(this_color_metrics[key])
                for key in set(global_metrics_accumulated.keys()).union(set(rollout_episode.global_metrics.keys())):
                    global_metrics_accumulated[key].append(rollout_episode.global_metrics[key])
        # convert from list of dicts of arrays to a single dict of arrays with large batch dimension
        if len(disc_obs) > 0:
            disc_obs = stack_dicts(disc_obs)
            disc_next_obs = stack_dicts(disc_next_obs)
            disc_obs, disc_next_obs, disc_labels, _ = add_symmetries(disc_obs, disc_next_obs, disc_labels, None)

        # gen_obs is a list of length episodes_per_iteration; 
        # If we aren't training the generator, they're all empty
        if len(gen_obs[0]) > 0:
            gen_obs = {k: np.concatenate([s[k] for s in gen_obs], axis=0) for k in gen_obs[0]}
            gen_labels = np.concatenate(gen_labels, axis=0)
            gen_actions = np.concatenate(gen_actions, axis=0)
            assert gen_actions.shape == (gen_labels.shape[0], 2), gen_actions.shape
            gen_obs, gen_next_obs, gen_labels, gen_actions = add_symmetries(gen_obs, {}, gen_labels, gen_actions)
            assert gen_actions.shape == (gen_labels.shape[0], 2), gen_actions.shape

        metrics_logger.log_metrics({k: sum(v)/self.episodes_per_iteration for k, v in global_metrics_accumulated.items()}, prefix=f'rollouts/game')
        for color in (0, 1):
            metrics_logger.log_metrics({k: sum(v)/self.episodes_per_iteration for k, v in player_metrics_accumulated[color].items()}, prefix=f'rollouts/game/{color}')
        return {
            "discriminator": RolloutBatch(
                obs=disc_obs, 
                next_obs=disc_next_obs, 
                actions=None, 
                next_maxq=disc_labels, 
                terminal_action=None,
                reward=None,
            ),
            "generator": RolloutBatch(
                obs=gen_obs, 
                next_obs=gen_next_obs, 
                actions=gen_actions, 
                next_maxq=gen_labels,
                terminal_action=None, # TODO
                reward=None, # TODO
            ),
        }

    @abc.abstractmethod
    def next_rollout(self, iteration, episode_idx) -> RolloutEpisode:
        raise NotImplementedError

    def launch_rollouts(self, iteration: int, hparams: Dict) -> None:
        pass

def add_symmetries(obs, next_obs, labels, actions=None):
    """
    Add symmetries to the observations and labels.
    """
    symmetrized_obs, symmetrized_actions = Translator.symmetries(obs, actions)
    # Now combine them into one big states dict
    obs = stack_dicts(symmetrized_obs)
    labels = np.concatenate([labels]*len(symmetrized_obs), axis=0)
    next_obs = stack_dicts([next_obs] * len(symmetrized_obs))
    return obs, next_obs, labels, symmetrized_actions

class InProcessRolloutSource(OptimizerRolloutSource):
    """
    Runs rollouts in the main process
    """
    def __init__(self, episodes_per_iteration, game_kwargs, agent, lambda_until_episodes=5000) -> None:
        super().__init__(episodes_per_iteration, game_kwargs, lambda_until_episodes)
        self.runner = RolloutRunner(game_kwargs, agent)

    def next_rollout(self, iteration, episode_idx) -> RolloutEpisode:
        return self.runner.single_rollout(iteration, episode_idx)

    def launch_rollouts(self, iteration, hparams) -> None:
        self.runner.update(hparams)
