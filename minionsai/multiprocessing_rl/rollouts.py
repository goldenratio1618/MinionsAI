import abc
from collections import defaultdict
from typing import Dict
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

    def get_rollouts(self, iteration: int) -> RolloutBatch:
        disc_obs = []
        disc_labels = []

        gen_obs = []
        gen_labels = []

        games = 0
        metrics_accumulated = (defaultdict(list), defaultdict(list))

        # Early in training use td-lambda to reduce variance of gradients
        # Late in training, turn this off since it introduces a bias when epsilon-greedy actions are taken.
        lam = None if iteration * self.episodes_per_iteration >= self.lambda_until_episodes else 1 - iteration * self.episodes_per_iteration / self.lambda_until_episodes
        hparams = {"lambda": lam}
        metrics_logger.log_metrics(hparams)

        self.launch_rollouts(iteration, hparams)

        for _ in tqdm.tqdm(range(self.episodes_per_iteration)):
            with metrics_logger.timing('single_episode'):

                rollout_episode = self.next_rollout()
                disc_obs.extend(rollout_episode.disc_obs)
                disc_labels.extend(rollout_episode.disc_labels)
                gen_obs.extend(rollout_episode.gen_obs)
                gen_labels.extend(rollout_episode.gen_labels)
                games += 1
                for color, this_color_metrics in enumerate(rollout_episode.metrics):
                    for key in set(metrics_accumulated[color].keys()).union(set(this_color_metrics.keys())):
                        metrics_accumulated[color][key].append(this_color_metrics[key])
        # convert from list of dicts of arrays to a single dict of arrays with large batch dimension
        if len(disc_obs) > 0:
            disc_obs = {k: np.concatenate([s[k] for s in disc_obs], axis=0) for k in disc_obs[0]}
            disc_obs, disc_labels = add_symmetries(disc_obs, disc_labels)

        if len(gen_obs) > 0:
            gen_obs = {k: np.concatenate([s[k] for s in gen_obs], axis=0) for k in gen_obs[0]}
            gen_obs, gen_labels = add_symmetries(gen_obs, gen_labels)

        for color in (0, 1):
            metrics_logger.log_metrics({k: sum(v)/self.episodes_per_iteration for k, v in metrics_accumulated[color].items()}, prefix=f'rollouts/game/{color}')
        return {
            "discriminator": RolloutBatch(disc_obs, disc_labels, num_games=self.episodes_per_iteration),
            "generator": RolloutBatch(gen_obs, gen_labels, num_games=self.episodes_per_iteration),
        }

    @abc.abstractmethod
    def next_rollout(self) -> RolloutEpisode:
        raise NotImplementedError

    def launch_rollouts(self, iteration: int, hparams: Dict) -> None:
        pass

def add_symmetries(obs, labels):
    """
    Add symmetries to the observations and labels.
    """
    symmetrized_obs = Translator.symmetries(obs)
    # Now combine them into one big states dict
    obs = {k: np.concatenate([s[k] for s in symmetrized_obs], axis=0) for k in obs}
    labels = np.concatenate([labels]*len(symmetrized_obs), axis=0)
    return obs, labels

class InProcessRolloutSource(OptimizerRolloutSource):
    """
    Runs rollouts in the main process
    """
    def __init__(self, episodes_per_iteration, game_kwargs, agent, lambda_until_episodes=5000) -> None:
        super().__init__(episodes_per_iteration, game_kwargs, lambda_until_episodes)
        self.runner = RolloutRunner(game_kwargs, agent)

    def next_rollout(self) -> RolloutEpisode:
        return self.runner.single_rollout()

    def launch_rollouts(self, iteration, hparams) -> None:
        self.runner.update(hparams)
