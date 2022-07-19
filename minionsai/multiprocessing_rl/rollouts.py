import abc
from collections import defaultdict
from typing import Dict
import tqdm

from .rollouts_data import RolloutBatch, RolloutEpisode
from .rollout_runner import RolloutRunner
from ..metrics_logger import metrics_logger
from ..discriminator_only.translator import Translator

class OptimizerRolloutSource(abc.ABC):
    """
    A source of rollout data from the optimizer's point of view
    """
    def __init__(self, episodes_per_iteration, game_kwargs) -> None:
        self.episodes_per_iteration = episodes_per_iteration
        self.game_kwargs = game_kwargs

    def get_rollouts(self, iteration: int) -> Dict[str, RolloutBatch]:
        disc_rollout_batch = None
        gen_rollout_batch = None

        games = 0
        player_metrics_accumulated = (defaultdict(list), defaultdict(list))
        global_metrics_accumulated = defaultdict(list)
        hparams = {}
        metrics_logger.log_metrics(hparams)

        self.launch_rollouts(iteration, hparams)
        print(f"Launching rollouts iteration {iteration}.")
        for idx in tqdm.tqdm(range(self.episodes_per_iteration)):
            with metrics_logger.timing('single_episode'):

                rollout_episode = self.next_rollout(iteration, idx)
                disc_rollout_batch = RolloutBatch.optional_add(disc_rollout_batch, rollout_episode.disc_rollout_batch)
                gen_rollout_batch = RolloutBatch.optional_add(gen_rollout_batch, rollout_episode.gen_rollout_batch)
                games += 1
                for color, this_color_metrics in enumerate(rollout_episode.player_metrics):
                    for key in set(player_metrics_accumulated[color].keys()).union(set(this_color_metrics.keys())):
                        player_metrics_accumulated[color][key].append(this_color_metrics[key])
                for key in set(global_metrics_accumulated.keys()).union(set(rollout_episode.global_metrics.keys())):
                    global_metrics_accumulated[key].append(rollout_episode.global_metrics[key])
        
        # TODO get these flags from somewhere rather than deducing it here?
        train_generator = gen_rollout_batch is not None
        train_discriminator = disc_rollout_batch is not None

        if train_discriminator:
            disc_rollout_batch = disc_rollout_batch.add_symmetries()
        if train_generator:
            gen_rollout_batch = gen_rollout_batch.add_symmetries()

        metrics_logger.log_metrics({k: sum(v)/self.episodes_per_iteration for k, v in global_metrics_accumulated.items()}, prefix=f'rollouts/game')
        for color in (0, 1):
            metrics_logger.log_metrics({k: sum(v)/self.episodes_per_iteration for k, v in player_metrics_accumulated[color].items()}, prefix=f'rollouts/game/{color}')
        return {
            "discriminator": disc_rollout_batch,
            "generator": gen_rollout_batch,
        }

    @abc.abstractmethod
    def next_rollout(self, iteration, episode_idx) -> RolloutEpisode:
        raise NotImplementedError

    def launch_rollouts(self, iteration: int, hparams: Dict) -> None:
        pass

class InProcessRolloutSource(OptimizerRolloutSource):
    """
    Runs rollouts in the main process
    """
    def __init__(self, episodes_per_iteration, game_kwargs, agent) -> None:
        super().__init__(episodes_per_iteration, game_kwargs)
        self.runner = RolloutRunner(game_kwargs, agent)

    def next_rollout(self, iteration, episode_idx) -> RolloutEpisode:
        return self.runner.single_rollout(iteration, episode_idx)

    def launch_rollouts(self, iteration, hparams) -> None:
        self.runner.update(hparams)
