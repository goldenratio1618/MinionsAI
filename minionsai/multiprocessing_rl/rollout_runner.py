
from collections import defaultdict
from minionsai.gen_disc.generator import QGenerator
from ..game_util import seed_everything, stack_dicts
import torch as th
import numpy as np
import random

from .rollouts_data import RolloutEpisode, RolloutTrajectory
from ..engine import Game


class RolloutRunner():
    def __init__(self, game_kwargs, agent, hparams={}):
        self.game_kwargs = game_kwargs
        self.hparams = hparams
        self.agent = agent

    def update(self, hparams):
        self.hparams.update(hparams)

    def make_game(self) -> Game:
        # TODO make these randomizations more configurable
        game_kwargs = self.game_kwargs.copy()
        game_kwargs["money"] = (random.randint(1, 4), random.randint(1, 4))
        return Game(**game_kwargs)

    def single_rollout(self, iteration, episode_idx) -> RolloutEpisode:
        # TODO Make this seed a more reasonable function of iteration & episode_idx
        # TODO Also make it depend on the original SEED set in the main script file, so that different main seeds 
        # have different rollout games.
        seed = iteration * 100000 + episode_idx
        seed_everything(seed)

        game = self.make_game()
        disc_trajectories = [RolloutTrajectory(obs=[], maxq=[], actions=None, previous_next_obs=[]) for _ in range(2)]

        all_gen_obs = []
        all_gen_labels = []
        all_gen_actions = []
        gen_metrics = [defaultdict(list) for _ in self.agent.generators]  # one for each generator, dict of key: array of measured values.
        while True:
            game.next_turn()
            if game.done:
                break
            active_player = game.active_player_color
            actionlist, gen_info, disc_info = self.agent.act_with_info(game)
            game.full_turn(actionlist)
            if "chosen_final_obs" in disc_info:
                # In this case we're optimizing the discriminator.
                # TODO have a better way to know if we're optimizing than checking info dict keys.
                disc_obs = disc_info["chosen_final_obs"]
                disc_best_obs = disc_info["best_final_obs"]
                max_winprob = disc_info["max_winprob"]
                disc_trajectories[active_player].obs.append(disc_obs)
                disc_trajectories[active_player].previous_next_obs.append(disc_best_obs)
                disc_trajectories[active_player].maxq.append(max_winprob)
            if "training_datas" in gen_info[0]:
                # Then we're training the generator also
                # TODO have a better way to know if we're optimizing than checking info dict keys.

                # Assume we're always training the first generator in the list.
                training_datas = gen_info[0]["training_datas"]
                train_generator, rollouts_per_turn = self.agent.generators[0]

                all_ending_labels = disc_info["all_winprobs"]  # This is shape [total rollouts_per_turn]
                ending_labels = all_ending_labels[:rollouts_per_turn]
                assert ending_labels.shape == (rollouts_per_turn,), ending_labels.shape

                for traj_training_data, label in zip(training_datas, ending_labels):
                    traj_training_data.next_maxq = np.concatenate([traj_training_data.next_maxq, [label]])
                all_gen_obs.append(stack_dicts([t.obs for t in training_datas]))

                gen_labels = np.concatenate([np.array(t.next_maxq) for t in training_datas])
                all_gen_labels.append(gen_labels)

                all_gen_actions.append(np.concatenate([np.array(t.actions) for t in training_datas]))

                # Accumulate metrics from the generators
                for accumulator, info in zip(gen_metrics, gen_info):
                    for key, value in info["metrics"].items():
                        accumulator[key].append(value)
        winner = game.winner

        # TODO: get these flags from somewhere rather than deducing them here.
        train_generator = len(all_gen_obs) > 0
        train_discriminator = len(disc_trajectories[0].obs) > 0

        player_metrics = (game.get_metrics(0), game.get_metrics(1))
        if train_discriminator:
            player_metrics[0]["pfirstturn"] = disc_trajectories[0].maxq[0]
            player_metrics[1]["pfirstturn"] = disc_trajectories[1].maxq[0]
            player_metrics[winner]["pfinal"] = disc_trajectories[winner].maxq[-1]
            player_metrics[1 - winner]["pfinal"] = 1 - disc_trajectories[1 - winner].maxq[-1]

            disc_winner_batch = disc_trajectories[winner].assemble(final_reward=1.0, lam=self.hparams['lambda'])
            disc_loser_batch = disc_trajectories[1 - winner].assemble(final_reward=0.0, lam=self.hparams['lambda'])

            disc_batch=disc_winner_batch + disc_loser_batch
        else:
            disc_batch = None

        if train_generator:
            all_gen_obs = stack_dicts(all_gen_obs)
            all_gen_labels = np.concatenate(all_gen_labels)
            all_gen_actions = np.concatenate(all_gen_actions)
            num = len(all_gen_labels)
            assert all_gen_labels.shape == (num,), all_gen_labels.shape
            assert all_gen_actions.shape == (num, 2), all_gen_actions.shape
            for key, value in all_gen_obs.items():
                assert value.shape[0] == num, (key, value.shape)
        else:
            all_gen_obs = {}
            all_gen_labels = []
            all_gen_actions = []
        global_metrics = {}
        if len(gen_metrics[0]) > 0:
            for i, metrics_dict in enumerate(gen_metrics):
                # i is the index of the generator in agent.generators list.
                for key, list_of_values in metrics_dict.items():
                    # check that we have the right number; one per turn of the game.
                    assert len(list_of_values) == len(disc_batch.next_maxq)
                    mean = sum(list_of_values) / len(list_of_values)
                    global_metrics[f"generators/{i}/{key}"] = mean
        result = RolloutEpisode(
            disc_rollout_batch=disc_batch,
            gen_obs=all_gen_obs,
            gen_labels=all_gen_labels,
            gen_actions=all_gen_actions,
            global_metrics = global_metrics,
            player_metrics=player_metrics)

        return result