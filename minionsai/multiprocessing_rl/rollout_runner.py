
from collections import defaultdict
from minionsai.gen_disc.generator import QGenerator
from ..game_util import seed_everything, stack_dicts
import torch as th
import numpy as np
import random

from .rollouts_data import RolloutEpisode
from .td_lambda import smooth_labels
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
        disc_obs_buffers = [[], []]  # one for each player
        disc_label_buffers = [[], []]  # one for each player
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
                max_winprob = disc_info["max_winprob"]
                disc_obs_buffers[active_player].append(disc_obs)
                disc_label_buffers[active_player].append(max_winprob)
            if "obs" in gen_info[0]:
                # Then we're training the generator also
                # TODO have a better way to know if we're optimizing than checking info dict keys.

                # Assume we're always training the first generator in the list.
                train_gen_info = gen_info[0]
                train_generator, rollouts_per_turn = self.agent.generators[0]
                all_gen_obs.append(train_gen_info["obs"])

                gen_labels = train_gen_info["next_maxq"]  # This is shape [actions_per_turn - 1, rollouts_per_turn]
                assert gen_labels.shape == (train_generator.actions_per_turn - 1, rollouts_per_turn), gen_labels.shape

                all_ending_labels = disc_info["all_winprobs"]  # This is shape [total rollouts_per_turn]
                ending_labels = all_ending_labels[:rollouts_per_turn]
                ending_labels = np.expand_dims(ending_labels, axis=0)  # [1, rollouts_per_turn]
                assert ending_labels.shape == (1, self.agent.rollouts_per_turn), ending_labels.shape

                gen_labels = np.concatenate([gen_labels, ending_labels], axis=0)
                assert gen_labels.shape == (self.agent.generator.actions_per_turn, self.agent.rollouts_per_turn), gen_labels.shape
                all_gen_labels.append(gen_labels)

                assert train_gen_info["numpy_actions"].shape == (self.agent.generator.actions_per_turn, self.agent.rollouts_per_turn, 2), gen_info["numpy_actions"].shape
                all_gen_actions.append(train_gen_info["numpy_actions"])
            
                # Accumulate metrics from the generators
                for accumulator, info in zip(gen_metrics, gen_info):
                    for key, value in info["metrics"].items():
                        accumulator[key].append(value)
        winner = game.winner

        player_metrics = (game.get_metrics(0), game.get_metrics(1))
        if len(disc_obs_buffers[0]) > 0:
            player_metrics[winner]["pfinal"] = disc_label_buffers[winner][-1]
            player_metrics[1 - winner]["pfinal"] = 1 - disc_label_buffers[1 - winner][-1]

        winner_obs = disc_obs_buffers[winner]
        loser_obs = disc_obs_buffers[1 - winner]

        disc_label_buffers[winner].append(1)
        disc_label_buffers[1 - winner].append(0)
        winner_disc_labels = disc_label_buffers[winner][1:]
        loser_disc_labels = disc_label_buffers[1 - winner][1:]

        if self.hparams.get('lambda', None) is not None:
            winner_disc_labels = smooth_labels(winner_disc_labels, self.hparams['lambda'])
            loser_disc_labels = smooth_labels(loser_disc_labels, self.hparams['lambda'])

        all_disc_obs = winner_obs + loser_obs
        all_disc_labels = np.concatenate([winner_disc_labels, loser_disc_labels])


        if len(all_gen_obs) > 0:
            total_turns = len(all_gen_labels) * self.agent.rollouts_per_turn * self.agent.generator.actions_per_turn
            all_gen_obs = stack_dicts(all_gen_obs)
            all_gen_obs = {k: v.reshape(total_turns, *v.shape[2:]) for k, v in all_gen_obs.items()}
            all_gen_labels = np.concatenate(all_gen_labels)  # shape=[actions_per_turn * turns, rollouts_per_turn]
            all_gen_labels = all_gen_labels.reshape(total_turns) 
            all_gen_actions = np.concatenate(all_gen_actions)  # shape=[actions_per_turn * turns, rollouts_per_turn, 2]
            all_gen_actions = all_gen_actions.reshape(total_turns, 2)

        global_metrics = {}
        if len(gen_metrics[0]) > 0:
            for i, metrics_dict in enumerate(gen_metrics):
                for key, list_of_values in metrics_dict.items():
                    # check that we have the right number
                    assert len(list_of_values) == len(all_disc_labels)
                    mean = sum(list_of_values) / len(list_of_values)
                    global_metrics[f"rollouts/generators/i/{key}"] = mean
        result = RolloutEpisode(
            disc_obs=all_disc_obs, 
            disc_labels=all_disc_labels, 
            gen_obs=all_gen_obs,
            gen_labels=all_gen_labels,
            gen_actions=all_gen_actions,
            global_metrics = global_metrics,
            player_metrics=player_metrics)

        return result