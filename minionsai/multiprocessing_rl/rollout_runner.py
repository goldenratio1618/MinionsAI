
from minionsai.game_util import stack_dicts
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

    def single_rollout(self) -> RolloutEpisode:
        game = self.make_game()
        disc_obs_buffers = [[], []]  # one for each player
        disc_label_buffers = [[], []]  # one for each player
        all_gen_obs = []
        all_gen_labels = []
        all_gen_actions = []
        while True:
            game.next_turn()
            if game.done:
                break
            active_player = game.active_player_color
            actionlist, gen_info, disc_info = self.agent.act_with_info(game)
            game.full_turn(actionlist)
            obs = disc_info["chosen_final_obs"]
            max_winprob = disc_info["max_winprob"]
            disc_obs_buffers[active_player].append(obs)
            disc_label_buffers[active_player].append(max_winprob)
            if gen_info is not None:
                # Then we're training the generator also
                all_gen_obs.append(gen_info["obs"])
                gen_labels = gen_info["next_maxq"]  # This is shape [actions - 1, num_games]
                ending_labels = disc_info["all_winprobs"]  # This is shape [num_games]
                ending_labels.expand_dims(0)
                gen_labels = np.concatenate([gen_labels, ending_labels], axis=0)
                all_gen_labels.append(gen_labels)

                all_gen_labels.append(gen_info["next_maxq"])
                all_gen_actions.append(gen_info["numpy_actions"])
            
        winner = game.winner

        metrics = (game.get_metrics(0), game.get_metrics(1))
        metrics[winner]["pfinal"] = disc_label_buffers[winner][-1]
        metrics[1 - winner]["pfinal"] = 1 - disc_label_buffers[1 - winner][-1]

        winner_obs = disc_obs_buffers[winner]
        loser_obs = disc_obs_buffers[1 - winner]

        disc_label_buffers[winner].append(1)
        disc_label_buffers[1 - winner].append(0)
        winner_disc_labels = disc_label_buffers[winner][1:]
        loser_disc_labels = disc_label_buffers[1 - winner][1:]


        if self.hparams['lambda'] is not None:
            winner_disc_labels = smooth_labels(winner_disc_labels, self.hparams['lambda'])
            loser_disc_labels = smooth_labels(loser_disc_labels, self.hparams['lambda'])

        all_disc_obs = winner_obs + loser_obs
        all_disc_labels = np.concatenate([winner_disc_labels, loser_disc_labels])

        if len(all_gen_obs) > 0:
            all_gen_obs = stack_dicts(all_gen_obs)
            all_gen_labels = np.concatenate(all_gen_labels)
            all_gen_actions = np.concatenate(all_gen_actions)

        result = RolloutEpisode(
            disc_obs=all_disc_obs, 
            disc_labels=all_disc_labels, 
            gen_obs=all_gen_obs,
            gen_labels=all_gen_labels,
            gen_actions=all_gen_actions,
            metrics=metrics)

        return result