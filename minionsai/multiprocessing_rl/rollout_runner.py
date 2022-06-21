
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
        obs_buffers = [[], []]  # one for each player
        label_buffers = [[], []]  # one for each player
        while True:
            game.next_turn()
            if game.done:
                break
            active_player = game.active_player_color
            actionlist, best_winprob = self.agent.act_with_winprob(game)
            game.full_turn(actionlist)
            obs = self.agent.translator.translate(game)
            obs_buffers[active_player].append(obs)
            label_buffers[active_player].append(best_winprob)
            
        winner = game.winner

        metrics = (game.get_metrics(0), game.get_metrics(1))
        metrics[winner]["pfinal"] = label_buffers[winner][-1]
        metrics[1 - winner]["pfinal"] = 1 - label_buffers[1 - winner][-1]

        winner_obs = obs_buffers[winner]
        loser_obs = obs_buffers[1 - winner]

        label_buffers[winner].append(1)
        label_buffers[1 - winner].append(0)
        winner_labels = label_buffers[winner][1:]
        loser_labels = label_buffers[1 - winner][1:]


        if self.hparams['lambda'] is not None:
            winner_labels = smooth_labels(winner_labels, self.hparams['lambda'])
            loser_labels = smooth_labels(loser_labels, self.hparams['lambda'])

        all_obs = winner_obs + loser_obs
        all_labels = np.concatenate([winner_labels, loser_labels])

        return RolloutEpisode(
            disc_obs=all_obs, 
            disc_labels=all_labels, 
            gen_obs=[],  # TODO - produce generator rollout data
            gen_labels=[],  # TODO - produce generator rollout data
            metrics=metrics)
