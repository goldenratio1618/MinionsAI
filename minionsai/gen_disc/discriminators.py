import abc
from typing import List

from ..game_util import adjacent_zombies, equal_np_dicts, stack_dicts
from ..unit_type import NECROMANCER, ZOMBIE
from ..engine import Game, dist, print_n_games

import numpy as np
import torch as th

class BaseDiscriminator(abc.ABC):
    @abc.abstractmethod
    def choose_option(self, games: List[Game]) -> int:
        raise NotImplementedError()

    # @abc.abstractmethod
    # def save(self, directory: str):
    #     raise NotImplementedError()
    
    # @abc.abstractmethod
    # def load(self, directory: str):
    #     raise NotImplementedError()

class ScriptedDiscriminator(BaseDiscriminator):
    def score(self, game):
        result = 0

        scores = game.get_scores
        scores_advantage = scores[game.active_player_color] - scores[game.inactive_player_color]
        result += scores_advantage

        for unit, (i, j) in game.units_with_locations():
            if unit.color == game.active_player_color:
                if unit.type == ZOMBIE:
                    if len(adjacent_zombies(game.board, (i, j), game.inactive_player_color)) > 2:
                        result -= 2
                if unit.type == NECROMANCER:
                    dist_to_center = dist(i, j, 2, 2)
                    result -= dist_to_center * 0.4
        return result

    def choose_option(self, games: List[Game]) -> int:
        return np.argmax([self.score(g) for g in games])

class HumanDiscriminator(BaseDiscriminator):
    def __init__(self, ncols=8, nrows=2, filter_agent=None):
        self.ncols = ncols
        self.nrows = nrows
        self.n_options = ncols * nrows
        self.filter_agent = filter_agent

    def choose_option(self, games: List[Game]) -> int:
        if self.filter_agent is not None:
            obs = [self.filter_agent.translator.translate(g) for g in games]
            stacked_obs = stack_dicts(obs)
            logprobs = self.filter_agent.policy(stacked_obs)

            # Find non-equivalent games
            equivalent_games = []
            for i, this_obs in enumerate(obs):
                copy = False
                for j in equivalent_games:
                    if equal_np_dicts(obs[i], obs[j]):
                        copy = True
                        break
                if not copy:
                    equivalent_games.append(i)

            sorted_idxes = sorted(equivalent_games, key=lambda i: logprobs[i], reverse=True)
        else:
            sorted_idxes = list(range(len(games)))
            logprobs = [None] * len(games)

        for r in range(self.nrows):
            print("-" * 16 * self.ncols)
            idxs = sorted_idxes[r * self.ncols : (r + 1) * self.ncols]
            print_n_games([games[i] for i in idxs])
            print("".join([f"Opt {i:<3} ({th.sigmoid(logprobs[i]).item():.1%})".ljust(15)+"|" for i in idxs]))

        return int(input("Choose an option: "))