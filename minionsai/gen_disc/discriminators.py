import abc
from audioop import reverse
from typing import Dict, List, Optional, Tuple

from ..game_util import adjacent_zombies, equal_np_dicts, sigmoid, stack_dicts
from ..unit_type import NECROMANCER, ZOMBIE, flexible_unit_type
from ..engine import Game, dist, print_n_games

import numpy as np
import torch as th

class BaseDiscriminator(abc.ABC):
    @abc.abstractmethod
    def choose_option(self, games: List[Game]) -> Tuple[int, Optional[Dict]]:
        """
        Return the best index, and optionally a dictionary of extra info.
        """
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

    def choose_option(self, games: List[Game]) -> Tuple[int, Optional[Dict]]:
        scores = [sigmoid(self.score(g)) for g in games]
        return np.argmax(scores), {"all_winprobs": scores}

class HumanDiscriminator(BaseDiscriminator):
    def __init__(self, ncols=8, nrows=2, filter_agent=None):
        self.ncols = ncols
        self.nrows = nrows
        self.n_options = ncols * nrows
        self.filter_agent = filter_agent

    def choose_option(self, games: List[Game]) -> Tuple[int, Optional[Dict]]:
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

            sorted_idxes = sorted(equivalent_games, key=lambda i: i)
        else:
            sorted_idxes = list(range(len(games)))
            logprobs = [None] * len(games)

        chosen = self.display_and_choose(sorted_idxes, logprobs, games)
        print(f"Chosen option {chosen} with win prob {th.sigmoid(logprobs[chosen]).item():.1%}")
        return chosen, None

    def display_and_choose(self, sorted_idxes, logprobs, games):
        for r in range(min(self.nrows, len(sorted_idxes) // self.ncols + 1)):
            print("-" * 16 * self.ncols)
            idxs = sorted_idxes[r * self.ncols : (r + 1) * self.ncols]
            print_n_games([games[i] for i in idxs])
            if self.filter_agent is not None:
                print("".join([f"Opt {i:<3}".ljust(15)+"|" for i in idxs]))

        print("-" * 16 * self.ncols)
        print(f"{len(sorted_idxes)} unique choices of {len(games)} total generated.")
        option = input("Choose an option: ")
        # If it's a valid number less than 256, it's a choice and we return it.
        if option.isdigit() and int(option) < len(games):
            return int(option)
        # If it starts with "f" or "filter" then it's a request to filter the games
        elif option.startswith("f") or option.startswith("filter"):
            try:
                _f, unit_type, i, j = option.split(" ")
                i, j = int(i), int(j)
                unit_type = flexible_unit_type(unit_type)
                # resort indices by whether the unit is at the given location
                def filter(game):
                    unit = game.board.board[i][j].unit
                    return unit is not None and unit.type == unit_type
                resorted_idxes = [i for i in sorted_idxes if filter(games[i])]
                return self.display_and_choose(resorted_idxes, logprobs, games)
            except Exception as e:
                print("Invalid filter request")
                print(e)
                return self.display_and_choose(sorted_idxes, logprobs, games)
        # If it starts with "c" or "clear" then it's a request to clear the filters
        elif option.startswith("c") or option.startswith("clear"):
            return self.choose_option(games)            
        else:
            print("Invalid option")
            return self.display_and_choose(sorted_idxes, logprobs, games)

class QDiscriminator(BaseDiscriminator):
    def __init__(self, translator, model, epsilon_greedy):
        self.translator = translator
        self.model = model
        self.epsilon_greedy = epsilon_greedy

    def choose_option(self, games: List[Game]) -> Tuple[int, Optional[Dict]]:
        obs_list = [self.translator.translate(g) for g in games]

        obs = stack_dicts(obs_list)
        logprobs = self.model(obs).detach().cpu().numpy()  # shape (n_options, 1)
        logprobs = np.squeeze(logprobs)
        result = self.sample(logprobs)
        max_logprob = np.max(logprobs)
        max_winprob = sigmoid(max_logprob).item()
        return result, {"max_winprob": max_winprob, "chosen_final_obs": obs_list[result], "all_winprobs": sigmoid(logprobs)}

    def sample(self, logprobs):
        if np.random.random() < self.epsilon_greedy:
            return np.random.choice(len(logprobs))
        else:
            return np.argmax(logprobs).item()

