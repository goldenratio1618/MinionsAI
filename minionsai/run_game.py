from collections import defaultdict
from typing import Tuple

import tqdm
from .engine import Game
from .agent import Agent
import numpy as np

class AgentException(Exception):
    def __init__(self, error, agent_index):
        self.error = error
        self.agent_index = agent_index

def run_game(game: Game, agents: Tuple[Agent, Agent], verbose=False, randomize_player_order=False) -> int:
    winner_agent, metrics = run_game_with_metrics(game, agents, verbose=verbose, randomize_player_order=randomize_player_order)
    return winner_agent

def run_game_with_metrics(game: Game, agents: Tuple[Agent, Agent], verbose=False, randomize_player_order=False) -> int:
    if randomize_player_order and np.random.random() < 0.5:
        agents = agents[::-1]
        reversed_agents = True
    else:
        reversed_agents = False
    while True:
        if verbose:
            print("===================================")
            print("Remaining turns:", game.remaining_turns)
            game.pretty_print()

        game.next_turn()
        if game.done:
            break
        active_agent = agents[game.active_player_color]
        game_copy = game.copy()
        try:
            actionlist = active_agent.act(game_copy)
        except Exception as e:
            print("Error in agent's act() function!")
            raise AgentException(e, game.active_player_color)
        if verbose:
            print(actionlist)
        game.full_turn(actionlist, verbose=verbose)
    metrics = game.get_metrics(0), game.get_metrics(1)
    if reversed_agents:
        metrics = metrics[::-1]

    winner_color = game.winner
    if reversed_agents:
        winner_agent = 1 - winner_color
    else:
        winner_agent = winner_color
    return winner_agent, metrics

def accumulate_metrics(metrics_list, num_games):
    metrics = defaultdict(list)
    for metrics_dict in metrics_list:
        for key, value in metrics_dict.items():
            metrics[key].append(value)
    for key, value in metrics.items():
        metrics[key] = sum(value) / num_games
    return metrics

def run_n_games(game_fn, agents, n, randomize_player_order=True, progress_bar=True):
    """
    Runs n games. If num_threads > 1, runs n games in parallel.

    Returns:
    - wins, a tuple of ints of how many times each agent won
    - metrics, a tuple of average metrics for each agent.
    """
    # if agents are a string, treat it as a path to a file containing the agent
    agents = [Agent.load(agent) if isinstance(agent, str) else agent for agent in agents]
    iterator = range(n)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)
    results = [run_game_with_metrics(game_fn(), agents, randomize_player_order=randomize_player_order) for _ in iterator]
    # Count how many times each agent won
    winners = [result[0] for result in results]
    wins = [0, 0]
    for winner in winners:
        wins[winner] += 1
    metrics = [accumulate_metrics([result[1][agent_idx] for result in results], n) for agent_idx in range(2)]
    return wins, metrics
