from typing import Tuple
from .engine import Game
from .agent import Agent

class AgentException(Exception):
    def __init__(self, error, agent_index):
        self.error = error
        self.agent_index = agent_index

def run_game(game: Game, agents: Tuple[Agent, Agent], verbose=False) -> int:
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
    return game.winner