from typing import Tuple
from engine import Game
from agent import Agent, RandomAIAgent

def run_game(game: Game, agents: Tuple[Agent, Agent], verbose=False) -> int:
    while True:
        print("===================================")
        game.next_turn()
        if verbose:
            print("Remaining turns:", game.remaining_turns)
            game.pretty_print()
        if game.done:
            break
        active_agent = agents[game.active_player_color]
        game_copy = game.copy()
        actionlist = active_agent.act(game_copy)
        if verbose:
            print(actionlist)
        game.full_turn(actionlist)
    return game.winner

if __name__ == "__main__":
    test_game = Game()
    run_game(test_game, (RandomAIAgent(), RandomAIAgent()), verbose=True)