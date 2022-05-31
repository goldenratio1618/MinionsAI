from typing import Tuple
from engine import Game
from agent import Agent, RandomAIAgent, CLIAgent

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
        actionlist = active_agent.act(game_copy)
        if verbose:
            print(actionlist)
        game.full_turn(actionlist)
    return game.winner

if __name__ == "__main__":
    test_game = Game()
    run_game(test_game, (RandomAIAgent(), CLIAgent(["python3", "-u", "randomAI.py"])), verbose=True)
