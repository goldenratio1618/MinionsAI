from typing import Tuple
from engine import Game
from agent import Agent, RandomAIAgent

def run_game(game: Game, agents: Tuple[Agent, Agent], verbose=False) -> int:
    while not game.done:
        print("===================================")
        active_agent = agents[game.active_player_color]
        game.pretty_print()
        game_copy = game.copy()
        game_copy.pretty_print()
        actionlist = active_agent.act(game_copy)
        if verbose:
            print(actionlist)
        game.full_turn(actionlist)
        if verbose:
            print("Remaining turns:", game.remaining_turns)
            game.board.print_board_state()
    return game.winner

if __name__ == "__main__":
    test_game = Game()
    run_game(test_game, (RandomAIAgent(), RandomAIAgent()), verbose=True)