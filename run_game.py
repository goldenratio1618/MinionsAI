from typing import Tuple
from engine import Game
from agent import Agent, RandomAIAgent

def run_game(game: Game, agents: Tuple[Agent, Agent], verbose=False) -> int:
    while not game.done:
        active_agent = agents[game.active_player_color]
        action = active_agent.act(game)
        game.process_single_action(action)
        if verbose:
            print("Remaining turns:", game.remaining_turns)
            game.board.print_board_state()
    return game.winner

if __name__ == "__main__":
    test_game = Game()
    run_game(test_game, (RandomAIAgent(), RandomAIAgent()), verbose=True)