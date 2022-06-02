from minionsai.agent import RandomAIAgent, NullAgent
from minionsai.engine import Game
from minionsai.run_game import run_game

def test_run_game():
    test_game = Game()
    run_game(test_game, (RandomAIAgent(), NullAgent()), verbose=True)