from minionsai.agent import CLIAgent, RandomAIAgent
from minionsai.engine import Game
from minionsai.run_game import run_game

def test_run_game():
    test_game = Game()
    run_game(test_game, (RandomAIAgent(), CLIAgent(["python3", "u", "minionsai/scripts/randomAI.py"])), verbose=True)
