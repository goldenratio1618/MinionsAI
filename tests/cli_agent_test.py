from minionsai.agent import CLIAgent, RandomAIAgent
from minionsai.engine import Game
from minionsai.run_game import run_game

def test_cli_agent_game():
    test_game = Game()
    run_game(test_game, (RandomAIAgent(), CLIAgent(["python3", "-u", "scripts/randomAI.py"])), verbose=True)

test_cli_agent_game()
