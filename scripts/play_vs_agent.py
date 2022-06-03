import sys
from minionsai.agent import Agent, HumanCLIAgent, RandomAIAgent
from minionsai.run_game import run_game
from minionsai.engine import Game

def main(agent_dir=None):
    if agent_dir is None:
        agent = RandomAIAgent()
    else:
        agent = Agent.load(agent_dir)
    game = Game()
    run_game(game, (HumanCLIAgent(), agent))

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) >= 2 else None
    main(arg)