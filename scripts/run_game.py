from minionsai.agent import NullAgent, RandomAIAgent
from minionsai.run_game import run_game
from minionsai.engine import Game

agents = [NullAgent(), RandomAIAgent()]
game = Game()
run_game(game, agents, verbose=True)