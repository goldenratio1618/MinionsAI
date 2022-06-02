from minionsai.agent import Agent, NullAgent, RandomAIAgent
from minionsai.run_game import run_game
from minionsai.engine import Game

agents = [RandomAIAgent(), RandomAIAgent()]
game = Game()
run_game(game, agents, verbose=True)