from minionsai.agent import Agent
from minionsai.engine import Game, Unit
import torch as th
import random

from minionsai.unit_type import NECROMANCER, ZOMBIE

agent_path = "C:\\Users/Maple/AppData/Local/Temp/MinionsAI/d3/checkpoints/iter_45"
agent = Agent.load(agent_path)
agent.verbose_level = 2

random.seed(123)

game = Game()
for (i, j), hex in game.board.hexes():
    hex.is_graveyard = False
    hex.unit = None

game.board.board[0][1].is_graveyard = True
game.board.board[1][1].is_graveyard = True
game.board.board[4][3].is_graveyard = True
game.board.board[3][3].is_graveyard = True

game.board.board[2][2].add_unit(Unit(0, NECROMANCER))
game.board.board[4][2].add_unit(Unit(1, NECROMANCER))

game.board.board[0][1].add_unit(Unit(0, ZOMBIE))
game.board.board[1][1].add_unit(Unit(0, ZOMBIE))
game.board.board[0][4].add_unit(Unit(0, ZOMBIE))
game.board.board[1][3].add_unit(Unit(0, ZOMBIE))
game.board.board[3][1].add_unit(Unit(0, ZOMBIE))

game.board.board[1][4].add_unit(Unit(1, ZOMBIE))
# game.board.board[2][3].add_unit(Unit(1, ZOMBIE))
game.board.board[3][2].add_unit(Unit(1, ZOMBIE))
game.board.board[4][1].add_unit(Unit(1, ZOMBIE))
game.board.board[2][4].add_unit(Unit(1, ZOMBIE))
game.board.board[3][3].add_unit(Unit(1, ZOMBIE))
game.board.board[2][4].add_unit(Unit(1, ZOMBIE))
game.board.board[3][4].add_unit(Unit(1, ZOMBIE))
game.board.board[4][3].add_unit(Unit(1, ZOMBIE))
# game.board.board[4][4].add_unit(Unit(1, ZOMBIE))

game.money = [9, 3]
game.remaining_turns -= 10
game.pretty_print()
print(game.get_scores)

obs = agent.translator.translate(game.copy())
print(obs)
logprob = agent.policy(obs)
win_prob = th.sigmoid(logprob).item()
print(f"Win probability: {win_prob:.3f}")
game.next_turn()
game.full_turn(agent.act(game.copy()))
game.next_turn()
game.full_turn(agent.act(game.copy()))