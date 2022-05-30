

from engine import BOARD_SIZE, unitList
import random

class RandomGenerator():
    def rollout(self, game_state):
        moves = self.random_moves(game_state)
        spawns = self.random_spawns(game_state)
        game_state.turn(moves, spawns, auto_continue=False)
        return (moves, spawns)

    def redo(prev_actions, game_state):
        moves, spawns = prev_actions
        game_state.turn(moves, spawns, auto_continue=False)

    def random_moves(self, game_state):
        random_moves = []
        for (i, j), unit in game_state.units(color=game_state.active_player_color):
            for _ in range(4):
                dest = [random.randint(0, BOARD_SIZE) for _ in range(4)]
                random_moves.append([i, j, *dest])
        return random_moves

    def random_spawns(self, game_state):
        return [(random.randint(0, len(unitList)), random.randint(0, BOARD_SIZE), random.randint(0, BOARD_SIZE)) for _ in range(4)]
        