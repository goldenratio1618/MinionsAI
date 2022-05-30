

from action import FinishPhaseAction, MoveAction, SpawnAction
from engine import adjacent_hexes
from unit_type import NECROMANCER, ZOMBIE
import random

class RandomGenerator():
    def rollout(self, game_state):
        moves = self.random_moves(game_state)
        spawns = self.random_spawns(game_state)
        all_actions = moves + [FinishPhaseAction()] + spawns + [FinishPhaseAction()]
        for action in all_actions:
            game_state.process_single_action(action)
        return all_actions

    def redo(self, prev_actions, game_state):
        for action in prev_actions:
            game_state.process_single_action(action)

    def random_moves(self, game_state):
        random_moves = []
        for unit, (i, j) in game_state.units_with_locations(color=game_state.active_player_color):
            dest = random.choice(adjacent_hexes(i, j))
            random_moves.append(MoveAction((i, j), dest))
        return random_moves

    def random_spawns(self, game_state):
        necromancer_location = None
        for unit, (i, j) in game_state.units_with_locations(color=game_state.active_player_color):
            if unit.type.name == NECROMANCER.name:
                necromancer_location = (i, j)
                break
        if necromancer_location is None:
            print("No necromancer found")
            game_state.pretty_print()
            return []
        adjacent_targets = adjacent_hexes(*necromancer_location)
        return [
            SpawnAction(ZOMBIE, dest)
            for dest in random.sample(adjacent_targets, 2)
        ]
        