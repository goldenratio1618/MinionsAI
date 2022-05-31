import abc
import random
from typing import List

from action import ActionList, SpawnAction, MoveAction
from engine import Game, adjacent_hexes
from unit_type import ZOMBIE, NECROMANCER

class Agent(abc.ABC):
    @abc.abstractmethod
    def act(self, game_copy: Game) -> ActionList:
        raise NotImplementedError()

class CLIAgent(Agent):
    def act(self, game_copy: Game) -> ActionList:
        # Do stuff with parse_input(yellow)
        raise NotImplementedError()

class RandomAIAgent(Agent):
    def act(self, game_copy: Game) -> ActionList:
        necromancer_location = None
        necromancer_destination = None
        for unit, (i, j) in game_copy.units_with_locations(color=game_copy.active_player_color):
            if unit.type.name == NECROMANCER.name:
                necromancer_location = (i, j)
                break
        
        move_actions = []
        for unit, (i, j) in game_copy.units_with_locations(color=game_copy.active_player_color):
            if random.random() < 0.2:
                # Don't move this guy
                dest = (i, j)
            else:
                dest = random.choice(adjacent_hexes(i, j))
            move_actions.append(MoveAction((i, j), dest))
            if (i, j) == necromancer_location:
                necromancer_destination = dest

        if necromancer_location is None:
            print("No necromancer found")
            game_copy.pretty_print()
            spawn_actions = []
        else:
            adjacent_targets = adjacent_hexes(*necromancer_location)
            spawn_actions = [
                SpawnAction(ZOMBIE, dest)
                for dest in random.sample(adjacent_targets, 2)
            ]
            # Also try spawning some spces two away from the necromancer
            # in case he moved.
            adjacent_targets = adjacent_hexes(*necromancer_destination)
            spawn_actions += [
                SpawnAction(ZOMBIE, dest)
                for dest in random.sample(adjacent_targets, 2)
            ]
        return ActionList(move_actions, spawn_actions)