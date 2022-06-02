import abc
import random
import subprocess
import sys

from .action import ActionList, SpawnAction, MoveAction
from .engine import Game, adjacent_hexes
from .unit_type import ZOMBIE, NECROMANCER, unitList

import os
import shutil
import importlib

class Agent(abc.ABC):
    """
    Actions:
        * Agent takes a Game object and returns an ActionList.
        * The game is a copy of the actual Game, so feel free to destructively do whatever with it.
            Use game.copy() to get another copy for backup.
        * You need to ultimately return an ActionList containing your entire turn,
            but you can use game.process_single_action() to see what happens after each single action within the turn.
    """
    @abc.abstractmethod
    def act(self, game_copy: Game) -> ActionList:
        raise NotImplementedError()

    @classmethod
    def deserialize_build(cls, directory: str) -> "Agent":
        """
        Override this if you need to do any special processing to load any extra data you saved in your override of serialize()
        """
        print("Deserializing agent", cls)
        return cls()

    def seed(self, seed: int):
        """
        Seeds any relevant random number generators.
        """
        pass

    def serialize(self, directory: str, exists_ok=False):
        """
        Creates a snapshot of this agent that can be passed around and run on other codebases inside `directory`
        It should expose an API like this:

        from directory import build_agent()
        agent = build_agent()  # gives back this Agent object.

        To do that we need to store 3 things:
        1. The current codebase
        2. A __init__.py file with build_agent()
        3. Your subclass may need to store other stuff as well; you should do that by overriding this method
        and doing whatever want after the super() call, e.g.:

        def serialize(self, directory: str):
            super().serialize(directory)
            agent_dir = os.path.join(directory, 'agent')
            os.mkdir(agent_dir)
            # Put stuff there.
        """
        print(f"Serializing agent into {directory}")
        if os.path.exists(directory):
            if not exists_ok:
                raise ValueError(f"Serialization failed - directory {directory} already exists")
        else:
            os.makedirs(directory)

        ####### 1. Store the codebase #######
        # No recursive copying
        ignore_patterns = [".git", "__pycache__"]
        ignore_patterns.append("*" + os.path.split(directory)[-1]+"*")

        # Copy all of MinionsAI/ into directory, ignoring files that match ignore_patterns
        # In a cross-platform compatible way
        
        source = os.path.join(os.path.dirname(__file__), '..')
        dest = os.path.join(directory, 'code')
        shutil.copytree(source, dest, ignore=shutil.ignore_patterns(*ignore_patterns))

        ####### 2. Make __init__.py #######
        module = self.__module__
        class_name = self.__class__.__name__
        with open(os.path.join(directory, "__init__.py"), "w") as f:
            init_contents = build_agent_init(module, class_name)
            f.write(init_contents)

    @staticmethod
    def deserialize(directory: str):
            print(f"Loading from directory...")
            print(f"  Temporarily adding to sys.path: {os.path.dirname(directory)}")
            sys.path.append(os.path.dirname(directory))
            module_name = os.path.basename(directory)
            module = importlib.import_module(module_name)
            sys.path.remove(os.path.dirname(directory))
            return module.build_agent()

class NullAgent(Agent):
    """
    Agent that does nothing.
    """
    def act(self, game_copy: Game) -> ActionList:
        return ActionList([], [])

class CLIAgent(Agent):
    def parse_input(self):
        input_list = []
        line = self.proc.stdout.readline().strip()
        while line != "":
            ints = [int(s) for s in line.split(" ") if s != ""]
            # move actions have length 4
            if len(ints) == 4:
                input_list.append(MoveAction((ints[0], ints[1]), (ints[2], ints[3])))
            elif len(ints) == 3:
                input_list.append(SpawnAction(unitList[ints[0]], (ints[1], ints[2])))
            line = self.proc.stdout.readline().strip()
        return input_list

    def __init__(self, commands):
        self.proc = subprocess.Popen(commands, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
        # send initial board config to process
        self.original_stdout = sys.stdout

    def act(self, game_copy: Game) -> ActionList:
        # send board state to process and then signal that turn begins
        sys.stdout = self.proc.stdin
        game_copy.board.print_board_properties()
        print()
        game_copy.board.print_board_state()
        print()
        print("Your turn")
        sys.stdout = self.original_stdout

        # collect input from process
        move_actions = self.parse_input()
        spawn_actions = self.parse_input()
        return ActionList(move_actions, spawn_actions)


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

def build_agent_init(module, class_name):
    return f"""
from .code.{module} import {class_name}
import os
import json

def build_agent():
    return {class_name}.deserialize_build(os.path.dirname(__file__))
"""