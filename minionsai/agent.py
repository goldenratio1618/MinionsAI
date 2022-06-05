import abc
import random
import subprocess
import sys

from .action import ActionList, SpawnAction, MoveAction
from .engine import Game, Phase, adjacent_hexes
from .unit_type import ZOMBIE, NECROMANCER, unitList, unit_type_from_name

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

    def save_instance(self, directory):
        """
        Save any extra data into `directory` needed to build this agent instance.
        You should override `save` and `load` so that this is a noop:

        example_agent = ExampleAgent()
        example_agent.save(dir)
        example_agent = ExampleAgent.load(dir)
        """
        pass

    @classmethod
    def load_instance(cls, directory: str) -> "Agent":
        print(f"Loading instance of agent {cls.__name__}")
        return cls()

    def seed(self, seed: int):
        """
        Seeds any relevant random number generators.
        """
        pass

    def save(self, directory: str, exists_ok=False):
        """
        Creates a snapshot of this agent that can be passed around and run on other codebases inside `directory`
        It should expose an API like this:

        from directory import build_agent()
        agent = build_agent()  # gives back this Agent object.

        To do that we need to store 3 things:
        1. The current codebase
        2. A __init__.py file with build_agent() entry point
        3. Your subclass may need to store other stuff as well to reproduce an instance;
            you should do that by overriding save_instance():
        """
        print(f"Saving agent into {directory}")
        if os.path.exists(directory):
            if exists_ok:
                shutil.rmtree(directory)
            else:
                raise ValueError(f"Save failed - directory {directory} already exists")
        else:
            os.makedirs(directory)

        ####### 1. Store the codebase #######
        # No recursive copying
        ignore_patterns = [".git", "__pycache__", "scoreboard*", "tests", "scripts"]
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

        ####### 3. Save extra data #######
        agent_dir = os.path.join(directory, 'agent')
        os.makedirs(agent_dir)
        self.save_instance(agent_dir)

    @staticmethod
    def load(directory: str):
            print(f"Loading from directory...")
            outer_dir = os.path.dirname(directory)
            print(f"  Temporarily adding to sys.path: {outer_dir}")
            added_to_path = False
            if not outer_dir in sys.path:
                sys.path.append(outer_dir)
                added_to_path = True
            try:
                module_name = os.path.basename(directory)
                module = importlib.import_module(module_name)
            except Exception as e:
                raise
            finally:
                # Make sure to remove it no matter what, even if tehre was an error
                if added_to_path:
                    sys.path.remove(outer_dir)
            return module.build_agent()

class NullAgent(Agent):
    """
    Agent that does nothing.
    """
    def act(self, game_copy: Game) -> ActionList:
        return ActionList([], [])

class HumanCLIAgent(Agent):
    """
    Agent that uses the command line to interact with the user.
    """
    def act(self, game_copy: Game) -> ActionList:
        print("========= NEW TURN =========")
        active_game_copy = game_copy.copy()
        move_actions = []
        spawn_actions = []
        while True:
            print("---> Your turn")
            print("Actions so far:")
            print(f"  Move phase: {move_actions}")
            print(f"  Spawn phase: {spawn_actions}")
            print("Type h for help")
            active_game_copy.pretty_print()
            input = sys.stdin.readline().strip()
            if input == "":
                continue
            elif input in ["e", "end"]:
                break
            elif input in ["h", "help"]:
                print("""
                (h)elp - show this help
                (e)nd - end the turn
                (m)ove i j k l - move a unit from (i, j) to (k, l)
                (s)pawn X i j - spawn a unit of type X at (i, j)
                (u)ndo - undo the turn
                """)
                continue
            elif input in ["u", "undo"]:
                active_game_copy = game_copy.copy()
                move_actions = []
                spawn_actions = []
                continue
            input_list = input.split()
            if input_list[0] in ["m", "move"]:
                if len(spawn_actions) > 0:
                    print("You cannot move a unit after entering the spawn phase.")
                    continue
                try:
                    i, j, k, l = [int(x) for x in input_list[1:]]
                except ValueError:
                    print("Invalid move command; should be `(m)ove i j k l`")
                    continue
                action = MoveAction((i, j), (k, l))
                move_actions.append(action)
                active_game_copy.process_single_action(action)
            if input_list[0] in ["s", "spawn"]:
                try:
                    unit_type_str = input_list[1]
                    # if it's an int, use it as an index; if it's one letter, look for one with that first letter
                    if unit_type_str.isdigit():
                        unit_type = unitList[int(unit_type_str)]
                    elif len(unit_type_str) == 1:
                        unit_type = next(unit for unit in unitList if unit.name[0].lower() == unit_type_str.lower())
                    else:
                        unit_type = unit_type_from_name(unit_type_str)
                    (i, j) = [int(x) for x in input_list[2:]]
                except ValueError:
                    print("Invalid spawn command; should be `(s)pawn type i j`")
                    continue
                action = SpawnAction(unit_type, (i, j))
                spawn_actions.append(action)
                if active_game_copy.phase == Phase.MOVE:
                    active_game_copy.end_move_phase()
                active_game_copy.process_single_action(action)
        return ActionList(move_actions, spawn_actions)


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
            if unit.type == NECROMANCER:
                # Attack if possible
                attack_targets = adjacent_hexes(i, j)
                random.shuffle(attack_targets)
                for (k, l) in attack_targets:
                    unit_there = game_copy.board.board[k][l].unit
                    if unit_there is not None and unit_there.color != game_copy.active_player_color:
                        move_actions.append(MoveAction((i, j), (k, l)))
                        break

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
    return {class_name}.load_instance(os.path.join(os.path.dirname(__file__), 'agent'))
"""