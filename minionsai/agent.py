import abc
import random
import subprocess
import sys

from minionsai.experiment_tooling import find_device

from .game_util import adjacent_zombies
from .action import ActionList, SpawnAction, MoveAction
from .engine import Board, Game, Phase, adjacent_hexes
from .unit_type import ZOMBIE, NECROMANCER, flexible_unit_type, unitList

import torch as th
import io
import pickle
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

    def save_extra(self, directory):
        """
        Save any extra data into `directory` needed to build this agent instance.
        You should override `save` and `load` so that this is a noop:

        example_agent = ExampleAgent()
        example_agent.save_extra(dir)
        pickle.dumps(example_agent, "agent.pkl")
        example_agent = pickle.loads("agent.pkl")
        example_agent.load_extra(dir)
        """
        pass

    def load_extra(self, directory: str):
        pass

    def seed(self, seed: int):
        """
        Seeds any relevant random number generators.
        """
        pass

    def save(self, directory: str, exists_ok=False, copy_code_from=None):
        """
        Creates a snapshot of this agent that can be passed around and run on other codebases inside `directory`
        It stores a pickle of the agent, along with the codebase needed to read that pickle.

        To do that we need to store 3 things:
        agent_name/
            agent.pkl                   # the pickle of the agent
            agent_name_module/          # the codebase needed to load the agent. I think all these nested directories are actually needed, sadly.
                __init__.py
                code/
                    minionsai/
            agent/                      # Any class-specific stuff needed, saved in save_extra() and loaded in load_extra()
                ...

        If `exists_ok` is True, then the directory will be overwritten if it exists.
        If `copy_code_from` is not None, then the codebase will be copied from that directory.
            That directory should be equivalent to MinionsAI/
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
        module_name = f'{os.path.basename(directory)}_module'

        if copy_code_from is None:
            copy_code_from = os.path.join(os.path.dirname(__file__), '..')
        dest = os.path.join(directory, module_name, 'code')
        shutil.copytree(copy_code_from, dest, ignore=shutil.ignore_patterns(*ignore_patterns))

        ####### 2. Make agent.pkl #######
        pickle.dump(self, open(os.path.join(directory, 'agent.pkl'), 'wb'))

        ####### 3. Save extra data #######
        agent_dir = os.path.join(directory, 'agent')
        os.makedirs(agent_dir)
        self.save_extra(agent_dir)

    # class-level dict of {module_name: module_path} of all agents we've loaded so far.
    loaded_agents = {}

    @staticmethod
    def load_deprecated(directory: str, already_in_path_ok=False):
        # TODO delete this once we have no old agents anymore.
        name = os.path.split(directory)[-1]
        if name in Agent.loaded_agents and Agent.loaded_agents[name] != directory:
            raise ValueError(f"Can't load a second agent with the same name! (Loading {directory} but already loaded {Agent.loaded_agents[name]}.")
        if name not in Agent.loaded_agents:
            if directory in sys.path and not already_in_path_ok:
                raise ValueError(f"Agent is already in sys.path somehow: {directory}.")
            Agent.loaded_agents[name] = directory
            sys.path.append(directory)

        print(f"Loading legacy agent from {directory}")
        module_name = f"{os.path.basename(directory)}_module"
        module = importlib.import_module(module_name)
        return module.build_agent()

    @staticmethod
    def load(directory: str, already_in_path_ok=False):
        if not os.path.exists(os.path.join(directory, 'agent.pkl')):
            return Agent.load_deprecated(directory, already_in_path_ok)
        name = os.path.split(directory)[-1]
        if name in Agent.loaded_agents and Agent.loaded_agents[name] != directory:
            raise ValueError(f"Can't load a second agent with the same name! (Loading {directory} but already loaded {Agent.loaded_agents[name]}.")
        if name not in Agent.loaded_agents:
            if directory in sys.path and not already_in_path_ok:
                raise ValueError(f"Agent is already in sys.path somehow: {directory}.")
            Agent.loaded_agents[name] = directory
            sys.path.append(directory)

        print(f"Loading {directory}...")
        outer_module_name = f'{os.path.basename(directory)}_module'
        with open(os.path.join(directory, 'agent.pkl'), 'rb') as f:
            agent = LocalCodeUnpickler(f, outer_module_name).load()
        agent.load_extra(os.path.join(directory, "agent"))
        return agent

class LocalCodeUnpickler(pickle.Unpickler):
    def __init__(self, file, outer_module_name):
        super().__init__(file)
        self.outer_module_name = outer_module_name
    def find_class(self, module, name):
        if 'minionsai' in module:
            sub_minions_part = module.split('minionsai.')[-1]
            new_module = f'{self.outer_module_name}.code.minionsai.{sub_minions_part}'
            return super().find_class(new_module, name)      
        
        # Copied from https://stackoverflow.com/questions/57081727/load-pickle-file-obtained-from-gpu-to-cpu
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: th.load(io.BytesIO(b), map_location=find_device())

        return super().find_class(module, name)

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
            active_game_copy.pretty_print()
            # Read user input
            inpt = input("Enter action (h for help): ")
            inpt = inpt.strip()
            if inpt == "":
                continue
            elif inpt in ["e", "end"]:
                break
            elif inpt in ["h", "help"]:
                print("""
     00         (h)elp - show this help
   01  10       (e)nd - end the turn
 02  11  20     (m)ove i j k l - move a unit from (i, j) to (k, l)
     ...        (s)pawn X i j - spawn a unit of type X at (i, j)
                (u)ndo - undo the turn
                """)
                continue
            elif inpt in ["u", "undo"]:
                active_game_copy = game_copy.copy()
                move_actions = []
                spawn_actions = []
                continue
            input_list = inpt.split()
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
                success, error_msg = active_game_copy.process_single_action(action)
            if input_list[0] in ["s", "spawn"]:
                try:
                    unit_type_str = input_list[1]
                    unit_type = flexible_unit_type(unit_type_str)
                    (i, j) = [int(x) for x in input_list[2:]]
                except ValueError:
                    print("Invalid spawn command; should be `(s)pawn type i j`")
                    continue
                action = SpawnAction(unit_type, (i, j))
                spawn_actions.append(action)
                if active_game_copy.phase == Phase.MOVE:
                    active_game_copy.end_move_phase()
                success, error_msg = active_game_copy.process_single_action(action)
            if not success:
                print(f"Error: {error_msg}")
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
        self.commands = commands
        # send initial board config to process
        self.original_stdout = sys.stdout

    def act(self, game_copy: Game) -> ActionList:
        # send board state to process and then signal that turn begins
        sys.stdout = self.proc.stdin
        print("Your color")
        print(game_copy.active_player_color)
        game_copy.board.print_board_properties()
        print()
        game_copy.board.print_board_state()
        print()
        print("Money")
        print(game_copy.money[0])
        print(game_copy.money[1])
        print()
        print("Your turn")
        sys.stdout = self.original_stdout

        # collect input from process
        move_actions = self.parse_input()
        spawn_actions = self.parse_input()
        return ActionList(move_actions, spawn_actions)

    def load_extra(self, directory: str):
        os.chdir(directory)
        os.chdir("../code")
        # set execution bit
        # we don't know which word of commands is file, so try all of them
        for i in self.commands: os.system("chmod +x " + i)
        self.__init__(self.commands)

class RandomAIAgent(Agent):
    def act(self, game_copy: Game) -> ActionList:
        necromancer_location = None
        necromancer_destination = None
        for unit, (i, j) in game_copy.units_with_locations(color=game_copy.active_player_color):
            if unit.type.name == NECROMANCER.name:
                necromancer_location = (i, j)
                break
        
        move_actions = []

        # If any enemy zombies can be killed, probably do that
        dead_zombie_attackers = []
        for unit, (i, j) in game_copy.units_with_locations(color=game_copy.inactive_player_color):
            if unit.type == ZOMBIE:
                adjacent_my_zombies = adjacent_zombies(game_copy.board, (i, j), game_copy.active_player_color)
                if len(adjacent_my_zombies) >= 2 and random.random() < 0.8:
                    dead_zombie_attackers.append(((i, j), adjacent_my_zombies))

        random.shuffle(dead_zombie_attackers)
        for enemy_zombie, attackers in dead_zombie_attackers:
            random.shuffle(attackers)
            for my_zombie in attackers[:2]:
                move_actions.append(MoveAction(my_zombie, enemy_zombie))

        my_units = list(game_copy.units_with_locations(color=game_copy.active_player_color))
        random.shuffle(my_units)
        for unit, (i, j) in my_units:
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
                attack_targets = list(adjacent_hexes(*dest))
                random.shuffle(attack_targets)
                for (k, l) in attack_targets:
                    unit_there = game_copy.board.board[k][l].unit
                    if unit_there is not None and unit_there.color != game_copy.active_player_color:
                        move_actions.append(MoveAction(dest, (k, l)))
                        break

        if necromancer_location is None:
            print("No necromancer found")
            game_copy.pretty_print()
            spawn_actions = []
        else:
            adjacent_targets = list(adjacent_hexes(*necromancer_destination))
            random.shuffle(adjacent_targets)
            spawn_actions = [
                SpawnAction(ZOMBIE, dest)
                for dest in random.sample(adjacent_targets, 2)
            ]
            # Also try spawning some spaces two away from the necromancer
            # in case he moved.
            adjacent_targets = list(adjacent_hexes(*necromancer_destination))
            random.shuffle(adjacent_targets)
            spawn_actions += [
                SpawnAction(ZOMBIE, dest)
                for dest in random.sample(adjacent_targets, 2)
            ]
        return ActionList(move_actions, spawn_actions)
