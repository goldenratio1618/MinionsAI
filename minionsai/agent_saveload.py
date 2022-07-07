import importlib
import io
import os
import pickle
import shutil
import sys
import torch as th
from typing import List, Tuple

from .action import ActionList
from .game_util import seed_everything
from .engine import Game
from .experiment_tooling import find_device
from .agent import Agent

def save(agent: Agent, directory: str, exists_ok=False, copy_code_from=None):
    """
    Creates a snapshot of this agent that can be passed around and run on other codebases inside `directory`
    It stores a pickle of the agent, along with the codebase needed to read that pickle.

    To do that we need to store a bunch of things:
    agent_name/
        agent.pkl                   # the pickle of the agent
        agent_name_module/          # the codebase needed to load the agent. I think all these nested directories are actually needed, sadly.
            __init__.py
            code/
                minionsai/
        agent/                      # Any class-specific stuff needed, saved in save_extra() and loaded in load_extra()
            ...
        sample_games.pkl            # A bunch of games so we can confirm upon loading that everything is in order.

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
    pickle.dump(agent, open(os.path.join(directory, 'agent.pkl'), 'wb'))

    ####### 3. Save extra data #######
    agent_dir = os.path.join(directory, 'agent')
    os.makedirs(agent_dir)
    agent.save_extra(agent_dir)

    ####### 4. Make sample_games.pkl #######
    sample_games = play_n_recording_actions(Game, [agent, agent], 2, seed=Agent.SAMPLE_GAMES_SEED)
    pickle.dump(sample_games, open(os.path.join(directory, 'sample_games.pkl'), 'wb'))

# dict of {module_name: module_path} of all agents we've loaded so far.
_loaded_agents = {}

def load(directory: str, already_in_path_ok=False, test_load_equivalence=True) -> Agent:
    if not os.path.exists(directory):
        raise ValueError(f"Can't load agent from {directory} - directory does not exist.")
    if not os.path.exists(os.path.join(directory, 'agent.pkl')):
        return load_deprecated(directory, already_in_path_ok)
    name = os.path.split(directory)[-1]
    if name in _loaded_agents and _loaded_agents[name] != directory:
        raise ValueError(f"Can't load a second agent with the same name! (Loading {directory} but already loaded {_loaded_agents[name]}.")
    if name not in _loaded_agents:
        if directory in sys.path and not already_in_path_ok:
            raise ValueError(f"Agent is already in sys.path somehow: {directory}.")
        _loaded_agents[name] = directory
        sys.path.append(directory)

    print(f"Loading {directory}...")
    outer_module_name = f'{os.path.basename(directory)}_module'
    with open(os.path.join(directory, 'agent.pkl'), 'rb') as f:
        agent = LocalCodeUnpickler(f, outer_module_name).load()
    agent.load_extra(os.path.join(directory, "agent"))

    if test_load_equivalence:
        if not os.path.exists(os.path.join(directory, 'sample_games.pkl')):
            print("Can't test load equivalence because sample_games.pkl doesn't exist in prior agent. This should onyl happen with legacy agents.")
        else:
            print("Testing load equivalence...")
            old_sample_games = pickle.load(open(os.path.join(directory, 'sample_games.pkl'), 'rb'))
            new_sample_games = play_n_recording_actions(Game, [agent, agent], len(old_sample_games), seed=Agent.SAMPLE_GAMES_SEED)
            ok = compare_multigame_histories(old_sample_games, new_sample_games)
            if not ok:
                raise ValueError("Loaded agent doesn't seem to be equivalent to the one it was saved from! Print statements above might provide some clues.")
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

def load_deprecated(directory: str, already_in_path_ok=False):
    # TODO delete this once we have no old agents anymore.
    name = os.path.split(directory)[-1]
    if name in _loaded_agents and _loaded_agents[name] != directory:
        raise ValueError(f"Can't load a second agent with the same name! (Loading {directory} but already loaded {_loaded_agents[name]}.")
    if name not in _loaded_agents:
        if directory in sys.path and not already_in_path_ok:
            raise ValueError(f"Agent is already in sys.path somehow: {directory}.")
        _loaded_agents[name] = directory
        sys.path.append(directory)

    print(f"Loading legacy agent from {directory}")
    module_name = f"{os.path.basename(directory)}_module"
    module = importlib.import_module(module_name)
    return module.build_agent()

def _play_game_recording_actions(game: Game, agents: Tuple[Agent, Agent]) -> List[ActionList]:
    actions = []
    while True:
        game.next_turn()
        if game.done:
            return actions
        action = agents[game.active_player_color].act(game.copy())
        actions.append(action)
        game.full_turn(action)

def play_n_recording_actions(game_fn, agents, n, seed=1234) -> List[List[ActionList]]:
    seed_everything(seed)
    [a.seed(seed=seed) for a in agents]
    return [_play_game_recording_actions(game_fn(), agents) for _ in range(n)]

def compare_game_histories(history1, history2) -> bool:
    result = True
    shared_history = []
    if len(history1) != len(history2):
        print(f"The two games are different lengths: {len(history1)} vs {len(history2)}")
        result = False
        # Continue to compare the histories anyway, to proviude a helpful error message
    for i, (action1, action2) in enumerate(zip(history1, history2)):
        if str(action1) == str(action2):
            shared_history.append(action1)
        else:
            print(f"The two games differ")
            print(f"They both start with: {shared_history}")
            print(f"At index {i} they differ: {action1} vs {action2}")
            result = False
            break
    return result

def compare_multigame_histories(histories1, histories2) -> bool:
    if len(histories1) != len(histories2):
        print(f"The two games are different lengths: {len(histories1)} vs {len(histories2)}")
        return False
    for i, (game1, game2) in enumerate(zip(histories1, histories2)):
        if not compare_game_histories(game1, game2):
            print(f"Mismatch above was detected on game #{i}/{len(histories1)}!")
            print("====================")
            return False
    return True