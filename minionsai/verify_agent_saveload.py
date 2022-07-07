import pickle
from typing import List, Tuple

from .game_util import seed_everything
from .action import ActionList
from .agent import Agent, NullAgent
from .agent_saveload import compare_multigame_histories, play_n_recording_actions, save, load
from .engine import Game
import numpy as np
import random
import os
import tempfile
import shutil

def verify_agent_saveload(agent, game_fn, num_games=10, mode='full') -> List[ActionList]:
    """
    agent: Agent to test
    game_fn: function that returns a game object playablke by this agent
    mode:
        * 'full': Test full save/load cycle including serializing all the code.
        * 'save': Test only the save_extra() and load_extra() methods.
        * 'seed': Just check that the agent is deterministic after seeding.
    """
    
    agent0 = agent
    if mode == 'seed':
        agent1 = agent
    elif mode == 'save':
        dir = tempfile.mkdtemp()
        print("Using temp dir: ", dir)
        try:
            agent.save_extra(dir)
            agent1 = pickle.loads(pickle.dumps(agent))
            agent1.load_extra(dir)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(dir)

    elif mode == 'full':
        dir = tempfile.mkdtemp()
        print("Using temp dir: ", dir)
        print(os.path.exists(dir))
        try:
            save(agent, dir, exists_ok=True)
            agent1 = load(dir)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    success = compare_agents(agent0, agent1, game_fn, num_games)
    return success

def verify_all_modes(agent, game_fn, num_games=30):
    success = verify_agent_saveload(agent, game_fn, num_games, mode='seed')
    if success:
        print("Passed seeding test.")
    else:
        print("FAILURE!")
        print("Determinism test failed; your agent is not deterministic after seeding.")
        print("Aborting saveload test.")
        return False

    success = verify_agent_saveload(agent, game_fn, num_games, mode='save')
    if success:
        print("Passed save_extra / load_extra test.")
    else:
        print("FAILURE!")
        print("save_extra & load_extra are not faithful.")
        print("Aborting saveload test.")
        return False

    success = verify_agent_saveload(agent, game_fn, num_games, mode='full')
    if success:
        print("SUCCESS!")
        print("All tests passed!")
        return True
    else:
        print("FAILURE!")
        print("Saveload test failed; your agent does not match after full code serialization & deserialization.")
        return False

def compare_agents(agent0, agent1, game_fn, num_games):
    actions_take_1 = play_n_recording_actions(game_fn, (agent0, NullAgent()), num_games)
    actions_take_2 = play_n_recording_actions(game_fn, (agent1, NullAgent()), num_games)
    return compare_multigame_histories(actions_take_1, actions_take_2)
