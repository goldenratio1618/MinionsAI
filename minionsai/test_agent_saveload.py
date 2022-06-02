from typing import List, Tuple
from .action import ActionList
from .agent import Agent, NullAgent
from .engine import Game
import numpy as np
import random
import os
import tempfile
import shutil

def test_agent_saveload(agent, game_fn, num_games=10, mode='full') -> List[ActionList]:
    """
    agent: Agent to test
    game_fn: function that returns a game object playablke by this agent
    mode:
        * 'full': Test full save/load cycle including serializing all the code.
        * 'save': Test only the save_instance() and load_instance() methods.
        * 'seed': Just check that the agent is deterministic after seeding.
    """
    
    agent0 = agent
    if mode == 'seed':
        agent1 = agent
    elif mode == 'save':
        dir = tempfile.mkdtemp()
        print("Using temp dir: ", dir)
        try:
            agent.save_instance(dir)
            agent1 = agent.__class__.load_instance(dir)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(dir)

    elif mode == 'full':
        dir = tempfile.mkdtemp()
        print("Using temp dir: ", dir)
        print(os.path.exists(dir))
        try:
            agent.save(dir, exists_ok=True)
            agent1 = Agent.load(dir)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    success = compare_agents(agent0, agent1, game_fn, num_games)
    return success

def test_all_modes(agent, game_fn, num_games=30):
    success = test_agent_saveload(agent, game_fn, num_games, mode='seed')
    if success:
        print("Passed seeding test.")
    else:
        print("FAILURE!")
        print("Determinism test failed; your agent is not deterministic after seeding.")
        print("Aborting saveload test.")
        return False

    success = test_agent_saveload(agent, game_fn, num_games, mode='save')
    if success:
        print("Passed save_instance / load_instance test.")
    else:
        print("FAILURE!")
        print("save_instance & load_instance are not faithful.")
        print("Aborting saveload test.")
        return False

    success = test_agent_saveload(agent, game_fn, num_games, mode='full')
    if success:
        print("SUCCESS!")
        print("All tests passed!")
        return True
    else:
        print("FAILURE!")
        print("Saveload test failed; your agent does not match after full code serialization & deserialization.")
        return False

def _play_game_recording_actions(game: Game, agents: Tuple[Agent, Agent]) -> List[ActionList]:
    actions = []
    while True:
        game.next_turn()
        if game.done:
            return actions
        action = agents[game.active_player_color].act(game.copy())
        actions.append(action)
        game.full_turn(action)

def _play_n_recording_actions(game_fn, agents, n, seed=1234) -> List[List[ActionList]]:
    random.seed(seed)
    np.random.seed(seed)
    [a.seed(seed=seed) for a in agents]
    return [_play_game_recording_actions(game_fn(), agents) for _ in range(n)]

def compare_game_histores(history1, history2) -> bool:
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

def compare_agents(agent0, agent1, game_fn, num_games):
    actions_take_1 = _play_n_recording_actions(game_fn, (agent0, NullAgent()), num_games)
    actions_take_2 = _play_n_recording_actions(game_fn, (agent1, NullAgent()), num_games)
    for i, (game1, game2) in enumerate(zip(actions_take_1, actions_take_2)):
        if not compare_game_histores(game1, game2):
            print(f"Mismatch above was detected on game #{i}/{num_games}!")
            print("====================")
            return False
    return True
