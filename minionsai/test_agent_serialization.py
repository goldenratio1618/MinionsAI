from typing import List, Tuple
from .action import ActionList
from .agent import Agent, NullAgent
from .engine import Game
import numpy as np
import random
import os
import tempfile
import shutil

def test_agent_serialization(agent, game_fn, num_games=10, serialize=False) -> List[ActionList]:
    """
    agent: Agent to test
    game_fn: function that returns a game object playablke by this agent
    serialize:
        * False: just check that the agent is deterministic after seeding.
        * True: check that the agent can be fully serialized and deserialized.
    """
    
    agent0 = agent
    if not serialize:
        agent1 = agent
    else:
        dir = tempfile.mkdtemp()
        print("Using temp dir: ", dir)
        print(os.path.exists(dir))
        try:
            agent.serialize(dir, exists_ok=True)
            agent1 = Agent.deserialize(dir)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(dir)

        
    success = compare_agents(agent0, agent1, game_fn, num_games)
    return success

def test_all_modes(agent, game_fn, num_games=30):
    success = test_agent_serialization(agent, game_fn, num_games, serialize=False)
    if success:
        print("Passed seeding test.")
    else:
        print("FAILURE!")
        print("Determinism test failed; your agent is not deterministic after seeding.")
        print("Aborting serialization test.")
        return False
    success = test_agent_serialization(agent, game_fn, num_games,serialize=True)
    if success:
        print("SUCCESS!")
        print("All tests passed!")
        return True
    else:
        print("FAILURE!")
        print("Serialization test failed; your agent does not match after serialization & deserialization.")
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
