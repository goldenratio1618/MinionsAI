from minionsai.agent import NullAgent, RandomAIAgent
from minionsai.engine import Game
from minionsai.verify_agent_saveload import verify_all_modes

def test_null_agent():
    verify_all_modes(NullAgent(), Game)

def test_random_agent():
    verify_all_modes(RandomAIAgent(), Game)