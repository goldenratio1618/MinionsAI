from minionsai.agent import NullAgent, RandomAIAgent
from minionsai.engine import Game
from minionsai.test_agent_saveload import test_all_modes

test_all_modes(NullAgent(), Game)
test_all_modes(RandomAIAgent(), Game)