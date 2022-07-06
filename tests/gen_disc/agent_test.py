from minionsai.agent import RandomAIAgent
from minionsai.engine import Game
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import ScriptedDiscriminator
from minionsai.gen_disc.generator import AgentGenerator
from minionsai.verify_agent_saveload import verify_all_modes
import torch as th


def test_saveload_no_models():
    agent = GenDiscAgent(ScriptedDiscriminator(), AgentGenerator(RandomAIAgent()), 4)
    verify_all_modes(agent, Game, num_games=6)

def test_saveload_with_models():
    pass