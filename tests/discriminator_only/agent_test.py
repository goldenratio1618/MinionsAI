import torch as th
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.agent import TrainedAgent
from minionsai.discriminator_only.translator import Translator
from minionsai.agent import RandomAIAgent
from minionsai.engine import Game
from minionsai.verify_agent_saveload import verify_all_modes

def test_saveload():
    d_model = 128
    depth = 2
    th.manual_seed(1112)
    model = MinionsDiscriminator(d_model, depth)
    model.to('cpu')
    agent = TrainedAgent(model, Translator(mode="discriminator"), RandomAIAgent(), 4)
    verify_all_modes(agent, Game, num_games=6)