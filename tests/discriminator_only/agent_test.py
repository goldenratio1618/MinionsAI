import torch as th
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.agent import TrainedAgent
from minionsai.discriminator_only.translator import Translator
from minionsai.agent import RandomAIAgent
from minionsai.engine import Game
from minionsai.test_agent_saveload import test_all_modes

def test_saveload():
    d_model = 128
    th.manual_seed(1112)
    model = MinionsDiscriminator(d_model)
    model.to('cpu')
    agent = TrainedAgent(model, Translator(), RandomAIAgent(), 4)
    test_all_modes(agent, Game, num_games=6)

test_saveload()