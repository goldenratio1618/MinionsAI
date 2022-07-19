from minionsai.action_bot.model import MinionsActionBot
from minionsai.agent import RandomAIAgent
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.engine import Game
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import QDiscriminator, ScriptedDiscriminator
from minionsai.gen_disc.generator import AgentGenerator, QGenerator
from minionsai.verify_agent_saveload import verify_all_modes
import torch as th


def test_saveload_no_models():
    agent = GenDiscAgent(ScriptedDiscriminator(), [(AgentGenerator(RandomAIAgent()), 4)])
    verify_all_modes(agent, Game, num_games=6)

def test_saveload_with_qdisc():
    D_MODEL = 16
    DEPTH=2
    disc_translator = Translator(mode='discriminator')
    disc_model = MinionsDiscriminator(D_MODEL, depth=DEPTH)
    disc_model.to(th.device('cpu'))
    agent = GenDiscAgent(QDiscriminator(disc_translator, disc_model, epsilon_greedy=0.1), [(AgentGenerator(RandomAIAgent()), 4)])
    verify_all_modes(agent, Game, num_games=6)

def test_saveload_with_qgen():
    D_MODEL = 16
    DEPTH=2
    gen_translator = Translator(mode='generator')
    gen_model = MinionsActionBot(D_MODEL, depth=DEPTH)
    gen_model.to(th.device('cpu'))
    agent = GenDiscAgent(ScriptedDiscriminator(), [(QGenerator(gen_translator, gen_model, epsilon_greedy=0.04), 4)])
    verify_all_modes(agent, Game, num_games=3)

def test_saveload_qgen_qdisc():
    D_MODEL = 16
    DEPTH=2
    gen_translator = Translator(mode='generator')
    gen_model = MinionsActionBot(D_MODEL, depth=DEPTH)
    gen_model.to(th.device('cpu'))

    disc_translator = Translator(mode='discriminator')
    disc_model = MinionsDiscriminator(D_MODEL, depth=DEPTH)
    disc_model.to(th.device('cpu'))
    agent = GenDiscAgent(QDiscriminator(disc_translator, disc_model, epsilon_greedy=0.1), [(QGenerator(gen_translator, gen_model, epsilon_greedy=0.04), 4)])
    verify_all_modes(agent, Game, num_games=2)