from minionsai.agent import RandomAIAgent
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import QDiscriminator, ScriptedDiscriminator
from minionsai.gen_disc.generator import AgentGenerator
from minionsai.multiprocessing_rl.rollout_runner import RolloutRunner
import numpy as np

def test_runner_determinism():
    generator = AgentGenerator(RandomAIAgent())
    disc_model = MinionsDiscriminator(depth=1, d_model=8)
    disc_model.to('cpu')
    discriminator = QDiscriminator(Translator(), disc_model, epsilon_greedy=0.5)
    agent = GenDiscAgent(discriminator, generator, rollouts_per_turn=2)
    game_kwargs = {'max_turns': 4}
    runner = RolloutRunner(game_kwargs, agent)
    num_tests = 1
    data1 = [runner.single_rollout(iteration=i, episode_idx=7*i+3) for i in range(num_tests)]
    data2 = [runner.single_rollout(iteration=i, episode_idx=7*i+3) for i in range(num_tests)]
    for data1_episode, data2_episode in zip(data1, data2):
        np.testing.assert_allclose(data1_episode.disc_labels, data2_episode.disc_labels)
