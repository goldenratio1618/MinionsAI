import tempfile
from minionsai.action_bot.model import MinionsActionBot
from minionsai.agent import RandomAIAgent
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.game_util import seed_everything
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import QDiscriminator, ScriptedDiscriminator
from minionsai.gen_disc.generator import AgentGenerator, QGenerator
from minionsai.metrics_logger import allow_repeated_logging
from minionsai.multiprocessing_rl.multiproc_rollouts import MultiProcessRolloutSource
from minionsai.multiprocessing_rl.rollout_runner import RolloutRunner
from minionsai.multiprocessing_rl.rollouts import InProcessRolloutSource
from minionsai.multiprocessing_rl.rollouts_data import RolloutBatch
import numpy as np
import pytest

def agent_fn_disc():
    generator = AgentGenerator(RandomAIAgent())
    disc_model = MinionsDiscriminator(depth=1, d_model=8)
    disc_model.to('cpu')
    discriminator = QDiscriminator(Translator(mode="discriminator"), disc_model, epsilon_greedy=0.5)
    agent = GenDiscAgent(discriminator, [(generator, 2)])
    return agent

# If things are broken, this test can hang. So we add a timeout.
@pytest.mark.timeout(10)
def test_runner_same_as_multiproc_disc():
    seed_everything(1112)
    allow_repeated_logging()
    agent = agent_fn_disc()
    game_kwargs = {'max_turns': 4}

    # state_dict = agent.discriminator.model.state_dict()
    # agent.discriminator.model.load_state_dict(state_dict)
    agent.discriminator.model.eval()
    local_rollouts = InProcessRolloutSource(episodes_per_iteration=4, game_kwargs=game_kwargs, agent=agent)
    multiproc_rollouts = MultiProcessRolloutSource(agent_fn_disc, agent, logs_directory=tempfile.mkdtemp(), episodes_per_iteration=4, game_kwargs=game_kwargs, num_procs=2, device='cpu', train_discriminator=True, train_generator=False)


    for iter in range(4):
        data_local: RolloutBatch = local_rollouts.get_rollouts(iter)["discriminator"]
        data_multiproc: RolloutBatch = multiproc_rollouts.get_rollouts(iter)["discriminator"]
        for key in data_local.obs:
            np.testing.assert_allclose(data_local.obs[key], data_multiproc.obs[key]), key
        np.testing.assert_allclose(data_local.next_maxq, data_multiproc.next_maxq)

def agent_fn_gen():
    gen_model = MinionsActionBot(depth=1, d_model=8)
    gen_model.to('cpu')
    generator = QGenerator(Translator(mode="generator"), gen_model, epsilon_greedy=0.5)
    discriminator = ScriptedDiscriminator()
    agent = GenDiscAgent(discriminator, [(generator, 2)])
    return agent

@pytest.mark.timeout(10)
def test_runner_same_as_multiproc_gen():
    seed_everything(1112)
    allow_repeated_logging()
    agent = agent_fn_gen()
    game_kwargs = {'max_turns': 4}

    state_dict = agent.generators[0][0].model.state_dict()
    agent.generators[0][0].model.load_state_dict(state_dict)
    agent.generators[0][0].model.eval()
    local_rollouts = InProcessRolloutSource(episodes_per_iteration=4, game_kwargs=game_kwargs, agent=agent)
    multiproc_rollouts = MultiProcessRolloutSource(agent_fn_gen, agent, logs_directory=tempfile.mkdtemp(), episodes_per_iteration=4, game_kwargs=game_kwargs, num_procs=2, device='cpu', train_discriminator=False, train_generator=True)


    for iter in range(4):
        data_local: RolloutBatch = local_rollouts.get_rollouts(iter)["generator"]
        data_multiproc: RolloutBatch = multiproc_rollouts.get_rollouts(iter)["generator"]
        
        for key in data_local.obs:
            np.testing.assert_allclose(data_local.obs[key], data_multiproc.obs[key]), key
        np.testing.assert_allclose(data_local.next_maxq, data_multiproc.next_maxq)
