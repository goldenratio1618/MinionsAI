from minionsai.agent import RandomAIAgent
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.game_util import seed_everything
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import QDiscriminator, ScriptedDiscriminator
from minionsai.gen_disc.generator import AgentGenerator
from minionsai.metrics_logger import allow_repeated_logging
from minionsai.multiprocessing_rl.multiproc_rollouts import MultiProcessRolloutSource
from minionsai.multiprocessing_rl.rollout_runner import RolloutRunner
from minionsai.multiprocessing_rl.rollouts import InProcessRolloutSource
from minionsai.multiprocessing_rl.rollouts_data import RolloutBatch
import numpy as np

def agent_fn():
    generator = AgentGenerator(RandomAIAgent())
    disc_model = MinionsDiscriminator(depth=1, d_model=8)
    disc_model.to('cpu')
    discriminator = QDiscriminator(Translator(mode="discriminator"), disc_model, epsilon_greedy=0.5)
    agent = GenDiscAgent(discriminator, generator, rollouts_per_turn=2)
    return agent

def test_runner_same_as_multiproc():
    seed_everything(1112)
    allow_repeated_logging()
    agent = agent_fn()
    game_kwargs = {'max_turns': 4}

    state_dict = agent.discriminator.model.state_dict()
    agent.discriminator.model.load_state_dict(state_dict)
    agent.discriminator.model.eval()
    local_rollouts = InProcessRolloutSource(episodes_per_iteration=3, game_kwargs=game_kwargs, agent=agent)
    multiproc_rollouts = MultiProcessRolloutSource(agent_fn, agent, episodes_per_iteration=3, game_kwargs=game_kwargs, num_procs=2, device='cpu')


    for iter in range(4):
        data_local: RolloutBatch = local_rollouts.get_rollouts(iter)["discriminator"]
        data_multiproc: RolloutBatch = multiproc_rollouts.get_rollouts(iter)["discriminator"]
        
        assert data_local.num_games == data_multiproc.num_games
        for key in data_local.obs:
            np.testing.assert_allclose(data_local.obs[key], data_multiproc.obs[key]), key
        np.testing.assert_allclose(data_local.next_maxq, data_multiproc.next_maxq)
