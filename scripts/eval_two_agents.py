from minionsai.experiment_tooling import find_device, get_experiments_directory
from minionsai.game_util import seed_everything
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import ScriptedDiscriminator
from minionsai.gen_disc.generator import AgentGenerator
from minionsai.scoreboard_envs import ENVS
import os
from minionsai.run_game import run_n_games
from minionsai.agent import Agent, HumanCLIAgent, RandomAIAgent
from minionsai.agent_saveload import load
import numpy as np

NUM_THREADS = 1
TOTAL_GAMES = 100

if __name__ == "__main__":
    # Update these to the agents you want to play
    agent0 = os.path.join(get_experiments_directory(), "gen_conveps396_5", "checkpoints", "iter_232")
    agent1 = os.path.join(get_experiments_directory(), "conv_big", "dfarhi_0613_conveps_256rolls_iter400_adapt")

    agents = [agent0, agent1]
    verbose = True

    slow_mode = TOTAL_GAMES==1
    if TOTAL_GAMES == 1:
        seed = np.random.randint(0, 2**10)
        print(f"Seeding with {seed}")
        seed_everything(seed)
        agents = [load(agent, test_load_equivalence=False) if isinstance(agent, str) else agent for agent in agents]

        agents[0] = GenDiscAgent(agents[0].discriminator, [
            (agents[0].generator, 2),
            (AgentGenerator(RandomAIAgent()), 8),
        ])

        # If we're only playing one game be more verbose
        agents[0].verbose_level = 2
        agents[1].verbose_level = 0

    agents = [load(agent, test_load_equivalence=False) if isinstance(agent, str) else agent for agent in agents]
    agents[0].discriminator.model.eval()
    agents[0].discriminator.epsilon_greedy = 0
    agents[0].generator.epsilon_greedy = 0
    agents[0].generator.sampling_temperature = 0.02
    agents[0].rollouts_per_turn = 128
    agents[1].rollouts_per_turn = 128
    # agents[0] = GenDiscAgent(agents[0].discriminator, [
    #     (agents[0].generator, 64),
    #     (AgentGenerator(RandomAIAgent()), 64),
    # ])
    # agents[1].rollouts_per_turn = 64
    wins, metrics = run_n_games(
        ENVS['zombies5x5'], agents, TOTAL_GAMES, num_threads=NUM_THREADS,
        randomize_player_order=True, progress_bar=verbose)

    print("Agent 0 wins:", wins[0] / TOTAL_GAMES)
    if (verbose):
        print("Agent 1 wins:", wins[1] / TOTAL_GAMES)
        print("Total games:", TOTAL_GAMES)
        print("=========================")
        print("Agent 0 metrics:")
        for metric, value in metrics[0].items():
            print(f"{metric}: {value}")
        print("=========================")
        print("Agent 1 metrics:")
        for metric, value in metrics[1].items():
            print(f"{metric}: {value}")
        print("=========================")
