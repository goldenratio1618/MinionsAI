from minionsai.experiment_tooling import get_experiments_directory
from minionsai.scoreboard_envs import ENVS
import os
from minionsai.run_game import run_n_games
from minionsai.agent import Agent, RandomAIAgent

NUM_THREADS = 1
TOTAL_GAMES = 1

if __name__ == "__main__":
    # Update these to the agents you want to play
    agent0 = RandomAIAgent()
    agent1 = os.path.join(get_experiments_directory(), "conv_big", "checkpoints", "iter_600")

    # while not os.path.exists(agent0_path):
    #     print("Waiting for agent 1 to finish training...")
    #     time.sleep(60)

    # while not os.path.exists(agent1_path):
    #     print("Waiting for agent 1 to finish training...")
    #     time.sleep(60)

    # agent1 = RandomAIAgent()
    agents = [agent0, agent1]
    #agents = [CLIAgent(["./stonkfish/a.out"]), CLIAgent(["./stonkfish/a.out.old"])]

    verbose = True

    slow_mode = TOTAL_GAMES==1
    if TOTAL_GAMES == 1:
        agents = [Agent.load(agent) if isinstance(agent, str) else agent for agent in agents]

        # If we're only playing one game be more verbose
        agents[0].verbose_level = 2
        agents[1].verbose_level = 2

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
