from collections import defaultdict
from minionsai.experiment_tooling import get_experiments_directory
from minionsai.scoreboard_envs import ENVS
import tqdm
import os
from minionsai.run_game import run_game_with_metrics, run_n_games
from minionsai.agent import Agent, CLIAgent, RandomAIAgent

# If this is 1, it runs in a single thread.
# Increase it above 1 for multiple threads.
# If you make it too high, you might consume all the RAM on your machine.
NUM_THREADS = 1

TOTAL_GAMES = 10

if __name__ == "__main__":
    # Update these to the agents you want to play
    agent0_path = os.path.join(get_experiments_directory(), "scan_batch_size_1024", "checkpoints", "iter_9")
    agent1_path = os.path.join(get_experiments_directory(), "scan_batch_size_256", "checkpoints", "iter_36")

    agents = [Agent.load(agent0_path), RandomAIAgent()]
    #agents = [CLIAgent(["./stonkfish/a.out"]), CLIAgent(["./stonkfish/a.out.old"])]

    verbose = True

    slow_mode = TOTAL_GAMES==1
    if TOTAL_GAMES == 1:
        # Log win probs from agent 0
        agents[0].verbose_level = 2
        agents[1].verbose_level = 2

    if TOTAL_GAMES > 1:
        wins, metrics = run_n_games(
            ENVS['zombies5x5'], agents, TOTAL_GAMES, 
            num_threads=NUM_THREADS, randomize_player_order=True, progress_bar=verbose)
    else:
        winner, metrics = run_game_with_metrics(ENVS['zombies5x5'](), agents, verbose=True, randomize_player_order=True)
        wins = [0, 0]
        wins[winner] += 1

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
