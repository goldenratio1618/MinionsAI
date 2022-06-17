from minionsai.experiment_tooling import get_experiments_directory
from minionsai.scoreboard_envs import ENVS
import time
import os
from minionsai.run_game import accumulate_metrics, run_game_with_metrics, run_n_games
from minionsai.agent import Agent, CLIAgent, RandomAIAgent
import torch.multiprocessing as mp
import tqdm
# If this is 1, it runs in a single thread.
# Increase it above 1 for multiple threads.
# If you make it too high, you might consume all the RAM on your machine.
NUM_THREADS = 1
TOTAL_GAMES = 100


class GameRunner():
    def __init__(self, game_fn, agents, num_games, output_queue):
        self.game_fn = game_fn
        self.agents = [Agent.load(agent) if isinstance(agent, str) else agent for agent in agents]
        self.num_games = num_games
        self.output_queue = output_queue

    def run(self):
        for _ in range(self.num_games):
            winner, metrics = run_game_with_metrics(self.game_fn(), self.agents, verbose=False, randomize_player_order=True)
            self.output_queue.put((winner, metrics))

def game_runner_thread(game_fn, agents, num_games, output_queue):
    runner = GameRunner(game_fn, agents, num_games, output_queue)
    runner.run()

if __name__ == "__main__":
    # Update these to the agents you want to play
    agent0 = os.path.join(get_experiments_directory(), "epi1024", "checkpoints", "iter_150")
    agent1 = RandomAIAgent()
    agents = [agent0, agent1]

    assert TOTAL_GAMES % NUM_THREADS == 0
    games_per_thread = TOTAL_GAMES // NUM_THREADS
    output_queue = mp.Queue()
    threads = []
    for i in range(NUM_THREADS):
        thread = mp.Process(target=game_runner_thread, args=(ENVS['zombies5x5'], agents, games_per_thread, output_queue))
        thread.start()
        threads.append(thread)

    wins = [0, 0]
    all_metrics = [[], []]
    for _ in tqdm.tqdm(range(TOTAL_GAMES)):
        winner, metrics = output_queue.get()
        wins[winner] += 1
        all_metrics[0].append(metrics[0])
        all_metrics[1].append(metrics[1])
    
    for thread in threads:
        thread.join()

    metrics0 = accumulate_metrics(all_metrics[0], TOTAL_GAMES)
    metrics1 = accumulate_metrics(all_metrics[1], TOTAL_GAMES)
    print("Agent 0 wins:", wins[0] / TOTAL_GAMES)
    print("Agent 1 wins:", wins[1] / TOTAL_GAMES)
    print("Total games:", TOTAL_GAMES)
    print("=========================")
    print("Agent 0 metrics:")
    for metric, value in metrics0.items():
        print(f"{metric}: {value}")
    print("=========================")
    print("Agent 1 metrics:")
    for metric, value in metrics1.items():
        print(f"{metric}: {value}")
    print("=========================")
