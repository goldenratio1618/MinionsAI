from collections import defaultdict
from minionsai.scoreboard_envs import ENVS
import tqdm
from minionsai.run_game import run_game
from minionsai.engine import Game
from minionsai.agent import Agent, CLIAgent

# Update these to the agents you want to play
agent0_path = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/shuffle_spawn/checkpoints/iter_30"
agent1_path = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/shuffle_spawn/checkpoints/iter_16"

agents = [Agent.load(agent0_path), Agent.load(agent1_path)]
#agents = [CLIAgent(["./stonkfish/a.out"]), CLIAgent(["./stonkfish/a.out.old"])]

total_games = 100
verbose = True

slow_mode = total_games==1
if total_games == 1:
    # Log win probs from agent 0
    agents[0].verbose_level = 2
    agents[1].verbose_level = 2

wins = [0, 0]
games = 0
metrics_accumulated = (defaultdict(list), defaultdict(list))
iterator = range(total_games)
if (verbose): iterator = tqdm.tqdm(iterator)
for i in iterator:
    player0 = i % 2
    agents_shuffled = [agents[player0], agents[1 - player0]]
    game = ENVS['zombies5x5']()
    winner = run_game(game, agents=agents_shuffled, verbose=total_games==1)
    if winner == 0:
        wins[player0] += 1
    else:
        wins[1 - player0] += 1
    for color, metrics in enumerate(metrics_accumulated):
        this_game_metrics = game.get_metrics(color)
        for key in set(metrics.keys()).union(set(this_game_metrics.keys())):
            metrics[key].append(this_game_metrics[key])
    games += 1

print("Agent 0 wins:", wins[0] / games)
if (verbose):
    print("Agent 1 wins:", wins[1] / games)
    print("Total games:", games)
    print("=========================")
    print("Agent 0 metrics:")
    for metric, values in metrics_accumulated[0].items():
        print(f"{metric}: {sum(values) / len(values)}")
    print("=========================")
    print("Agent 1 metrics:")
    for metric, values in metrics_accumulated[1].items():
        print(f"{metric}: {sum(values) / len(values)}")
    print("=========================")
