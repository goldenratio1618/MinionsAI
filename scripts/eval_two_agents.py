import tqdm
from minionsai.run_game import run_game
from minionsai.engine import Game
from minionsai.discriminator_only.agent import TrainedAgent

# Update these to the agents you want to play
agent0_path = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/test/6"
agent1_path = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/test/0"

agents = [TrainedAgent.deserialize(agent0_path), TrainedAgent.deserialize(agent1_path)]

total_games = 100

slow_mode = total_games==1
if total_games == 1:
    # Log win probs from agent 0
    agents[0].verbose_level = 1

wins = [0, 0]
games = 0
for i in tqdm.tqdm(range(total_games)):
    player0 = i % 2
    agents_shuffled = [agents[player0], agents[1 - player0]]
    game = Game(money=(1, 1))
    winner = run_game(game, agents=agents_shuffled, verbose=total_games==1)
    if winner == 0:
        wins[player0] += 1
    else:
        wins[1 - player0] += 1
    games += 1

print("Total games:", games)
print("Agent 0 wins:", wins[0] / games)
print("Agent 1 wins:", wins[1] / games)
