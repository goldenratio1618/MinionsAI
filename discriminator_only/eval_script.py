import tqdm
from run_game import run_game
from engine import Game
from discriminator_only.agent import TrainedAgent

agent0_path = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/test/0"
agent1_path = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/test/2"

agents = [TrainedAgent.load(agent0_path), TrainedAgent.load(agent1_path)]

total_games = 1
wins = [0, 0]
games = 0
for i in tqdm.tqdm(range(total_games)):
    player0 = i % 2
    agents_shuffled = [agents[player0], agents[1 - player0]]
    game = Game(money=(1, 1))
    winner = run_game(game, agents=agents_shuffled, verbose=True)
    if winner == 0:
        wins[player0] += 1
    else:
        wins[1 - player0] += 1
    games += 1

print("Total games:", games)
print("Agent 0 wins:", wins[0] / games)
print("Agent 1 wins:", wins[1] / games)
