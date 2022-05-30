import tqdm
from discriminator_only.run_game import run_game
from discriminator_only.agent import TrainedAgent

agent0_path = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/test/2"
agent1_path = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/test/2"

agents = [TrainedAgent.load(agent0_path), TrainedAgent.load(agent1_path)]

wins = [0, 0]
games = 0
for i in tqdm.tqdm(range(100)):
    player0 = i % 2
    agents_shuffled = [agents[player0], agents[1 - player0]]

    winner = run_game(game_kwargs={}, agents=agents_shuffled)
    if winner == 0:
        wins[player0] += 1
    else:
        wins[1 - player0] += 1
    games += 1

print("Total games:", games)
print("Agent 0 wins:", wins[0] / games)
print("Agent 1 wins:", wins[1] / games)
