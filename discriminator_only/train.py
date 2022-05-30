"""
Work in progress, but as of this commit I get to 95% winrate vs random bot in 5x5 world with 25% randomly graveyards.

Reproduce by simply running `train.py`. After setting BOARD_SIZE and graveyard_locs in engine.py
"""


from tabnanny import check
from discriminator_only.agent import TrainedAgent
from discriminator_only.model import MinionsDiscriminator
from discriminator_only.random_generator import RandomGenerator
from discriminator_only.translator import Translator
from engine import Game
import torch as th
import numpy as np
import os
import tqdm
import shutil

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = th.device('cpu')
if(th.cuda.is_available()): 
    device = th.device('cuda:0') 
    th.cuda.empty_cache()
    print("Device set to : " + str(th.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

ROLLOUTS_PER_TURN = 4
EPISODES_PER_ITERATION = 128
SAMPLE_REUSE = 3
BATCH_SIZE = 32
EVAL_EVERY = 2
CHECKPOINT_EVERY = 2
EVAL_COMPUTE_BOOST = 1

run_name = 'test'
# TODO: make this location more reasonable
checkpoint_dir = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/{run_name}"
# create the directory if it doesn't exist and clear its contents if it does iexist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
else:
    print(f"Duplicate run name {run_name}; clearing checkpoint directory f{checkpoint_dir}")
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)

generator = RandomGenerator()

print("Creating policy...")
policy = MinionsDiscriminator(d_model=128, device=device)
print("Policy intiialzied:")
print(policy)

translator = Translator()
agent = TrainedAgent(policy, translator, generator, ROLLOUTS_PER_TURN)

# TODO - use run_game instead, with a custom Agent subclass that remembers the states.
def single_rollout(game_kwargs, agents = (agent, agent)):
    game = Game(**game_kwargs)
    state_buffers = [[], []]  # one for each player
    while not game.done:
        agents[game.active_player_color].act(game)
        state_buffers[game.active_player_color].append(translator.translate(game))
        game.next_turn()

    winner = game.winner
    winner_states = state_buffers[winner]
    winner_labels = np.ones(len(winner_states))
    loser_states = state_buffers[1 - winner]
    loser_labels = np.zeros(len(loser_states))
    all_states = winner_states + loser_states
    all_labels = np.concatenate([winner_labels, loser_labels])
    return all_states, all_labels, winner

def rollouts(game_kwargs):
    states = []
    labels = []

    for _ in tqdm.tqdm(range(EPISODES_PER_ITERATION)):
        states_, labels_, _ = single_rollout(game_kwargs)
        states.extend(states_)
        labels.extend(labels_)
    return states, labels

optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)

def eval_vs_random():
    wins = 0
    games = 0
    for i in tqdm.tqdm(range(100)):
        null_policy = lambda x: th.Tensor([0]).to(device)
        random_agent = TrainedAgent(null_policy, translator, generator, 2)
        good_agent = TrainedAgent(policy, translator, generator, ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST)
        good_idx = i % 2

        agents = [None, None]
        agents[good_idx] = good_agent
        agents[1 - good_idx] = random_agent

        _, _, winner = single_rollout(game_kwargs={}, agents=agents)
        if winner == good_idx:
            wins += 1
        games += 1
    return wins / games

iteration = 0
while True:
    print("===================================")
    print(f"=========== Iteration: {iteration} ===========")
    print("===================================")
    if iteration % CHECKPOINT_EVERY == 0:
        print("Saving checkpoint...")
        agent.save(os.path.join(checkpoint_dir, f"{iteration}"))
    if iteration % EVAL_EVERY == 0:
        print("Evaluating...")
        eval_winrate = eval_vs_random()
        print(f"Win rate vs random = {eval_winrate}")

    print("Starting rollouts...")
    states, labels = rollouts({})
    print("Starting training...")
    for epoch in range(SAMPLE_REUSE):
        print(f"Epoch {epoch}")
        all_idxes = np.random.permutation(len(states))
        n_batches = len(all_idxes) // BATCH_SIZE
        final_loss = None
        for idx in tqdm.tqdm(range(n_batches)):
            batch_idxes = all_idxes[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]
            batch_obs = {}
            for key in states[0]:
                batch_obs[key] = np.concatenate([states[i][key] for i in batch_idxes], axis=0)
            batch_labels = np.array(labels)[batch_idxes]
            batch_labels = th.from_numpy(batch_labels).to(device)
            optimizer.zero_grad()
            disc_logprob = policy(batch_obs) # [batch, 1]
            batch_labels = th.unsqueeze(batch_labels, 1)
            loss = th.nn.BCEWithLogitsLoss()(disc_logprob, batch_labels)
            loss.backward()
            optimizer.step()
            if idx == n_batches - 1:
                final_loss = loss.item()
        print(f"Loss: {final_loss}")

    iteration += 1

