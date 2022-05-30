"""
Work in progress, but as of this commit I get to 95% winrate vs random bot in 5x5 world with 25% randomly graveyards.

Reproduce by simply running `train.py`. After setting BOARD_SIZE and graveyard_locs in engine.py
"""


from tabnanny import check
from discriminator_only.model import MinionsDiscriminator
from discriminator_only.random_generator import RandomGenerator
from discriminator_only.translator import translate
from engine import Game
import torch as th
import numpy as np
import os
import tqdm

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
EPISODES_PER_ITERATION = 16
SAMPLE_REUSE = 2
BATCH_SIZE = 32
EVAL_EVERY = 5
CHECKPOINT_EVERY = 2

run_name = 'test'
# TODO: make this location more reasonable
checkpoint_dir = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/{run_name}"
# create the directory if it doesn't exist and clear its contents if it does iexist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
else:
    print(f"Duplicate run name {run_name}; clearing checkpoint directory")
    for file in os.listdir(checkpoint_dir):
        os.remove(os.path.join(checkpoint_dir, file))

generator = RandomGenerator()

print("Creating policy...")
policy = MinionsDiscriminator(d_model=128).to(device)
print("Policy intiialzied:")
print(policy)

def single_rollout(game_kwargs, rollouts_per_turn_by_player=(ROLLOUTS_PER_TURN, ROLLOUTS_PER_TURN)):
    game = Game(**game_kwargs)
    state_buffers = [[], []]  # one for each player
    while not game.done:
        # print(f"  Game turns remaining: {game.remaining_turns}")
        options = []
        scores = []
        for i in range(rollouts_per_turn_by_player[game.active_player_color]):
            # print(f"    Turn rollout {i}")
            actions = generator.rollout(game)
            obs = translate(game)
            obs = {k: th.Tensor(v).int().to(device) for k, v in obs.items()}
            disc_logprob = policy(obs).detach().cpu().numpy()
            scores.append(disc_logprob)
            options.append(actions)
        best_option = options[np.argmax(scores)]
        generator.redo(best_option, game)
        game.next_turn()
        state_buffers[game.active_player_color].append(translate(game))
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
        good_idx = i % 2
        rollouts_per_turn_by_player = [1, 1]
        rollouts_per_turn_by_player[good_idx] = ROLLOUTS_PER_TURN * 4

        _, _, winner = single_rollout(game_kwargs={}, rollouts_per_turn_by_player=rollouts_per_turn_by_player)
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
        policy.save(os.path.join(checkpoint_dir, f"{iteration}.pt"))

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
                batch_obs[key] = th.from_numpy(batch_obs[key]).to(device)    
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

    if iteration % EVAL_EVERY == 0:
        print("Evaluating...")
        eval_winrate = eval_vs_random()
        print(f"Win rate vs random = {eval_winrate}")