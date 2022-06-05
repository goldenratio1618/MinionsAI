"""
Run with a name for the folder to keep agents in:

> python3 train.py --name my_run

Then agent checkpoints & logs are saved in <your temp dir>/MinionsAI/my_run
"""

import argparse
from collections import defaultdict
from tkinter import Y
from minionsai.run_game import run_game
from minionsai.discriminator_only.agent import TrainedAgent
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.engine import Game
from minionsai.agent import RandomAIAgent
import torch as th
import numpy as np
import os
import tqdm
import shutil
import random
import tempfile
import logging
from minionsai.metrics_logger import metrics_logger

logger = logging.getLogger(__name__)

# How many rollouts do we run of each turn before picking the best
ROLLOUTS_PER_TURN = 64

# How many episodes of data do we collect each iteration, before running a few epochs of optimization?
# Potentially good to use a few times bigger EPISODES_PER_ITERATION than BATCH_SIZE, to minimize correlation within batches
EPISODES_PER_ITERATION = 128  

# Once we've collected the data, how many times do we go over it for optimization (within one iteration)?
SAMPLE_REUSE = 3

# Frequency of running evals vs random agent
EVAL_EVERY = 4

# Frequency of storing a saved agent
CHECKPOINT_EVERY = 4

# During evals, run this many times extra rollouts compared to during rollout generation
EVAL_COMPUTE_BOOST = 1

# Model Size
DEPTH = 2
D_MODEL = 64 * DEPTH

# Optimizer hparams
BATCH_SIZE = 32
LR = 3e-5

# kwargs to create a game (passed to Game)
game_kwargs = {}
eval_game_kwargs = {}

def find_device():
    logger.info("=========================")
    # set device to cpu or cuda
    if(th.cuda.is_available()): 
        device = th.device('cuda:0') 
        th.cuda.empty_cache()
        logger.info("Device set to : " + str(th.cuda.get_device_name(device)))
    else:
        device = th.device('cpu')
        logger.info("Device set to : cpu")
    logger.info("=========================")
    return device

def setup_directory(run_name):
    """
    Set up logging and checkpoint directories for a run.
    Returns the subdirectory for checkpoints.
    """
    tempdir = tempfile.gettempdir()
    run_directory = os.path.join(tempdir, 'MinionsAI', run_name)
    checkpoint_dir = os.path.join(run_directory, 'checkpoints')
    # If the directory already exists, warn the user and check if it's ok to overwrite it.
    if os.path.exists(run_directory):
        print(f"Run directory already exists at {run_directory}")
        ok = input("OK to overwrite? (y/n) ")
        if ok != "y":
            exit()
        shutil.rmtree(run_directory)
    os.makedirs(checkpoint_dir)

    logging.basicConfig(filename=os.path.join(run_directory, 'logs.txt'), level=logging.DEBUG, 
                        format='[%(levelname)s %(asctime)s] %(name)s: %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    metrics_logger.configure(os.path.join(run_directory, 'metrics.csv'))
    return checkpoint_dir

def build_agent():
    generator = RandomAIAgent()

    logger.info("Creating policy...")
    policy = MinionsDiscriminator(d_model=D_MODEL, depth=DEPTH)
    logger.info("Policy initialized:")
    logger.info(policy)
    logger.info(f"Policy total parameter count: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")

    translator = Translator()
    agent = TrainedAgent(policy, translator, generator, ROLLOUTS_PER_TURN)
    return agent

# TODO - use run_game instead, with a custom Agent subclass that remembers the states.
def single_rollout(game_kwargs, agents):
    # Randomize starting money
    game_kwargs["money"] = (random.randint(1, 4), random.randint(1, 4))

    game = Game(**game_kwargs)
    state_buffers = [[], []]  # one for each player
    while True:
        game.next_turn()
        if game.done:
            break
        active_player = game.active_player_color
        actionlist = agents[active_player].act(game)
        game.full_turn(actionlist)
        state_buffers[active_player].append(agents[active_player].translator.translate(game))
    winner = game.winner
    # game.pretty_print()
    # print(winner)
    winner_states = state_buffers[winner]
    winner_labels = np.ones(len(winner_states))
    loser_states = state_buffers[1 - winner]
    loser_labels = np.zeros(len(loser_states))
    all_states = winner_states + loser_states
    all_labels = np.concatenate([winner_labels, loser_labels])
    return all_states, all_labels, winner

def rollouts(game_kwargs, agents):
    states = []
    labels = []

    games = 0
    first_player_wins = 0
    for _ in tqdm.tqdm(range(EPISODES_PER_ITERATION)):
        states_, labels_, winning_color = single_rollout(game_kwargs, agents)
        states.extend(states_)
        labels.extend(labels_)
        games += 1
        if winning_color == 0:
            first_player_wins += 1
    # Log instantaneous metrics here, and send cumulative out to the main control flow to integrate
    metrics_logger.log_metrics({'first_player_winrate': first_player_wins / games})
    return states, labels, {'rollout_games': games, 'rollout_states': len(states)}

def eval_vs_random(agent):
    wins = 0
    games = 0
    for i in tqdm.tqdm(range(100)):
        random_agent = RandomAIAgent()
        good_agent = TrainedAgent(agent.policy, agent.translator, agent.generator, ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST)
        good_idx = i % 2

        agents = [None, None]
        agents[good_idx] = good_agent
        agents[1 - good_idx] = random_agent

        game = Game(**eval_game_kwargs)
        winner = run_game(game, agents=agents)
        if winner == good_idx:
            wins += 1
        games += 1
    return wins / games

def main(run_name):
    checkpoint_dir = setup_directory(run_name)
    logger.info(f"Starting run {run_name}")

    device = find_device()

    agent = build_agent()
    policy = agent.policy
    policy.to(device)
    optimizer = th.optim.Adam(policy.parameters(), lr=LR)

    iteration = 0
    turns_optimized = 0
    rollout_stats = defaultdict(int)
    while True:
        metrics_logger.log_metrics({'iteration': iteration})
        print()
        print("====================================")
        logger.info(f"=========== Iteration: {iteration} ===========")
        print("====================================")
        if iteration % CHECKPOINT_EVERY == 0:
            logger.info("Saving checkpoint...")
            # Save with more rollouts_per_turn. TODO - clean up this hack.
            agent.rollouts_per_turn = ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST
            agent.save(os.path.join(checkpoint_dir, f"iter_{iteration}"))
            agent.rollouts_per_turn = ROLLOUTS_PER_TURN

        logger.info("Starting rollouts...")
        states, labels, rollout_info = rollouts(game_kwargs, [agent, agent])
        for k, v in rollout_info.items():
            rollout_stats[k] += v
        metrics_logger.log_metrics(rollout_stats)
        logger.info("Starting training...")
        for epoch in range(SAMPLE_REUSE):
            logger.info(f"  Epoch {epoch}/{SAMPLE_REUSE}...")
            all_idxes = np.random.permutation(len(states))
            n_batches = len(all_idxes) // BATCH_SIZE
            final_loss = None
            for idx in range(n_batches):
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
                turns_optimized += len(batch_idxes)
        logger.info(f"Iteration {iteration} complete.")
        param_norm = sum([th.norm(param, p=2) for param in policy.parameters()]).item()
        metrics_logger.log_metrics({"loss": final_loss, 'turns_optimized': turns_optimized, 'param_norm': param_norm})
        metrics_logger.flush()

        iteration += 1

        if iteration % EVAL_EVERY == 0:
            logger.info("Evaluating...")
            eval_winrate = eval_vs_random(agent)
            metrics_logger.log_metrics({"eval_winrate": eval_winrate})
            logger.info(f"Win rate vs random = {eval_winrate}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    args = parser.parse_args()

    main(run_name=args.name)