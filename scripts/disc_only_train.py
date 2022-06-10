"""
Run with a name for the folder to keep agents in:

> python3 train.py --name my_run

Then agent checkpoints & logs are saved in <your temp dir>/MinionsAI/my_run
"""

import argparse
from collections import defaultdict
from tkinter import Y
from minionsai.experiment_tooling import find_device, setup_directory
from minionsai.run_game import run_game
from minionsai.discriminator_only.agent import TrainedAgent
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.engine import Game
from minionsai.agent import RandomAIAgent
from minionsai.scoreboard_envs import ENVS
import torch as th
import numpy as np
import os
import tqdm
import random
import logging
import tempfile
from minionsai.metrics_logger import metrics_logger

logger = logging.getLogger(__name__)

# How many rollouts do we run of each turn before picking the best
ROLLOUTS_PER_TURN = 64

# How many episodes of data do we collect each iteration, before running a few epochs of optimization?
# Potentially good to use a few times bigger EPISODES_PER_ITERATION * DATA_AUG_FACTOR than BATCH_SIZE, to minimize correlation within batches
# (DATA_AUG_FACTOR = 4)
EPISODES_PER_ITERATION = 256

# Once we've collected the data, how many times do we go over it for optimization (within one iteration)?
SAMPLE_REUSE = 2

# Frequency of running evals vs random agent
EVAL_EVERY = 2
EVAL_VS_PAST_ITERS = [2, 8, 16]
EVAL_VS_AGENTS = [os.path.join(tempfile.gettempdir(), "MinionsAI", "shuffle_spawn", "checkpoints", "iter_8")]
EVAL_VS_RANDOM_UNTIL = 5
EVAL_TRIALS = 100

# Frequency of storing a saved agent
CHECKPOINT_EVERY = 1

# During evals, run this many times extra rollouts compared to during rollout generation
EVAL_COMPUTE_BOOST = 4

# Model Size
DEPTH = 2
D_MODEL = 64 * DEPTH

# Optimizer hparams
BATCH_SIZE = 256
LR = 3e-5

# kwargs to create a game (passed to Game)
game_kwargs = {'symmetrize': False}
# Eval env registered in scoreboard_envs.py
EVAL_ENV_NAME = 'zombies5x5'

MAX_ITERATIONS = None

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
    return all_states, all_labels, (game.get_metrics(0), game.get_metrics(1))

def rollouts(game_kwargs, agents):
    states = []
    labels = []

    games = 0
    metrics_accumulated = (defaultdict(list), defaultdict(list))
    for _ in tqdm.tqdm(range(EPISODES_PER_ITERATION)):
        with metrics_logger.timing('single_episode'):
           states_, labels_, this_game_metrics = single_rollout(game_kwargs, agents)
        states.extend(states_)
        labels.extend(labels_)
        games += 1
        for color, this_color_metrics in enumerate(this_game_metrics):
            for key in set(metrics_accumulated[color].keys()).union(set(this_color_metrics.keys())):
                metrics_accumulated[color][key].append(this_color_metrics[key])
    rollout_states = len(states)
    # convert from list of dicts of arrays to a single dict of arrays with large batch dimension
    states = {k: np.concatenate([s[k] for s in states], axis=0) for k in states[0]}
    
    # Add symmetries
    symmetrized_states = Translator.symmetries(states)

    # Now combine them into one big states dict
    states = {k: np.concatenate([s[k] for s in symmetrized_states], axis=0) for k in states}
    labels = np.concatenate([labels]*len(symmetrized_states), axis=0)
    # Log instantaneous metrics here, and send cumulative out to the main control flow to integrate
    for color in (0, 1):
        metrics_logger.log_metrics({k: sum(v)/EPISODES_PER_ITERATION for k, v in metrics_accumulated[color].items()}, prefix=f'rollouts/game/{color}')
    return states, labels, {'rollout_games': games, 'rollout_states': rollout_states}

def eval_vs_other_by_path(agent, eval_agent_path):
    logger.info(f"Looking for eval agent at {eval_agent_path}...")
    if os.path.exists(eval_agent_path):
        agent_name = os.path.basename(eval_agent_path)
        eval_agent = TrainedAgent.load(eval_agent_path)
        eval_vs_other(agent, eval_agent, agent_name)

def eval_vs_other(agent, eval_agent, name):
    wins = 0
    games = 0
    for i in tqdm.tqdm(range(EVAL_TRIALS)):
        good_agent = TrainedAgent(agent.policy, agent.translator, agent.generator, ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST)
        good_idx = i % 2

        agents = [None, None]
        agents[good_idx] = good_agent
        agents[1 - good_idx] = eval_agent

        game = ENVS[EVAL_ENV_NAME]()
        winner = run_game(game, agents=agents)
        if winner == good_idx:
            wins += 1
        games += 1
    winrate = wins / games
    metrics_logger.log_metrics({f"eval_winrate/{name}": winrate})
    logger.info(f"Win rate vs {name} = {winrate}")  

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
    while MAX_ITERATIONS is None or iteration < MAX_ITERATIONS:
        metrics_logger.log_metrics({'iteration': iteration})
        print()
        print("====================================")
        logger.info(f"=========== Iteration: {iteration} ===========")
        print("====================================")
        if iteration % CHECKPOINT_EVERY == 0:
            with metrics_logger.timing('checkpointing'):
                logger.info("Saving checkpoint...")
                # Save with more rollouts_per_turn. TODO - clean up this hack.
                agent.rollouts_per_turn = ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST
                agent.save(os.path.join(checkpoint_dir, f"iter_{iteration}"))
                agent.rollouts_per_turn = ROLLOUTS_PER_TURN
        with metrics_logger.timing('rollouts'):
            logger.info("Starting rollouts...")
            policy.eval()  # Set policy to non-training mode
            states, labels, rollout_info = rollouts(game_kwargs, [agent, agent])
            policy.train()  # Set policy back to training mode
            for k, v in rollout_info.items():
                rollout_stats[k] += v
            metrics_logger.log_metrics(rollout_stats)
        with metrics_logger.timing('training'):
            logger.info("Starting training...")
            num_states = states['board'].shape[0]
            for epoch in range(SAMPLE_REUSE):
                logger.info(f"  Epoch {epoch}/{SAMPLE_REUSE}...")
                all_idxes = np.random.permutation(num_states)
                n_batches = num_states // BATCH_SIZE
                for idx in range(n_batches):
                    with metrics_logger.timing('training_batch'):
                        batch_idxes = all_idxes[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]
                        batch_obs = {}
                        for key in states:
                            batch_obs[key] = states[key][batch_idxes]
                        batch_labels = np.array(labels)[batch_idxes]
                        batch_labels = th.from_numpy(batch_labels).to(device)
                        optimizer.zero_grad()
                        disc_logprob = policy(batch_obs) # [batch, 1]
                        batch_labels = th.unsqueeze(batch_labels, 1)
                        loss = th.nn.BCEWithLogitsLoss()(disc_logprob, batch_labels)
                        loss.backward()
                        optimizer.step()
                        if idx in [0, n_batches // 2, n_batches - 1]:
                            max_batch_digits = len(str(n_batches))
                            metrics_logger.log_metrics({f"loss/epoch_{epoch}/batch_{idx:0>{max_batch_digits}}": loss.item()})
                        turns_optimized += len(batch_idxes)
            logger.info(f"Iteration {iteration} complete.")
            param_norm = sum([th.norm(param, p=2) for param in policy.parameters()]).item()
            metrics_logger.log_metrics({'turns_optimized': turns_optimized, 'param_norm': param_norm})
        metrics_logger.flush()

        iteration += 1

        if iteration % EVAL_EVERY == 0:
            with metrics_logger.timing('eval'):
                logger.info("Evaluating...")
                policy.eval()  # Set policy to non-training mode
                if iteration < EVAL_VS_RANDOM_UNTIL:
                    eval_agent = RandomAIAgent()
                    eval_vs_other(agent, eval_agent, 'random')
                for eval_agent_path in EVAL_VS_AGENTS:
                    eval_vs_other_by_path(agent, eval_agent_path)                  
                for iter in EVAL_VS_PAST_ITERS:
                    eval_vs_other_by_path(agent, os.path.join(checkpoint_dir, f"iter_{iter}"))

                policy.train()  # Set policy back to training mode



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    args = parser.parse_args()

    main(run_name=args.name)