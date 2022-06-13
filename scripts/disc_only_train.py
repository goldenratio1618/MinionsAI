"""
Run with a name for the folder to keep agents in:

> python3 train.py --name my_run

Then agent checkpoints & logs are saved in <your temp dir>/MinionsAI/my_run
"""

import argparse
from collections import defaultdict
import tempfile
from minionsai.experiment_tooling import find_device, setup_directory
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import ScriptedDiscriminator
from minionsai.run_game import run_game, run_n_games
from minionsai.discriminator_only.agent import TrainedAgent
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.engine import Game
from minionsai.agent import Agent, RandomAIAgent
from minionsai.scoreboard_envs import ENVS
import torch as th
import numpy as np
import os
import tqdm
import random
import logging
from minionsai.metrics_logger import metrics_logger

logger = logging.getLogger(__name__)

# How many rollouts do we run of each turn before picking the best
ROLLOUTS_PER_TURN = 64
EPSILON_GREEEDY = 0.1

# How many episodes of data do we collect each iteration, before running a few epochs of optimization?
# Potentially good to use a few times bigger EPISODES_PER_ITERATION * DATA_AUG_FACTOR than BATCH_SIZE, to minimize correlation within batches
# (DATA_AUG_FACTOR = 4)
EPISODES_PER_ITERATION = 256

# Once we've collected the data, how many times do we go over it for optimization (within one iteration)?
SAMPLE_REUSE = 2

# Frequency of running evals
EVAL_EVERY = 2
# Put iteration numbers here to eval vs past versions of this train run.
EVAL_VS_PAST_ITERS = []
# Specific agent instances to eval vs
EVAL_VS_AGENTS = [GenDiscAgent(ScriptedDiscriminator(), RandomAIAgent(), rollouts_per_turn=16)]
# Eval against random up until this iteration
EVAL_VS_RANDOM_UNTIL = 5
EVAL_TRIALS = 100
EVAL_THREADS = 1

# Frequency of storing a saved agent
CHECKPOINT_EVERY = 1

# During evals, run this many times extra rollouts compared to during rollout generation
EVAL_COMPUTE_BOOST = 4

# Model Size
DEPTH = 2
D_MODEL = 64 * DEPTH

# Optimizer hparams
BATCH_SIZE = EPISODES_PER_ITERATION
LR = 3e-5

LAMBDA = 0.95

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
    agent = TrainedAgent(policy, translator, generator, ROLLOUTS_PER_TURN, epsilon_greedy=EPSILON_GREEEDY)
    return agent

def smooth_labels(labels, lam):
    """
    Takes a list of win probs with a 0/1 at the end
    and smooths them into targets for the model
    using exponential moving average
    """
    reversed_labels = np.array(labels)[::-1]
    lambda_powers = lam ** np.arange(len(labels), 0, -1)
    turn_contribs = np.cumsum(reversed_labels * lambda_powers)
    norm = np.cumsum(lambda_powers)
    return (turn_contribs / norm)[::-1]

# TODO put this in a separate file
array = [1, 2]
desired = np.array([2/3 * 1 + 1/3 * 2, 2.])
np.testing.assert_allclose(smooth_labels(array, 0.5), desired)

array = [1, 2, 3]
np.testing.assert_allclose(smooth_labels(array, 0.5), [4/7 * 1 + 2/7 * 2 + 1/7 * 3, 2/3 * 2 + 1/3 * 3, 3])

# TODO - use run_game instead, with a custom Agent subclass that remembers the states.
def single_rollout(game_kwargs, agents, lam=None):
    # Randomize starting money
    game_kwargs["money"] = (random.randint(1, 4), random.randint(1, 4))

    game = Game(**game_kwargs)
    state_buffers = [[], []]  # one for each player
    label_buffers = [[], []]  # one for each player
    while True:
        game.next_turn()
        if game.done:
            break
        active_player = game.active_player_color
        actionlist, best_winprob = agents[active_player].act_with_winprob(game)
        game.full_turn(actionlist)
        state_buffers[active_player].append(agents[active_player].translator.translate(game))
        label_buffers[active_player].append(best_winprob)
        
    winner = game.winner

    metrics = (game.get_metrics(0), game.get_metrics(1))
    metrics[winner]["pfinal"] = label_buffers[winner][-1]
    metrics[1 - winner]["pfinal"] = 1 - label_buffers[1 - winner][-1]

    label_buffers[winner].append(1)
    label_buffers[1 - winner].append(0)
    # game.pretty_print()
    # print(winner)
    winner_states = state_buffers[winner]
    winner_labels = label_buffers[winner][1:]
    loser_states = state_buffers[1 - winner]
    loser_labels = label_buffers[1 - winner][1:]
    if lam is not None:
        winner_labels = smooth_labels(winner_labels, lam)
        loser_labels = smooth_labels(loser_labels, lam)
    all_states = winner_states + loser_states
    all_labels = np.concatenate([winner_labels, loser_labels])
    return all_states, all_labels, metrics

def rollouts(game_kwargs, agents, lam=None):
    states = []
    labels = []

    games = 0
    metrics_accumulated = (defaultdict(list), defaultdict(list))
    for _ in tqdm.tqdm(range(EPISODES_PER_ITERATION)):
        with metrics_logger.timing('single_episode'):
           states_, labels_, this_game_metrics = single_rollout(game_kwargs, agents, lam=lam)
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
    good_agent = TrainedAgent(agent.policy, agent.translator, agent.generator, ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST)
    wins, _metrics = run_n_games(ENVS[EVAL_ENV_NAME], [good_agent, eval_agent], n=EVAL_TRIALS, num_threads=EVAL_THREADS)
    winrate = wins[0] / EVAL_TRIALS
    metrics_logger.log_metrics({f"eval_winrate/{name}": winrate})
    logger.info(f"Win rate vs {name} = {winrate}")  

def main(run_name):
    checkpoint_dir, code_dir = setup_directory(run_name)
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
                agent.save(os.path.join(checkpoint_dir, f"iter_{iteration}"), copy_code_from=code_dir)
                agent.rollouts_per_turn = ROLLOUTS_PER_TURN
        with metrics_logger.timing('rollouts'):
            logger.info("Starting rollouts...")
            policy.eval()  # Set policy to non-training mode
            
            # Early in training use td-lambda to reduce variance of gradients
            # Late in training, turn this off since it introduces a bias when epsilon-greedy actions are taken.
            lam = None if iteration >= 20 else 1 - iteration / 20
            metrics_logger.log_metrics({'lambda': lam})
            states, labels, rollout_info = rollouts(game_kwargs, [agent, agent], lam=lam)
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
                for eval_agent in EVAL_VS_AGENTS:
                    if isinstance(eval_agent, str):
                        eval_vs_other_by_path(agent, eval_agent)
                    elif isinstance(eval_agent, Agent):                  
                        eval_vs_other(agent, eval_agent, name=eval_agent.__class__.__name__)
                for iter in EVAL_VS_PAST_ITERS:
                    eval_vs_other_by_path(agent, os.path.join(checkpoint_dir, f"iter_{iter}"))

                policy.train()  # Set policy back to training mode



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    args = parser.parse_args()

    main(run_name=args.name)