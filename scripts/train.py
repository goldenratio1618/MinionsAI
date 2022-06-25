"""
Run with a name for the folder to keep agents in:

> python3 train.py --name my_run

Then agent checkpoints & logs are saved in <your experiments dir>/MinionsAI/my_run
"""

import argparse
from collections import defaultdict
import random
from minionsai.experiment_tooling import find_device, get_experiments_directory, setup_directory
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import QDiscriminator, ScriptedDiscriminator
from minionsai.gen_disc.generator import AgentGenerator, QGenerator
from minionsai.multiprocessing_rl.multiproc_rollouts import MultiProcessRolloutSource
from minionsai.multiprocessing_rl.rollouts import InProcessRolloutSource
from minionsai.run_game import run_n_games
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.agent import Agent, RandomAIAgent
from minionsai.scoreboard_envs import ENVS
import torch as th
import numpy as np
import os
import logging
from minionsai.metrics_logger import metrics_logger

logger = logging.getLogger(__name__)

TRAIN_GENERATOR = False

# How many rollouts do we run of each turn before picking the best
ROLLOUTS_PER_TURN = 64
DISC_EPSILON_GREEDY = 0.1
GEN_EPSILON_GREEDY = 0.1
GEN_SAMPLING_TEMPERATURE = 0.01

# How many episodes of data do we collect each iteration, before running a few epochs of optimization?
# Potentially good to use a few times bigger EPISODES_PER_ITERATION than BATCH_SIZE, to minimize correlation within batches
EPISODES_PER_ITERATION = 256
ROLLOUT_PROCS = 1

# Once we've collected the data, how many times do we go over it for optimization (within one iteration)?
SAMPLE_REUSE = 2

# Frequency of running evals
EVAL_EVERY = 8
# Put iteration numbers here to eval vs past versions of this train run.
EVAL_VS_PAST_ITERS = []
# Specific agent instances to eval vs
EVAL_VS_AGENTS = [
    # GenDiscAgent(ScriptedDiscriminator(), RandomAIAgent(), rollouts_per_turn=16),
    os.path.join(get_experiments_directory(), "conv_big", "checkpoints", "iter_200")
]
# Eval against random up until this iteration
EVAL_VS_RANDOM_UNTIL = 5
EVAL_TRIALS = 100

# Frequency of storing a saved agent
CHECKPOINT_EVERY = 4

# During evals, run this many times extra rollouts compared to during rollout generation
EVAL_COMPUTE_BOOST = 4

# Model Size
DEPTH = 2
D_MODEL = 64 * DEPTH
GEN_DEPTH = 1
GEN_D_MODEL = 64 * GEN_DEPTH

# Optimizer hparams
BATCH_SIZE = EPISODES_PER_ITERATION
DISC_LR = 1e-4
GEN_LR = 1e-4

# kwargs to create a game (passed to Game)
game_kwargs = {'symmetrize': False}
# Eval env registered in scoreboard_envs.py
EVAL_ENV_NAME = 'zombies5x5'

MAX_ITERATIONS = 400

SEED = 12345

def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    th.cuda.manual_seed_all(SEED)

def build_agent():
    logger.info("Creating generator...")
    if TRAIN_GENERATOR:
        gen_model = MinionsDiscriminator(d_model=GEN_D_MODEL, depth=GEN_DEPTH)  # TODO - make generator model instead
        logger.info("Generator model initialized:")
        logger.info(gen_model)
        gen_translator = Translator()  # TODO - make gen translator
        generator = QGenerator(model=gen_model, translator=gen_translator, sampling_temperature=GEN_SAMPLING_TEMPERATURE, epsilon_greedy=GEN_EPSILON_GREEDY)
    else:
        generator = AgentGenerator(RandomAIAgent())

    logger.info("Creating discriminator...")
    disc_model = MinionsDiscriminator(d_model=D_MODEL, depth=DEPTH)
    logger.info("Discriminator model initialized:")
    logger.info(disc_model)
    logger.info(f"Discriminator model total parameter count: {sum(p.numel() for p in disc_model.parameters() if p.requires_grad):,}")

    disc_translator = Translator()
    discriminator = QDiscriminator(translator=disc_translator, model=disc_model, epsilon_greedy=DISC_EPSILON_GREEDY)

    logger.info(f"Discriminator model total parameter count: {sum(p.numel() for p in disc_model.parameters() if p.requires_grad):,}")
    if TRAIN_GENERATOR:
            logger.info(f"Generator model total parameter count: {sum(p.numel() for p in gen_model.parameters() if p.requires_grad):,}")

    agent = GenDiscAgent(discriminator, generator, rollouts_per_turn=ROLLOUTS_PER_TURN)
    return agent

def eval_vs_other_by_path(agent, eval_agent_path):
    logger.info(f"Looking for eval agent at {eval_agent_path}...")
    if os.path.exists(eval_agent_path):
        agent_name = os.path.basename(eval_agent_path)
        eval_agent = Agent.load(eval_agent_path)
        eval_vs_other(agent, eval_agent, agent_name)

def eval_vs_other(agent, eval_agent, name):
    # Hack to temporarily change the agent's rollouts_per_turn
    agent.rollouts_per_turn = ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST
    wins, _metrics = run_n_games(ENVS[EVAL_ENV_NAME], [agent, eval_agent], n=EVAL_TRIALS)
    agent.rollouts_per_turn = ROLLOUTS_PER_TURN
    winrate = wins[0] / EVAL_TRIALS
    metrics_logger.log_metrics({f"eval_winrate/{name}": winrate})
    logger.info(f"Win rate vs {name} = {winrate}")  

def main(run_name):
    seed()
    checkpoint_dir, code_dir = setup_directory(run_name)
    logger.info(f"Starting run {run_name}")

    device = find_device()

    agent = build_agent()
    disc_model = agent.discriminator.model
    disc_model.to(device)
    disc_optimizer = th.optim.Adam(disc_model.parameters(), lr=DISC_LR)

    if TRAIN_GENERATOR:
        gen_model = agent.generator.model
        gen_model.to(device)
        gen_optimizer = th.optim.Adam(gen_model.parameters(), lr=GEN_LR)

    def model_mode_eval():
        disc_model.eval()
        if TRAIN_GENERATOR:
            gen_model.eval()

    def model_mode_train():
        disc_model.train()
        if TRAIN_GENERATOR:
            gen_model.train()

    if ROLLOUT_PROCS == 1:
        rollout_source = InProcessRolloutSource(EPISODES_PER_ITERATION, game_kwargs, agent)
    else:
        rollout_source = MultiProcessRolloutSource(build_agent, agent, EPISODES_PER_ITERATION, game_kwargs, ROLLOUT_PROCS)

    iteration = 0
    turns_optimized = 0
    rollout_stats = defaultdict(int)
    while MAX_ITERATIONS is None or iteration <= MAX_ITERATIONS:
        with metrics_logger.timing('iteration'):
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

            metrics_logger.log_metrics(rollout_stats)
            disc_param_norm = sum([th.norm(param, p=2) for param in disc_model.parameters()]).item()
            metrics_logger.log_metrics({
                    'disc/epsilon_greedy': agent.discriminator.epsilon_greedy, 
                    'turns_optimized': turns_optimized, 
                    'disc/param_norm': disc_param_norm})
            if TRAIN_GENERATOR:
                gen_param_norm = sum([th.norm(param, p=2) for param in gen_model.parameters()]).item()
                metrics_logger.log_metrics({
                    'gen/epsilon_greedy': agent.generator.epsilon_greedy, 
                    'gen/param_norm': gen_param_norm})

            with metrics_logger.timing('rollouts'):
                logger.info("Starting rollouts...")
                model_mode_eval()

                rollout_batch = rollout_source.get_rollouts(iteration=iteration)
                num_turns = rollout_batch['discriminator'].labels.shape[0]
                model_mode_train()
                rollout_stats['rollouts/games'] += rollout_batch['discriminator'].num_games
                rollout_stats['rollouts/turns'] += num_turns

            with metrics_logger.timing('training/disc'):
                logger.info("Starting training discriminator...")
                disc_rollout_batch = rollout_batch['discriminator']
                for epoch in range(SAMPLE_REUSE):
                    logger.info(f"  Epoch {epoch}/{SAMPLE_REUSE}...")
                    all_idxes = np.random.permutation(num_turns)
                    n_batches = num_turns // BATCH_SIZE
                    for idx in range(n_batches):
                        with metrics_logger.timing('training_batch'):
                            batch_idxes = all_idxes[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]
                            batch_obs = {}
                            for key in disc_rollout_batch.obs:
                                batch_obs[key] = disc_rollout_batch.obs[key][batch_idxes]
                            batch_labels = disc_rollout_batch.labels[batch_idxes]
                            batch_labels = th.from_numpy(batch_labels).to(device)
                            disc_optimizer.zero_grad()
                            disc_logprob = disc_model(batch_obs) # [batch, 1]
                            batch_labels = th.unsqueeze(batch_labels, 1)
                            loss = th.nn.BCEWithLogitsLoss()(disc_logprob, batch_labels)
                            loss.backward()
                            disc_optimizer.step()
                            if idx in [0, n_batches // 2, n_batches - 1]:
                                max_batch_digits = len(str(n_batches))
                                metrics_logger.log_metrics({f"loss/epoch_{epoch}/batch_{idx:0>{max_batch_digits}}": loss.item()})
                            turns_optimized += len(batch_idxes)
            # TODO - train genereator here
            logger.info(f"Iteration {iteration} complete.")
            iteration += 1
            # agent.epsilon_greedy = EPSILON_GREEEDY * (1 - 0.8 * iteration / MAX_ITERATIONS)

        metrics_logger.flush()
        if iteration % EVAL_EVERY == 0:
            with metrics_logger.timing('eval'):
                logger.info("Evaluating...")
                model_mode_eval()
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
                model_mode_train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    args = parser.parse_args()

    main(run_name=args.name)