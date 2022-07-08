"""
Run with a name for the folder to keep agents in:

> python3 train.py --name my_run

Then agent checkpoints & logs are saved in <your experiments dir>/MinionsAI/my_run
"""

import argparse
from collections import defaultdict
import random
from minionsai.action_bot.model import MinionsActionBot
from minionsai.experiment_tooling import find_device, get_experiments_directory, setup_directory
from minionsai.game_util import seed_everything
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import QDiscriminator, ScriptedDiscriminator
from minionsai.gen_disc.generator import AgentGenerator, QGenerator
from minionsai.multiprocessing_rl.multiproc_rollouts import MultiProcessRolloutSource
from minionsai.multiprocessing_rl.rollouts import InProcessRolloutSource
from minionsai.run_game import run_n_games
from minionsai.discriminator_only.model import MinionsDiscriminator
from minionsai.discriminator_only.translator import Translator
from minionsai.agent import Agent, CLIAgent, RandomAIAgent
from minionsai.agent_saveload import load, save
from minionsai.scoreboard_envs import ENVS
import torch as th
import numpy as np
import os
import logging
from minionsai.metrics_logger import metrics_logger
import tqdm

logger = logging.getLogger(__name__)

TRAIN_GENERATOR = True
TRAIN_DISCRIMINATOR = False
LOAD_DISCRIMINATOR_MODEL = os.path.join(get_experiments_directory(), "conveps_repro_0704", "checkpoints", "conveps_repro_iter_400_adapt")
LOAD_GENERATOR_MODEL = None # os.path.join(get_experiments_directory(), "gen_convbig396", "checkpoints", "iter_396_adapt")

# How many rollouts do we run of each turn before picking the best
ROLLOUTS_PER_TURN = 16
DISC_EPSILON_GREEDY = 0.1
GEN_EPSILON_GREEDY = 0.04  # (1 - 0.04)^10 ~ 66%
GEN_SAMPLING_TEMPERATURE = 0.03

# How many episodes of data do we collect each iteration, before running a few epochs of optimization?
# Potentially good to use a few times bigger EPISODES_PER_ITERATION than BATCH_SIZE, to minimize correlation within batches
EPISODES_PER_ITERATION = 32
ROLLOUT_PROCS = 4

# Once we've collected the data, how many times do we go over it for optimization (within one iteration)?
SAMPLE_REUSE = 2
GEN_SAMPLE_REUSE = 1  # There is so much data, no need to reuse it.

# Frequency of running evals
EVAL_EVERY = 8
# Put iteration numbers here to eval vs past versions of this train run.
EVAL_VS_PAST_ITERS = []
# Specific agent instances to eval vs
EVAL_VS_AGENTS = [
    GenDiscAgent(ScriptedDiscriminator(), [(AgentGenerator(RandomAIAgent()), 16)]),
    os.path.join(get_experiments_directory(), "conv_big", "dfarhi_0613_conveps_256rolls_iter400_adapt")
]
# Eval against random up until this iteration
EVAL_VS_RANDOM_UNTIL = 3
EVAL_TRIALS = 50

# Frequency of storing a saved agent
CHECKPOINT_EVERY = 4

# During evals, run this many times extra rollouts compared to during rollout generation
EVAL_COMPUTE_BOOST = 16

# Model Size
DEPTH = 2
D_MODEL = 64 * DEPTH
GEN_DEPTH = 1
GEN_D_MODEL = 64 * GEN_DEPTH

# Optimizer hparams
DISC_BATCH_SIZE = EPISODES_PER_ITERATION
DISC_LR = 1e-4
GEN_BATCH_SIZE = EPISODES_PER_ITERATION * 64
GEN_LR = 2e-4

# kwargs to create a game (passed to Game)
game_kwargs = {'symmetrize': False}
# Eval env registered in scoreboard_envs.py
EVAL_ENV_NAME = 'zombies5x5'

MAX_ITERATIONS = 400

SEED = 12345

def build_agent():
    logger.info("Creating generator...")
    if TRAIN_GENERATOR:
        gen_model = MinionsActionBot(d_model=GEN_D_MODEL, depth=GEN_DEPTH)  # TODO - make generator model instead
        logger.info("Generator model initialized:")
        logger.info(gen_model)
        gen_translator = Translator("generator")  # TODO - make gen translator
        generator = QGenerator(model=gen_model, translator=gen_translator, sampling_temperature=GEN_SAMPLING_TEMPERATURE, epsilon_greedy=GEN_EPSILON_GREEDY)
    elif LOAD_GENERATOR_MODEL is None:
        generator = AgentGenerator(RandomAIAgent())
    else:
        generator_agent = load(LOAD_GENERATOR_MODEL, already_in_path_ok=True)  # ok if a thread loads this after main has already done so.
        generator = generator_agent.generator
        gen_model = generator.model
        gen_model.to(find_device())

    logger.info("Creating discriminator...")
    if TRAIN_DISCRIMINATOR:
        disc_model = MinionsDiscriminator(d_model=D_MODEL, depth=DEPTH)
        logger.info("Discriminator model initialized:")
        logger.info(disc_model)
        logger.info(f"Discriminator model total parameter count: {sum(p.numel() for p in disc_model.parameters() if p.requires_grad):,}")

        disc_translator = Translator("discriminator")
        discriminator = QDiscriminator(translator=disc_translator, model=disc_model, epsilon_greedy=DISC_EPSILON_GREEDY)
    elif LOAD_DISCRIMINATOR_MODEL is None:
        discriminator = ScriptedDiscriminator()
    else:
        disc_agent = load(LOAD_DISCRIMINATOR_MODEL, already_in_path_ok=True)  # ok if a thread loads this after main has already done so.
        discriminator = disc_agent.discriminator
        disc_model = discriminator.model
        disc_model.to(find_device())
    if TRAIN_DISCRIMINATOR or LOAD_DISCRIMINATOR_MODEL:
        logger.info(f"Discriminator model total parameter count: {sum(p.numel() for p in disc_model.parameters() if p.requires_grad):,}")
    if TRAIN_GENERATOR or LOAD_GENERATOR_MODEL:
        logger.info(f"Generator model total parameter count: {sum(p.numel() for p in gen_model.parameters() if p.requires_grad):,}")

    agent = GenDiscAgent(discriminator, [(generator, ROLLOUTS_PER_TURN), (AgentGenerator(RandomAIAgent()), 64)])
    return agent

def eval_vs_other_by_path(agent, eval_agent_path):
    logger.info(f"Looking for eval agent at {eval_agent_path}...")
    if os.path.exists(eval_agent_path):
        agent_name = os.path.basename(eval_agent_path)
        eval_agent = load(eval_agent_path)
        eval_vs_other(agent, eval_agent, agent_name)

def eval_vs_other(agent, eval_agent, name):
    # Hack to temporarily change the agent's rollouts_per_turn & epsilon greedy values
    # TODO - make it easier to set an agent into "eval" mode.
    agent.rollouts_per_turn = ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST
    if TRAIN_DISCRIMINATOR:
        agent.discriminator.epsilon_greedy = 0.0
    wins, _metrics = run_n_games(ENVS[EVAL_ENV_NAME], [agent, eval_agent], n=EVAL_TRIALS)
    agent.rollouts_per_turn = ROLLOUTS_PER_TURN
    if TRAIN_DISCRIMINATOR:
        agent.discriminator.epsilon_greedy = DISC_EPSILON_GREEDY
    winrate = wins[0] / EVAL_TRIALS
    metrics_logger.log_metrics({f"eval_winrate/{name}": winrate})
    logger.info(f"Win rate vs {name} = {winrate}")  

def main(run_name):
    seed_everything(SEED)
    assert TRAIN_DISCRIMINATOR or TRAIN_GENERATOR, "What are you doing?"

    checkpoint_dir, code_dir = setup_directory(run_name)
    logger.info(f"Starting run {run_name}")

    device = find_device()

    agent = build_agent()
    if TRAIN_DISCRIMINATOR:
        disc_model = agent.discriminator.model
        disc_model.to(device)
        disc_optimizer = th.optim.Adam(disc_model.parameters(), lr=DISC_LR)

    if TRAIN_GENERATOR:
        # Assume we train the first generator.
        train_gen, rollouts_per_turn = agent.generators[0]
        gen_model = train_gen.model
        gen_model.to(device)
        gen_optimizer = th.optim.Adam(gen_model.parameters(), lr=GEN_LR)

    def model_mode_eval():
        if TRAIN_DISCRIMINATOR:
            disc_model.eval()
        if TRAIN_GENERATOR:
            gen_model.eval()

    def model_mode_train():
        if TRAIN_DISCRIMINATOR:
            disc_model.train()
        if TRAIN_GENERATOR:
            gen_model.train()

    if ROLLOUT_PROCS == 1:
        rollout_source = InProcessRolloutSource(EPISODES_PER_ITERATION, game_kwargs, agent)
    else:
        rollout_source = MultiProcessRolloutSource(build_agent, agent, EPISODES_PER_ITERATION, game_kwargs, ROLLOUT_PROCS, 
                                    train_generator=TRAIN_GENERATOR, train_discriminator=TRAIN_DISCRIMINATOR)

    iteration = 0
    turns_optimized = 0
    gen_actions_optimized = 0
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
                    if TRAIN_DISCRIMINATOR:
                        agent.discriminator.epsilon_greedy = 0.0
                    model_mode_eval()
                    save(agent, os.path.join(checkpoint_dir, f"iter_{iteration}"), copy_code_from=code_dir)
                    model_mode_train()
                    agent.rollouts_per_turn = ROLLOUTS_PER_TURN
                    if TRAIN_DISCRIMINATOR:
                        agent.discriminator.epsilon_greedy = DISC_EPSILON_GREEDY

            metrics_logger.log_metrics(rollout_stats)
            if TRAIN_DISCRIMINATOR:
                disc_param_norm = sum([th.norm(param, p=2) for param in disc_model.parameters()]).item()
                metrics_logger.log_metrics({
                        'disc/turns_optimized': turns_optimized, 
                        'disc/param_norm': disc_param_norm})
            if TRAIN_GENERATOR:
                gen_param_norm = sum([th.norm(param, p=2) for param in gen_model.parameters()]).item()
                metrics_logger.log_metrics({
                    'gen/actions_optimized': gen_actions_optimized,
                    'gen/param_norm': gen_param_norm})

            with metrics_logger.timing('rollouts'):
                logger.info("Starting rollouts...")
                model_mode_eval()

                rollout_batch = rollout_source.get_rollouts(iteration=iteration)
                model_mode_train()
                rollout_stats['rollouts/games'] += rollout_batch['discriminator'].num_games


            if TRAIN_DISCRIMINATOR:
                disc_rollout_batch = rollout_batch['discriminator']
                num_turns = disc_rollout_batch.labels.shape[0]
                rollout_stats['disc/rollouts/turns'] += num_turns
                with metrics_logger.timing('training/disc'):
                    logger.info("Starting training discriminator...")
                    for epoch in range(SAMPLE_REUSE):
                        logger.info(f"  Epoch {epoch}/{SAMPLE_REUSE}...")
                        all_idxes = np.random.permutation(num_turns)
                        n_batches = num_turns // DISC_BATCH_SIZE
                        for idx in range(n_batches):
                            with metrics_logger.timing('training_batch/disc'):
                                batch_idxes = all_idxes[idx * DISC_BATCH_SIZE: (idx + 1) * DISC_BATCH_SIZE]
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
                                    metrics_logger.log_metrics({f"disc/loss/epoch_{epoch}/batch_{idx:0>{max_batch_digits}}": loss.item()})
                                turns_optimized += len(batch_idxes)
            if TRAIN_GENERATOR:
                gen_rollout_batch = rollout_batch['generator']
                num_actions = gen_rollout_batch.labels.shape[0]
                rollout_stats['gen/rollouts/actions'] += num_actions
                with metrics_logger.timing('training/gen'):
                    logger.info("Starting training generator...")
                    for epoch in range(GEN_SAMPLE_REUSE):
                        logger.info(f"  Epoch {epoch}/{GEN_SAMPLE_REUSE}...")
                        all_idxes = np.random.permutation(num_actions)
                        n_batches = num_actions // GEN_BATCH_SIZE
                        for idx in tqdm.tqdm(range(n_batches)):
                            with metrics_logger.timing('training_batch/gen'):
                                batch_idxes = all_idxes[idx * GEN_BATCH_SIZE: (idx + 1) * GEN_BATCH_SIZE]
                                batch_obs = {}
                                for key in gen_rollout_batch.obs:
                                    batch_obs[key] = gen_rollout_batch.obs[key][batch_idxes]
                                batch_labels = gen_rollout_batch.labels[batch_idxes]
                                batch_labels = th.from_numpy(batch_labels).to(device)
                                assert batch_labels.shape == (GEN_BATCH_SIZE,)
                                batch_actions = gen_rollout_batch.actions[batch_idxes]
                                batch_actions = th.from_numpy(batch_actions).to(device)  # [batch, 2]
                                assert batch_actions.shape == (GEN_BATCH_SIZE, 2), batch_actions.shape
                                gen_logits = gen_model(batch_obs) # [batch, N, N]
                                assert gen_logits.shape[0] == GEN_BATCH_SIZE and gen_logits.shape[1] == gen_logits.shape[2], gen_logits.shape
                                # Now we need to index the taken actions into the logits.
                                # Seems torch can only do this if we flatten first.
                                batch_actions = batch_actions[:, 0] * gen_logits.shape[1] + batch_actions[:, 1]
                                batch_actions = th.unsqueeze(batch_actions, 1)
                                gen_logits = gen_logits.view(GEN_BATCH_SIZE, -1)

                                selected_logits = th.gather(gen_logits, 1, batch_actions)
                                selected_logits = th.squeeze(selected_logits, 1)
                                assert selected_logits.shape == (GEN_BATCH_SIZE, ), selected_logits.shape
                                gen_optimizer.zero_grad()
                                loss = th.nn.BCEWithLogitsLoss()(selected_logits, batch_labels)
                                loss.backward()
                                gen_optimizer.step()
                                if idx in [0, n_batches // 2, n_batches - 1]:
                                    max_batch_digits = len(str(n_batches))
                                    metrics_logger.log_metrics({f"gen/loss/epoch_{epoch}/batch_{idx:0>{max_batch_digits}}": loss.item()})
                                gen_actions_optimized += len(batch_idxes)
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