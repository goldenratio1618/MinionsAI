"""
Run with a name for the folder to keep agents in:

> python3 train.py --name my_run

Then agent checkpoints & logs are saved in <your experiments dir>/MinionsAI/my_run
"""

import argparse
from collections import defaultdict
from functools import partial
from minionsai.action_bot.model import MinionsActionBot
from minionsai.ema import ema_avg
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
from torch.optim.swa_utils import AveragedModel
import numpy as np
import os
import logging
from minionsai.metrics_logger import metrics_logger
import tqdm

logger = logging.getLogger(__name__)

TRAIN_GENERATOR = True
TRAIN_DISCRIMINATOR = True
LOAD_DISCRIMINATOR_MODEL = None # os.path.join(get_experiments_directory(), "conveps_repro_0704", "checkpoints", "conveps_repro_iter_400_adapt")
LOAD_GENERATOR_MODEL = None # os.path.join(get_experiments_directory(), "tree_2", "checkpoints", "iter_328")

# How many rollouts do we run of each turn before picking the best
ROLLOUTS_PER_TURN = 16
DISC_EPSILON_GREEDY = 0.03
GEN_EPSILON_GREEDY = 0.1

# How many episodes of data do we collect each iteration, before running a few epochs of optimization?
# Potentially good to use a few times bigger EPISODES_PER_ITERATION than BATCH_SIZE, to minimize correlation within batches
EPISODES_PER_ITERATION = 32
ROLLOUT_PROCS = 4

# Once we've collected the data, how many times do we go over it for optimization (within one iteration)?
SAMPLE_REUSE = 2
GEN_SAMPLE_REUSE = 1  # There is so much data, no need to reuse it.

# During evals, run this many times extra rollouts compared to during rollout generation
EVAL_COMPUTE_BOOST = 4

# Put iteration numbers here to eval vs past versions of this train run.
EVAL_VS_PAST_ITERS = []
# Specific agent instances to eval vs
EVAL_VS_AGENTS = [
    os.path.join(get_experiments_directory(), "conv_big", "dfarhi_0613_conveps_256rolls_iter400_adapt")
]
# Eval against random up until this iteration
EVAL_VS_RANDOM_UNTIL = 3
EVAL_TRIALS = 128
# Frequency of running evals
# We want to spend 25% of the time on evals, so:
EVAL_EVERY = 4 * EVAL_TRIALS * EVAL_COMPUTE_BOOST // EPISODES_PER_ITERATION
print(f"Going to evaluate every {EVAL_EVERY} iterations")

# Frequency of storing a saved agent
CHECKPOINT_EVERY = EVAL_EVERY

# Model Size
DEPTH = 2
D_MODEL = 64 * DEPTH
GEN_DEPTH = 1
GEN_D_MODEL = 64 * GEN_DEPTH

# Optimizer hparams
DISC_BATCH_SIZE = EPISODES_PER_ITERATION
DISC_LR = 3e-5
DISC_EMA_HORIZON_ITERATIONS = 20 # 2.5% of 2000 iterations, rounded down a bit to be conservative
DISC_EMA_HORIZON_BATCHES = 20 * 4 * EPISODES_PER_ITERATION / DISC_BATCH_SIZE * SAMPLE_REUSE * DISC_EMA_HORIZON_ITERATIONS
DISC_EMA_DECAY = 1 - 1 / DISC_EMA_HORIZON_BATCHES
GEN_BATCH_SIZE = EPISODES_PER_ITERATION * 64
GEN_LR = 2e-4

# kwargs to create a game (passed to Game)
game_kwargs = {'symmetrize': False}
# Eval env registered in scoreboard_envs.py
EVAL_ENV_NAME = 'zombies5x5'

MAX_ITERATIONS = 4096

SEED = 12345

def build_agents():
    logger.info("Creating generator...")
    if TRAIN_GENERATOR:
        gen_model = MinionsActionBot(d_model=GEN_D_MODEL, depth=GEN_DEPTH)
        logger.info("Generator model initialized:")
        logger.info(gen_model)
        gen_translator = Translator("generator")
        gen_model.eval()
        gen_model.to(find_device())
        rollout_generator = QGenerator(model=gen_model, translator=gen_translator, epsilon_greedy=GEN_EPSILON_GREEDY)
        eval_generator = QGenerator(model=gen_model, translator=gen_translator, epsilon_greedy=0.0)
    elif LOAD_GENERATOR_MODEL is None:
        gen_model = None
        eval_generator = rollout_generator = AgentGenerator(RandomAIAgent())
    else:
        generator_agent = load(LOAD_GENERATOR_MODEL, already_in_path_ok=True)  # ok if a thread loads this after main has already done so.
        generator = generator_agent.generators[0][0]
        generator.epsilon_greedy = GEN_EPSILON_GREEDY
        gen_model = generator.model
        gen_model.to(find_device())
        gen_model.eval()
        eval_generator = rollout_generator = generator

    logger.info("Creating discriminator...")
    if TRAIN_DISCRIMINATOR:
        disc_model = MinionsDiscriminator(d_model=D_MODEL, depth=DEPTH)
        logger.info("Discriminator model initialized:")
        logger.info(disc_model)
        logger.info(f"Discriminator model total parameter count: {sum(p.numel() for p in disc_model.parameters() if p.requires_grad):,}")

        disc_translator = Translator("discriminator")
        disc_model.to(find_device())
        disc_model.eval()
        disc_model_ema = AveragedModel(disc_model, avg_fn=partial(ema_avg, decay=DISC_EMA_DECAY))
        disc_model_ema.to(find_device())
        disc_model_ema.update_parameters(disc_model)
        rollout_discriminator = QDiscriminator(translator=disc_translator, model=disc_model_ema, epsilon_greedy=DISC_EPSILON_GREEDY)
        eval_discriminator = QDiscriminator(translator=disc_translator, model=disc_model_ema, epsilon_greedy=0.0)

    elif LOAD_DISCRIMINATOR_MODEL is None:
        disc_model = disc_model_ema = None
        eval_discriminator = rollout_discriminator = ScriptedDiscriminator()
    else:
        disc_agent = load(LOAD_DISCRIMINATOR_MODEL, already_in_path_ok=True)  # ok if a thread loads this after main has already done so.
        discriminator = disc_agent.discriminator
        discriminator.epsilon_greedy = DISC_EPSILON_GREEDY
        disc_model = discriminator.model
        disc_model.to(find_device())
        disc_model.eval()
        eval_discriminator = rollout_discriminator = discriminator
        disc_model_ema = None

    if TRAIN_DISCRIMINATOR or LOAD_DISCRIMINATOR_MODEL:
        logger.info(f"Discriminator model total parameter count: {sum(p.numel() for p in disc_model.parameters() if p.requires_grad):,}")
    if TRAIN_GENERATOR or LOAD_GENERATOR_MODEL:
        logger.info(f"Generator model total parameter count: {sum(p.numel() for p in gen_model.parameters() if p.requires_grad):,}")

    rollout_agent = GenDiscAgent(rollout_discriminator, [
        (rollout_generator, ROLLOUTS_PER_TURN), 
        (AgentGenerator(RandomAIAgent()), 64)
        ])

    eval_agent = GenDiscAgent(eval_discriminator, [
        (eval_generator, ROLLOUTS_PER_TURN * EVAL_COMPUTE_BOOST), 
        (AgentGenerator(RandomAIAgent()), 64 * EVAL_COMPUTE_BOOST)
        ])
    return gen_model, disc_model, disc_model_ema, rollout_agent, eval_agent

def build_agent():
    _, _, _, agent, _ = build_agents()
    return agent

def eval_vs_other_by_path(agent, eval_opponent_path):
    logger.info(f"Looking for eval agent at {eval_opponent_path}...")
    if os.path.exists(eval_opponent_path):
        agent_name = os.path.basename(eval_opponent_path)
        eval_opponent = load(eval_opponent_path)
        eval_vs_other(agent, eval_opponent, agent_name)

def eval_vs_other(agent, eval_opponent, name):
    wins, _metrics = run_n_games(ENVS[EVAL_ENV_NAME], [agent, eval_opponent], n=EVAL_TRIALS)
    winrate = wins[0] / EVAL_TRIALS
    metrics_logger.log_metrics({f"eval_winrate/{name}": winrate})
    logger.info(f"Win rate vs {name} = {winrate}")  

def main(run_name):
    seed_everything(SEED)
    assert TRAIN_DISCRIMINATOR or TRAIN_GENERATOR, "What are you doing?"

    checkpoint_dir, code_dir = setup_directory(run_name)
    logger.info(f"Starting run {run_name}")

    device = find_device()

    gen_model, disc_model, disc_model_ema, rollout_agent, eval_agent = build_agents()
    if TRAIN_DISCRIMINATOR:
        disc_optimizer = th.optim.Adam(disc_model.parameters(), lr=DISC_LR)

    if TRAIN_GENERATOR:
        # Assume we train the first generator.
        gen_optimizer = th.optim.Adam(gen_model.parameters(), lr=GEN_LR)

    if ROLLOUT_PROCS == 1:
        rollout_source = InProcessRolloutSource(EPISODES_PER_ITERATION, game_kwargs, rollout_agent)
    else:
        rollout_source = MultiProcessRolloutSource(build_agent, rollout_agent, EPISODES_PER_ITERATION, game_kwargs, ROLLOUT_PROCS, 
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
            if iteration % CHECKPOINT_EVERY == 0 or iteration == MAX_ITERATIONS:
                with metrics_logger.timing('checkpointing'):
                    logger.info("Saving checkpoint...")
                    save(eval_agent, os.path.join(checkpoint_dir, f"iter_{iteration}"), copy_code_from=code_dir)

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
                assert (gen_model is None or not gen_model.training) and \
                    (disc_model is None or not disc_model.training), \
                    "Shouldn't be training during rollouts."
                rollout_batch = rollout_source.get_rollouts(iteration=iteration)
                # rollout_stats['rollouts/games'] += rollout_batch['discriminator'].num_games

            if TRAIN_DISCRIMINATOR:
                disc_rollout_batch = rollout_batch['discriminator']
                num_turns = disc_rollout_batch.next_maxq.shape[0]
                rollout_stats['disc/rollouts/turns'] += num_turns
                with metrics_logger.timing('training/disc'):
                    logger.info("Starting training discriminator...")
                    disc_model.train()
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

                                # Construct labels from max of next obs
                                with th.no_grad():
                                    batch_next_obs = {}
                                    for key in disc_rollout_batch.next_obs:
                                        batch_next_obs[key] = disc_rollout_batch.next_obs[key][batch_idxes]
                                    next_maxq = th.sigmoid(disc_model(batch_next_obs))
                                    assert next_maxq.shape == (DISC_BATCH_SIZE, 1)
                                    terminal_action = th.from_numpy(disc_rollout_batch.terminal_action[batch_idxes]).to(device)
                                    terminal_action = th.unsqueeze(terminal_action, 1)
                                    assert terminal_action.shape == (DISC_BATCH_SIZE, 1), terminal_action.shape
                                    reward = th.from_numpy(disc_rollout_batch.reward[batch_idxes]).to(device)
                                    reward = th.unsqueeze(reward, 1)
                                    assert reward.shape == (DISC_BATCH_SIZE, 1), reward.shape
                                    batch_labels = th.where(terminal_action, reward, next_maxq)
                                    assert batch_labels.shape == (DISC_BATCH_SIZE, 1), batch_labels.shape

                                disc_optimizer.zero_grad()
                                disc_logprob = disc_model(batch_obs) # [batch, 1]
                                loss = th.nn.BCEWithLogitsLoss()(disc_logprob, batch_labels)
                                loss.backward()
                                disc_optimizer.step()
                                if idx in [0, n_batches // 2, n_batches - 1]:
                                    max_batch_digits = max(len(str(n_batches)), 3)
                                    metrics_logger.log_metrics({f"disc/loss/epoch_{epoch}/batch_{idx:0>{max_batch_digits}}": loss.item()})
                                turns_optimized += len(batch_idxes)
                            with metrics_logger.timing("ema/disc"):
                                disc_model_ema.update_parameters(disc_model)
                    disc_model.eval()

            if TRAIN_GENERATOR:
                gen_rollout_batch = rollout_batch['generator']
                num_actions = gen_rollout_batch.next_maxq.shape[0]
                rollout_stats['gen/rollouts/actions'] += num_actions
                with metrics_logger.timing('training/gen'):
                    logger.info("Starting training generator...")
                    gen_model.train()
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
                                batch_labels = gen_rollout_batch.next_maxq[batch_idxes]
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
                                if idx in range(0, n_batches, 50):
                                    max_batch_digits = max(len(str(n_batches)), 3)
                                    metrics_logger.log_metrics({f"gen/loss/epoch_{epoch}/batch_{idx:0>{max_batch_digits}}": loss.item()})
                                gen_actions_optimized += len(batch_idxes)
                    gen_model.eval()

            logger.info(f"Iteration {iteration} complete.")
            iteration += 1
            # agent.epsilon_greedy = EPSILON_GREEEDY * (1 - 0.8 * iteration / MAX_ITERATIONS)

        metrics_logger.flush()
        if iteration % EVAL_EVERY == 0 or iteration == MAX_ITERATIONS:
            with metrics_logger.timing('eval'):
                logger.info("Evaluating...")
                if iteration < EVAL_VS_RANDOM_UNTIL:
                    eval_opponent = RandomAIAgent()
                    eval_vs_other(eval_agent, eval_opponent, 'random')
                for eval_opponent in EVAL_VS_AGENTS:
                    if isinstance(eval_opponent, str):
                        eval_vs_other_by_path(eval_agent, eval_opponent)
                    elif isinstance(eval_opponent, Agent):                  
                        eval_vs_other(eval_agent, eval_opponent, name=eval_opponent.__class__.__name__)
                for iter in EVAL_VS_PAST_ITERS:
                    eval_vs_other_by_path(eval_agent, os.path.join(checkpoint_dir, f"iter_{iter}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    args = parser.parse_args()

    main(run_name=args.name)