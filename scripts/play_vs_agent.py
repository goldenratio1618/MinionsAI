"""
Play against an agent with
> python play_vs_agent.py path/to/agent
"""
import sys
from minionsai.agent import Agent, HumanCLIAgent, RandomAIAgent
from minionsai.gen_disc.agent import GenDiscAgent
from minionsai.gen_disc.discriminators import HumanDiscriminator
from minionsai.run_game import run_game
from minionsai.engine import Game
import argparse
import tempfile
import os

def main(agent_dir=None, disc_mode=False):
    if agent_dir is None:
        agent = RandomAIAgent()
    else:
        agent = Agent.load(agent_dir)
        agent.verbose_level = 2

    if disc_mode:
        agent = GenDiscAgent(HumanDiscriminator(filter_agent=agent), RandomAIAgent(), rollouts_per_turn=agent.rollouts_per_turn)
    game = Game()
    winner = run_game(game, (agent, HumanCLIAgent()), verbose=True)
    print("Game over.\nWinner:", winner)
    print("Game metrics (player 0):")
    print(game.get_metrics(0))
    print("Game metrics (player 1):")
    print(game.get_metrics(1))

if __name__ == "__main__":
    # Use argparse to parse arguments "path", "name" and "iter"
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Full path to agent directory")
    parser.add_argument("--name", help="Experiment name (will automatically find path in <tempdir>/MinionsAI/<name>/checkpoints)")
    parser.add_argument("--iter", help="Iteration number (if unspecified, uses latest)")
    parser.add_argument("--disc_mode", type=bool, default=False, help="Choose among n random options like a discriminator.")
    args = parser.parse_args()

    if args.iter is not None:
        assert args.path is None, "Cannot specify both --path and --iter"
        assert args.name is not None, "Must specify --name when using --iter"

    if args.path is not None:
        assert args.name is None, "Cannot specify both --path and --name"
        path = args.path
    elif args.name is not None:
        agents_dir = os.path.join(tempfile.gettempdir(), "MinionsAI", args.name, "checkpoints")
        print(f"Finding path by experiment name: {agents_dir}")
        if args.iter is None:
            print("Finding latest iter ...")
            # Find the latest iteration based on the filenames
            choices = os.listdir(agents_dir)
            choices_iters = [int(choice.split('_')[-1]) for choice in choices]
            iter = max(choices_iters)
        else:
            print(f"Using specified iter: {args.iter}")
            iter = args.iter
        path = os.path.join(agents_dir, f"iter_{iter}")
    else:
        path = None
    print(f"Getting agent from: {path}")

    main(path, args.disc_mode)