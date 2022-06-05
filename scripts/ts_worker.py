"""
Run this to evaluate all the agents in a directory against each other.

> python ts_worker.py --dir=path/to/directory/with/agents
"""

from minionsai.trueskill_worker import TrueskillWorker
from minionsai.engine import Game
import os
from minionsai.agent import NullAgent, RandomAIAgent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str)
parser.add_argument("--add_defaults", type=bool, default="False", help="Add null & random agents to the pool.")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--restart", type=bool, default=False, help="Restart all scores from scratch even if previous data is available.")
args = parser.parse_args()

if args.add_defaults:
    NullAgent().save(os.path.join(args.dir, 'null'), exists_ok=True)
    RandomAIAgent().save(os.path.join(args.dir, 'random_agent'), exists_ok=True)
worker = TrueskillWorker(args.dir, Game, batch_size=args.batch_size, restart=args.restart)
worker.run()