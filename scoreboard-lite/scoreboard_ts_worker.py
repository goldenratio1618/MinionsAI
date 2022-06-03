"""
Run this to evaluate all the agents in a directory against each other.
"""

from minionsai.trueskill_worker import TrueskillWorker
from scoreboard_envs import ENVS
from util import env_dir
import sys

if __name__ == "__main__":
    # Get env name from argv
    env_name = sys.argv[1]
    dir = env_dir(env_name)
    worker = TrueskillWorker(dir, ENVS[env_name], batch_size=10)
    worker.run()