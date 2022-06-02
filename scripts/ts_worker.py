"""
Run this to evaluate all the agents in a directory against each other.
"""

from minionsai.trueskill_worker import TrueskillWorker
from minionsai.engine import Game

worker = TrueskillWorker(dir, Game)
worker.run()