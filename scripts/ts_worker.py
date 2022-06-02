"""
Run this to evaluate all the agents in a directory against each other.
"""

from minionsai.trueskill_worker import TrueskillWorker
from minionsai.engine import Game
import os
from minionsai.agent import NullAgent, RandomAIAgent

dir = f"C:\\Users/Maple/AppData/Local/Temp/MinionsAI/test"
add_default = True
if add_default:
    NullAgent().save(os.path.join(dir, 'null'), exists_ok=True)
    RandomAIAgent().save(os.path.join(dir, 'random_agent'), exists_ok=True)

worker = TrueskillWorker(dir, Game, batch_size=10)
worker.run()