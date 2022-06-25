from functools import lru_cache
import os
import shutil
import tempfile
import torch as th
import logging
from .metrics_logger import metrics_logger

logger = logging.getLogger(__name__)

@lru_cache()
def get_experiments_directory():
    try:
        from .local_config import EXPERIMENTS_DIRECTORY
        return EXPERIMENTS_DIRECTORY
    except ImportError:
        return os.path.join(tempfile.gettempdir(), 'MinionsAI')

def find_device():
    logger.info("=========================")
    # set device to cpu or cuda
    if (th.cuda.is_available()): 
        device = th.device('cuda:0') 
        th.cuda.empty_cache()
        logger.info("Device set to : " + str(th.cuda.get_device_name(device)))
    else:
        device = th.device('cpu')
        logger.info("Device set to : cpu")
    logger.info("=========================")
    return device

def setup_directory(run_name):
    """
    Set up logging and checkpoint directories for a run.
    Returns the subdirectory for checkpoints.
    """
    run_directory = os.path.join(get_experiments_directory(), run_name)
    checkpoint_dir = os.path.join(run_directory, 'checkpoints')
    # If the directory already exists, warn the user and check if it's ok to overwrite it.
    if os.path.exists(run_directory):
        print(f"Run directory already exists at {run_directory}")
        ok = input("OK to overwrite? (y/n) ")
        if ok != "y":
            exit()
        shutil.rmtree(run_directory)
    os.makedirs(checkpoint_dir)

    ####### Configure logger #######
    logging.basicConfig(filename=os.path.join(run_directory, 'logs.txt'), level=logging.DEBUG, 
                        format='[%(levelname)s %(asctime)s] %(name)s: %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    metrics_logger.configure(os.path.join(run_directory, 'metrics.csv'))

    ####### Snapshot the codebase #######
    # Copy all of MinionsAI/ into directory/code, ignoring files that match ignore_patterns
    # In a cross-platform compatible way

    # No recursive copying
    ignore_patterns = [".git", "__pycache__", "scoreboard*", "tests", "scripts"]
    ignore_patterns.append("*" + run_name + "*")

    code_dir = os.path.join(run_directory, 'code')

    code_source = os.path.join(os.path.dirname(__file__), '..')
    shutil.copytree(code_source, code_dir, ignore=shutil.ignore_patterns(*ignore_patterns))

    return checkpoint_dir, code_dir
