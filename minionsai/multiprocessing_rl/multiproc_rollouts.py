from typing import Dict, Tuple
import multiprocessing as mp

from ..gen_disc.agent import GenDiscAgent

from ..experiment_tooling import configure_logger, find_device
from .rollout_runner import RolloutRunner
from .rollouts_data import RolloutEpisode
from .rollouts import OptimizerRolloutSource
import os
import logging
logger = logging.getLogger(__name__)

def worker_proc_main(*args):
    """
    Main function for worker processes
    """
    worker = Worker(*args)
    worker.run()

class Worker():
    """
    Worker process that runs rollouts
    """
    def __init__(self, rank: int, game_kwargs, agent_fn, 
                    requests_queue: mp.Queue, episodes_queue: mp.Queue, iteration_info_queue: mp.Queue, 
                    device, logs_filename: str):
        configure_logger(logs_filename, also_stdout=False)
        self.rank = rank
        self.requests_queue = requests_queue
        self.episodes_queue = episodes_queue
        self.iteration_info_queue = iteration_info_queue
        self.agent = agent_fn()
        if isinstance(self.agent, GenDiscAgent):
            if hasattr(self.agent.discriminator, "model"):
                self.agent.discriminator.model.to(device)
                self.agent.discriminator.model.eval()
            for generator, _ in self.agent.generators:
                if hasattr(generator, "model"):
                    generator.model.to(device)
                    generator.model.eval()
        self.runner = RolloutRunner(game_kwargs, self.agent)
        self.iteration = -1

    def wait_for_request(self) -> Tuple[int, int]:
        return self.requests_queue.get()

    def update_for_new_iteration(self, iteration: int) -> None:
        logger.info(f"Updating to new iteration: {self.iteration} -> {iteration}")
        info_iter, hparams, models_state = self.iteration_info_queue.get()
        self.iteration = info_iter
        if info_iter < iteration:
            # This can happen if somehow this worker missed an entire iteration
            logger.warning(f"This worker seems to have missed an entire iteration, since we received a request for an episode from iteration {iteration}, but the next iteration in the queue is {info_iter}. Hoping we can just update again and it will be ok.")
            self.update_for_new_iteration(iteration)
        if info_iter > iteration:
            # TODO this might not be an error; maybe we are just processing a stale request.
            raise ValueError(f"Worker {self.rank} received request for iteration {iteration} but found info for iteration {info_iter}")
        self.runner.update(hparams)
        if isinstance(self.agent, GenDiscAgent):
            if "discriminator" in models_state:
                self.agent.discriminator.model.load_state_dict(models_state["discriminator"])
            if "generator" in models_state:
                # Assume we only train the first generator
                generator, _ = self.agent.generators[0]
                generator.model.load_state_dict(models_state["generator"])

    def run(self):
        while True:
            logger.info("Worker waiting for new request... ")
            iteration, episode_idx = self.wait_for_request()
            logger.info(f"Worker {self.rank} found request: {iteration}, {episode_idx}")
            if iteration != self.iteration:
                self.update_for_new_iteration(iteration)
            result = self.runner.single_rollout(iteration, episode_idx)
            logger.info("Done with episode.")
            self.episodes_queue.put((iteration, episode_idx, result))
            import sys
            sys.stdout.flush()

class MultiProcessRolloutSource(OptimizerRolloutSource):
    """
    Starts multiple worker processes to run rollouts in parallel
    """
    def __init__(self, agent_fn, main_thread_agent, episodes_per_iteration, game_kwargs, logs_directory,
                num_procs=4, device=None, train_generator=False, train_discriminator=False):
        super().__init__(episodes_per_iteration, game_kwargs)
        if device is None:
            device = find_device()
        self.iteration = -1
        self.train_generator = train_generator
        self.train_discriminator = train_discriminator

        self.agent_fn = agent_fn
        self.main_thread_agent = main_thread_agent

        # Queue for incoming data from workers
        self.episodes_queue = mp.Queue()

        # Queue for outgoing requests to workers
        self.requests_queue = mp.Queue()

        # Need to ensure we return the rollouts in the correct order
        # So we keep this buffer in case workers send them to us out of order.
        self.local_data_reordering_buffer = {}

        # Place to send ireation metadata to each worker
        self.iteration_info_queues = []
        self.procs = []

        # Make worker processes
        mp.set_start_method("spawn", force=True)
        for rank in range(num_procs):
            iteration_info_queue = mp.Queue()
            self.iteration_info_queues.append(iteration_info_queue)
            proc = mp.Process(
                target=worker_proc_main, 
                args=(rank, self.game_kwargs, self.agent_fn, self.requests_queue, self.episodes_queue, iteration_info_queue, device, os.path.join(logs_directory, f"logs_rollout_proc_{rank}.txt")), daemon=True)
            self.procs.append(proc)
        for p in self.procs:
            p.start()

    
    def next_rollout(self, iteration, episode_idx) -> RolloutEpisode:
        assert iteration == self.iteration
        # If we previously received this episode_idx, return it
        if episode_idx in self.local_data_reordering_buffer:
            return self.local_data_reordering_buffer.pop(episode_idx)
        
        # Otherwise, wait for more data
        found_iteration, found_episode_idx, data = self.episodes_queue.get()
        if found_iteration != self.iteration:
            raise ValueError(f"Received episode from wrong iteration: {found_iteration} != {self.iteration}")
        if found_episode_idx == episode_idx:
            return data
        # Otherwise, save it for later, and try again.
        self.local_data_reordering_buffer[found_episode_idx] = data
        return self.next_rollout(iteration, episode_idx)

    def launch_rollouts(self, iteration: int, hparams: Dict) -> None:
        # Check that both queues are empty
        assert self.requests_queue.empty()
        assert self.episodes_queue.empty()
        assert self.local_data_reordering_buffer == {}
        self.iteration = iteration

        if isinstance(self.main_thread_agent, GenDiscAgent):
            # Hacketty hack hack
            agent_state = {}
            if self.train_discriminator:
                # Convert to cpu because it seems to get corrupted if you pass GPU tensors directly.
                agent_state['discriminator'] = {k: v.cpu() for k, v in self.main_thread_agent.discriminator.model.state_dict().items()}
            if self.train_generator:
                train_generator, _ = self.main_thread_agent.generators[0]
                agent_state['generator'] = {k: v.cpu() for k, v in train_generator.model.state_dict().items()}
        for q in self.iteration_info_queues:
            q.put((iteration, hparams, agent_state))

        # Send requests worker threads that they should begin this iteration
        for i in range(self.episodes_per_iteration):
            self.requests_queue.put((iteration, i))




