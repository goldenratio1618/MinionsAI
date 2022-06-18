from typing import Dict, Tuple
import multiprocessing as mp

from ..experiment_tooling import find_device
from .rollout_runner import RolloutRunner
from .rollouts_data import RolloutEpisode
from .rollouts import OptimizerRolloutSource

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
    def __init__(self, rank: int, game_kwargs, agent_fn, requests_queue: mp.Queue, episodes_queue: mp.Queue, iteration_info_queue: mp.Queue):
        self.rank = rank
        self.requests_queue = requests_queue
        self.episodes_queue = episodes_queue
        self.iteration_info_queue = iteration_info_queue
        self.agent = agent_fn()
        self.agent.policy.to(find_device())
        self.agent.policy.eval()
        self.runner = RolloutRunner(game_kwargs, self.agent)
        self.iteration = -1

    def wait_for_request(self) -> Tuple[int, int]:
        return self.requests_queue.get()

    def update_for_new_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        info_iter, hparams, model_state = self.iteration_info_queue.get()
        if info_iter < iteration:
            raise ValueError(f"Worker {self.rank} received request for iteration {iteration} but found info for iteration {info_iter}")
        if info_iter > iteration:
            # TODO this might not be an error; maybe we are just processing a stale request.
            raise ValueError(f"Worker {self.rank} received request for iteration {iteration} but found info for iteration {info_iter}")
        self.runner.update(hparams)
        self.agent.policy.load_state_dict(model_state)

    def run(self):
        while True:
            iteration, episode_idx = self.wait_for_request()
            if iteration != self.iteration:
                self.update_for_new_iteration(iteration)
            self.episodes_queue.put((iteration, episode_idx, self.runner.single_rollout()))

class MultiProcessRolloutSource(OptimizerRolloutSource):
    """
    Starts multiple worker processes to run rollouts in parallel
    """
    def __init__(self, agent_fn, main_thread_agent, episodes_per_iteration, game_kwargs, num_procs=4, lambda_until_episodes=5000):
        super().__init__(episodes_per_iteration, game_kwargs, lambda_until_episodes)
        self.iteration = -1

        self.agent_fn = agent_fn
        self.main_thread_agent = main_thread_agent

        # Queue for incoming data from workers
        self.episodes_queue = mp.Queue()

        # Queue for outgoing requests to workers
        self.requests_queue = mp.Queue()

        # Place to send ireation metadata to each worker
        self.iteration_info_queues = []
        self.procs = []

        # Make worker processes
        mp.set_start_method("spawn", force=True)
        for rank in range(num_procs):
            iteration_info_queue = mp.Queue()
            self.iteration_info_queues.append(iteration_info_queue)
            proc = mp.Process(target=worker_proc_main, args=(rank, self.game_kwargs, self.agent_fn, self.requests_queue, self.episodes_queue, iteration_info_queue), daemon=True)
            self.procs.append(proc)
        for p in self.procs:
            p.start()

    
    def next_rollout(self) -> RolloutEpisode:
        iteration, episode_idx, data = self.episodes_queue.get()
        if iteration != self.iteration:
            raise ValueError(f"Received episode from wrong iteration: {iteration} != {self.iteration}")
        return data

    def launch_rollouts(self, iteration: int, hparams: Dict) -> None:
        # Check that both queues are empty
        assert self.requests_queue.empty()
        assert self.episodes_queue.empty()
        self.iteration = iteration
        for q in self.iteration_info_queues:
            state_dict = self.main_thread_agent.policy.state_dict()
            # convert to cpu to avoid GPU corruption on interproces comms
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
            q.put((iteration, hparams, state_dict))

        # Send requests worker threads that they should begin this iteration
        for i in range(self.episodes_per_iteration):
            self.requests_queue.put((iteration, i))




