import abc
from typing import Any, Callable, List, Tuple

from minionsai.game_util import stack_dicts
from ..multiprocessing_rl.rollouts_data import TrainingData
import numpy as np

class NodePointer(abc.ABC):
    @abc.abstractmethod
    def hash_node(self):
        """
        Hash of the current node.
        """
        pass

    @abc.abstractmethod
    def evaluate_node(self) -> Tuple[Any, List[Any], List[float]]:
        """
        Return the obs of this node, the available actions, and the Q-values of the available actions.
        """
        pass

    @abc.abstractmethod
    def take_action(self, action) -> None:
        """
        Move along the tree to a new location.
        """
        pass

class DepthFirstTreeSearch:
    """
    A version of tree search optimized to minize copying of the state.

    Starts with a root node and explores it greedily to the end.

    Then goes back to the most promising fork, and pursues that greedily to the end.

    Rinse and repeat.
    """
    def __init__(self, root: Callable[[], NodePointer], verbose=False):
        self._root = root
        self._explored_nodes = {}  # dict of hashes of nodes we've already seen, with maxq as values.
        self._unexplored_branches = []  # list of unexplored branches  (Q-estimate, obs_this node, actions_to_return_here, action)
        self._explored_root = False
        self._verbose = verbose

    def _verbose_print(self, msg):
        if self._verbose:
            print(msg)

    def run_trajectory(self, extra_training_data=None, epsilon_greedy=0.0, max_retries=100):
        if extra_training_data is not None:
            training_data = extra_training_data
        else:
            training_data = {
                "obs": [],
                "actions": [],
                "next_maxq": [],
            }

        if len(self._unexplored_branches) == 0 and self._explored_root:
            # We've explored the entire tree already.
            return [], None, training_data

        node_pointer = self._root()
        if not self._explored_root:
            root_obs, root_actions, root_q_estimates = node_pointer.evaluate_node()
            for action, q_estimate in zip(root_actions, root_q_estimates):
                self._unexplored_branches.append((q_estimate, root_obs, [], action))
            self._explored_root = True

        # Initialize the trajectory with the actions to get here.
        self._unexplored_branches.sort(key=lambda x: x[0], reverse=True)
        # print(f"Choosing option among: {[(q, path, next) for q, obs, path, next in self._unexplored_branches]}")
        _, obs, all_actions, next_action = self._unexplored_branches.pop(0)
        # print(f"Starting trajetory from {next_action} (via {all_actions})")
        all_actions = all_actions.copy()

        for action in all_actions:
            node_pointer.take_action(action)
        while True:
            training_data['obs'].append(obs)
            training_data['actions'].append(next_action)

            all_actions.append(next_action)
            node_pointer.take_action(next_action)
            current_node_hash = node_pointer.hash_node()
            # node_pointer.game.pretty_print()
            if current_node_hash in self._explored_nodes:
                # We've already been here.
                maxq = self._explored_nodes[current_node_hash]
                if maxq is None:
                    # This is a terminal node, and we don't know its maxq
                    # So we can't use this transition.
                    training_data['obs'].pop()
                    training_data['actions'].pop()
                else:
                    training_data['next_maxq'].append(maxq)
                self._verbose_print(f"Found another way to duplicate node {current_node_hash}; trying again.")
                if max_retries == 0:
                    # print("Max retries reached.")
                    return [], None, training_data
                return self.run_trajectory(extra_training_data=training_data, epsilon_greedy=epsilon_greedy, max_retries=max_retries - 1)
            self._verbose_print(f"{current_node_hash} not in explored_nodes {self._explored_nodes}")
            
            current_node_actions = all_actions.copy()
            obs, action_choices, q_estimates = node_pointer.evaluate_node()
            self._verbose_print(f"Available actions: {action_choices}")
            if len(action_choices) > 0:
                maxq = max(q_estimates)
                best_idx = np.argmax(q_estimates) if np.random.random() > epsilon_greedy else np.random.choice(len(action_choices))
                next_action = action_choices[best_idx]
                for i, (action, q_estimate) in enumerate(zip(action_choices, q_estimates)):
                    if i != best_idx:
                        self._unexplored_branches.append((q_estimate, obs, current_node_actions, action))
                self._explored_nodes[current_node_hash] = maxq
                training_data['next_maxq'].append(maxq)
                self._verbose_print(f"Best idx is {best_idx}; maxq is {maxq}")
            else:
                self._explored_nodes[current_node_hash] = None
                self._verbose_print("Found terminal node.")
                return all_actions, node_pointer, TrainingData(
                    obs=stack_dicts(training_data['obs']),
                    actions=training_data['actions'],
                    next_maxq=training_data['next_maxq'],
                )
                
