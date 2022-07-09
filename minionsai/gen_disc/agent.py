from typing import List, Tuple

from ..experiment_tooling import find_device

from ..engine import print_n_games
from .generator import BaseGenerator
from .discriminators import BaseDiscriminator
from ..agent import Agent, RandomAIAgent
import numpy as np
import logging
import tabulate

logger = logging.getLogger(__name__)

class GenDiscAgent(Agent):
    """
    An agent that uses a generator to generate possible turns and a discriminator to choose among them.
    """
    def __init__(self, discriminator: BaseDiscriminator, generators: List[Tuple[BaseGenerator, int]], verbose_level=0):
        self.discriminator = discriminator
        self.generators = generators
        self.verbose_level = verbose_level

    def act(self, game):
        action, _generator_info, _discriminator_info = self.act_with_info(game)
        return action
    
    def act_with_info(self, game):
        options_action_list = []
        options_states = []
        generator_info = []
        actual_num_rollouts = []  # Actual count of rollouts produced by each generator.
        for gen, num_rollouts in self.generators:
            this_options_action_list, this_options_states, this_generator_info = gen.propose_n_actions(game, num_rollouts)
            options_action_list.extend(this_options_action_list)
            options_states.extend(this_options_states)
            generator_info.append(this_generator_info)
            actual_num_rollouts.append(len(this_options_action_list))

        best_option_idx, discriminator_info = self.discriminator.choose_option(options_states)

        #### Produce metrics ####
        all_ending_labels = discriminator_info["all_winprobs"]
        max_winprob = np.max(all_ending_labels)
        
        pointer = 0
        num_generators_who_contained_best_action = 0
        a_generator_who_contained_best_action = None
        for (gen, _), my_gen_info, my_actual_num_rollouts in zip(self.generators, generator_info, actual_num_rollouts):

            # Count how many unique states each generator proposed.
            # We assume that different states would have slightly different win probabilities.
            my_game_hashes = [hash(game) for game in options_states[pointer:pointer + my_actual_num_rollouts]]
            my_unique_options = np.unique(my_game_hashes).size / my_actual_num_rollouts

            # Check which generators proposed the chosen action (or one equivalent)
            # We again assume that different actions would have slightly different win probabilities.
            my_ending_labels = all_ending_labels[pointer:pointer + my_actual_num_rollouts]
            my_max_winprob = np.max(my_ending_labels)
            have_best_action = max_winprob - my_max_winprob < 1e-4

            my_gen_info["metrics"] = {
                "unique_options": my_unique_options,
                "have_best_action": have_best_action,
                "alone_have_best_action": False  # will be overwritten after the loop
            }

            num_generators_who_contained_best_action += have_best_action
            if have_best_action:
                a_generator_who_contained_best_action = my_gen_info["metrics"]
            pointer += num_rollouts

        # Did only one generator contain the best action?
        if num_generators_who_contained_best_action == 0:
            raise ValueError("No generator contained the best action!")
        elif num_generators_who_contained_best_action == 1:
            a_generator_who_contained_best_action["alone_have_best_action"] = True

        #### Verbose logging ####
        if self.verbose_level >= 1:
            print("Generated choices:")
            print_n_games(options_states[:8])
            print("|".join([f"  pwin={p:.1%}".ljust(15) for p in discriminator_info.get('all_winprobs', [0] * len(options_states))[:8]]))
        if self.verbose_level >= 2 and len(options_states) > 8:
            print_n_games(options_states[8:16])
            print("|".join([f"  pwin={p:.1%}".ljust(15) for p in discriminator_info.get('all_winprobs', [0] * len(options_states))[8:16]]))
        if self.verbose_level >= 2 and "metrics" in generator_info[0]:
            keys = list(generator_info[0]["metrics"].keys())
            data=[
                ["cls"] + [gen.__class__.__name__ for gen, _ in self.generators],
                ["num"] + [num_rollouts for _, num_rollouts in self.generators],
            ]
            data.extend([[k] + [info_dict["metrics"][k] for info_dict in generator_info] for k in keys])
            print(tabulate.tabulate(data, headers=[""] + [f"gen {i}" for i in range(len(self.generators))]))
        if self.verbose_level >= 1:
            logger.info(f"Chosen option: {best_option_idx}")

        best_option = options_action_list[best_option_idx]
        return best_option, generator_info, discriminator_info

    def load_extra(self, directory: str):
        # Hacketty hack hack
        if hasattr(self.discriminator, "model"):
            self.discriminator.model.to(find_device())
        for gen, _ in self.generators:
            if hasattr(gen, "model"):
                gen.model.to(find_device())
