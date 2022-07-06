import pickle

from ..experiment_tooling import find_device

from ..engine import print_n_games
from .generator import BaseGenerator
from .discriminators import BaseDiscriminator
from ..agent import Agent, RandomAIAgent
import os
import json
import torch as th
import io
import logging

logger = logging.getLogger(__name__)

class GenDiscAgent(Agent):
    """
    An agent that uses a generator to generate possible turns and a discriminator to choose among them.
    """
    def __init__(self, discriminator: BaseDiscriminator, generator: BaseGenerator, rollouts_per_turn, verbose_level=0):
        self.discriminator = discriminator
        self.generator = generator
        self.rollouts_per_turn = rollouts_per_turn
        self.verbose_level = verbose_level

    def act(self, game):
        action, _generator_info, _discriminator_info = self.act_with_info(game)
        return action
    
    def act_with_info(self, game):
        options_action_list, options_states, generator_info = self.generator.propose_n_actions(game, self.rollouts_per_turn)

        best_option_idx, discriminator_info = self.discriminator.choose_option(options_states)
        if self.verbose_level >= 1:
            print("Generated choices:")
            print_n_games(options_states[:8])
            print("|".join([f"  pwin={p:.1%}".ljust(15) for p in discriminator_info.get('all_winprobs', [0] * len(options_states))[:8]]))
        if self.verbose_level >= 2 and len(options_states) > 8:
            print_n_games(options_states[8:16])
            print("|".join([f"  pwin={p:.1%}".ljust(15) for p in discriminator_info.get('all_winprobs', [0] * len(options_states))[8:16]]))

        if self.verbose_level >= 1:
            logger.info(f"Chosen option: {best_option_idx}")


        best_option = options_action_list[best_option_idx]
        return best_option, generator_info, discriminator_info

    def load_extra(self, directory: str):
        # Hacketty hack hack
        if hasattr(self.discriminator, "model"):
            self.discriminator.model.to(find_device())
        if hasattr(self.generator, "model"):
            self.generator.model.to(find_device())
