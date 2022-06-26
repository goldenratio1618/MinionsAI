from .generator import BaseGenerator
from .discriminators import BaseDiscriminator
from ..agent import Agent, RandomAIAgent
import os
import numpy as np
import json

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
        best_option = options_action_list[best_option_idx]
        return best_option, generator_info, discriminator_info

    # TODO save/load on this class needs to be rethought.
    def save_instance(self, directory: str):
        pass

    @classmethod
    def load_instance(cls, directory: str):
        pass
