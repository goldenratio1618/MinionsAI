from .discriminators import BaseDiscriminator
from ..agent import Agent, RandomAIAgent
import os
import numpy as np
import json

class GenDiscAgent(Agent):
    """
    An agent that uses a generator to generate possible turns and a discriminator to choose among them.
    """
    def __init__(self, discriminator: BaseDiscriminator, generator: Agent, rollouts_per_turn, verbose_level=0):
        self.discriminator = discriminator
        self.generator = generator
        self.rollouts_per_turn = rollouts_per_turn
        self.verbose_level = verbose_level
    
    def act(self, game):
        options_action_list = []
        options_states = []
        for _ in range(self.rollouts_per_turn):
            game_copy = game.copy()
            actions = self.generator.act(game_copy)
            game_copy.full_turn(actions)
            options_states.append(game_copy)
            options_action_list.append(actions)

        best_option_idx = self.discriminator.choose_option(options_states)
        best_option = options_action_list[best_option_idx]
        return best_option

    # TODO save/load on this class needs to be rethought.
    def save_instance(self, directory: str):
        generator_path = os.path.join(directory, "generator")
        os.makedirs(generator_path)
        self.generator.save(generator_path)
        discriminator_path = os.path.join(directory, "discriminator")
        os.makedirs(discriminator_path)
        self.discriminator.save(discriminator_path)
        json.dump({
            'rollouts_per_turn': self.rollouts_per_turn,
        }, open(os.path.join(directory, 'config.json'), 'w'))

    @classmethod
    def load_instance(cls, directory: str):
        config = json.load(open(os.path.join(directory, 'config.json')))
        rollouts_per_turn = config['rollouts_per_turn']
        generator_path = os.path.join(directory, "generator")
        generator = Agent.load(generator_path)
        discriminator_path = os.path.join(directory, "discriminator")
        discriminator = BaseDiscriminator.load(discriminator_path)
        agent = GenDiscAgent(discriminator, generator, rollouts_per_turn)
        return agent

