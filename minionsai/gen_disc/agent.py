import pickle
from .generator import BaseGenerator
from .discriminators import BaseDiscriminator
from ..agent import Agent, RandomAIAgent
import os
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

    def save_instance(self, directory: str):
        # Put the discriminator and generator in the their own directories
        discriminator_dir = os.path.join(directory, "discriminator")
        generator_dir = os.path.join(directory, "generator")
        os.makedirs(discriminator_dir)
        os.makedirs(generator_dir)
        with  open(os.path.join(discriminator_dir, "instance.pkl"), "wb") as file:
            pickle.dump(self.discriminator, file)
        with  open(os.path.join(generator_dir, "instance.pkl"), "wb") as file:
            pickle.dump(self.generator, file)
        json.dump({
            'rollouts_per_turn': self.rollouts_per_turn,
        }, open(os.path.join(directory, 'config.json'), 'w'))

    @classmethod
    def load_instance(cls, directory: str):
        config = json.load(open(os.path.join(directory, 'config.json')))
        rollouts_per_turn = config['rollouts_per_turn']
        discriminator_dir = os.path.join(directory, "discriminator")
        generator_dir = os.path.join(directory, "generator")
        code_dir = os.path.join(directory, "..", "code")
        with  open(os.path.join(discriminator_dir, "instance.pkl"), "rb") as file:
            discriminator = LocalCodeUnpickler(file, code_dir).load()
        with  open(os.path.join(generator_dir, "instance.pkl"), "rb") as file:
            generator = LocalCodeUnpickler(file, code_dir).load()
        return cls(discriminator, generator, rollouts_per_turn, 0)

class LocalCodeUnpickler(pickle.Unpickler):
    # TODO can we use something like this to save the entire Agent,
    # Rather than the current hacky thing with Agent.load() and AgentSubclass.load_instance()?
    def __init__(self, file, code_dir):
        super().__init__(file)
        self.code_dir = code_dir
    def find_class(self, module, name):
        if module.startswith('minionsai'):
            import sys
            minionsai_path = os.path.join(self.code_dir, 'minionsai')
            previously_in_sys_path = minionsai_path in sys.path
            if not previously_in_sys_path:
                sys.path.append(minionsai_path)
            result = super().find_class(module, name)      
            if not previously_in_sys_path:
                sys.path.remove(minionsai_path)
            return result
        return super().find_class(module, name)