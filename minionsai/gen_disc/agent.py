import pickle

from minionsai.experiment_tooling import find_device

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
            # Hacketty hack hack
            if hasattr(discriminator, "model"):
                discriminator.model.to(find_device())
        with  open(os.path.join(generator_dir, "instance.pkl"), "rb") as file:
            generator = LocalCodeUnpickler(file, code_dir).load()
            # Hacketty hack hack
            if hasattr(generator, "model"):
                generator.model.to(find_device())
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
        
        # Copied from https://stackoverflow.com/questions/57081727/load-pickle-file-obtained-from-gpu-to-cpu
        if module == 'torch.storage' and name == '_load_from_bytes':
            print("Loading model to device.")
            return lambda b: th.load(io.BytesIO(b), map_location=find_device())

        return super().find_class(module, name)