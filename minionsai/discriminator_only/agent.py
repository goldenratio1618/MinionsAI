from ..game_util import equal_np_dicts
from ..engine import print_n_games
from ..experiment_tooling import find_device
from .model import MinionsDiscriminator
from ..agent import Agent, RandomAIAgent
import os
import numpy as np
from .translator import Translator
import torch as th
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# TODO - make this class a subclass of GenDiscAgent
# Hard part is getting save / load to work.
class TrainedAgent(Agent):
    def __init__(self, policy, translator, generator, rollouts_per_turn, verbose_level=0):
        self.translator = translator
        self.policy = policy
        self.generator = generator
        self.rollouts_per_turn = rollouts_per_turn
        self.verbose_level = verbose_level
    
    def act(self, game):
        action, _ = self.act_with_winprob(game)
        return action

    def act_with_winprob(self, game):
        options_action_list = []
        options_obs = []
        options_final_states = []  # used for printing only
        for i in range(self.rollouts_per_turn):
            game_copy = game.copy()
            actions = self.generator.act(game_copy)
            game_copy.full_turn(actions)
            options_obs.append(self.translator.translate(game_copy))
            options_action_list.append(actions)
            options_final_states.append(game_copy)

        obs_flat = {k: np.concatenate([o[k] for o in options_obs]) for k in options_obs[0]}
        disc_logprobs = self.policy(obs_flat).detach().cpu().numpy()  # [num_options, 1]
        # Remove extra dimension
        disc_logprobs = disc_logprobs.squeeze(1)
        best_option_idx = np.argmax(disc_logprobs)
        if self.verbose_level >= 2:
            # Find non-equivalent games
            equivalent_games = []
            for i, obs in enumerate(options_obs):
                copy = False
                for j in equivalent_games:
                    if equal_np_dicts(options_obs[i], options_obs[j]):
                        copy = True
                        break
                if not copy:
                    equivalent_games.append(i)
            k = 8
            # best k options in descending order
            best_k_options = sorted(equivalent_games, key=lambda i: disc_logprobs[i], reverse=True)[:k]
            print_n_games([options_final_states[i] for i in best_k_options])
            print("|".join([f"Opt {i:<3}: {sigmoid(disc_logprobs[i]).item():.1%}".ljust(15) 
                for i in best_k_options]))
        best_prob = sigmoid(disc_logprobs[best_option_idx]).item()
        if self.verbose_level >= 1:
            print(f"Choosing option {best_option_idx}; win prob = {best_prob * 100:.1f}%")
        best_option = options_action_list[best_option_idx]
        return best_option, best_prob

    def save_instance(self, directory: str):
        th.save(self.policy.state_dict(), os.path.join(directory, 'weights.pt'))
        json.dump({
            'rollouts_per_turn': self.rollouts_per_turn,
            'd_model': self.policy.d_model,
            'depth': self.policy.depth,
        }, open(os.path.join(directory, 'config.json'), 'w'))

    @classmethod
    def load_instance(cls, directory: str):
        config = json.load(open(os.path.join(directory, 'config.json')))
        d_model = config['d_model']
        depth = config['depth']
        rollouts_per_turn = config['rollouts_per_turn']
        model = MinionsDiscriminator(d_model=d_model, depth=depth)
        model.load_state_dict(th.load(os.path.join(directory, 'weights.pt'), map_location=th.device('cpu')))
        model.to(find_device())
        model.eval()  # Set to eval mode to disable dropout, etc.

        agent = TrainedAgent(model, Translator(), RandomAIAgent(), rollouts_per_turn)
        return agent

