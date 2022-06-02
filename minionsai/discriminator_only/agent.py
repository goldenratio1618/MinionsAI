from .model import MinionsDiscriminator
from ..agent import Agent, RandomAIAgent
import os
import numpy as np
from .translator import Translator
import torch as th
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TrainedAgent(Agent):
    def __init__(self, policy, translator, generator, rollouts_per_turn, verbose_level=0):
        self.translator = translator
        self.policy = policy
        self.generator = generator
        self.rollouts_per_turn = rollouts_per_turn
        self.verbose_level = verbose_level
    
    def act(self, game):
        options = []
        scores = []
        for i in range(self.rollouts_per_turn):
            game_copy = game.copy()
            actions = self.generator.act(game_copy)
            game_copy.full_turn(actions)
            obs = self.translator.translate(game_copy)
            disc_logprob = self.policy(obs).detach().cpu().numpy()
            scores.append(disc_logprob)
            options.append(actions)
            if self.verbose_level >= 2:
                print(f"Option {i}")
                game_copy.pretty_print()
        best_option_idx = np.argmax(scores)
        if self.verbose_level >= 1:
            print(f"Choosing option {best_option_idx}; win prob = {sigmoid(scores[best_option_idx]).item() * 100:.1f}%")
        best_option = options[best_option_idx]
        return best_option

    def serialize(self, directory: str, exists_ok: bool = False):
        super().serialize(directory, exists_ok = exists_ok)
        agent_dir = os.path.join(directory, "agent")
        os.makedirs(agent_dir)
        th.save(self.policy.state_dict(), os.path.join(agent_dir, 'weights.pt'))
        json.dump({
            'rollouts_per_turn': self.rollouts_per_turn,
            'd_model': self.policy.d_model,
        }, open(os.path.join(agent_dir, 'config.json'), 'w'))

    @classmethod
    def deserialize_build(cls, directory: str):
        agent_dir = os.path.join(directory, "agent")
        config = json.load(open(os.path.join(agent_dir, 'config.json')))
        d_model = config['d_model']
        rollouts_per_turn = config['rollouts_per_turn']
        model = MinionsDiscriminator(d_model=d_model)
        model.load_state_dict(th.load(os.path.join(agent_dir, 'weights.pt')))
        model.to(th.device('cpu'))  # TODO get better device if present.

        agent = TrainedAgent(model, Translator(), RandomAIAgent(), rollouts_per_turn)
        return agent

