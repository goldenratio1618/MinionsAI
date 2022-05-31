from agent import Agent
import pickle
import os
import numpy as np

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

    def save(self, checkpoint_path):
        os.makedirs(checkpoint_path)
        self.policy.save(os.path.join(checkpoint_path, "weights.pt"))
        pickle.dump(self, open(os.path.join(checkpoint_path, "agent.pkl"), "wb"))

    @classmethod
    def load(cls, checkpoint_path):
        agent = pickle.load(open(os.path.join(checkpoint_path, "agent.pkl"), "rb"))
        agent.policy.load(os.path.join(checkpoint_path, "weights.pt"))
        return agent
    