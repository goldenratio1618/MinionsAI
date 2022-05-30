from agent import Agent
import pickle
import os
import numpy as np

class TrainedAgent(Agent):
    def __init__(self, policy, translator, generator, rollouts_per_turn):
        self.translator = translator
        self.policy = policy
        self.generator = generator
        self.rollouts_per_turn = rollouts_per_turn
    
    def act(self, game):
        options = []
        scores = []
        for i in range(self.rollouts_per_turn):
            actions = self.generator.rollout(game)
            obs = self.translator.translate(game)
            disc_logprob = self.policy(obs).detach().cpu().numpy()
            scores.append(disc_logprob)
            options.append(actions)
        best_option = options[np.argmax(scores)]
        self.generator.redo(best_option, game)
        

    def save(self, checkpoint_path):
        print(os.path.exists(checkpoint_path))
        os.makedirs(checkpoint_path)
        self.policy.save(os.path.join(checkpoint_path, "weights.pt"))
        pickle.dump(self, open(os.path.join(checkpoint_path, "agent.pkl"), "wb"))

    @classmethod
    def load(cls, checkpoint_path):
        agent = pickle.load(open(os.path.join(checkpoint_path, "agent.pkl"), "rb"))
        agent.policy.load(os.path.join(checkpoint_path, "weights.pt"))
        return agent
    