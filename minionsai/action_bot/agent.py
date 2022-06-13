from minionsai.action import ActionList
from minionsai.action_bot.actions import possibly_legal_moves, possibly_legal_spawns
from minionsai.engine import Phase
from .model import MinionsDiscriminator
from ..agent import Agent, RandomAIAgent
import os
import numpy as np
import random
from .translator import ActionTranslator
import torch as th
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ActionAgent(Agent):
    def __init__(self, policy, translator, generator, rollouts_per_turn, attempts_per_action, n_actions, verbose_level=0):
        self.translator = translator
        self.policy = policy
        self.generator = generator
        self.rollouts_per_turn = rollouts_per_turn
        self.attempts_per_action = attempts_per_action
        self.verbose_level = verbose_level
        self.n_actions = n_actions
        self.eps_threshold = 1.0
        self.possible_moves = possibly_legal_moves()
        self.possible_spawns = possibly_legal_spawns()
        self.num_possible_moves = len(self.possible_moves)
        self.num_possible_actions = self.num_possible_moves + len(self.possible_spawns)

    def set_epsilon(self, eps_threshold):
        self.eps_threshold = eps_threshold
    

    def select_action(self, state, test=False):
        # print(state)
        global steps_done
        sample = random.random()
        if test or sample > self.eps_threshold:
            with th.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                s = self.policy(state)
                # print(s)
                s_max = s.max(1)[1]
                return s_max
                # s_max = th.tensor([[s_max]], device=device, dtype=th.long)
                # print("Selected")
                # print(s_max)
                # return s_max
        else:
            s_rand = random.randrange(self.n_actions)
            # print("Random")
            return s_rand

    def try_action(self, game, s_max):
        action = None
        if s_max <= self.num_possible_moves:
            try:
                action = self.possible_moves[s_max]
                game.process_single_action(action)
            except e:
                return None
        else:
            if game.phase == Phase.MOVE:
                game.end_move_phase()
            try:
                action = self.possible_spawns[s_max - self.num_possible_moves]
                game.process_single_action(action)
            except e:
                return None
        return action

    def act(self, game, test=False):
        game_copy = game.copy()
        action_list_move = []
        action_list_spawn = []
        while True:
            obs = self.translator.translate(game_copy)
            action_nn = self.select_action(obs, test)
            action = self.try_action(game_copy, action_nn)
            if action is None:
                break
            if game_copy.phase == Phase.MOVE:
                action_list_move.push(action)
            else:
                action_list_spawn.push(action)
        turn = ActionList(action_list_move, action_list_spawn)

        options_action_list = []
        options_obs = []
        for i in range(self.rollouts_per_turn):
            game_copy = game.copy()
            actions = self.generator.act(game_copy)
            game_copy.full_turn(actions)
            options_obs.append(self.translator.translate(game_copy))
            options_action_list.append(actions)

            if self.verbose_level >= 2:
                print(f"Option {i}")
                game_copy.pretty_print()
        obs_flat = {k: np.concatenate([o[k] for o in options_obs]) for k in options_obs[0]}
        disc_logprobs = self.policy(obs_flat).detach().cpu().numpy()  # [num_options]
        best_option_idx = np.argmax(disc_logprobs)
        if self.verbose_level >= 1:
            print(f"Choosing option {best_option_idx}; win prob = {sigmoid(disc_logprobs[best_option_idx]).item() * 100:.1f}%")
        best_option = options_action_list[best_option_idx]
        return best_option

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
        model.to(th.device('cpu'))  # TODO get better device if present.

        agent = TrainedAgent(model, Translator(), RandomAIAgent(), rollouts_per_turn)
        return agent

