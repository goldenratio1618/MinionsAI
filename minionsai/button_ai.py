# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import random
import math
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple, deque

n_actions = 700
board_size = 4
depth = 2
d_model = 8

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ButtonAI(th.nn.Module):
    # Network defined by the Deepmind paper
    def __init__(self, d_model, depth, n_actions):
        super().__init__()
        self.board_embedding = th.nn.Embedding(board_size, d_model)
        self.transformer = th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model, batch_first=True),
            num_layers=depth
        )

        self._device = None
        self.depth = depth
        self.d_model = d_model

        self.input_linear1 = th.nn.Linear(d_model, d_model)
        self.input_linear2 = th.nn.Linear(d_model, d_model)
        self.value_linear1 = th.nn.Linear(d_model, d_model)
        self.value_linear2 = th.nn.Linear(d_model, n_actions)

    def to(self, device):
        super().to(device)
        self._device = device
    
    @property
    def device(self):
        if self._device is None:
            raise ValueError("Didn't tell policy what device to use!")
        return self._device

    def process_input(self, obs: th.Tensor):
        # print(obs)
        obs = obs.to(device)
        embs = th.cat([self.board_embedding(obs)], dim=1)
        # print(embs)
        return embs

    def process_output_into_scalar(self, trunk_out):
        # print(trunk_out)
        flat, _ = th.max(trunk_out, dim=1)
        # print(flat)
        # flat = trunk_out
        # flat, _ = th.max(trunk_out, dim=1)  # [batch, d_model]
        x = self.value_linear1(flat)  # [batch, d_model]
        x = th.nn.ReLU()(x)  # [batch, d_model]
        logit = self.value_linear2(flat)  # [batch, 1]
        return logit

    def forward(self, state):
        # print("FORWARD")
        # print(state)
        obs = self.process_input(state)  # [batch, num_things, d_model]
        # print(obs)
        obs = self.input_linear1(obs)  # [batch, num_things, d_model]
        # print(obs)
        obs = th.nn.ReLU()(obs)  # [batch, num_things, d_model]
        # print(obs)
        obs = self.input_linear2(obs)  # [batch, num_things, d_model]
        # print(obs)
        trunk_out = self.transformer(obs)  # [batch, num_things, d_model]
        # print(obs)
        # trunk_out = obs
        output = self.process_output_into_scalar(trunk_out)  # [batch, 1]
        # print(output)
        return output

    def save(self, checkpoint_path):
        th.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(th.load(checkpoint_path, map_location=lambda storage, loc: storage))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0#0.9
EPS_END = 1.0#0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = ButtonAI(d_model, depth, n_actions)
policy_net.to(device)
target_net = ButtonAI(d_model, depth, n_actions)
target_net.to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

def select_action(state, test=False):
    # print(state)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if test or sample > eps_threshold:
        with th.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            s = policy_net(state)
            # print(s)
            s_max = s.max(1)[1]
            s_max = th.tensor([[s_max]], device=device, dtype=th.long)
            # print("Selected")
            # print(s_max)
            return s_max
    else:
        s_rand = th.tensor([[random.randrange(n_actions)]], device=device, dtype=th.long)
        # print("Random")
        return s_rand



episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # print(batch.state)
    # print(batch.action)
    # print(batch.reward)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = th.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=th.bool)
    non_final_next_states = th.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = th.cat(batch.state)
    action_batch = th.cat(batch.action)
    reward_batch = th.cat(batch.reward)

    # print("state batch = " + str(state_batch))
    # print("action batch = " + str(action_batch))
    # print("reward batch = " + str(reward_batch))


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)
    # print("state actions 1 = " + str(state_action_values))
    state_action_values = state_action_values.gather(1, action_batch)
    # print("state actions 2 = " + str(state_action_values))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = th.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # print(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def init_state():
    win = np.random.choice(range(board_size))
    state = win
    state = th.from_numpy(np.array([[state]]))
    return state, win

num_episodes = 51
durations = 500
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state, win = init_state()

    for t in range(durations):
        # Select and perform an action

        action = select_action(state)
        reward = (action == win)
        # print(str(reward) + " " + str(action) + " " + str(win))
        done = (t == durations)
        reward = th.tensor([reward], device=device)

        if not done:
            next_state, next_win = init_state()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        win = next_win

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        target_net.save("/home/aaatanas/MinionsAI_tests/buttonAI/test3/"+ str(i_episode))
        n_success = 0
        for t in range(durations):
            state, win = init_state()
            # Select and perform an action
            action = select_action(state, True)
            # print(str(action) + ", " + str(win) + ": " + str(reward))
            reward = (action == win)
            done = False
            n_success += reward
        print("Iteration " + str(i_episode) + " success rate: " + str(n_success/durations))

print('Complete')
