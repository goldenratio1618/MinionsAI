from ..discriminator_only.model import MinionsDiscriminator
from ..engine import BOARD_SIZE
import torch as th
from ..unit_type import MAX_UNIT_HEALTH, unitList

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.memory = deque([],maxlen=capacity)

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)


class MinionsActionBot(MinionsDiscriminator):
    def __init__(self, d_model, depth):
        super().__init__(d_model, depth)
        self.unit_embedding = th.nn.Embedding(len(unitList) * 2 * 2 * 2 * MAX_UNIT_HEALTH + 1, d_model)
        self.phase_embedding = th.nn.Embedding(2, d_model)
        self.input_linear1 = th.nn.Linear(d_model, d_model)
        self.input_linear2 = th.nn.Linear(d_model, d_model)
        # self.money_embedding = th.nn.Embedding(max_money_emb, d_model)
        # self.opponent_money_embedding = th.nn.Embedding(max_money_emb, d_model)
        self.transformer = th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model, batch_first=True),
            num_layers=depth,
        )
        self.value_linear1 = th.nn.Linear(d_model, d_model)

    def to(self, device):
        super().to(device)
        self._device = device
    
    @property
    def device(self):
        if self._device is None:
            raise ValueError("Didn't tell policy what device to use!")
        return self._device

    def process_input(self, obs: th.Tensor):
        # obs is dict of:
        #   board: [batch, num_hexes, 3]
        #   remaining_turns: [batch, 1]
        #
        # Extract these tensors, keeping the int type
        obs = {k: th.Tensor(v).int().to(self.device) for k, v in obs.items()}

        board_obs = obs['board']
        assert tuple(board_obs.shape[1:]) == (BOARD_SIZE ** 2, 3), board_obs.shape
        hex_embs = self.hex_embedding(board_obs[:, :, 0])  # [batch, num_hexes, d_model]
        terrain_emb = self.terrain_embedding(board_obs[:, :, 1])  # [batch, num_hexes, d_model]
        unit_type_embs = self.unit_embedding(board_obs[:, :, 2])  # [batch, num_hexes, d_model]
        board_embs = hex_embs + terrain_emb + unit_type_embs # [batch, num_hexes, d_model]
        board_embs = self.activation(board_embs)

        # Reshape to [batch, height, width, channels]
        board_embs_conv = board_embs.view(board_embs.shape[0], BOARD_SIZE, BOARD_SIZE, self.d_model)
        # But the conv wants [batch, channels, height, width]
        board_embs_conv = board_embs_conv.permute(0, 3, 1, 2)
        board_embs_conv = self.input_conv1(board_embs_conv)
        board_embs_conv = board_embs_conv.permute(0, 2, 3, 1)  # [batch, height, width, d_model]
        board_embs_conv = board_embs_conv.view(board_embs_conv.shape[0], BOARD_SIZE ** 2, self.d_model)
        board_embs = board_embs + self.conv_ln(board_embs_conv)
        
        money_emb = self.money_embedding(obs['money'])
        remaining_turns_emb = self.remaining_turns_embedding(obs['remaining_turns'])
        opp_money_emb = self.opp_money_embedding(obs['opp_money'])
        score_diff_emb = self.score_diff_embedding(obs['score_diff'])
        phase_emb = self.phase_embedding(obs['phase'])
        # legal_actions_emb = self.legal_moves_embedding(obs['legal_actions'])
        embs = th.cat([board_embs, money_emb, remaining_turns_emb, opp_money_emb, score_diff_emb, phase_emb], dim=1) # legal_actions_emb ?
        return embs

    def process_output_into_scalar(self, trunk_out):
        # print(trunk_out.size()) # [B, N, 8]
        transposed = th.transpose(trunk_out, 1, 2)  # [B, 8, N]
        # print(transposed.size())
        post_linear = self.value_linear1(trunk_out) # [B, N, 8]
        # print(post_linear.size())
        output = th.matmul(post_linear, transposed)   # shape [B, N, N]
        return output


    def forward(self, state):
        return super().forward(state)
