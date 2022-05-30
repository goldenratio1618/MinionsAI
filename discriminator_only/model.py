from discriminator_only.translator import HEXES
import torch as th
from engine import BOARD_SIZE, unitList

class MinionsDiscriminator(th.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.unit_embedding = th.nn.Embedding(len(unitList) * 2 + 1, d_model)
        self.location_embedding = th.nn.Embedding(BOARD_SIZE ** 2, d_model)
        self.hex_embedding = th.nn.Embedding(BOARD_SIZE ** 2, d_model)
        self.terrain_embedding = th.nn.Embedding(3, d_model)
        self.input_linear1 = th.nn.Linear(d_model, d_model)
        self.input_linear2 = th.nn.Linear(d_model, d_model)
        # self.money_embedding = th.nn.Embedding(max_money_emb, d_model)
        # self.opponent_money_embedding = th.nn.Embedding(max_money_emb, d_model)
        self.transformer = th.nn.Identity() # For now.
        self.value_linear1 = th.nn.Linear(d_model, d_model)
        self.value_linear2 = th.nn.Linear(d_model, 1)

        self.d_model = d_model

    def process_input(self, obs: th.Tensor):
        # obs is dict of:
        # board: [batch, num_hexes, 3]
        # Extract these tensors, keeping the int type
        board_obs = obs['board']
        assert tuple(board_obs.shape[1:]) == (BOARD_SIZE ** 2, 3), board_obs.shape
        hex_embs = self.hex_embedding(board_obs[:, :, 0])  # [batch, num_hexes, d_model]
        terrain_emb = self.hex_embedding(board_obs[:, :, 1])  # [batch, num_hexes, d_model]
        unit_type_embs = self.unit_embedding(board_obs[:, :, 2])  # [batch, num_hexes, d_model]
        embs = th.cat([hex_embs + terrain_emb + unit_type_embs], dim=1)
        assert tuple(embs.shape[1:]) == (BOARD_SIZE ** 2, self.d_model), embs.shape
        return embs

    def process_output_into_action_logits(self, state, trunk_out):
        my_units = trunk_out[len(state.hexes):len(state.units) + len(state.hexes)]
        hexes = trunk_out[:len(state.hexes)]
        
        # Maybe an extra dense layer on each? Probably not needed.
        action_logits = th.matmul(my_units, hexes)
        return action_logits

    def process_output_into_scalar(self, trunk_out):
        flat, _ = th.max(trunk_out, dim=1)  # [batch, d_model]
        x = self.value_linear1(flat)  # [batch, d_model]
        x = th.nn.ReLU()(x)  # [batch, d_model]
        logit = self.value_linear2(flat)  # [batch, 1]
        return logit

    def forward(self, state):
        obs = self.process_input(state)  # [batch, num_things, d_model]
        obs = self.input_linear1(obs)  # [batch, num_things, d_model]
        obs = th.nn.ReLU()(obs)  # [batch, num_things, d_model]
        obs = self.input_linear2(obs)  # [batch, num_things, d_model]
        # Can skip the transformer entirely at first for simplicity.
        trunk_out = self.transformer(obs)  # [batch, num_things, d_model]
        output = self.process_output_into_scalar(trunk_out)  # [batch, num_units, num_hexes]
        return output

    def save(self, checkpoint_path):
        th.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(th.load(checkpoint_path, map_location=lambda storage, loc: storage))