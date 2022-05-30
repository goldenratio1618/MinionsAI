from discriminator_only.translator import HEXES
import torch as th
from MinionsAI.engine import BOARD_SIZE, unitList



class MinionsDiscriminator(th.nn.Module):
    def __init__(self, d_model, board_size=10):
        self.unit_embedding = th.nn.Embedding(len(unitList) * 2, d_model)
        self.location_embedding = th.nn.Embedding(board_size ** 2, d_model)
        self.hex_embedding = th.nn.Embedding(board_size ** 2, d_model)
        self.terrain_embedding = th.nn.Embedding(3, d_model)
        # self.money_embedding = th.nn.Embedding(max_money_emb, d_model)
        # self.opponent_money_embedding = th.nn.Embedding(max_money_emb, d_model)
        self.transformer = th.nn.Identity() # For now.
        self.value_linear = th.nn.Linear(d_model, 1)


    def process_input(self, obs):
        # obs is dict of:
        # hexes: [batch, num_hexes, 1]
        # units: [batch, num_units, 2]
        hex_embs = self.hex_embedding(obs['hexes'][:, :, 0])  # [batch, num_hexes, d_model]
        terrain_emb = self.hex_embedding(obs['hexes'][:, :, 1])  # [batch, num_hexes, d_model]
        unit_type_embs = self.unit_embedding(obs['units'][:, :, 0])  # [batch, num_units, d_model]
        unit_location_embs = self.location_embedding(obs['units'][:, :, 1])  # [batch, num_units, d_model]
        embs = th.cat([hex_embs + terrain_emb, unit_type_embs + unit_location_embs], dim=1)
        return embs

    def process_output_into_action_logits(self, state, trunk_out):
        my_units = trunk_out[len(state.hexes):len(state.units) + len(state.hexes)]
        hexes = trunk_out[:len(state.hexes)]
        
        # Maybe an extra dense layer on each? Probably not needed.
        action_logits = th.matmul(my_units, hexes)
        return action_logits

    def process_output_into_scalar(self, trunk_out):
        flat = th.max(trunk_out, dim=1)  # [batch, d_model]
        logit = self.value_linear(flat)  # [batch, 1]
        return logit

    def forward(self, state):
        obs = self.process_input(state)  # [batch, num_things, d_model]
        # Can skip the transformer entirely at first for simplicity.
        trunk_out = self.transformer(obs)  # [batch, num_things, d_model]
        output = self.process_output(state, trunk_out)  # [batch, num_units, num_hexes]
        return output