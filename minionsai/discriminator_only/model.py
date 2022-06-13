import torch as th
from ..engine import BOARD_SIZE
from ..unit_type import unitList

class MinionsDiscriminator(th.nn.Module):
    def __init__(self, d_model, depth):
        super().__init__()
        # TODO - get dimension from ObservationEnum objects
        self.unit_embedding = th.nn.Embedding(len(unitList) * 2 + 1, d_model)
        self.location_embedding = th.nn.Embedding(BOARD_SIZE ** 2, d_model)
        self.hex_embedding = th.nn.Embedding(BOARD_SIZE ** 2, d_model)
        self.terrain_embedding = th.nn.Embedding(3, d_model)
        self.remaining_turns_embedding = th.nn.Embedding(21, d_model)
        self.money_embedding = th.nn.Embedding(21, d_model)
        self.opp_money_embedding = th.nn.Embedding(21, d_model)
        self.score_diff_embedding = th.nn.Embedding(41, d_model)

        # TODO We process the board with a conv net to help the model understand adjacency
        # This is ok when everything has speed 1, range 1.
        # But eventually we'll want to upgrade this to relative attention in the transformer
        # Essentially telling each entry in the transformer how far it is from each other entry.
        self.input_conv1 = th.nn.Conv2d(d_model, d_model, 3, padding='same')
        self.conv_ln = th.nn.LayerNorm(d_model)
        self.transformer = th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model, batch_first=True),
            num_layers=depth,
        )
        self.value_linear1 = th.nn.Linear(d_model, d_model)
        self.value_linear2 = th.nn.Linear(d_model, 1)

        self.activation = th.nn.ReLU()

        self.d_model = d_model
        self.depth = depth
        self._device = None

    def to(self, device):
        super().to(device)
        self._device = device

    @property
    def device(self):
        if self._device is None:
            raise ValueError("Didn't tell policy what device to use!")
        return self._device

    def process_input(self, obs: th.Tensor):
        # obs is dict of ints to be embedded:
        #   board: [batch, num_hexes, 3]
        #   remaining_turns, money, etc: each [batch, 1]

        # Extract these tensors, keeping the int type
        obs = {k: th.Tensor(v).int().to(self.device) for k, v in obs.items()}

        board_obs = obs['board']
        assert tuple(board_obs.shape[1:]) == (BOARD_SIZE ** 2, 3), board_obs.shape
        hex_embs = self.hex_embedding(board_obs[:, :, 0])  # [batch, num_hexes, d_model]
        terrain_emb = self.terrain_embedding(board_obs[:, :, 1])  # [batch, num_hexes, d_model]
        unit_type_embs = self.unit_embedding(board_obs[:, :, 2])  # [batch, num_hexes, d_model]
        board_embs = hex_embs + terrain_emb + unit_type_embs # [batch, num_hexes, d_model]
        
        # Reshape to [batch, height, width, channels]
        board_embs_conv = board_embs.view(board_embs.shape[0], BOARD_SIZE, BOARD_SIZE, self.d_model)
        # But the conv wants [batch, channels, height, width]
        board_embs_conv = board_embs_conv.permute(0, 3, 1, 2)
        board_embs_conv = self.input_conv1(board_embs_conv)
        board_embs_conv = board_embs_conv.permute(0, 2, 3, 1)  # [batch, height, width, d_model]
        board_embs_conv = board_embs_conv.view(board_embs_conv.shape[0], BOARD_SIZE ** 2, self.d_model)
        board_embs = board_embs + self.conv_ln(board_embs_conv)

        remaining_turns_emb = self.remaining_turns_embedding(obs['remaining_turns'])
        money_emb = self.money_embedding(obs['money'])
        opp_money_emb = self.opp_money_embedding(obs['opp_money'])
        score_diff_emb = self.score_diff_embedding(obs['score_diff'])
        embs = th.cat([board_embs, remaining_turns_emb, money_emb, opp_money_emb, score_diff_emb], dim=1)
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
        x = self.activation(x)  # [batch, d_model]
        logit = self.value_linear2(flat)  # [batch, 1]
        return logit

    def forward(self, state):
        obs = self.process_input(state)  # [batch, num_things, d_model]
        trunk_out = self.transformer(obs)  # [batch, num_things, d_model]
        output = self.process_output_into_scalar(trunk_out)  # [batch, 1]
        return output

    def save(self, checkpoint_path):
        th.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(th.load(checkpoint_path, map_location=lambda storage, loc: storage))