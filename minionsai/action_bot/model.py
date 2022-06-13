from minionsai.discriminator_only.model import MinionsDiscriminator

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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


class MinionsActionBot(MinionsDiscriminator):
    def __init__(self, d_model, depth, n_actions):
        super.__init__(d_model, depth)
        self.unit_embedding = th.nn.Embedding(len(unitList) * 2 * 2 * 2 * 7 + 1, d_model)
        self.unit_embedding = th.nn.Embedding(len(unitList) * 2 + 1, d_model)
        self.location_embedding = th.nn.Embedding(BOARD_SIZE ** 2, d_model)
        self.hex_embedding = th.nn.Embedding(BOARD_SIZE ** 2, d_model)
        self.phase_embedding = th.nn.Embedding(1, d_model)
        self.terrain_embedding = th.nn.Embedding(3, d_model)
        self.remaining_turns_embedding = th.nn.Embedding(21, d_model)
        self.money_embedding = th.nn.Embedding(21, d_model)
        self.opp_money_embedding = th.nn.Embedding(21, d_model)
        self.score_diff_embedding = th.nn.Embedding(41, d_model)
        self.input_linear1 = th.nn.Linear(d_model, d_model)
        self.input_linear2 = th.nn.Linear(d_model, d_model)
        # self.money_embedding = th.nn.Embedding(max_money_emb, d_model)
        # self.opponent_money_embedding = th.nn.Embedding(max_money_emb, d_model)
        self.transformer = th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model, batch_first=True),
            num_layers=depth,
        )
        self.value_linear1 = th.nn.Linear(d_model, d_model)
        self.value_linear2 = th.nn.Linear(d_model, n_actions)

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
        embs = th.cat([hex_embs + terrain_emb + unit_type_embs], dim=1)
        assert tuple(embs.shape[1:]) == (BOARD_SIZE ** 2, self.d_model), embs.shape
        phase_emb = self.phase_embedding(obs['phase'])
        remaining_turns_emb = self.remaining_turns_embedding(obs['remaining_turns'])
        money_emb = self.money_embedding(obs['money'])
        opp_money_emb = self.opp_money_embedding(obs['opp_money'])
        score_diff_emb = self.score_diff_embedding(obs['score_diff'])
        embs = th.cat([embs, phase_emb, remaining_turns_emb, money_emb, opp_money_emb, score_diff_emb], dim=1)
        return embs

    def process_output_into_scalar(self, trunk_out):
        # print(trunk_out)
        flat, _ = th.max(trunk_out, dim=1)
        # print(flat)
        # flat = trunk_out
        # flat, _ = th.max(trunk_out, dim=1)  # [batch, d_model]
        x = self.value_linear1(flat)  # [batch, d_model]
        x = th.nn.ReLU()(x)  # [batch, d_model]
        logit = self.value_linear2(flat)  # [batch, n_actions]
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
        output = self.process_output_into_scalar(trunk_out)  # [batch, n_actions]
        # print(output)
        return output

    def save(self, checkpoint_path):
        th.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(th.load(checkpoint_path, map_location=lambda storage, loc: storage))
