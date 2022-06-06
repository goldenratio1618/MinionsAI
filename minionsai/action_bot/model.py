from minionsai.discriminator_only.model import MinionsDiscriminator

class MinionsActionBot(MinionsDiscriminator):
    def __init__(self, d_model, depth):
        super.__init__(d_model, depth)
        self.unit_embedding = th.nn.Embedding(len(unitList) * 2 * 2 * 2 * 7 + 1, d_model)
        self.value_linear2 = th.nn.Linear(d_model, 625)
