# Multihead Attention Class for Kasual Ai Model and training


from torch import nn


class KasualAttention(nn.Module):
    def __init__(self):
        super().__init__()
