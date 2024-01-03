'''
Multi-headed attention
https://medium.com/@prudhviraju.srivatsavaya/implementing-multiheaded-attention-a321dcb5aab8
https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/973a6c6c63211b6c7ab6fdf50e026e458d1f6e4e/lux_ai/nns/attn_blocks.py
'''
import torch
from torch import nn
import torch.nn.functional as F

class AttnVector(nn.Module):
    def __init__(
            self,
            in_channels,
            heads,
    ):
        super(AttnVector, self).__init__()
        self.heads = heads

        self.w = nn.Linear(in_channels, in_channels)

        # nn.init.normal_(self.w, std=0.01)

    def forward(self, x: torch.Tensor):
        out = self.w(x)
        return out


class AttnBlock(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias: bool = False
    ):
        super().__init__()

        assert dim % heads == 0, f"dim:{dim} | heads:{heads}"
        self.heads = heads
        self.qkv_dim = dim // heads

        print(f"dim: {dim} | heads: {heads} | qkv_dim: {dim // heads}")
        self.q = AttnVector(dim, heads)
        self.k = AttnVector(dim, heads)
        self.v = AttnVector(dim, heads)

        self.scale = self.qkv_dim ** -0.5

        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask):
        # print(f"embds shape: {x.shape}")
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        print(f"query shape: {query.shape}\nkey shape: {key.shape}\nvalue shape: {value.shape}\n")
        weights = torch.einsum('bhwc,bpqd->bhwpq', query, key)
        # print(f"weights shape: {weights.shape}")
        if mask is not None:
            raise NotImplementedError

        weights = self.drop(weights)
        weights *= self.scale
        weights = F.softmax(weights, dim=-1)
        # print(f"value shape: {value.shape}")
        attn = weights @ value

        return attn
'''
bs = batch size
h = height
w = width
c = channels
e = embeddings
'''
matrix = torch.randint(low=0, high=2, size=[8,6,7,4], dtype=torch.float32)
# emb = nn.Embedding(2,16)
# embs = emb(matrix)
attn_block = AttnBlock(matrix.shape[-1], 4)
print(attn_block(matrix, None))