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
            proj_channels,
    ):
        super(AttnVector, self).__init__()
        self.heads = heads
        self.w = nn.Parameter(torch.Tensor(in_channels, heads, proj_channels // heads))

        nn.init.normal_(self.w, std=0.01)

    def forward(self, x: torch.Tensor):
        out = torch.einsum('bhwc,cnp->bhwnp', x, self.w)
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

        print(f"dim: {dim} | heads: {heads} | qkv_dim: {self.qkv_dim}")
        self.q = AttnVector(dim, heads, proj_channels=1)
        self.k = AttnVector(dim, heads, proj_channels=1)
        self.v = AttnVector(dim, heads, proj_channels=1)

        self.scale = self.qkv_dim ** -0.5

        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask):
        print(f"embds shape: {x.shape}")
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        print(f"query shape: {query.shape}\nkey shape: {key.shape}\nvalue shape: {value.shape}\n")
        weights = torch.einsum('bhwc,bpqd->bhwpq', query, key)
        print(f"weights shape: {weights.shape}")
        if mask is not None:
            raise NotImplementedError

        weights = self.drop(weights)
        weights *= self.scale
        weights = F.softmax(weights, dim=-1)
        print(f"value shape: {value.shape}")
        attn = weights @ value

        return attn
'''
bs = batch size
h = height
w = width
c = channels
'''
matrix = torch.randint(low=1, high=10, size=[8,6,7,4])
print(f"matrix shape: {matrix.shape}")

attn_block = AttnBlock(matrix.shape[-1], 4)
print(attn_block(matrix, None))