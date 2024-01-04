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
            bias
    ):
        super(AttnVector, self).__init__()
        self.heads = heads
        self.qkv_dim = in_channels // heads

        self.w = nn.Linear(in_channels, in_channels, bias=bias)

        # nn.init.normal_(self.w, std=0.01)

    def forward(self, x: torch.Tensor):
        print(f'---  x:{x.shape}  ---')
        shape = x.shape[:-1]
        out = self.w(x)
        out = out.view(*shape, self.heads, self.qkv_dim)
        print(f'---  out:{out.shape}  ---')
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
        self.q = AttnVector(dim, heads, bias)
        self.k = AttnVector(dim, heads, bias)
        self.v = AttnVector(dim, heads, bias)

        self.scale = self.qkv_dim ** -0.5

        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask):
        query = self.q(x)
        key = self.k(x).mT
        value = self.v(x)
        print(f"query shape: {query.shape}\nkey shape: {key.shape}\nvalue shape: {value.shape}\n")
        weights = torch.einsum('bhwnd, bhwdn -> bnhw', query, key)
        print(f"weights shape: {weights.shape}")
        print(query)
        print(key)
        print(weights)
        if mask is not None:
            raise NotImplementedError

        weights = self.drop(weights)
        weights *= self.scale
        weights = F.softmax(weights, dim=-1)

        attn = torch.einsum('bnhwd, bnxyd -> bnhwxy', weights, value)
        #
        return attn
'''
bs = batch size
h = height
w = width
c = channels
e = embeddings
'''
matrix = torch.randint(low=0, high=2, size=[1,2,2,4], dtype=torch.float32)
attn_block = AttnBlock(matrix.shape[-1], 2, False)
print(attn_block(matrix, None))

# print(f'---  matrix  ---')
# shapes = matrix.shape[:-1]
# matrix = matrix.view(*shapes, 3, 2)
# print(matrix.shape)
# print(matrix)
# print(matrix.mT)
# weights = torch.einsum('bhwnd, bhwnd -> bhwn', matrix, matrix)
# print(weights)
# attn = None
# a = torch.Tensor([[2,4],[3,1],[1,3]])
# b = torch.Tensor([[2,3,4],[3,1,4]])
# print(a, a.shape)
# print(b, b.shape)
# print(torch.einsum('xy,yz->xz', b, a))