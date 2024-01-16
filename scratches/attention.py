'''
Multi-headed attention
https://medium.com/@prudhviraju.srivatsavaya/implementing-multiheaded-attention-a321dcb5aab8
https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/973a6c6c63211b6c7ab6fdf50e026e458d1f6e4e/lux_ai/nns/attn_blocks.py
'''
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

        # print(f"dim: {dim} | heads: {heads} | qkv_dim: {dim // heads}")
        self.q = AttnVector(dim, heads, bias)
        self.k = AttnVector(dim, heads, bias)
        self.v = AttnVector(dim, heads, bias)

        self.scale = self.qkv_dim ** -0.5

        self.drop = nn.Dropout(0.1)

        self.final = nn.Linear(dim, 1)

    def forward(self, x, mask):
        query = self.q(x)
        key = self.k(x).mT
        value = self.v(x)

        weights = torch.einsum('bhwnd, bxydn -> bnhwxy', query, key)

        if mask is not None:
            raise NotImplementedError

        weights = self.drop(weights)
        weights *= self.scale
        weights = F.softmax(weights, dim=-1)

        attn = torch.einsum('bnhwxy, bxynd -> bhwnd', weights, value)
        attn = torch.flatten(attn, start_dim=-2, end_dim=-1)

        attn = self.final(attn)

        return attn

if __name__=='__main__':
    '''
    bs = batch size
    h = height
    w = width
    c = channels
    n = heads
    d = embeddings
    '''
    matrix = torch.randint(low=0, high=2, size=[8,6,7,4], dtype=torch.float32)
    # matrix = torch.eye(6,7, dtype=torch.float32).reshape([1,6,7,1]).repeat(8,1,1,4)

    attn_block = AttnBlock(matrix.shape[-1], 2, False)
    out = attn_block(matrix, None).detach()

    plt.figure(1)
    rows = 2
    cols = 4
    index = int(f"{rows}{cols}0")
    for idx in range(out.shape[0]):
        index += 1
        plt.subplot(index)
        plt.imshow(out[idx], cmap='hot', interpolation='nearest')
    plt.show()