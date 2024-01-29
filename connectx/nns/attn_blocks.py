from typing import Callable, Tuple

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

    def forward(self, x: torch.Tensor):
        x = x.permute([0,2,3,1])
        shape = x.shape[:-1]

        out = self.w(x)

        out = out.view(*shape, self.heads, self.qkv_dim)
        return out


class MHABlock(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias: bool = False
    ):
        super(MHABlock, self).__init__()

        assert dim % heads == 0, f"dim:{dim} | heads:{heads}"
        self.heads = heads
        self.qkv_dim = dim // heads

        self.q = AttnVector(dim, heads, bias)
        self.k = AttnVector(dim, heads, bias)
        self.v = AttnVector(dim, heads, bias)

        self.scale = self.qkv_dim ** -0.5

        self.drop = nn.Dropout(0.1)

        self.merge_fc = nn.Linear(dim, 1)

    def forward(self, x):
        query = self.q(x)
        key = self.k(x).mT
        value = self.v(x)

        weights = torch.einsum('bhwnd, bxydn -> bnhwxy', query, key)

        weights = self.drop(weights)
        weights *= self.scale
        weights = F.softmax(weights, dim=-1)

        attn = torch.einsum('bnhwxy, bxynd -> bhwnd', weights, value)
        attn = torch.flatten(attn, start_dim=-2, end_dim=-1).permute([0,3,1,2])
        # attn = self.merge_fc(attn)

        return attn

class ViTBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            mhsa_layer: nn.Module = MHABlock,
            normalize: bool = True,
            activation: Callable = nn.GELU
    ):
        super(ViTBlock, self).__init__()

        self.norm1 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.mhsa = mhsa_layer

        self.norm2 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1)
            ),
            activation(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.mhsa(self.norm1(x))
        x = x + identity
        return self.mlp(self.norm2(x)) + x


# if __name__=="__main__":
#     import matplotlib.pyplot as plt
#     '''
#     bs = batch size
#     h = height
#     w = width
#     c = channels
#     n = heads
#     d = embeddings
#     '''
#     # matrix = torch.randint(low=0, high=2, size=[8, 6, 7, 4], dtype=torch.float32)
#     matrix = torch.eye(6,7, dtype=torch.float32).reshape([1,6,7,1]).repeat(8,1,1,4)
#
#     attn_block = MHABlock(matrix.shape[-1], 2, False)
#     attention = attn_block(matrix, None).detach()
#
#     plt.figure(1)
#     rows = 2
#     cols = 4
#     index = int(f"{rows}{cols}0")
#     for idx in range(attention.shape[0]):
#         index += 1
#         plt.subplot(index)
#         plt.imshow(attention[idx], cmap='hot', interpolation='nearest')
#     plt.show()