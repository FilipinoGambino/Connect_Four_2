'''
Multi-headed attention
https://medium.com/@prudhviraju.srivatsavaya/implementing-multiheaded-attention-a321dcb5aab8
https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/973a6c6c63211b6c7ab6fdf50e026e458d1f6e4e/lux_ai/nns/attn_blocks.py
'''
import torch
from torch import nn
import torch.nn.functional as F


class AttnBlock:
    def __init__(self, emb_dim):
        self.heads = 3
        self.q_dim = emb_dim * self.heads
        self.k_dim = 24
        self.v_dim = 32

        self.q = nn.Parameter(torch.rand(self.heads, self.q_dim, emb_dim))
        self.k = nn.Parameter(torch.rand(self.heads, self.k_dim, emb_dim))
        self.v = nn.Parameter(torch.rand(self.heads, self.v_dim, emb_dim))
        print(f"query shape: {self.q.shape}")
        print(f"key shape: {self.k.shape}")
        print(f"value shape: {self.v.shape}")

    def split_heads(self, inputs):


    def forward(self, x):
        print(f"transposed shape: {x.mT.shape}")
        query = self.q.matmul(x.mT)
        query = F.softmax(query * (float(self.heads) ** -0.5))
        torch.matmul(query, key.transpose(-2, -1))
        k_out = self.q.matmul(q_out)

        v_out = self.q.matmul(x.mT)

        weights = q_out.dot(k_out)

        attn = v_out * weights

        return F.softmax(attn / (self.q_dim ** 1/2), dim=0)

matrix = torch.randint(low=1, high=10, size=[5,10])
print(f"matrix shape: {matrix.shape}")
embedding = nn.Embedding(matrix.shape[-1], 16)
embds = embedding(matrix)
print(f"embds shape: {embds.shape}")
attn_block = AttnBlock(embds.shape[-1])
print(attn_block.forward(embds))