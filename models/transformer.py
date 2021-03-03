import math
import torch
from torch import nn
from torch.nn import functional as F

EMBD_DIM = 2048
N_HEAD = 4
attn_pdrop = 0.
resid_pdrop = 0.
EMBD_PDROP = 0.
N_LAYER = 1
OUT_DIM = 1


class NarrowSelfAttention(nn.Module):
    def __init__(self, embd_dim, n_head):
        super().__init__()
        assert embd_dim % n_head == 0

        self.n_head = n_head

        self.key = nn.Linear(embd_dim, embd_dim)
        self.query = nn.Linear(embd_dim, embd_dim)
        self.value = nn.Linear(embd_dim, embd_dim)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(embd_dim, embd_dim)

    def forward(self, input):
        x, mask = input['x'], input['mask']
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = mask.unsqueeze(1)  # (B, T, T) -> (B, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att[att != att] = 0.
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBD_DIM)
        self.ln2 = nn.LayerNorm(EMBD_DIM)
        self.attn = NarrowSelfAttention(EMBD_DIM, N_HEAD)
        self.mlp = nn.Sequential(
            nn.Linear(EMBD_DIM, 4 * EMBD_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBD_DIM, EMBD_DIM),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, input):
        x, mask = input['x'], input['mask']
        x_ = self.ln1(x)
        x = x + self.attn({'x': x_, 'mask': mask})
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.drop = nn.Dropout(EMBD_PDROP)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])

        self.ln_f = nn.LayerNorm(EMBD_DIM)
        self.head = nn.Linear(EMBD_DIM, OUT_DIM, bias=False)

    def forward(self, input):
        x, mask = input['x'], input['mask']
        x = self.drop(x)
        x = self.blocks({'x': x, 'mask': mask})
        x = self.ln_f(x)
        x = self.head(x)

        return x
