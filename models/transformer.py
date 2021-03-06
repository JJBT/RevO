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
        att = att.masked_fill(mask == 0, float('-inf')).clone()
        att = F.softmax(att, dim=-1)
        att = torch.where(torch.isnan(att), torch.tensor(0.), att)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, embd_dim, n_head, resid_pdrop):
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop

        super().__init__()
        self.ln1 = nn.LayerNorm(self.embd_dim)
        self.ln2 = nn.LayerNorm(self.embd_dim)
        self.attn = NarrowSelfAttention(self.embd_dim, self.n_head)
        self.mlp = nn.Sequential(
            nn.Linear(self.embd_dim, 4 * self.embd_dim),
            nn.GELU(),
            nn.Linear(4 * self.embd_dim, self.embd_dim),
            nn.Dropout(self.resid_pdrop),
        )

    def forward(self, input):
        x, mask = input['x'], input['mask']
        x_ = self.ln1(x)
        x = x + self.attn({'x': x_, 'mask': mask})
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, embd_dim, n_head, attn_pdrop, resid_pdrop, embd_pdrop, n_layer, out_dim):
        super().__init__()
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.n_layer = n_layer
        self.out_dim = out_dim

        self.drop = nn.Dropout(self.embd_dim)
        self.blocks = nn.Sequential(*[Block(self.embd_dim, self.n_head, self.resid_pdrop) for _ in range(self.n_layer)])

        self.ln_f = nn.LayerNorm(self.embd_dim)
        self.head = nn.Linear(self.embd_dim, self.out_dim, bias=False)

    def forward(self, input):
        x, mask = input['x'], input['mask']
        x = self.drop(x)
        x = self.blocks({'x': x, 'mask': mask})
        x = self.ln_f(x)
        x = self.head(x)

        return x
