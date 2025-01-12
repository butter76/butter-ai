import torch
import torch.nn as nn
import math

class GPTConfig:
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.size()
        # [B, T, C]
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # [B, n_head, T, head_size]
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # [B, n_head, T, head_size]
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # [B, n_head, T, head_size]

        # compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # [B, n_head, T, T]

        # causal mask
        causal_mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        # optional additional padding mask from dataset
        if attn_mask is not None:
            # attn_mask shape: [B, 1, 1, T] or [B, 1, T, T]
            # broadcast to [B, n_head, T, T] for multiplication or addition
            att = att + attn_mask

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # [B, n_head, T, head_size]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size

    def forward(self, idx: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length should not exceed block_size"

        token_embeddings = self.token_emb(idx)  # [B, T, n_embd]
        position_embeddings = self.pos_emb[:, :T, :]  # [1, T, n_embd]
        x = self.drop(token_embeddings + position_embeddings)

        # pass the input through transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]
        return logits