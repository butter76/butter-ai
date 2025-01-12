from typing import Optional
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.n_head = config['n_head']
		self.n_embd = config['n_embd']
		assert self.n_embd % self.n_head == 0
		
		self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
		self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
		self.attn_dropout = nn.Dropout(config['dropout'])
		self.resid_dropout = nn.Dropout(config['dropout'])
		
	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
		
		# Calculate query, key, values for all heads in batch
		q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # [B, T, C]
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
		
		# Scaled dot-product attention
		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B, nh, T, T]
		if mask is not None:
			att = att.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
		att = F.softmax(att, dim=-1)
		att = self.attn_dropout(att)
		y = att @ v  # [B, nh, T, hs]
		
		# Re-assemble all head outputs side by side
		y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
		
		# Output projection
		y = self.resid_dropout(self.c_proj(y))
		return y

class Block(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.ln_1 = nn.LayerNorm(config['n_embd'])
		self.attn = MultiHeadAttention(config)
		self.ln_2 = nn.LayerNorm(config['n_embd'])
		self.mlp = nn.Sequential(
			nn.Linear(config['n_embd'], 4 * config['n_embd']),
			nn.GELU(),
			nn.Linear(4 * config['n_embd'], config['n_embd']),
			nn.Dropout(config['dropout']),
		)

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = x + self.attn(self.ln_1(x), mask)
		x = x + self.mlp(self.ln_2(x))
		return x

class GPT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.max_seq_len = config['max_seq_len']
		
		self.tok_emb = nn.Embedding(config['vocab_size'], config['n_embd'])
		self.pos_emb = nn.Parameter(torch.zeros(1, config['max_seq_len'], config['n_embd']))
		self.drop = nn.Dropout(config['dropout'])
		
		self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])
		self.ln_f = nn.LayerNorm(config['n_embd'])
		self.head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
		
		# Initialize weights
		self.apply(self._init_weights)
		
	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if isinstance(module, nn.Linear) and module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.LayerNorm):
			torch.nn.init.zeros_(module.bias)
			torch.nn.init.ones_(module.weight)

	def forward(self, idx: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		B, T = idx.size()
		assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max len is {self.max_seq_len}"
		
		# Get token embeddings and add positional embeddings
		tok_emb = self.tok_emb(idx)  # [B, T, C]
		pos_emb = self.pos_emb[:, :T, :]  # [1, T, C]
		x = self.drop(tok_emb + pos_emb)  # [B, T, C]
		
		# Apply transformer blocks
		for block in self.blocks:
			x = block(x, mask)
		
		x = self.ln_f(x)  # [B, T, C]
		logits = self.head(x)  # [B, T, vocab_size]
		
		return logits