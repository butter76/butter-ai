import torch
import torch.nn as nn
from typing import Optional
import yaml

class CausalSelfAttention(nn.Module):
	def __init__(self, config: dict):
		super().__init__()
		assert config['n_embd'] % config['n_head'] == 0
		
		self.n_head = config['n_head']
		self.n_embd = config['n_embd']
		self.dropout = config['dropout']
		
		# key, query, value projections
		self.key = nn.Linear(config['n_embd'], config['n_embd'])
		self.query = nn.Linear(config['n_embd'], config['n_embd'])
		self.value = nn.Linear(config['n_embd'], config['n_embd'])
		
		# output projection
		self.proj = nn.Linear(config['n_embd'], config['n_embd'])
		
		# causal mask
		self.register_buffer("mask", torch.tril(torch.ones(config['seq_length'], config['seq_length'])))
		
		self.dropout1 = nn.Dropout(config['dropout'])
		self.dropout2 = nn.Dropout(config['dropout'])

	def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
		
		# Calculate query, key, values for all heads in batch
		k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # [B, nh, T, hs]
		q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
		v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
		
		# Scaled dot-product attention
		att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))  # [B, nh, T, T]
		
		# Causal mask
		att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
		
		# Padding mask
		if padding_mask is not None:
			padding_mask = padding_mask.view(B, 1, 1, T)  # [B, 1, 1, T]
			att = att.masked_fill(padding_mask == 0, float('-inf'))
		
		att = torch.softmax(att, dim=-1)
		att = self.dropout1(att)
		
		y = att @ v  # [B, nh, T, hs]
		y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
		
		return self.dropout2(self.proj(y))

class Block(nn.Module):
	def __init__(self, config: dict):
		super().__init__()
		self.ln1 = nn.LayerNorm(config['n_embd'])
		self.attn = CausalSelfAttention(config)
		self.ln2 = nn.LayerNorm(config['n_embd'])
		self.mlp = nn.Sequential(
			nn.Linear(config['n_embd'], 4 * config['n_embd']),
			nn.GELU(),
			nn.Linear(4 * config['n_embd'], config['n_embd']),
			nn.Dropout(config['dropout'])
		)

	def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = x + self.attn(self.ln1(x), padding_mask)
		x = x + self.mlp(self.ln2(x))
		return x

class GPT(nn.Module):
	def __init__(self, config_path: str):
		super().__init__()
		
		# Load config
		with open(config_path, 'r') as f:
			config = yaml.safe_load(f)['model']
		
		self.seq_length = config['seq_length']
		
		# Input embedding
		self.tok_emb = nn.Embedding(config['vocab_size'], config['n_embd'])
		self.pos_emb = nn.Parameter(torch.zeros(1, config['seq_length'], config['n_embd']))
		self.drop = nn.Dropout(config['dropout'])
		
		# Transformer blocks
		self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])
		
		# Final layer norm
		self.ln_f = nn.LayerNorm(config['n_embd'])
		
		# Output head
		self.head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

	def forward(self, idx: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		B, T = idx.size()
		
		# Get token embeddings and add positional embeddings
		tok_emb = self.tok_emb(idx)  # [B, T, C]
		pos_emb = self.pos_emb[:, :T, :]  # [1, T, C]
		x = self.drop(tok_emb + pos_emb)
		
		# Apply transformer blocks
		for block in self.blocks:
			x = block(x, padding_mask)
		
		x = self.ln_f(x)
		logits = self.head(x)  # [B, T, vocab_size]
		
		return logits