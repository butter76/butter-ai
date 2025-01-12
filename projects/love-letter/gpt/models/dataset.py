import torch
from torch.utils.data import Dataset
import os
from typing import List, Tuple
from .tokenizer import LoveLetterTokenizer
import yaml

class LoveLetterDataset(Dataset):
	def __init__(self, data_dir: str, tokenizer: LoveLetterTokenizer, config_path: str):
		self.tokenizer = tokenizer
		
		# Load config
		with open(config_path, 'r') as f:
			config = yaml.safe_load(f)
		self.seq_length = config['model']['seq_length']
		
		# Load all log files and tokenize them
		self.examples: List[List[int]] = []
		for filename in os.listdir(data_dir):
			if filename.endswith('.log'):
				with open(os.path.join(data_dir, filename), 'r') as f:
					text = f.read()
					tokens = self.tokenizer.tokenize(text)
					if tokens:  # Only add non-empty sequences
						self.examples.append(tokens)

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		tokens = self.examples[idx]
		
		# Create input and target sequences
		# Left pad the sequence with PAD tokens
		if len(tokens) > self.seq_length:
			tokens = tokens[:self.seq_length]
		
		pad_length = self.seq_length - len(tokens)
		padded_tokens = [self.tokenizer.special_tokens['PAD']] * pad_length + tokens
		
		# Input is all tokens except last, target is all tokens except first
		x = torch.tensor(padded_tokens[:-1], dtype=torch.long)
		y = torch.tensor(padded_tokens[1:], dtype=torch.long)
		
		return x, y

	def get_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
		"""Create padding mask where 1 indicates non-pad tokens"""
		return (tokens != self.tokenizer.special_tokens['PAD']).float()