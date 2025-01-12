import os
import glob
from typing import Optional
import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from gpt.models.tokenizer import LoveLetterTokenizer
from gpt.models.gpt_model import GPT

class LoveLetterDataset(Dataset):
	def __init__(self, data_dir: str, tokenizer: LoveLetterTokenizer, max_seq_len: int):
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.examples = []
		
		# Load all log files
		log_files = glob.glob(os.path.join(data_dir, "*.log"))
		for file_path in tqdm(log_files, desc="Loading data"):
			with open(file_path, 'r') as f:
				log_content = f.read()
				tokens = self.tokenizer.tokenize(log_content)
				if len(tokens) > 1:  # Ensure we have at least input and target
					self.examples.append(tokens)
	
	def __len__(self) -> int:
		return len(self.examples)
	
	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		tokens = self.examples[idx]
		
		# Calculate actual sequence length
		seq_len = min(len(tokens), self.max_seq_len)
		
		# Create left-padded sequence
		if len(tokens) >= self.max_seq_len:
			# Take last max_seq_len tokens
			tokens = tokens[-self.max_seq_len:]
			# All positions are valid (no padding)
			attention_mask = torch.ones(self.max_seq_len - 1, dtype=torch.bool)
		else:
			# Add padding to the left
			pad_length = self.max_seq_len - len(tokens)
			tokens = [self.tokenizer.special_tokens['PAD']] * pad_length + tokens
			# Create attention mask (False for padding, True for actual tokens)
			attention_mask = torch.zeros(self.max_seq_len - 1, dtype=torch.bool)
			attention_mask[-len(tokens)+1:] = True  # -1 because we drop last token for target
		
		# Create input and target sequences
		x = torch.tensor(tokens[:-1], dtype=torch.long)  # [seq_len-1]
		y = torch.tensor(tokens[1:], dtype=torch.long)   # [seq_len-1]
		
		return x, y, attention_mask

def train():
	# Load config
	with open('config/model_config.yaml', 'r') as f:
		config = yaml.safe_load(f)
	
	# Initialize tokenizer and model
	tokenizer = LoveLetterTokenizer()
	model = GPT(config['model'])
	
	# Setup device and move model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	
	# Create dataset and dataloader
	dataset = LoveLetterDataset(
		config['data']['logs_dir'],
		tokenizer,
		config['model']['max_seq_len']
	)
	dataloader = DataLoader(
		dataset,
		batch_size=config['training']['batch_size'],
		shuffle=True,
		num_workers=4
	)
	
	# Setup optimizer
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=config['training']['learning_rate'],
		weight_decay=config['training']['weight_decay']
	)
	
	# Training loop
	for epoch in range(config['training']['max_epochs']):
		model.train()
		total_loss = 0
		progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
		
		for batch_idx, (x, y, attention_mask) in enumerate(progress_bar):
			# Move batch to device
			x = x.to(device)                    # [batch_size, seq_len]
			y = y.to(device)                    # [batch_size, seq_len]
			attention_mask = attention_mask.to(device)  # [batch_size, seq_len]
			
			# Forward pass
			logits = model(x, attention_mask)   # [batch_size, seq_len, vocab_size]

			
			# Calculate loss only on non-padding tokens
			loss = F.cross_entropy(
				logits.reshape(-1, logits.size(-1)),
				y.reshape(-1),
				ignore_index=tokenizer.special_tokens['PAD'],
				reduction='mean'
			)
			
			# Backward pass
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
			optimizer.step()
			
			# Update progress
			total_loss += loss.item()
			progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
		
		# Save checkpoint
		checkpoint_path = f"gpt/checkpoints/model_epoch_{epoch+1}.pt"
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': total_loss / len(dataloader),
		}, checkpoint_path)

if __name__ == "__main__":
	train()