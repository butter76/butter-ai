import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from gpt.models.tokenizer import LoveLetterTokenizer
from gpt.models.dataset import LoveLetterDataset
from gpt.models.model import GPT
import yaml
from typing import Optional
import argparse
from tqdm import tqdm

def train(
	model: nn.Module,
	train_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
	device: torch.device,
	config: dict
) -> float:
	model.train()
	total_loss = 0
	
	progress_bar = tqdm(train_loader, desc="Training")
	for batch_idx, (x, y) in enumerate(progress_bar):
		x, y = x.to(device), y.to(device)
		
		# Create padding mask
		padding_mask = (x != 0).float()  # 0 is PAD token
		
		# Forward pass
		logits = model(x, padding_mask)
		
		# Calculate loss (ignore padding tokens)
		loss = nn.functional.cross_entropy(
			logits.view(-1, logits.size(-1)),
			y.view(-1),
			ignore_index=0  # ignore PAD token
		)
		
		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])
		optimizer.step()
		
		if scheduler is not None:
			scheduler.step()
		
		total_loss += loss.item()
		progress_bar.set_postfix({'loss': loss.item()})
	
	return total_loss / len(train_loader)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='config/model_config.yaml')
	parser.add_argument('--logs_dir', type=str, default='../../../love-letter-logs/logs/')
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
	args = parser.parse_args()
	
	# Load config
	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)
	
	# Setup device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Create tokenizer and dataset
	tokenizer = LoveLetterTokenizer()
	dataset = LoveLetterDataset(args.logs_dir, tokenizer, args.config)
	
	# Create data loader
	train_loader = DataLoader(
		dataset,
		batch_size=config['training']['batch_size'],
		shuffle=True,
		num_workers=4
	)
	
	# Create model
	model = GPT(args.config).to(device)
	
	# Create optimizer
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=config['training']['learning_rate'],
		weight_decay=config['training']['weight_decay']
	)
	
	# Create learning rate scheduler
	scheduler = torch.optim.lr_scheduler.LinearLR(
		optimizer,
		start_factor=1.0,
		end_factor=0.0,
		total_iters=config['training']['warmup_steps']
	)
	
	# Create checkpoint directory
	os.makedirs(args.checkpoint_dir, exist_ok=True)
	
	# Training loop
	for epoch in range(config['training']['max_epochs']):
		loss = train(model, train_loader, optimizer, scheduler, device, config)
		print(f'Epoch {epoch + 1}/{config["training"]["max_epochs"]}, Loss: {loss:.4f}')
		
		# Save checkpoint
		if (epoch + 1) % 10 == 0:
			checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
			torch.save({
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss,
			}, checkpoint_path)

if __name__ == '__main__':
	main()