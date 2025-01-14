import argparse
from typing import TypedDict
import yaml
from pathlib import Path
from .config_types import ModelConfig, DataConfig, TrainingConfig, Config

def load_config(config_path: str) -> Config:
	"""Load and parse the YAML config file."""
	with open(config_path, 'r') as f:
		return yaml.safe_load(f)

def get_base_parser() -> argparse.ArgumentParser:
	"""Create base argument parser with shared arguments."""
	parser = argparse.ArgumentParser(description='Love Letter GPT Model')
	parser.add_argument(
		'--config', 
		type=str,
		default='gpt/config/model_config.yaml',
		help='Path to config file'
	)
	
	# Model architecture arguments
	parser.add_argument(
		'--d-model',
		type=int,
		help='Dimension of the model'
	)
	parser.add_argument(
		'--nhead',
		type=int,
		help='Number of attention heads'
	)
	parser.add_argument(
		'--num-layers',
		type=int,
		help='Number of transformer layers'
	)
	parser.add_argument(
		'--seq-length',
		type=int,
		help='Maximum sequence length'
	)
	parser.add_argument(
		'--dropout',
		type=float,
		help='Dropout rate'
	)
	parser.add_argument(
		'--device',
		type=str,
		choices=['cuda', 'cpu'],
		help='Device to run the model on'
	)
	return parser

def get_training_parser() -> argparse.ArgumentParser:
	"""Create argument parser for training with specific training arguments."""
	parser = get_base_parser()
	
	# Training specific arguments
	parser.add_argument(
		'--batch-size', 
		type=int,
		help='Override batch size from config'
	)
	parser.add_argument(
		'--epochs', 
		type=int,
		help='Override number of epochs from config'
	)
	parser.add_argument(
		'--learning-rate', 
		type=float,
		help='Override learning rate from config'
	)
	parser.add_argument(
		'--checkpoint', 
		type=str,
		help='Override checkpoint path from config'
	)
	parser.add_argument(
		'--save-every',
		type=int,
		help='Override checkpoint saving frequency (epochs)'
	)
	parser.add_argument(
		'--num-workers',
		type=int,
		help='Override number of workers for data loading'
	)
	parser.add_argument(
		'--prefetch-factor',
		type=int,
		help='Override prefetch factor for data loading'
	)
	
	# Data specific arguments
	parser.add_argument(
		'--data-dir',
		type=str,
		help='Override data directory path'
	)
	parser.add_argument(
		'--data-type',
		type=str,
		choices=['mixed', 'pov', 'full'],
		help='Override data type (mixed or pov or full)'
	)
	parser.add_argument(
		'--train-split',
		type=float,
		help='Override train split ratio (0-1)'
	)
	parser.add_argument(
		'--val-split',
		type=float,
		help='Override validation split ratio (0-1)'
	)
	parser.add_argument(
		'--max-logs',
		type=int,
		help='Override maximum number of logs to process'
	)
	return parser

def get_generate_parser() -> argparse.ArgumentParser:
	"""Create argument parser for generation with specific generation arguments."""
	parser = get_base_parser()
	parser.add_argument(
		'--checkpoint',
		type=str,
		help='Path to model checkpoint'
	)
	parser.add_argument(
		'--temperature',
		type=float,
		default=1.0,
		help='Sampling temperature'
	)
	parser.add_argument(
		'--max-tokens',
		type=int,
		default=30,
		help='Maximum number of tokens to generate'
	)
	parser.add_argument(
		'--prompt',
		type=str,
		default='|gamestart\n|p1|hidden|draw|1',
		help='Starting prompt for generation'
	)
	return parser

def update_config_with_args(config: Config, args: argparse.Namespace) -> Config:
	"""Update config with command line arguments if they are provided."""
	# Create a new config to avoid modifying the original
	updated_config = Config(
		model=config['model'],
		data=config['data'],
		training=config['training'],
		generation=config['generation']
	)
	
	# Model architecture updates
	if hasattr(args, 'd_model') and args.d_model is not None:
		updated_config['model']['d_model'] = args.d_model
	if hasattr(args, 'nhead') and args.nhead is not None:
		updated_config['model']['nhead'] = args.nhead
	if hasattr(args, 'num_layers') and args.num_layers is not None:
		updated_config['model']['num_layers'] = args.num_layers
	if hasattr(args, 'seq_length') and args.seq_length is not None:
		updated_config['model']['seq_length'] = args.seq_length
	if hasattr(args, 'dropout') and args.dropout is not None:
		updated_config['model']['dropout'] = args.dropout
	if hasattr(args, 'device') and args.device is not None:
		updated_config['model']['device'] = args.device
	
	# Training specific updates
	if hasattr(args, 'batch_size') and args.batch_size is not None:
		updated_config['training']['batch_size'] = args.batch_size
	if hasattr(args, 'epochs') and args.epochs is not None:
		updated_config['training']['epochs'] = args.epochs
	if hasattr(args, 'learning_rate') and args.learning_rate is not None:
		updated_config['training']['learning_rate'] = args.learning_rate
	if hasattr(args, 'checkpoint') and args.checkpoint is not None:
		updated_config['training']['checkpoint_path'] = args.checkpoint
	if hasattr(args, 'save_every') and args.save_every is not None:
		updated_config['training']['save_every'] = args.save_every
	if hasattr(args, 'num_workers') and args.num_workers is not None:
		updated_config['training']['num_workers'] = args.num_workers
	if hasattr(args, 'prefetch_factor') and args.prefetch_factor is not None:
		updated_config['training']['prefetch_factor'] = args.prefetch_factor
	
	# Data specific updates
	if hasattr(args, 'data_dir') and args.data_dir is not None:
		updated_config['data']['data_dir'] = args.data_dir
	if hasattr(args, 'data_type') and args.data_type is not None:
		updated_config['data']['type'] = args.data_type
	if hasattr(args, 'train_split') and args.train_split is not None:
		updated_config['data']['train_split'] = args.train_split
	if hasattr(args, 'val_split') and args.val_split is not None:
		updated_config['data']['val_split'] = args.val_split
	if hasattr(args, 'max_logs') and args.max_logs is not None:
		updated_config['data']['max_logs'] = args.max_logs
	
	return updated_config