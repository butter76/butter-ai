import torch

from gpt.models.config_types import Config
torch.set_float32_matmul_precision('high')
torch.set_printoptions(profile="full")
from torch.utils.data import DataLoader
from gpt.models.dataset import LoveLetterDataset
from gpt.models.gpt_ll import LoveLetterTransformer
from gpt.models.tokenizer import LoveLetterTokenizer
from gpt.models.cli import get_validate_parser, load_config, update_config_with_args
from tqdm import tqdm

def validate(config: Config):	
	model_config = config['model']
	data_config = config['data']
	training_config = config['training']
	generation_config = config['generation']
	
	# Setup device
	device = torch.device(model_config['device'] if torch.cuda.is_available() else "cpu")
	
	# Initialize tokenizer and dataset
	tokenizer = LoveLetterTokenizer()
	dataset = LoveLetterDataset(
		tokenizer=tokenizer,
		data_config=data_config,
		model_config=model_config
	)

	# Create dataloader
	dataloader = DataLoader(
		dataset,
		batch_size=training_config['batch_size'],
		shuffle=False,
		num_workers=training_config['num_workers'],
		pin_memory=True,
		prefetch_factor=training_config['prefetch_factor']
	)

	# Load checkpoint
	checkpoint = torch.load(generation_config['checkpoint_path'])

	# Initialize model
	model = LoveLetterTransformer(
		vocab_size=tokenizer.vocab_size,
		model_config=checkpoint['model_config'],
	).to(device)
	model.load_state_dict(checkpoint['model_state_dict'])
	print(f"Loaded model from checkpoint: {generation_config['checkpoint_path']}")
	
	# Validation loop
	model.eval()
	total_loss = 0
	total_tokens = 0
	
	with torch.inference_mode():
		pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating')
		for batch_idx, (x, y) in pbar:
			x, y = x.to(device), y.to(device)
			
			# Forward pass
			logits = model.get_policy(x)
			
			# Compute loss
			loss = torch.nn.functional.cross_entropy(
				logits.view(-1, logits.size(-1)),
				y.view(-1),
				ignore_index=tokenizer.special_tokens['PAD'],
				reduction='sum'
			)
			
			# Count non-padding tokens
			non_pad_mask = (y != tokenizer.special_tokens['PAD'])
			num_tokens = non_pad_mask.sum().item()
			
			total_loss += loss.item()
			total_tokens += num_tokens
			
			# Update progress bar
			avg_loss = total_loss / total_tokens
			pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
	
	# Final metrics
	avg_loss = total_loss / total_tokens
	perplexity = torch.exp(torch.tensor(avg_loss)).item()
	
	print("\nValidation Results:")
	print(f"Average Loss: {avg_loss:.4f}")
	print(f"Perplexity: {perplexity:.4f}")
	print(f"Total Tokens: {total_tokens}")

if __name__ == "__main__":
	parser = get_validate_parser()
	args = parser.parse_args()
	config = load_config(args.config)
	config = update_config_with_args(config, args)
	validate(config)