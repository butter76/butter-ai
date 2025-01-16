import os
import torch

from gpt.models.config_types import Config
torch.set_float32_matmul_precision('high')
torch.set_printoptions(profile="full")
from torch.utils.data import DataLoader
from gpt.models.dataset import LoveLetterDataset
from gpt.models.gpt_ll import LoveLetterTransformer
from gpt.models.tokenizer import LoveLetterTokenizer
from gpt.models.cli import get_training_parser, load_config, update_config_with_args
from tqdm import tqdm

def train(config: Config):
    # Load and update config with CLI arguments
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']

    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and dataset
    tokenizer = LoveLetterTokenizer()
    dataset = LoveLetterDataset(
        tokenizer=tokenizer,
        data_config=data_config,
        model_config=model_config
    )

    # Create train/val split
    total_size = len(dataset)
    train_size = int(total_size * data_config['train_split'])
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create train and val dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        pin_memory=True,
        prefetch_factor=training_config['prefetch_factor']
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=True,
        prefetch_factor=training_config['prefetch_factor']
    )

    # Load from checkpoint if it exists
    checkpoint_path = training_config.get('checkpoint_path')
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        # Create model that matches the checkpoint
        model = LoveLetterTransformer(
            vocab_size=tokenizer.vocab_size,
            model_config=checkpoint['model_config'],
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        # Initialize model
        model = LoveLetterTransformer(
            vocab_size=tokenizer.vocab_size,
            model_config=model_config,
        ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    # After creating the optimizer, add the scheduler:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_config['learning_rate'],
        epochs=training_config['epochs'],
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3,  # Spend 30% of training time in warmup
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1e4  # Final lr = initial_lr/10000
    )

    
    # Training loop
    for epoch in range(1, training_config['epochs'] + 1):
        model.train()
        total_loss = 0
        total_tokens = 0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch}')
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update totals
            total_loss += loss.item()
            total_tokens += num_tokens
            
            # Update progress bar
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Log epoch metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        print({
            "epoch_loss": avg_loss,
            "epoch": epoch,
            "total_tokens": total_tokens
        })
        
        # Save checkpoint and run validation
        if epoch % training_config['save_every'] == 0:
            model.eval()
            val_loss = 0
            val_tokens = 0
            
            with torch.inference_mode():
                val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Validation Epoch {epoch}')
                for batch_idx, (x, y) in val_pbar:
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
                    
                    val_loss += loss.item()
                    val_tokens += num_tokens
                    
                    # Update progress bar
                    avg_val_loss = val_loss / val_tokens if val_tokens > 0 else 0
                    val_pbar.set_postfix({'avg_val_loss': f'{avg_val_loss:.4f}'})

                avg_val_loss = val_loss / val_tokens if val_tokens > 0 else 0
                perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
                print({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss,
                    "val_perplexity": perplexity,
                    "train_tokens": total_tokens,
                    "val_tokens": val_tokens
                })
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'model_config': model.config,
            }, f"gpt/checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = update_config_with_args(config, args)
    train(config)