import os
import torch
torch.set_float32_matmul_precision('high')
torch.set_printoptions(profile="full")
from torch.utils.data import DataLoader
from gpt.models.dataset import LoveLetterDataset
from gpt.models.gpt_ll import LoveLetterTransformer
from gpt.models.tokenizer import LoveLetterTokenizer
from gpt.models.cli import get_training_parser, load_config, update_config_with_args
from tqdm import tqdm

def train(args):
    # Load and update config with CLI arguments
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
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
    
    # Initialize model
    model = LoveLetterTransformer(
        vocab_size=tokenizer.vocab_size,
        model_config=model_config,
    ).to(device)

    # Load from checkpoint if it exists
    checkpoint_path = training_config.get('checkpoint_path')
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    
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
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch}')
        for batch_idx, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=tokenizer.special_tokens['PAD']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Inside your training loop, after optimizer.step():
            scheduler.step()

            # Update your logging to track the learning rate:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            
            total_loss += loss.item()
        
        # Log epoch metrics
        avg_loss = total_loss / len(train_dataloader)
        print({
            "epoch_loss": avg_loss,
            "epoch": epoch
        })
        
        # Save checkpoint
        if epoch % training_config['save_every'] == 0:
            # Add validation loop after training loop and before checkpoint saving
            model.eval()
            val_loss = 0
            with torch.no_grad():
                val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Validation Epoch {epoch}')
                for batch_idx, (x, y) in val_pbar:
                    x, y = x.to(device), y.to(device)
                    
                    # Forward pass
                    logits = model(x)
                    # Compute loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=tokenizer.special_tokens['PAD']
                    )
                    
                    val_loss += loss.item()
                    val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

                avg_val_loss = val_loss / len(val_dataloader)
                print({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss
                })
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"gpt/checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()
    train(args)