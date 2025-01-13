import torch
from torch.utils.data import DataLoader
from gpt.models.dataset import LoveLetterDataset
from gpt.models.gpt_ll import LoveLetterTransformer
from gpt.models.tokenizer import LoveLetterTokenizer
import yaml
from tqdm import tqdm

def train():
    # Load config
    config_path = 'gpt/config/model_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and dataset
    tokenizer = LoveLetterTokenizer()
    dataset = LoveLetterDataset(
        tokenizer=tokenizer,
        config_path='./gpt/config/model_config.yaml'
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model
    model = LoveLetterTransformer(
        vocab_size=tokenizer.vocab_size,
        config_path=config_path,
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # After creating the optimizer, add the scheduler:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        epochs=config['training']['epochs'],
        steps_per_epoch=len(dataloader),
        pct_start=0.3,  # Spend 30% of training time in warmup
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1e4  # Final lr = initial_lr/10000
    )

    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch}')
        for batch_idx, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            seq_len = x.size(1)
            num_heads = config['model']['nhead']  # 4 heads
            
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(device)
            
            # Get padding mask and transform it
            padding_mask = dataset.get_padding_mask(x)  # [batch_size, seq_len]
            padding_mask = padding_mask.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, seq_len]
            
            # Combine masks and expand for multiple heads
            mask = (~causal_mask & padding_mask).float()
            mask = mask.masked_fill(mask == 0, float('-inf'))
            mask = mask.masked_fill(mask == 1, float(0.0))
            mask = mask.repeat(num_heads, 1, 1)  # [num_heads * batch_size, seq_len, seq_len]
            
            # Forward pass
            logits = model(x, mask=mask)
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
        avg_loss = total_loss / len(dataloader)
        print({
            "epoch_loss": avg_loss,
            "epoch": epoch
        })
        
        # Save checkpoint
        if epoch % config['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()