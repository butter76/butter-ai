import torch
from torch.utils.data import DataLoader
import yaml
from gpt.models.gpt import GPTModel
from gpt.models.dataset import LoveLetterDataset
from gpt.models.tokenizer import LoveLetterTokenizer
import os
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train():
    # Load configuration
    config_path = 'gpt/config/model_config.yaml'
    config = load_config(config_path)
    model_config = config['model']
    data_config = config['data']
    
    # Set device
    device = torch.device(model_config['device'] if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and dataset
    tokenizer = LoveLetterTokenizer()
    train_dataset = LoveLetterDataset(
        data_config['train_logs_dir'],
        tokenizer,
        config_path
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model
    model = GPTModel(model_config).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config['lr'],
        weight_decay=0.01
    )
    
    # Training loop
    for epoch in range(model_config['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{model_config["epochs"]}')
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            # Move batch to device
            x = x.to(device)
            y = y.to(device)
            
            # Create padding mask
            padding_mask = train_dataset.get_padding_mask(x).to(device)
            
            # Forward pass
            logits = model(x, padding_mask)
            
            # Calculate loss (only on non-padded tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model_config['vocab_size']),
                y.view(-1),
                ignore_index=model_config['pad_token_id']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Save checkpoint
        if not os.path.exists(model_config['save_dir']):
            os.makedirs(model_config['save_dir'])
        
        checkpoint_path = os.path.join(
            model_config['save_dir'],
            f'model_epoch_{epoch+1}.pt'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

if __name__ == '__main__':
    train()
