import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embeddings = nn.Embedding(config['seq_length'], config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
        
        # Create transformer layers using PyTorch's TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_head'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            batch_first=True,
            norm_first=False
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['n_layer']
        )
        
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, padding_mask):
        b, t = x.size()
        device = x.device
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        token_embeddings = self.token_embeddings(x)
        pos_embeddings = self.position_embeddings(pos)
        
        x = token_embeddings + pos_embeddings
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = torch.triu(torch.full((t, t), float('-inf'), device=device), diagonal=1)
        
        # Use padding_mask directly - it should remain 2D [batch_size, seq_len]
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits




    def generate(self, tokens: list[int], max_new_tokens: int, temperature=1.0):
        device = next(self.parameters()).device
        
        for _ in range(max_new_tokens):
            # Left pad sequence to seq_length
            pad_length = self.config['seq_length'] - len(tokens)
            if pad_length > 0:
                padded_tokens = [0] * pad_length + tokens
            else:
                padded_tokens = tokens[-self.config['seq_length']:]
                
            # Create input tensor and padding mask
            x = torch.tensor([padded_tokens], dtype=torch.long).to(device)
            padding_mask = (x != 0).float().to(device)
            
            # Get predictions
            logits = self(x, padding_mask)
            logits = logits[:, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)[0].item()
            
            # Add new token to sequence
            tokens.append(next_token)
        
        return tokens
