import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
import yaml

class LoveLetterTransformer(nn.Module):
    def __init__(
        self,
        config_path: str,
        vocab_size: int,
    ):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['model']
            
        self.config = config
        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        max_seq_len = config['seq_length']
        dropout = config['dropout']
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Policy head (original output layer)
        self.policy_head = nn.Linear(d_model, vocab_size)
        
        # # Value head
        # self.value_head = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 1),
        #     nn.Tanh()  # Output between -1 and 1 for win probability
        # )
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        pad_token_id = 0
        
        x_embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        x_embedded = x_embedded + self.pos_encoding[-seq_len:, :]  # [batch_size, seq_len, d_model]
        
        features = self.transformer(x_embedded, mask)  # [batch_size, seq_len, d_model]
        
        policy_logits = self.policy_head(features)  # [batch_size, seq_len, vocab_size]
        
        return policy_logits
        
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
            x = torch.tensor([padded_tokens], dtype=torch.long).to(device)  # [1, seq_length]
            mask = self.generate_mask(x)
            
            # Get predictions
            logits = self(x, mask)  # [1, seq_length, vocab_size]
            logits = logits[:, -1, :] / temperature  # [1, vocab_size]
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)  # [1, vocab_size]
            next_token = torch.multinomial(probs, num_samples=1)[0].item()  # scalar
            
            # Add new token to sequence
            tokens.append(next_token)
        
        return tokens
    def generate_mask(self, x):
        num_heads = self.config['nhead']
        seq_len = x.size(1)
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Get padding mask and transform it
        padding_mask = x != 0   # [batch_size, seq_len]
        padding_mask = padding_mask.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, seq_len]
        
        # Combine masks and expand for multiple heads
        mask = (~causal_mask & padding_mask).float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        mask = mask.repeat(num_heads, 1, 1)  # [num_heads * batch_size, seq_len, seq_len]
    