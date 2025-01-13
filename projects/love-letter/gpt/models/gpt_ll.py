import torch
import torch.nn as nn
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pad_token_id = 0

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        # Create padding mask (1 for padding tokens, 0 for non-padding)
        padding_mask = (x == pad_token_id)  # [batch_size, seq_len]
        
        x_embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        x_embedded = x_embedded + self.pos_encoding[-seq_len:, :]  # [batch_size, seq_len, d_model]
        
        features = self.transformer(x_embedded, mask=causal_mask, src_key_padding_mask=padding_mask)  # [batch_size, seq_len, d_model]
        
        policy_logits = self.policy_head(features)  # [batch_size, seq_len, vocab_size]
        
        return policy_logits
    
    def generate(self, tokens: list[int], max_new_tokens: int, temperature=1.0) -> list[int]:
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
            
            # Get predictions
            logits = self(x)  # [1, seq_length, vocab_size]
            logits = logits[:, -1, :] / temperature  # [1, vocab_size]
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)  # [1, vocab_size]
            next_token = torch.multinomial(probs, num_samples=1)[0].item()  # scalar
            
            # Add new token to sequence
            tokens.append(next_token)
        
        return tokens