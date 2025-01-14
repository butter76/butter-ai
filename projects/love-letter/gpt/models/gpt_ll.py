from typing import cast, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config_types import ModelConfig

class LoveLetterTransformer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        vocab_size: int,
    ):
        super().__init__()
            
        self.config = model_config
        d_model = model_config['d_model']
        nhead = model_config['nhead']
        num_layers = model_config['num_layers']
        max_seq_len = model_config['seq_length']
        dropout = model_config['dropout']

        
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
        
        x_embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        x_embedded = x_embedded + self.pos_encoding[:seq_len, :]  # [batch_size, seq_len, d_model]
        
        features = self.transformer(x_embedded, mask=causal_mask)  # [batch_size, seq_len, d_model]
        
        policy_logits = self.policy_head(features)  # [batch_size, seq_len, vocab_size]
        
        return policy_logits
    
    def generate(self, tokens: list[int], max_new_tokens: int, temperature=1.0) -> list[int]:
        device = next(self.parameters()).device
        
        for _ in range(max_new_tokens):
            # Left pad sequence to seq_length
            pad_length = self.config['seq_length'] - len(tokens)
            if pad_length > 0:
                padded_tokens = tokens + [0] * pad_length
            else:
                padded_tokens = tokens[-self.config['seq_length']:]
                
            # Create input tensor and padding mask
            x = torch.tensor([padded_tokens], dtype=torch.long).to(device)  # [1, seq_length]
            
            # Get predictions
            logits = self(x)  # [1, seq_length, vocab_size]
            logits = logits[:, len(tokens) - 1, :] / temperature  # [1, vocab_size]
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)  # [1, vocab_size]
            next_token = cast(int, torch.multinomial(probs, num_samples=1)[0].item())  # scalar
            
            # Add new token to sequence
            tokens.append(next_token)
        
        return tokens