from typing import Any, cast, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from flash_attn.modules.mha import FlashSelfAttention
from .config_types import ModelConfig

class FlashAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # Pass causal=True directly in the constructor
        self.flash_attention = FlashSelfAttention(causal=True, attention_dropout=dropout)
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        batch_size, seq_len, d_model = x.shape
        
        # Fuse QKV projections into one operation
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Use flash_attn_func directly instead of FlashSelfAttention module
        output = cast(Any, flash_attn_func(q, k, v, causal=causal, dropout_p=self.dropout))
        
        output = output.contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(output)

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.flash_attention = FlashAttentionLayer(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        x = x + self.flash_attention(self.norm1(x))
        x = x + self.dropout(self.linear2(self.activation(self.linear1(self.norm2(x)))))
        return x

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
        ffn_mul = model_config['ffn_mul']

        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, d_model))
        
        # Replace standard transformer layers with Flash Attention layers
        self.transformer = nn.ModuleList([
            FlashTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * ffn_mul,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Policy head (original output layer)
        self.policy_head = nn.Linear(d_model, vocab_size)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()  # Output between -1 and 1 for win probability
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        
        x_embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        x_embedded = x_embedded + self.pos_encoding[:seq_len, :]  # [batch_size, seq_len, d_model]
        
        # Pass through transformer layers
        features = x_embedded  # [batch_size, seq_len, d_model]
        for layer in self.transformer:
            features = layer(features)  # [batch_size, seq_len, d_model]
        
        policy_logits = self.policy_head(features)  # [batch_size, seq_len, vocab_size]
        value = self.value_head(features) # [batch_size, seq_len, 1]
        return policy_logits, value
    
    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        policy_logits, _ = self(x)
        return policy_logits
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, value = self(x)
        return value
    
    def compute_loss(self, logits, value, y, y_value, non_pad_mask):
        # Create mask for non-padding tokens
        mask = non_pad_mask.view(-1)
        
        losses = {
            'policy': (torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='none'
            ) * mask).sum(),
            'value': (torch.nn.functional.mse_loss(
                value.view(-1),
                y_value.view(-1),
                reduction='none'
            ) * mask).sum()
        }
        
        return losses

    def generate_continuation(self, x: torch.Tensor, length: torch.Tensor, max_new_tokens: int, temperature=1.0) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        max_seq_len = self.config['seq_length']

        # Create mask for sequences that need continuation
        needs_continuation = (length < max_seq_len) & (x.gather(1, (length - 1).unsqueeze(1)).squeeze(1) != 27) & (x.gather(1, (length - 1).unsqueeze(1)).squeeze(1) != 28)

        # Continue generating while any sequence needs continuation and we haven't exceeded max_new_tokens
        for _ in range(max_new_tokens):
            if not needs_continuation.any():
                break
                
            # Get predictions for the next token for all sequences
            logits = self.get_policy(x)  # [batch_size, seq_len, vocab_size]
            
            # Select logits at the current sequence lengths
            next_token_logits = logits.gather(1, (length - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, logits.size(-1))).squeeze(1)
            next_token_logits = next_token_logits / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]
            
            # Update only sequences that need continuation
            mask = needs_continuation & (length < max_seq_len)
            x[mask, length[mask]] = next_tokens[mask]
            length[mask] += 1
            
            # Update continuation mask
            needs_continuation = mask & (next_tokens != 27) & (next_tokens != 28)
        
        return x, length
    
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
            logits = self.get_policy(x)  # [1, seq_length, vocab_size]
            logits = logits[:, len(tokens) - 1, :] / temperature  # [1, vocab_size]
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)  # [1, vocab_size]
            next_token = cast(int, torch.multinomial(probs, num_samples=1)[0].item())  # scalar
            
            # Add new token to sequence
            tokens.append(next_token)

            if next_token == 27 or next_token == 28:
                break
        
        return tokens
    

    
