from typing import Any, cast, List
from networkx import mycielski_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from flash_attn.modules.mha import FlashSelfAttention

from gpt.models.tokenizer import SPECIAL_TOKENS
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
        ffn_mul = model_config['ffn_mul']

        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, d_model))
        
        # Replace standard transformer layers with Flash Attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * ffn_mul),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Policy head (original output layer)
        self.policy_head = nn.Linear(d_model, vocab_size)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()  # Output between -1 and 1 for win probability
        )

        # Opponent's card head
        self.opp_card_head = nn.Linear(d_model, vocab_size)
        
        # My card head
        self.my_card_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        
        x_embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        x_embedded = x_embedded + self.pos_encoding[:seq_len, :]  # [batch_size, seq_len, d_model]


        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        # Pass through transformer layers
        features = x_embedded  # [batch_size, seq_len, d_model]
        features = self.transformer(features, mask=causal_mask)  # [batch_size, seq_len, d_model]
        
        policy_logits = self.policy_head(features)  # [batch_size, seq_len, vocab_size]
        value = self.value_head(features) # [batch_size, seq_len, 1]
        opp_card = self.opp_card_head(features)  # [batch_size, seq_len, vocab_size]
        my_card = self.my_card_head(features)  # [batch_size, seq_len, vocab_size]
    
        return {
            'policy': policy_logits,
            'value': value,
            'card_guess': opp_card,
            'my_card': my_card,
        }
    
    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        output = self(x)
        return output['policy']
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        output = self(x)
        return output['value']
    
    def compute_loss(self, output, target, non_pad_mask):
        # Create mask for non-padding tokens
        mask = non_pad_mask.view(-1)

        logits = output['policy']
        value = output['value']
        guesses = output['card_guess']
        my_card = output['my_card']
        
        losses = {
            'policy': (torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target['policy'].view(-1),
                reduction='none'
            ) * mask).sum(),
            'value': (torch.nn.functional.mse_loss(
                value.view(-1),
                target['value'].view(-1),
                reduction='none'
            ) * mask).sum(),
            'card_guess': (torch.nn.functional.cross_entropy(
                guesses.view(-1, guesses.size(-1)),
                target['card_guess'].view(-1),
                reduction='none'
            ) * mask).sum(),
            'my_card': (torch.nn.functional.cross_entropy(
                my_card.view(-1, my_card.size(-1)),
                target['my_card'].view(-1),
                reduction='none'
            ) * mask).sum(),
        }
        
        return losses

    def generate_continuation(self, x: torch.Tensor, length: torch.Tensor, max_new_tokens: int, temperature=1.0) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        max_seq_len = self.config['seq_length']

        # Create mask for sequences that need continuation
        needs_continuation = (length < max_seq_len) & (x.gather(1, (length - 1).unsqueeze(1)).squeeze(1) != SPECIAL_TOKENS['EOS1']) & (x.gather(1, (length - 1).unsqueeze(1)).squeeze(1) != SPECIAL_TOKENS['EOS2'])

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
            needs_continuation = mask & (next_tokens != SPECIAL_TOKENS['EOS1']) & (next_tokens != SPECIAL_TOKENS['EOS2'])
        
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

            if next_token == SPECIAL_TOKENS['EOS1'] or next_token == SPECIAL_TOKENS['EOS2']:
                break
        
        return tokens
    

    
