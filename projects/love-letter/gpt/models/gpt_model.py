import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)       # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Generate a causal (triangular) mask for self-attention so tokens cannot attend to future positions.
    Returns a [seq_len, seq_len] mask with True for blocked positions.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # [seq_len, seq_len]
    # In PyTorch Transformer, True means to mask out, so we invert.
    # We'll do that in the forward pass. Here we just return the lower-triangular matrix.
    return mask == 0


class LoveLetterGPT(nn.Module):
    """
    A GPT-like model for predicting tokens in Love Letter game logs.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        n_head: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        pad_token_id: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len] (token ids)
        attention_mask: [batch_size, seq_len], where 0 is pad, 1 is real token
        """
        b, seq_len = x.shape
        device = x.device

        # Create token embeddings
        token_emb = self.token_emb(x)  # [b, seq_len, d_model]

        # Apply positional encoding
        h = self.pos_emb(token_emb)    # [b, seq_len, d_model]

        # Build the combined mask for Transformer
        # We'll combine the padding mask and the causal mask
        causal_mask = generate_causal_mask(seq_len, device)  # [seq_len, seq_len], bool
        # In PyTorch, the attention mask is of shape [batch_size, seq_len], we need to convert that
        # to shape [batch_size, 1, seq_len] to broadcast to [batch_size, seq_len, seq_len].
        # Then we need to combine it with the causal mask by logical OR.
        if attention_mask is not None:
            # attention_mask: 1 = real token, 0 = pad
            # We want to produce "True" for positions we should mask out
            # So let's invert it => True for pad
            inverted_pad_mask = (attention_mask == 0).unsqueeze(1).bool()  # [b, 1, seq_len]
            # Expand to [b, seq_len, seq_len]
            inverted_pad_mask = inverted_pad_mask.expand(b, seq_len, seq_len)
            # Combine with causal_mask (which is [seq_len, seq_len])
            # We'll broadcast causal_mask across batch dimension.
            combined_mask = causal_mask.unsqueeze(0) | inverted_pad_mask  # OR
        else:
            # No pad mask, only causal
            combined_mask = causal_mask.unsqueeze(0)

        # The PyTorch transformer wants a float mask with -inf for masked positions, 0 for non-masked
        # We'll convert "True" => -inf
        combined_mask = combined_mask.masked_fill(combined_mask, float('-inf'))  # [b, seq_len, seq_len]
        
        # Pass through transformer
        # By default, TransformerEncoder expects src_mask of shape [batch_size, seq_len, seq_len]
        # or [seq_len, seq_len] if no batch dimension. We have [b, seq_len, seq_len].
        out = self.transformer(h, mask=combined_mask)  # [b, seq_len, d_model]

        # Final layernorm
        out = self.ln_f(out)  # [b, seq_len, d_model]

        # Predictions
        logits = self.head(out)  # [b, seq_len, vocab_size]
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss ignoring the pad tokens.
        logits: [b, seq_len, vocab_size]
        targets: [b, seq_len]
        """
        # shift logits for cross-entropy
        # In many GPT tasks, we predict next token => we can do that shift externally
        # or we just apply cross entropy to each position's target. We'll do it directly.
        # We'll flatten for efficiency
        b, seq_len, _ = logits.shape
        logits_flat = logits.view(b * seq_len, self.vocab_size)   # [b*seq_len, vocab_size]
        targets_flat = targets.view(b * seq_len)                  # [b*seq_len]

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fn(logits_flat, targets_flat)
        return loss