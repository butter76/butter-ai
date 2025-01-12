import torch
from torch.utils.data import Dataset
import os
from typing import List, Tuple
from .tokenizer import LoveLetterTokenizer
import yaml

class LoveLetterDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: LoveLetterTokenizer, config_path: str):
        self.tokenizer = tokenizer
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.seq_length = config['model']['seq_length']
        
        self.examples: List[List[int]] = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.log'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    text = f.read()
                    tokens = self.tokenizer.tokenize(text)
                    if tokens:  # Only add non-empty sequences
                        self.examples.append(tokens)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.examples[idx]
        
        # Create input and target sequences
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]

        pad_length = self.seq_length - len(tokens)
        padded_tokens = [self.tokenizer.special_tokens['PAD']] * pad_length + tokens
        
        x = torch.tensor(padded_tokens[:-1], dtype=torch.long)  # [seq_length]
        y = torch.tensor(padded_tokens[1:], dtype=torch.long)   # [seq_length]
        
        return x, y

    def get_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create padding mask where 1 indicates non-pad tokens"""
        return (tokens != self.tokenizer.special_tokens['PAD']).float()


##########################
# New CharacterDataset for GPT
##########################

class CharacterDataset(Dataset):
    """
    A dataset that processes the .log files at a character level.
    We'll encode each character to an integer ID (ASCII or extended),
    then produce sequences for next-character prediction.
    """
    def __init__(self, data_dir: str, block_size: int = 256, ascii_size: int = 128):
        super().__init__()
        self.block_size = block_size
        self.ascii_size = ascii_size

        # Read all .log files as a single text corpus
        texts = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".log"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        self.data = "\n".join(texts)

        # Convert to array of ASCII codes (clamped to [0..ascii_size-1])
        self.encoded = [min(ord(ch), self.ascii_size - 1) for ch in self.data]

    def __len__(self) -> int:
        # For next-character pred, length is number of possible 'block_size' segments
        return max(0, len(self.encoded) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.encoded[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # [block_size]
        y = torch.tensor(chunk[1:], dtype=torch.long)   # [block_size]
        return x, y