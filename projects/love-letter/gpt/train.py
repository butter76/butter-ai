import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from typing import Tuple
from .models.dataset import LoveLetterDataset
from .models.tokenizer import LoveLetterTokenizer
from .models.gpt_model import LoveLetterGPT

def parse_config(config_str: str) -> dict:
    with open(config_str, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def train_loop(
    model: LoveLetterGPT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_token_id: int
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        x, y = batch
        # x, y: [batch_size, seq_len]
        x, y = x.to(device), y.to(device)
        # Build attention mask => (x != pad_token_id)
        attention_mask = (x != pad_token_id).long()  # [b, seq_len]

        optimizer.zero_grad()
        logits = model(x, attention_mask=attention_mask)  # [b, seq_len, vocab_size]
        loss = model.compute_loss(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def val_loop(
    model: LoveLetterGPT,
    dataloader: DataLoader,
    device: torch.device,
    pad_token_id: int
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            attention_mask = (x != pad_token_id).long()  # [b, seq_len]

            logits = model(x, attention_mask=attention_mask)
            loss = model.compute_loss(logits, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(config_path: str):
    # Parse config
    config = parse_config(config_path)
    model_cfg = config["model"]
    data_cfg = config.get("data", {})
    train_logs_dir = data_cfg.get("train_logs_dir", "../../../love-letter-logs/logs/")
    val_logs_dir = data_cfg.get("val_logs_dir", None)

    device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    vocab_size = model_cfg["vocab_size"]
    seq_length = model_cfg["seq_length"]
    n_layer = model_cfg["n_layer"]
    n_head = model_cfg["n_head"]
    d_model = model_cfg["d_model"]
    d_ff = model_cfg["d_ff"]
    dropout = model_cfg["dropout"]
    lr = model_cfg["lr"]
    batch_size = model_cfg["batch_size"]
    epochs = model_cfg["epochs"]
    pad_token_id = model_cfg["pad_token_id"]

    tokenizer = LoveLetterTokenizer(debug=False)
    train_dataset = LoveLetterDataset(
        data_dir=train_logs_dir,
        tokenizer=tokenizer,
        config_path=config_path  # re-use to get seq_length or just read model_cfg
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Optionally do validation dataset if val_logs_dir is provided
    val_dataloader = None
    if val_logs_dir and os.path.exists(val_logs_dir):
        val_dataset = LoveLetterDataset(
            data_dir=val_logs_dir,
            tokenizer=tokenizer,
            config_path=config_path
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = LoveLetterGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        d_ff=d_ff,
        max_seq_len=seq_length,
        dropout=dropout,
        pad_token_id=pad_token_id
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    save_dir = model_cfg.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_loop(model, train_dataloader, optimizer, device, pad_token_id)
        print(f"Epoch {epoch} / {epochs}, Train Loss: {train_loss:.4f}")

        if val_dataloader is not None:
            val_loss = val_loop(model, val_dataloader, device, pad_token_id)
            print(f"Epoch {epoch} / {epochs}, Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(save_dir, f"model_epoch{epoch}_valloss{val_loss:.4f}.pt")
                torch.save(model.state_dict(), ckpt_path)
        else:
            # Save every epoch if no val set
            ckpt_path = os.path.join(save_dir, f"model_epoch{epoch}_trainloss{train_loss:.4f}.pt")
            torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to model config")
    args = parser.parse_args()
    main(args.config)