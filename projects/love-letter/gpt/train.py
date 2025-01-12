import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gpt.models.gpt_model import GPT, GPTConfig
from gpt.models.dataset import CharacterDataset
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_hparams(hparams_path: str) -> dict:
    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
    return hparams


def train():
    # Hard-coded path to hparams
    config_path = os.path.join(os.path.dirname(__file__), "config", "hparams.yaml")
    hparams = load_hparams(config_path)
    model_config = hparams["model"]
    train_config = hparams["training"]

    # Set seed
    set_seed(train_config.get("seed", 42))

    # Construct dataset
    # By default, logs might be stored in ../../../love-letter-logs/logs/
    # Adjust path as necessary
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../love-letter-logs/logs"))
    block_size = model_config["block_size"]
    dataset = CharacterDataset(data_dir=data_dir, block_size=block_size, ascii_size=model_config["vocab_size"])

    # Create DataLoader
    batch_size = train_config["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build GPT model
    config = GPTConfig(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        n_embd=model_config["n_embd"],
        dropout=model_config["dropout"]
    )
    model = GPT(config).to(device)

    # Training hyperparameters
    max_epochs = train_config["max_epochs"]
    learning_rate = train_config["learning_rate"]
    betas = tuple(train_config["betas"])
    weight_decay = train_config["weight_decay"]
    warmup_steps = train_config["warmup_steps"]
    gradient_clip_val = train_config["gradient_clip_val"]
    save_every = train_config.get("save_every", 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step+1)/warmup_steps, 1.0)
    )

    # Training loop
    step = 0
    for epoch in range(max_epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)  # [B, T]
            y = y.to(device)  # [B, T]

            # Forward
            logits = model(x)
            # logits shape: [B, T, vocab_size]

            # We want next-character prediction loss
            loss_fn = nn.CrossEntropyLoss()
            # Flatten
            B, T, V = logits.shape
            logits_flat = logits.view(B*T, V)        # [B*T, V]
            targets_flat = y.view(B*T)               # [B*T]
            loss = loss_fn(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Step {step}, Loss {loss.item():.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints")
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_path, f"gpt_epoch_{epoch+1}.pt"))
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training completed.")


if __name__ == "__main__":
    train()