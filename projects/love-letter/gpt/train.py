import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gpt.models.gpt_model import GPT, GPTConfig
from gpt.models.dataset import LoveLetterDataset
from gpt.models.tokenizer import LoveLetterTokenizer
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
    # Hard-coded path to training hyperparams
    hparams_path = os.path.join(os.path.dirname(__file__), "config", "hparams.yaml")
    hparams = load_hparams(hparams_path)

    # Hard-coded path to model config (used by dataset and model)
    model_config_path = os.path.join(os.path.dirname(__file__), "config", "model_config.yaml")
    with open(model_config_path, 'r') as f:
        model_config_yaml = yaml.safe_load(f)

    # Extract relevant config
    model_conf = model_config_yaml["model"]  # GPT model config
    train_conf = model_config_yaml["training"]  # general training config
    training_hparams = hparams.get("training", {})

    # Merge hyperparams: command line > hparams.yaml > model_config.yaml
    # Here we just combine them, with hparams.yaml taking precedence over model_config.yaml
    for k, v in training_hparams.items():
        train_conf[k] = v

    # Set seed
    seed = train_conf.get("seed", 42)
    set_seed(seed)

    # Create tokenizer
    tokenizer = LoveLetterTokenizer(debug=False)

    # Construct dataset using LoveLetterDataset
    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../love-letter-logs/logs")
    )

    dataset = LoveLetterDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        config_path=model_config_path
    )

    # Create DataLoader
    batch_size = train_conf["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build GPT model
    config = GPTConfig(
        vocab_size=model_conf["vocab_size"],
        block_size=model_conf["seq_length"],
        n_layer=model_conf["n_layer"],
        n_head=model_conf["n_head"],
        n_embd=model_conf["n_embd"],
        dropout=model_conf["dropout"]
    )
    model = GPT(config).to(device)

    # Training hyperparameters
    max_epochs = train_conf["max_epochs"]
    learning_rate = train_conf["learning_rate"]
    betas = (0.9, 0.99)
    if "betas" in train_conf:
        betas = tuple(train_conf["betas"])  # e.g. [0.9, 0.99]
    weight_decay = train_conf["weight_decay"]
    warmup_steps = train_conf["warmup_steps"]
    gradient_clip_val = train_conf["gradient_clip_val"]
    save_every = train_conf.get("save_every", 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, 1.0)
    )

    # Training loop
    step = 0
    for epoch in range(max_epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)  # [B, seq_length]
            y = y.to(device)  # [B, seq_length]

            # Forward
            logits = model(x)  # [B, seq_length, vocab_size]

            # Next-token prediction loss
            loss_fn = nn.CrossEntropyLoss()
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)  # [B*T, V]
            targets_flat = y.view(B * T)         # [B*T]
            loss = loss_fn(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
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