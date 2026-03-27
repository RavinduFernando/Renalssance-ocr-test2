"""
train.py

Training loop for CRNN with CTC loss.
Typically called from a notebook, but can also be run as a script.
"""

from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LineOCRDataset, collate_fn, build_vocab
from .models import CRNN


def prepare_dataloaders(config: Dict, splits=("train", "val")):
    """
    Build vocab from training CSV and create DataLoaders for each split.
    """
    train_csv = Path(config["data"]["train_csv"])
    df_train_texts = train_csv.read_text(encoding="utf-8").splitlines()[1:]  # simple hack
    # In practice, use pandas to load and get "text" column
    import pandas as pd
    df_train = pd.read_csv(train_csv)
    all_texts = df_train["text"].astype(str).tolist()

    char2idx, idx2char = build_vocab(all_texts)

    loaders = {}
    for split in splits:
        csv_path = config["data"][f"{split}_csv"]
        ds = LineOCRDataset(csv_path, char2idx, img_height=config["data"]["img_height"])
        loader = DataLoader(
            ds,
            batch_size=config["training"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["training"]["num_workers"],
            collate_fn=collate_fn,
        )
        loaders[split] = loader

    return loaders, char2idx, idx2char


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["images"].to(device)          # (N, 1, H, W)
        labels = batch["labels"].to(device)          # (sum_target_len,)
        label_lens = batch["label_lens"].to(device)  # (N,)

        # CTC expects (T, N, C)
        logits = model(images)                       # (T, N, C)
        log_probs = logits.log_softmax(2)

        # Input lengths: T is same for all in batch (width)
        T, N, _ = log_probs.size()
        input_lens = torch.full(
            size=(N,),
            fill_value=T,
            dtype=torch.long,
            device=device,
        )

        loss = criterion(log_probs, labels, input_lens, label_lens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)

        logits = model(images)
        log_probs = logits.log_softmax(2)
        T, N, _ = log_probs.size()
        input_lens = torch.full((N,), T, dtype=torch.long, device=device)

        loss = criterion(log_probs, labels, input_lens, label_lens)
        epoch_loss += loss.item()

    return epoch_loss / max(1, len(loader))


def run_training(config: Dict):
    """
    High-level training loop. You typically call this from a notebook:

        from src.train import run_training
        run_training(config)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, char2idx, idx2char = prepare_dataloaders(config)
    num_classes = len(char2idx)

    model = CRNN(num_classes=num_classes).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_loss = float("inf")
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        train_loss = train_one_epoch(model, loaders["train"],
                                     criterion, optimizer, device)
        val_loss = validate(model, loaders["val"], criterion, device)

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / f"crnn_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "config": config,
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    return model, char2idx, idx2char
