"""Training loop for the GRU4Rec baseline.

Cross-entropy loss over the item vocabulary (padding index ignored), Adam
optimizer, early stopping on validation Recall@20. Saves the best checkpoint
(weights + model config + item/category id mappings) to
checkpoints/gru4rec_baseline/, which future phases load and freeze.
"""
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

import config
import data
from evaluate import evaluate
from model import GRU4Rec


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    set_seed(config.SEED)

    mappings = data.load_mappings()
    num_items = len(mappings["item_id_to_idx"])
    print(f"[train] num_items={num_items}")

    train_loader = data.get_dataloader("train", config.BATCH_SIZE, shuffle=True)
    val_loader = data.get_dataloader("val", config.BATCH_SIZE, shuffle=False)

    model = GRU4Rec(
        num_items=num_items,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        pad_idx=config.PAD_IDX,
        dropout=config.DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_recall20 = -1.0
    epochs_no_improve = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for items, lengths, targets, cats in train_loader:
            optimizer.zero_grad()
            logits = model(items, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        val_metrics = evaluate(model, val_loader, ks=config.EVAL_KS)
        elapsed = time.time() - t0

        print(
            f"[train] epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"val_recall@20={val_metrics['recall@20']:.4f} val_mrr@20={val_metrics['mrr@20']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["recall@20"] > best_recall20:
            best_recall20 = val_metrics["recall@20"]
            epochs_no_improve = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_items": num_items,
                    "embedding_dim": config.EMBEDDING_DIM,
                    "hidden_dim": config.HIDDEN_DIM,
                    "pad_idx": config.PAD_IDX,
                    "dropout": config.DROPOUT,
                },
                "train_config": {
                    "batch_size": config.BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "pad_len": config.PAD_LEN,
                    "seed": config.SEED,
                },
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, config.CHECKPOINT_DIR / "best.pt")

            shutil.copy(config.PROCESSED_DIR / "item_id_to_idx.json", config.CHECKPOINT_DIR / "item_id_to_idx.json")
            shutil.copy(config.PROCESSED_DIR / "category_id_to_idx.json", config.CHECKPOINT_DIR / "category_id_to_idx.json")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"[train] early stopping at epoch {epoch} (no val_recall@20 improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
                break

    print(f"[train] best val_recall@20={best_recall20:.4f}, checkpoint saved to {config.CHECKPOINT_DIR / 'best.pt'}")
    return config.CHECKPOINT_DIR / "best.pt"


if __name__ == "__main__":
    train()
