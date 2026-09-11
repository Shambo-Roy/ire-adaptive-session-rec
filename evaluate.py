"""Evaluation: Recall@K / MRR@K on the test split, and results.json bookkeeping.

Kept separate from train.py so future phases can re-evaluate a frozen
checkpoint (with a drift mechanism plugged into its gate_fn) without
re-running training.
"""
import json

import torch

import config
import data
from model import GRU4Rec


def compute_ranks(logits, targets):
    """1-indexed rank of the true item for each row of logits, via a score
    comparison rather than a full sort (cheaper for a large vocabulary)."""
    true_scores = logits.gather(1, targets.unsqueeze(1))  # [B, 1]
    ranks = (logits > true_scores).sum(dim=1) + 1
    return ranks


@torch.no_grad()
def evaluate(model, loader, ks=config.EVAL_KS):
    model.eval()
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"mrr@{k}": 0.0 for k in ks})
    n = 0

    for items, lengths, targets, cats in loader:
        logits = model(items, lengths)
        ranks = compute_ranks(logits, targets)
        for k in ks:
            hits = ranks <= k
            sums[f"recall@{k}"] += hits.float().sum().item()
            mrr = torch.where(hits, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float))
            sums[f"mrr@{k}"] += mrr.sum().item()
        n += targets.size(0)

    return {name: total / n for name, total in sums.items()}


def load_checkpoint(path=None):
    path = path or (config.CHECKPOINT_DIR / "best.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    model = GRU4Rec(
        num_items=mc["num_items"],
        embedding_dim=mc["embedding_dim"],
        hidden_dim=mc["hidden_dim"],
        pad_idx=mc["pad_idx"],
        dropout=mc["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def append_result(result_row):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    if config.RESULTS_JSON.exists():
        with open(config.RESULTS_JSON) as f:
            results = json.load(f)
        results = [r for r in results if r.get("mechanism_name") != result_row["mechanism_name"]]
    results.append(result_row)
    with open(config.RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    return config.RESULTS_JSON


def main():
    model, ckpt = load_checkpoint()
    test_loader = data.get_dataloader("test", config.BATCH_SIZE, shuffle=False)
    metrics = evaluate(model, test_loader, ks=(10, 20))

    result_row = {
        "mechanism_name": "no_adaptation_baseline",
        "recall@10": metrics["recall@10"],
        "recall@20": metrics["recall@20"],
        "mrr@10": metrics["mrr@10"],
        "mrr@20": metrics["mrr@20"],
    }

    print("\n=== Test results ===")
    print(f"{'mechanism':<25}{'recall@10':>12}{'recall@20':>12}{'mrr@10':>12}{'mrr@20':>12}")
    print(
        f"{result_row['mechanism_name']:<25}"
        f"{result_row['recall@10']:>12.4f}{result_row['recall@20']:>12.4f}"
        f"{result_row['mrr@10']:>12.4f}{result_row['mrr@20']:>12.4f}"
    )

    out_path = append_result(result_row)
    print(f"\n[evaluate] saved to {out_path}")


if __name__ == "__main__":
    main()
