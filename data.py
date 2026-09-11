"""Raw CSV loading and all preprocessing for the session-recommendation pipeline.

Pipeline (see README.md for the full write-up):
    1. load train-item-views.csv + product-categories.csv (both ';'-separated),
       left-join categoryId onto item views.
    2. sort each session's events by timeframe.
    3. filter sessions with length < MIN_SESSION_LENGTH.
    4. filter items with global support < MIN_ITEM_SUPPORT, then re-filter
       session length (removing rare items can shorten sessions below 2).
    5/6. build item-id and category-id -> contiguous-index mappings (0 reserved
       for padding / "no category") over the whole filtered dataset.
    7. time-based split by each session's earliest eventdate.
    8. drop val/test events whose item never appears in the train split.
    9. sequence augmentation: (prefix -> next_item) pairs, per split.
    10/11. left-pad prefixes to PAD_LEN with index 0, track true lengths.
    12. category sequences aligned to the (padded) item sequences.
    13. cache everything under processed/.

This module has no PyTorch model/training logic in it beyond the thin
Dataset/DataLoader wrappers at the bottom, so it can be tested independently
of model.py / train.py.
"""
import json
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config


# ---------------------------------------------------------------------------
# Loading + filtering
# ---------------------------------------------------------------------------

def load_and_join():
    """Load both raw CSVs and left-join categoryId onto item-view events."""
    views = pd.read_csv(
        config.TRAIN_ITEM_VIEWS_CSV, sep=";",
        dtype={"sessionId": "int64", "itemId": "int64", "timeframe": "int64"},
    )
    cats = pd.read_csv(
        config.PRODUCT_CATEGORIES_CSV, sep=";", dtype={"itemId": "int64"},
    )
    cats = cats.drop_duplicates(subset="itemId")

    merged = views.merge(cats, on="itemId", how="left")
    n_missing = int(merged["categoryId"].isna().sum())
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} events reference an itemId with no matching categoryId; "
            f"filling categoryId with -1 for these."
        )
    merged["categoryId"] = merged["categoryId"].fillna(-1).astype("int64")
    return merged


def sort_sessions(df):
    return df.sort_values(["sessionId", "timeframe"], kind="mergesort")


def filter_min_session_length(df, min_len):
    sizes = df.groupby("sessionId").size()
    keep = sizes[sizes >= min_len].index
    return df[df["sessionId"].isin(keep)]


def filter_min_item_support(df, min_support):
    counts = df["itemId"].value_counts()
    keep = counts[counts >= min_support].index
    return df[df["itemId"].isin(keep)]


def preprocess_events():
    """Steps 1-4: load, join, sort, filter (session length + item support)."""
    df = load_and_join()
    df = sort_sessions(df)

    before_sessions = df["sessionId"].nunique()
    df = filter_min_session_length(df, config.MIN_SESSION_LENGTH)
    df = filter_min_item_support(df, config.MIN_ITEM_SUPPORT)
    df = filter_min_session_length(df, config.MIN_SESSION_LENGTH)  # re-filter after item drop
    after_sessions = df["sessionId"].nunique()

    print(f"[data] sessions: {before_sessions} -> {after_sessions} after length/support filtering")
    print(f"[data] events remaining: {len(df)}")
    print(f"[data] items remaining: {df['itemId'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# ID mappings (steps 5-6)
# ---------------------------------------------------------------------------

def build_item_mapping(df):
    """Contiguous item-id -> index mapping. Index 0 is reserved for padding."""
    unique_items = sorted(df["itemId"].unique().tolist())
    return {int(item): idx + 1 for idx, item in enumerate(unique_items)}


def build_category_mapping(df):
    """Contiguous category-id -> index mapping. Index 0 is reserved for
    "no category" (used for the -1 sentinel from a missing left-join match,
    though with 100% coverage this should not occur in practice)."""
    unique_cats = sorted(c for c in df["categoryId"].unique().tolist() if c != -1)
    return {int(cat): idx + 1 for idx, cat in enumerate(unique_cats)}


# ---------------------------------------------------------------------------
# Time-based split (step 7) + unseen-item drop (step 8)
# ---------------------------------------------------------------------------

def split_sessions_by_date(df, test_days=config.TEST_DAYS, val_days=config.VAL_DAYS):
    df = df.copy()
    df["eventdate"] = pd.to_datetime(df["eventdate"])

    sess_dates = df.groupby("sessionId")["eventdate"].min()
    max_date = sess_dates.max()

    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    val_end = test_start - pd.Timedelta(days=1)

    test_sessions = sess_dates[sess_dates >= test_start].index
    val_sessions = sess_dates[(sess_dates >= val_start) & (sess_dates <= val_end)].index
    train_sessions = sess_dates[sess_dates < val_start].index

    train_df = df[df["sessionId"].isin(train_sessions)]
    val_df = df[df["sessionId"].isin(val_sessions)]
    test_df = df[df["sessionId"].isin(test_sessions)]

    split_info = {
        "train_start": str(sess_dates.min().date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(max_date.date()),
        "train_sessions": int(len(train_sessions)),
        "val_sessions": int(len(val_sessions)),
        "test_sessions": int(len(test_sessions)),
    }
    print(
        f"[data] split -> train: {split_info['train_sessions']} sessions "
        f"(< {split_info['val_start']}), val: {split_info['val_sessions']} sessions "
        f"({split_info['val_start']} to {split_info['val_end']}), "
        f"test: {split_info['test_sessions']} sessions "
        f"({split_info['test_start']} to {split_info['test_end']})"
    )
    return train_df, val_df, test_df, split_info


def drop_unseen_items(train_df, val_df, test_df):
    """Drop val/test events whose item never appears in the train split."""
    train_items = set(train_df["itemId"].unique().tolist())

    def _drop(split_df, name):
        mask = split_df["itemId"].isin(train_items)
        dropped = int((~mask).sum())
        print(f"[data] {name}: dropping {dropped} / {len(split_df)} events with items unseen in train")
        return split_df[mask]

    val_df = _drop(val_df, "val")
    test_df = _drop(test_df, "test")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Sequence augmentation + padding (steps 9-12)
# ---------------------------------------------------------------------------

def report_prefix_length_distribution(df, name="train"):
    """Print the unpadded prefix-length distribution for a split (step 10)."""
    lengths = []
    for _, group in df.groupby("sessionId", sort=False):
        n = len(group)
        if n >= 2:
            lengths.extend(range(1, n))
    lengths = np.array(lengths)
    print(
        f"[data] {name} prefix-length distribution (unpadded, {len(lengths)} pairs): "
        f"mean={lengths.mean():.2f} median={np.median(lengths):.1f} "
        f"p95={np.percentile(lengths, 95):.1f} p99={np.percentile(lengths, 99):.1f} "
        f"max={int(lengths.max())}"
    )
    print(
        f"[data] using PAD_LEN={config.PAD_LEN} "
        f"(prefixes longer than this are truncated to their most recent PAD_LEN items)"
    )
    return lengths


def _augment_session(items, cats, pad_len):
    """Generate left-padded (prefix -> next_item) examples for one session's
    item/category index sequences, e.g. [i1,i2,i3,i4] ->
    ([i1]->i2), ([i1,i2]->i3), ([i1,i2,i3]->i4)."""
    n = len(items)
    examples = []
    for t in range(1, n):
        prefix_items = items[:t]
        prefix_cats = cats[:t]
        target = items[t]
        if len(prefix_items) > pad_len:
            prefix_items = prefix_items[-pad_len:]
            prefix_cats = prefix_cats[-pad_len:]
        length = len(prefix_items)
        pad_amount = pad_len - length
        padded_items = [0] * pad_amount + prefix_items
        padded_cats = [0] * pad_amount + prefix_cats
        examples.append((padded_items, length, target, padded_cats))
    return examples


def build_examples(df, item_mapping, category_mapping, pad_len):
    """Map raw ids -> indices and generate padded (prefix -> next_item)
    examples for every session in df (step 9, 11, 12)."""
    df = df.sort_values(["sessionId", "timeframe"], kind="mergesort").copy()
    df["item_idx"] = df["itemId"].map(item_mapping)
    df["cat_idx"] = df["categoryId"].map(lambda c: category_mapping.get(c, 0))

    all_items, all_lengths, all_targets, all_cats = [], [], [], []
    for _, group in df.groupby("sessionId", sort=False):
        items = group["item_idx"].tolist()
        cats = group["cat_idx"].tolist()
        if len(items) < 2:
            continue
        for padded_items, length, target, padded_cats in _augment_session(items, cats, pad_len):
            all_items.append(padded_items)
            all_lengths.append(length)
            all_targets.append(target)
            all_cats.append(padded_cats)

    return (
        np.array(all_items, dtype=np.int64).reshape(-1, pad_len),
        np.array(all_lengths, dtype=np.int64),
        np.array(all_targets, dtype=np.int64),
        np.array(all_cats, dtype=np.int64).reshape(-1, pad_len),
    )


# ---------------------------------------------------------------------------
# Orchestration + disk cache (step 13)
# ---------------------------------------------------------------------------

def preprocess_and_cache():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = preprocess_events()

    item_mapping = build_item_mapping(df)
    category_mapping = build_category_mapping(df)
    print(f"[data] item vocab size: {len(item_mapping)}")
    print(f"[data] category vocab size: {len(category_mapping)}")

    train_df, val_df, test_df, split_info = split_sessions_by_date(df)
    train_df, val_df, test_df = drop_unseen_items(train_df, val_df, test_df)

    report_prefix_length_distribution(train_df, name="train")

    counts = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        items, lengths, targets, cats = build_examples(split_df, item_mapping, category_mapping, config.PAD_LEN)
        out_dir = config.PROCESSED_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "items.npy", items)
        np.save(out_dir / "lengths.npy", lengths)
        np.save(out_dir / "targets.npy", targets)
        np.save(out_dir / "categories.npy", cats)
        counts[name] = int(len(targets))
        print(f"[data] {name}: {counts[name]} (prefix -> next_item) examples saved to {out_dir}")

    with open(config.PROCESSED_DIR / "item_id_to_idx.json", "w") as f:
        json.dump(item_mapping, f)
    with open(config.PROCESSED_DIR / "category_id_to_idx.json", "w") as f:
        json.dump(category_mapping, f)
    with open(config.PROCESSED_DIR / "split_info.json", "w") as f:
        json.dump(
            {
                **split_info,
                "pad_len": config.PAD_LEN,
                "num_items": len(item_mapping),
                "num_categories": len(category_mapping),
                "example_counts": counts,
            },
            f,
            indent=2,
        )

    print("[data] preprocessing complete.")


def load_mappings():
    with open(config.PROCESSED_DIR / "item_id_to_idx.json") as f:
        item_id_to_idx = {int(k): v for k, v in json.load(f).items()}
    with open(config.PROCESSED_DIR / "category_id_to_idx.json") as f:
        category_id_to_idx = {int(k): v for k, v in json.load(f).items()}
    return {"item_id_to_idx": item_id_to_idx, "category_id_to_idx": category_id_to_idx}


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoader
# ---------------------------------------------------------------------------

class SessionDataset(Dataset):
    def __init__(self, split):
        split_dir = config.PROCESSED_DIR / split
        self.items = np.load(split_dir / "items.npy")
        self.lengths = np.load(split_dir / "lengths.npy")
        self.targets = np.load(split_dir / "targets.npy")
        self.categories = np.load(split_dir / "categories.npy")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.items[idx]),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.from_numpy(self.categories[idx]),
        )


def get_dataloader(split, batch_size, shuffle):
    dataset = SessionDataset(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config.NUM_WORKERS)


if __name__ == "__main__":
    preprocess_and_cache()
