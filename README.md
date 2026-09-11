# Adaptive Session Recommendation Under Intent Drift — Phase 1

Phase 1 of a multi-phase project. This phase implements and evaluates only a
**GRU4Rec baseline** with no drift-adaptation. Later phases will add synthetic
drift injection (session-splicing using real item category labels) and
swappable drift-response mechanisms (fixed decay, CUSUM, ADWIN, a
KL-divergence gate, a learned classifier) that intervene on the GRU's hidden
state between timesteps. This phase's code is structured so those can be
added without rewriting the encoder, training loop, or data pipeline.

## Repo layout

```
config.py     - all hyperparameters and paths (single source of truth)
data.py       - raw CSV loading, all preprocessing, PyTorch Dataset/DataLoader
model.py      - GRU4Rec model + the per-timestep hidden-state hook
train.py      - training loop, early stopping, checkpointing
evaluate.py   - Recall@K / MRR@K on a split, results.json bookkeeping
archive/      - raw data (train-item-views.csv, product-categories.csv)
processed/    - cached preprocessed tensors + id mappings (generated)
checkpoints/gru4rec_baseline/ - best model checkpoint (generated)
results/results.json          - per-mechanism metrics table (generated)
```

## How to rerun each step

```bash
python data.py       # preprocess raw CSVs -> processed/
python train.py       # train GRU4Rec baseline -> checkpoints/gru4rec_baseline/best.pt
python evaluate.py    # evaluate best.pt on test -> results/results.json
```

Each step reads its inputs from disk (no shared in-memory state), so they can
be rerun independently once `processed/` exists.

## Data

Raw files (semicolon-separated despite the `.csv` extension):
- `archive/train-item-views.csv`: `sessionId;userId;itemId;timeframe;eventdate`
  (1,235,380 rows, 310,324 sessions, 122,993 items, 2016-01-01 to 2016-06-01).
  `userId` is ignored (mostly NA; this is session-based recommendation).
  `timeframe` is milliseconds elapsed within a session only — used to sort
  events within a session, not comparable across sessions.
- `archive/product-categories.csv`: `itemId;categoryId` (184,047 items, 1,217
  categories). Verified 100% coverage against train-item-views' itemIds
  (0 missing on a left join) — the "missing category -> -1 + warning" path in
  `data.load_and_join` is defensive and was not exercised on this data.

## Preprocessing (`data.py`)

1. Left-join `categoryId` onto item-view events on `itemId`.
2. Sort each session's events by `timeframe`.
3. Filter sessions with length < 2.
4. Filter items with global support < 5, then **re-filter** session length
   < 2 (dropping rare items can shorten a session below 2 events).
   Result: 204,789 sessions, 993,483 events, 43,136 items.
5/6. Build contiguous `item_id -> index` and `category_id -> index` mappings
   over the *whole* filtered dataset (all splits), reserving index 0 for
   padding / "no category". Saved to `processed/item_id_to_idx.json` and
   `processed/category_id_to_idx.json` — reused by every future phase.
7. **Time-based split**, by each session's *earliest* eventdate (only 260 of
   310,324 sessions span two calendar dates; for those, the earliest date is
   used). Cutoff: last 7 days → test, the 7 days before that → val, everything
   earlier → train. Resulting session counts (after the filtering in steps
   3-4, which is why these differ from the raw last-7/prior-7-day session
   counts on the unfiltered data):
   - train: 176,976 sessions (before 2016-05-19)
   - val: 11,835 sessions (2016-05-19 to 2016-05-25)
   - test: 15,978 sessions (2016-05-26 to 2016-06-01)

   A 7/7-day cutoff was chosen (and kept) because it gives a test set that is
   neither near-empty nor a large fraction of the data (~7.8% of sessions),
   with a comparably-sized validation set for early stopping.
8. Drop val/test events whose item never appeared in train (the model has no
   trained embedding for such items). This dropped 259 / 56,963 val events and
   495 / 77,061 test events (~0.5-0.6% each) — small, as expected given train
   covers the large majority of the item vocabulary.
9. Sequence augmentation per split independently: a session `[i1,i2,i3,i4]`
   produces `([i1]->i2)`, `([i1,i2]->i3)`, `([i1,i2,i3]->i4)`.
   Result: 682,483 train / 44,878 val / 60,600 test (prefix -> next_item) pairs.
10. **`PAD_LEN` choice.** The unpadded prefix-length distribution over the
    682,483 train pairs: mean 4.13, median 3, p95 = 12, p99 = 19, max = 69.
    `PAD_LEN = 19` was chosen (covers ~99% of prefixes without padding) —
    prefixes longer than 19 are truncated to their most recent 19 items
    (i.e. the tail closest to the target), rather than paying for a max
    length of 69 that only a long tail of sessions would use.
11. Left-pad item (and aligned category) sequences to `PAD_LEN` with index 0;
    true (unpadded, post-truncation) lengths are stored alongside.
12. **Category sequences are preserved.** Every saved example includes a
    `categories.npy` array aligned index-for-index with `items.npy` (same
    padding, same length, same truncation), even though this phase's model
    does not consume it. Confirmed present under `processed/{train,val,test}/categories.npy`.
13. All tensors + both id mappings + `split_info.json` (split dates, session
    counts, dropped-event counts, chosen `PAD_LEN`, vocab sizes) are cached
    under `processed/`.

## Model (`model.py`) and the drift-mechanism hook interface

`GRU4Rec` is a standard single-layer GRU session encoder: an
`nn.Embedding(num_items+1, embedding_dim, padding_idx=0)` feeding an
`nn.GRUCell(embedding_dim, hidden_dim)` driven by an **explicit per-timestep
Python loop** (rather than `nn.GRU`/`pack_padded_sequence`) specifically so a
hook can be inserted between timesteps:

```python
def default_gate(hidden_state, step_info):
    return hidden_state   # no-op; future drift mechanisms replace this
```

Inside `GRU4Rec.forward`, at every timestep `t` (for every session in the
batch, real or padding):

```python
h_candidate = self.gru_cell(x_t, h)
step_info = {"t": t, "mask": mask_t, "item_idx": item_t, "cat_idx": cat_t}
h_candidate = self.gate_fn(h_candidate, step_info)   # <-- hook, called every step
h = mask_t * h_candidate + (1 - mask_t) * h          # padding never updates h
```

`step_info` carries the timestep index, a `[batch,1]` mask of which sequences
are on a real (non-padding) token at this step, and the item/category index
just consumed — enough for a future mechanism (e.g. a CUSUM or KL-divergence
gate watching the category sequence) to decide whether/how to modify the
hidden state, without needing to change `forward()`.

**How a future drift mechanism plugs in:** implement a function with the
same signature as `default_gate` and pass it as `gate_fn=` when constructing
`GRU4Rec(...)` (or reassign `model.gate_fn` on an existing instance). No
changes to the recurrence loop, training loop, or data pipeline are needed —
the loop always calls `self.gate_fn(...)` at every step; it is never bypassed.

Padding is handled by masking (not `pack_padded_sequence`): `h` is only
updated at real-token timesteps, so padding tokens never influence the
recurrence, and left-padding is handled correctly regardless of the
padding's position (verified: running a padded sequence vs. its unpadded
tail alone produces identical output logits).

The final (last real-token) hidden state feeds a `Linear(hidden_dim,
num_items+1)` to produce logits over the full item vocabulary; the padding
index's logit is masked to `-1e9` before it's returned, excluding it from
loss and from top-K predictions.

## Training (`train.py`)

- `CrossEntropyLoss(ignore_index=0)` (padding index excluded from the loss).
- Adam, `lr=1e-3` (the standard default; no evidence in this phase to deviate).
- `batch_size=256`, up to `NUM_EPOCHS=20`, early stopping on validation
  Recall@20 with patience 5.
- Best checkpoint (by val Recall@20) saved to
  `checkpoints/gru4rec_baseline/best.pt`, containing model weights, the model
  config (needed to reconstruct the architecture), and the training config
  used. The item/category id mapping JSONs are copied alongside it. Future
  phases should load and **freeze** this checkpoint rather than retraining.

## Evaluation (`evaluate.py`)

Recall@10, Recall@20, MRR@10, MRR@20 on the test split, computed by ranking
the true next item among all item scores (padding index excluded). Results
are appended to `results/results.json` as a list of rows:

```json
[{"mechanism_name": "no_adaptation_baseline", "recall@10": ..., "recall@20": ..., "mrr@10": ..., "mrr@20": ...}]
```

Future phases append one row per drift-response mechanism to this same file
(re-running `evaluate.append_result` replaces any row with the same
`mechanism_name`, so re-evaluating a mechanism doesn't duplicate rows).
