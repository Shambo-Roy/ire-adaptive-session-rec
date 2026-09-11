# Design Document — Adaptive Session Recommendation Under Intent Drift

This is a **living document**, updated after every session/milestone. It is
the source of truth for the final project report: what was built, why, what
it costs (throughput/latency), and — critically — the full history of any
component that was changed or replaced, with the old version's details kept
rather than deleted.

How to read this doc: sections are organized by project phase. Within a
phase, each component has a "Decision" (what/why) and, where measured, a
"Performance" note. A **Changelog** at the bottom timestamps every revision.
When a future phase replaces something described here, the old entry is kept
under a "(superseded — see Phase N)" note, not deleted.

---

## Project overview

Multi-phase research project on session-based recommendation under
**intent drift** (a user's within-session interest shifting mid-session).

- **Phase 1 (this phase):** GRU4Rec baseline, no drift adaptation. Establishes
  the data pipeline, encoder, training/eval harness, and — importantly — an
  extensibility hook in the encoder that later phases plug into.
- **Phase 2 (planned, not yet started):** synthetic drift injection via
  session-splicing, using the real per-item category labels already carried
  through the Phase 1 pipeline.
- **Phase 3 (planned, not yet started):** 3–4 swappable drift-response
  mechanisms (fixed decay, CUSUM, ADWIN, a KL-divergence gate, a learned
  classifier) that intervene on GRU4Rec's hidden state between timesteps,
  via the hook built in Phase 1.
- **Phase 4 (planned, not yet started):** recovery-step and drift-specific
  evaluation.

---

## Phase 1 — GRU4Rec baseline

### 1. Environment / stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.12.10 | given |
| ML framework | PyTorch 2.13.0 (CPU build, no CUDA) | only backend available in this environment (`torch.cuda.is_available() == False`); see [Performance & known constraints](#performance--known-constraints) for the impact |
| Data handling | pandas 3.0.5, numpy 2.5.2 | standard, sufficient for a ~1M-row CSV; no need for a distributed/out-of-core tool at this scale |
| Threading | PyTorch default (14 CPU threads detected) | no explicit `torch.set_num_threads` tuning done in Phase 1 |

No GPU was available in the execution environment. This is the single
biggest performance constraint of Phase 1 and is expected to remain a
constraint through later phases unless the environment changes — see
throughput numbers below and revisit if/when a GPU becomes available
(nn.GRUCell-in-a-loop would very likely need to become batched/vectorized
differently to benefit from a GPU; noted as a future risk, not yet acted on).

### 2. Dataset

Source: RecSys Challenge 2015-style session click-stream data.

- `archive/train-item-views.csv` (semicolon-separated despite `.csv`):
  `sessionId;userId;itemId;timeframe;eventdate` — 1,235,380 rows, 310,324
  unique sessions, 122,993 unique items, 2016-01-01 to 2016-06-01.
  `userId` is ~70% NA and is ignored — this is **session-based**, not
  user-based, recommendation. `timeframe` (ms within a session) is only
  valid for intra-session ordering.
- `archive/product-categories.csv`: `itemId;categoryId` — 184,047 items,
  1,217 categories. Verified 100% coverage against train-item-views'
  itemIds (0 missing on a left join) before writing any code against it.

Both files were located locally under `archive/` (not at the
`/mnt/user-data/uploads/` path originally given — that path doesn't exist in
this Windows environment; the files were found in the project directory
instead, and their contents were verified against the stated
schema/row-counts before use, per the task's "stop and report if it differs"
instruction).

### 3. Preprocessing (`data.py`)

**Decision: filter before splitting, not after.** Session-length (≥2) and
item-support (≥5) filtering is applied to the *whole* dataset before the
time-based train/val/test split (rather than filtering each split
independently). This follows the literal step ordering given in the task
spec and is also standard practice in the GRU4Rec/session-rec literature —
it keeps the item vocabulary and support statistics consistent across the
whole timeline instead of being skewed by whichever split a rare item lands
in.
- Result: 310,324 → 204,789 sessions, 122,993 → 43,136 items, 993,483 events
  remaining.
- Re-filtering session length after the item-support filter was necessary:
  dropping rare items shortens some sessions below the length-2 threshold.

**Decision: build item/category id→index mappings over the whole filtered
dataset (pre-split), then drop val/test events for items unseen in train
specifically (post-split).** These are two different things and both were
implemented, matching the task's explicit step ordering (mapping built in
steps 5–6, split in step 7, unseen-item drop in step 8):
- The mapping (and therefore the embedding table size) covers every item
  that survived the global support filter, whether or not that item
  happens to fall inside the train date range.
- Separately, val/test *rows* whose item never appears in train are dropped
  (259/56,963 val events, 495/77,061 test events — ~0.5–0.6% each), because
  the model has no trained embedding for such an item and scoring against it
  would be predicting from a random, untrained vector.
- Index 0 is reserved for padding in both the item and category mappings
  (and doubles as "no category" for the category mapping, for the missing-
  category fallback path that data confirmed is never hit on this dataset).

**Decision: time-based split cutoff = last 7 days → test, prior 7 days →
val, rest → train**, based on each session's *earliest* eventdate (only 260
of 310,324 sessions span two calendar dates; those use the earlier date).
7/7 days was chosen after inspecting daily session-count histograms near the
end of the range — it produces a test set that's a reasonable ~8% of
sessions (not near-empty, not enormous) with a comparably-sized val set for
early stopping. Post-filtering session counts: train 176,976 (< 2016-05-19),
val 11,835 (2016-05-19–05-25), test 15,978 (2016-05-26–06-01).

**Decision: sequence augmentation, one (prefix → next_item) pair per
non-first event in a session**, generated independently per split (train
pairs only from train sessions, etc.) — this is the standard GRU4Rec /
"session-parallel mini-batch" style training-signal expansion, not a novel
choice. Result: 682,483 / 44,878 / 60,600 train/val/test pairs.

**Decision: `PAD_LEN = 19`**, chosen from the actual unpadded prefix-length
distribution over the 682,483 train pairs (mean 4.13, median 3, p95=12,
**p99=19**, max=69). 19 covers ~99% of prefixes with no padding waste;
prefixes longer than 19 are truncated to their most recent 19 items (the
tail closest to the target — the end of the session is assumed to be the
most predictive context), rather than paying full compute for the small
tail of sessions running up to length 69. This is a straightforward
recall/compute tradeoff — not revisited yet, but a candidate to sweep in a
later phase if baseline accuracy looks limited by context truncation.

**Decision: category sequences are computed and cached now, even though
Phase 1's model doesn't consume them** (`processed/{split}/categories.npy`,
aligned index-for-index with `items.npy`, same padding/truncation) — this is
purely to avoid re-deriving them from raw data in Phase 2/3, which need
per-item category labels for drift injection and drift detection.

**Performance:** preprocessing (`python data.py`) runs in well under a
minute end-to-end on the full 1.2M-row CSV on this machine (pandas
groupby/merge dominates; not separately profiled since it's a one-time,
non-bottleneck step — will revisit if this ever needs to run repeatedly,
e.g. once Phase 2's splicing needs its own pass over the data).

**Disk footprint of `processed/`:** train 209 MB, val 14 MB, test 19 MB
(item/length/target/category `.npy` arrays), item mapping JSON 676 KB
(43,136 items), category mapping JSON 12 KB (995 categories).

### 4. Model (`model.py`) — GRU4Rec + the drift-mechanism hook

**Decision: hand-rolled per-timestep recurrence loop using `nn.GRUCell`,
instead of `nn.GRU` + `pack_padded_sequence`.** This is a deliberate
architectural choice made *specifically* for extensibility, not for
accuracy or speed — `nn.GRU` is the faster, more standard choice for a plain
GRU4Rec baseline (it runs the recurrence in cuDNN/optimized C++ rather than
a Python loop), but it does not expose the per-step hidden state, so there
is nowhere for a future drift-response mechanism to intervene between
timesteps. The task requires that later phases be able to plug in without
rewriting the encoder, so the loop-based `GRUCell` approach was chosen up
front, accepting its throughput cost now (see Performance below) in exchange
for not having to re-architect the encoder in Phase 3.

  - **Hook interface:**
    ```python
    def default_gate(hidden_state, step_info):
        return hidden_state   # no-op; future drift mechanisms replace this
    ```
    Called at *every* timestep (real or padding) inside `forward()`,
    immediately after `nn.GRUCell` produces the candidate hidden state and
    before it's merged into the running hidden state:
    ```python
    h_candidate = self.gru_cell(x_t, h)
    step_info = {"t": t, "mask": mask_t, "item_idx": item_t, "cat_idx": cat_t}
    h_candidate = self.gate_fn(h_candidate, step_info)   # hook, never bypassed
    h = mask_t * h_candidate + (1 - mask_t) * h
    ```
    A future mechanism plugs in via `GRU4Rec(..., gate_fn=my_mechanism)` or
    by reassigning `model.gate_fn` — no change to `forward()`,
    `train.py`, or `data.py` needed. `step_info` already carries what a
    CUSUM/ADWIN/KL-gate/classifier mechanism is likely to need (timestep
    index, real-vs-padding mask, current item id, current category id)
    without having to thread new state through the loop later.
  - Verified (smoke test, not a unit test file yet) that: (a) padding never
    influences the hidden state — a left-padded sequence and its unpadded
    tail alone produce bit-identical output logits; (b) swapping in a
    non-identity `gate_fn` measurably changes model output, confirming the
    hook is live and not dead code.

**Decision: masking instead of `pack_padded_sequence`.** Padding-awareness
is implemented by zeroing the hidden-state update (not the hidden state
itself) on padding timesteps: `h = mask * h_candidate + (1-mask) * h`. This
was necessary because `pack_padded_sequence` is designed for right-padded,
length-sorted batches feeding `nn.GRU`'s optimized kernel — it doesn't
compose with a manual `GRUCell` loop over left-padded sequences the way this
architecture needs. The mask approach works correctly regardless of padding
position (left-padding, as chosen here, works exactly like right-padding
would) and was the direct enabler of also inserting the drift hook.

**Decision: left-padding** (not right-padding) for item/category sequences.
Chosen for consistency with common GRU4Rec/session-rec preprocessing
conventions where "the end of the array is always the most recent event" —
this made the truncation rule in preprocessing (§3) and the "final hidden
state = last real token" logic in the model align naturally, with no
special-casing needed for variable-length sequences within a fixed-size
tensor batch.

**Architecture summary:** `Embedding(num_items+1, embedding_dim=100,
padding_idx=0)` → `GRUCell(100, hidden_dim=100)` (looped over `T=19`
timesteps, gated by `default_gate`) → `Dropout(0.1)` → `Linear(100,
num_items+1)`. Padding index's output logit is masked to `-1e9` before
being returned (excluded from loss and top-K predictions).

### 5. Training (`train.py`)

- `CrossEntropyLoss(ignore_index=0)` — padding excluded from the loss.
- Adam, `lr=1e-3` (standard default; no tuning done in Phase 1, no evidence
  yet that it needs to change).
- `batch_size=256`, up to `NUM_EPOCHS=20`, early stopping on validation
  Recall@20, patience 5.
- Best-checkpoint selection by val Recall@20 (not train loss / val loss) —
  Recall@20 is the metric this baseline will ultimately be compared against
  future drift-response mechanisms on, so selecting the checkpoint by the
  same metric avoids a train/eval-objective mismatch.
- Checkpoint (`checkpoints/gru4rec_baseline/best.pt`, 34.9 MB) bundles model
  weights, model-reconstruction config, and training config used — plus the
  item/category id mapping JSONs are copied alongside it — so future phases
  can load and **freeze** it as a fixed encoder without needing anything
  else from this phase's run.

**Actual run (2026-09-11):** early-stopped at epoch 13 (no val Recall@20
improvement for 5 epochs after the peak). Best checkpoint at **epoch 8**:
train loss 4.0137, val Recall@20 = 0.3773, val MRR@20 = 0.1230.

Per-epoch train loss and val metrics (full curve, useful for the report):

| epoch | train loss | val recall@20 | val mrr@20 | epoch wall time |
|---|---|---|---|---|
| 1 | 9.4681 | 0.1988 | 0.0707 | 211.9s |
| 2 | 7.2369 | 0.3031 | 0.1043 | 211.8s |
| 3 | 6.0390 | 0.3440 | 0.1157 | 211.2s |
| 4 | 5.3292 | 0.3632 | 0.1217 | 210.2s |
| 5 | 4.8532 | 0.3717 | 0.1229 | 210.4s |
| 6 | 4.5041 | 0.3760 | 0.1235 | 209.9s |
| 7 | 4.2331 | 0.3768 | 0.1229 | 210.6s |
| **8** | **4.0137** | **0.3773 (best)** | 0.1230 | 206.0s |
| 9 | 3.8305 | 0.3737 | 0.1210 | 206.4s |
| 10 | 3.6750 | 0.3698 | 0.1181 | 227.0s |
| 11 | 3.5402 | 0.3682 | 0.1177 | 223.3s |
| 12 | 3.4207 | 0.3665 | 0.1162 | 230.9s |
| 13 | 3.3178 | 0.3622 | 0.1156 | 217.1s |

Overfitting is visible from epoch 8 onward (train loss keeps falling, val
Recall@20 falls) — expected and exactly what early stopping is for; not
treated as a problem requiring a design change in Phase 1.

### 6. Evaluation (`evaluate.py`)

Recall@10/20 and MRR@10/20 on the test split, computed by ranking the true
next item among all item scores (`rank = (logits > true_score).sum() + 1`,
i.e. a direct score-comparison rather than a full `argsort` — chosen for
throughput on a ~43K-item vocabulary, see Performance below).

**Decision: results stored as an append/replace-by-`mechanism_name` list in
`results/results.json`**, not overwritten wholesale — so Phase 3's multiple
drift-response mechanisms can each add a row without disturbing the others,
and re-running the same mechanism updates its row in place instead of
duplicating it.

**Test results (2026-09-11, `no_adaptation_baseline`):**

| mechanism | recall@10 | recall@20 | mrr@10 | mrr@20 |
|---|---|---|---|---|
| no_adaptation_baseline | 0.2622 | 0.3614 | 0.1093 | 0.1162 |

(Test Recall@20/MRR@20 are slightly below the best validation values above,
0.3773/0.1230 — consistent with normal val→test generalization gap, not
investigated further in Phase 1.)

### Performance & known constraints

CPU-only execution (no CUDA available) is the dominant constraint on this
phase's runtime. Component-level microbenchmarks (`batch_size=256`,
`embedding_dim=hidden_dim=100`, `PAD_LEN=19`, 14 CPU threads, measured
2026-09-11):

| component | throughput | latency |
|---|---|---|
| DataLoader iteration only (no model) | ~131,400 examples/s | ~1.95 ms/batch |
| Model forward pass only (`eval()`, no grad) | ~16,749 examples/s | ~15.3 ms/batch |
| Full train step (forward + backward + Adam) | ~3,800–4,600 examples/s | ~56–67 ms/batch |
| Full validation pass (44,878 examples, forward + ranking) | ~11,000 examples/s | — |
| Full train epoch (682,483 examples, incl. shuffling + train loop) | — | ~206–231s (~3.5–3.9 min) |

The gap between "forward-only" (~16.7K examples/s) and "full train step"
(~3.8–4.6K examples/s) shows the backward pass + optimizer step — not the
per-timestep Python loop itself — dominates training time; the DataLoader is
never the bottleneck (>30x faster than the model). This confirms the
`GRUCell`-loop-for-extensibility tradeoff (§4) costs real wall-clock time
(a plain `nn.GRU` would very likely close much of this gap by moving the
19-step recurrence out of the Python interpreter) but is not, on this
dataset size, prohibitively slow: a full training run (13 epochs to early
stopping) completed in **~46 minutes**. No component has been changed or
replaced for performance reasons yet in Phase 1 — this section exists to
give later phases a baseline to compare against if/when the `GRUCell` loop
or CPU-only execution does become a bottleneck (e.g. if Phase 3's mechanisms
add meaningful per-step compute inside the hook, or if the dataset/epoch
budget grows).

### Open questions / risks carried into later phases

- **GPU availability.** If a GPU becomes available, the `GRUCell`-loop
  architecture will not automatically get cuDNN-level speedups the way
  `nn.GRU` would — worth revisiting whether the loop can be vectorized
  (e.g. `torch.jit.script`, or restructuring the hook to operate on a whole
  batch of steps) without losing the per-step hook. Not addressed in Phase 1.
- **`PAD_LEN=19` truncation** drops the earliest part of the session for the
  ~1% of prefixes longer than 19 items — for a drift-detection use case
  where the *early* part of a session might contain the "before drift"
  signal, this truncation could matter more than it does for a plain
  baseline. Flagged for reconsideration when Phase 2 (drift injection) is
  designed, not changed now.
- **Item mapping built pre-split** (§3) means the embedding table has slots
  for a small number of items that never appear in train (only observed in
  val/test date ranges) — those slots are never trained. Harmless for
  Phase 1 (such items are excluded from val/test predictions per the
  unseen-item drop), but worth remembering if a future phase inspects
  embedding-table statistics directly.

---

## Changelog

- **2026-09-11** — Phase 1 initial implementation and this document created.
  Data verified against spec, preprocessing pipeline built, GRU4Rec baseline
  + hook interface implemented, trained (13 epochs, early stopped), evaluated
  on test split. No components replaced yet — nothing to supersede.
