"""Central configuration for the GRU4Rec baseline pipeline (Phase 1).

All paths and hyperparameters live here so later phases (synthetic drift
injection, drift-response mechanisms) can import and override values without
touching data.py / model.py / train.py / evaluate.py.
"""
from pathlib import Path

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "archive"
TRAIN_ITEM_VIEWS_CSV = RAW_DIR / "train-item-views.csv"
PRODUCT_CATEGORIES_CSV = RAW_DIR / "product-categories.csv"

PROCESSED_DIR = PROJECT_ROOT / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "gru4rec_baseline"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"

# --- preprocessing ---
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# Time-based split by each session's earliest eventdate:
# last TEST_DAYS days -> test, the VAL_DAYS days before that -> val, rest -> train.
# See README.md for the resulting session counts and why 7/7 was chosen.
TEST_DAYS = 7
VAL_DAYS = 7

# Left-pad item/category prefixes to this length; longer prefixes are
# truncated to their most recent PAD_LEN items. Chosen from the observed
# prefix-length distribution (mean 4.1, median 3, p95=12, p99=19) -- see
# README.md for the full distribution and justification.
PAD_LEN = 19

PAD_IDX = 0  # index 0 is reserved for padding in both item and category vocabularies

# --- model ---
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
DROPOUT = 0.1

# --- training ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0
SEED = 42

# --- evaluation ---
EVAL_KS = (10, 20)
