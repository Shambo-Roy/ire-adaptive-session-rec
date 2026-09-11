"""GRU4Rec baseline model with an explicit per-timestep hidden-state hook.

Future phases will implement drift-response mechanisms (fixed decay, CUSUM,
ADWIN, a KL-divergence gate, a learned classifier) that intervene on the
GRU's hidden state between timesteps -- e.g. resetting/attenuating it when a
change in intent is detected mid-session. To support that without touching
this module's recurrence loop, the forward pass is written as an explicit
per-step Python loop (using nn.GRUCell rather than nn.GRU) and calls a
`gate_fn(hidden_state, step_info) -> hidden_state` hook at *every* timestep,
right after the new hidden state is computed and before it is used as the
input to the next step.

For this phase, `gate_fn` defaults to `default_gate`, a no-op passthrough.
A future drift mechanism plugs in by passing a different `gate_fn` to
`GRU4Rec(...)` (or by monkey-patching `model.gate_fn`) -- no changes to
`forward()` are needed. `step_info` carries whatever a mechanism might need
to make a decision: the timestep index, which sequences are on a real
(non-padding) token at this step, and the item/category ids just consumed.
"""
import torch
import torch.nn as nn

import config


def default_gate(hidden_state, step_info):
    """No-op passthrough hook.

    hidden_state: [batch, hidden_dim] tensor, the GRU's hidden state after
        processing timestep `step_info['t']`.
    step_info: dict with keys:
        - 't': int, current timestep index (0-indexed, over the padded sequence)
        - 'mask': [batch, 1] float tensor, 1.0 where this position is a real
          (non-padding) token, 0.0 where it is padding
        - 'item_idx': [batch] long tensor, the item index consumed at this step
        - 'cat_idx': [batch] long tensor or None, the aligned category index
          consumed at this step (None if the caller didn't provide categories)

    Future drift-response mechanisms replace this function to intervene on
    the hidden state (e.g. decay/reset it) based on step_info. It must
    return a tensor of the same shape as hidden_state.
    """
    return hidden_state


class GRU4Rec(nn.Module):
    def __init__(
        self,
        num_items,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        pad_idx=config.PAD_IDX,
        dropout=config.DROPOUT,
        gate_fn=None,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.gate_fn = gate_fn if gate_fn is not None else default_gate

        # +1 for the padding token (index 0); real items occupy 1..num_items
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=pad_idx)
        self.gru_cell = nn.GRUCell(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, item_seq, lengths, cat_seq=None):
        """item_seq: [B, T] long, left-padded with pad_idx.
        lengths: [B] long, true (unpadded) sequence lengths -- used only for
            sanity/assertions here; masking is derived directly from item_seq
            so left-padding is handled correctly regardless of position.
        cat_seq: optional [B, T] long, category indices aligned with item_seq
            (passed through to the hook via step_info; unused by this model).

        Returns logits: [B, num_items+1] over the full item vocabulary,
        with the padding index masked out.
        """
        B, T = item_seq.shape
        device = item_seq.device

        embedded = self.embedding(item_seq)  # [B, T, E]
        h = torch.zeros(B, self.hidden_dim, device=device)

        for t in range(T):
            x_t = embedded[:, t, :]
            item_t = item_seq[:, t]
            mask_t = (item_t != self.pad_idx).float().unsqueeze(-1)  # [B, 1]

            h_candidate = self.gru_cell(x_t, h)

            step_info = {
                "t": t,
                "mask": mask_t,
                "item_idx": item_t,
                "cat_idx": cat_seq[:, t] if cat_seq is not None else None,
            }
            h_candidate = self.gate_fn(h_candidate, step_info)

            # Padding tokens must not influence the hidden state: only update
            # h on real-token timesteps, otherwise carry the previous h forward.
            h = mask_t * h_candidate + (1 - mask_t) * h

        h = self.dropout(h)
        logits = self.output_layer(h)  # [B, num_items+1]
        logits[:, self.pad_idx] = -1e9  # exclude padding index from predictions
        return logits
