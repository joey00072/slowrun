"""
NCA Pre-Pre-Training — Stage 1 (Tiny Track)
===========================================================
Implements the pre-pre-training stage from:
  "Training Language Models via Neural Cellular Automata"
  https://arxiv.org/abs/2603.10055

Trains the slowrun GPT architecture on synthetic NCA token sequences 
generated entirely in PyTorch (no JAX), then saves a checkpoint for 
Stage 2 (language pre-training via tiny/train.py --nca-checkpoint).

Usage:
    torchrun --standalone --nproc_per_node=8 tiny/nca_pretraining.py
    torchrun --standalone --nproc_per_node=8 tiny/nca_pretraining.py \
        --nca-tokens 164000000 \
        --save-checkpoint tiny_nca_ppt.pt \
        --run nca_ppt_tiny
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import argparse
import gzip
import io
from types import SimpleNamespace
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import wandb

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="NCA Pre-Pre-Training (Stage 1)")

# NCA hyperparameters
parser.add_argument(
    "--nca-grid", type=int, default=12, help="NCA grid size H=W (paper: 12)"
)
parser.add_argument(
    "--nca-patch",
    type=int,
    default=2,
    help="Patch size for tokenization (paper: 2, gives 10^4 vocab)",
)
parser.add_argument(
    "--nca-colors", type=int, default=10, help="Number of NCA states/colors (paper: 10)"
)
parser.add_argument(
    "--nca-steps",
    type=int,
    default=16,
    help="Number of NCA simulation steps per example",
)
parser.add_argument(
    "--nca-examples",
    type=int,
    default=0,
    help="Number of grids per trajectory; <=0 derives it from --ppt-seq-len like the reference code",
)
parser.add_argument(
    "--identity-bias",
    type=float,
    default=0.0,
    help="Identity bias for NCA transitions (reference default: 0.0)",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1e-4,
    help="NCA sampling temperature (reference script: 1e-4)",
)
parser.add_argument(
    "--init-rollout-steps",
    type=int,
    default=10,
    help="Burn-in steps before recording NCA frames (reference script: 10)",
)

# Token budget
parser.add_argument(
    "--nca-tokens",
    type=int,
    default=164_000_000,
    help="Total NCA tokens to train on (paper: 164M for improvement)",
)
parser.add_argument(
    "--device-batch-size", type=int, default=32, help="Sequences per GPU per micro-step"
)
parser.add_argument(
    "--total-batch-size",
    type=int,
    default=524288,
    help="Total tokens per optimizer step (matches tiny train.py)",
)

# Model (must match tiny/train.py defaults)
parser.add_argument("--n-layer", type=int, default=16)
parser.add_argument("--n-head", type=int, default=8)
parser.add_argument("--n-embd", type=int, default=1024)

# NCA complexity filtering (KEY from paper)
parser.add_argument(
    "--complexity-threshold",
    type=float,
    default=0.5,
    help="Min gzip compression ratio for rule filtering (paper: 0.5)",
)
parser.add_argument(
    "--complexity-upper-bound",
    type=float,
    default=1.0,
    help="Max gzip compression ratio (paper: 1.0)",
)
parser.add_argument(
    "--num-rules",
    type=int,
    default=16000,
    help="Number of unique NCA rules (paper: 16000)",
)
parser.add_argument(
    "--val-rules",
    type=int,
    default=2000,
    help="Number of held-out validation rules (reference script: 2000)",
)
parser.add_argument(
    "--filter-rules",
    action="store_true",
    default=True,
    help="Enable complexity filtering (paper: enabled)",
)
parser.add_argument(
    "--no-filter-rules", action="store_true", help="Disable complexity filtering"
)

# Optimizer
parser.add_argument(
    "--lr", type=float, default=1e-4, help="Peak learning rate for stage-1 Adam"
)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--warmup-frac", type=float, default=0.1)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--grad-clip-enable", action="store_true")
parser.add_argument(
    "--ppt-vocab-size",
    type=int,
    default=64000,
    help="Synthetic-stage model vocab size (reference script: 64000)",
)
parser.add_argument(
    "--ppt-seq-len",
    type=int,
    default=1024,
    help="Synthetic-stage transformer context length (reference script: 1024)",
)

# Output
parser.add_argument(
    "--save-checkpoint",
    type=str,
    default="tiny_nca_ppt.pt",
    help="Path to save NCA prepretrained checkpoint",
)
parser.add_argument("--save-result", type=str, default="tiny_nca_ppt_result.json")
parser.add_argument("--run", type=str, default=None, help="Wandb run name")
parser.add_argument("--wandb-group", type=str, default="nca_ppt", help="Wandb group")
parser.add_argument("--log-interval", type=int, default=50)
parser.add_argument("--eval-interval", type=int, default=500)
parser.add_argument("--eval-steps", type=int, default=20)
parser.add_argument(
    "--refresh-rules-every",
    type=int,
    default=500,
    help="Regenerate the filtered training rule bank every N optimizer steps",
)

args = parser.parse_args()

# =============================================================================
# Derived constants
# =============================================================================

# NCA vocabulary: each patch encodes patch^2 cells, each with num_colors values
# vocab_size = num_colors^(patch^2) + 2 special tokens (BOS/EOS per step)
NCA_PATCH = args.nca_patch
NCA_COLORS = args.nca_colors
NCA_GRID = args.nca_grid
NCA_VOCAB = NCA_COLORS ** (NCA_PATCH * NCA_PATCH)  # 10^4 = 10000
NCA_BOS = NCA_VOCAB  # special start-of-grid token
NCA_EOS = NCA_VOCAB + 1  # special end-of-grid token
NCA_VOCAB_TOTAL = NCA_VOCAB + 2  # 10002
PPT_MODEL_VOCAB = args.ppt_vocab_size

# Sequence layout: each NCA "grid" becomes (grid_patches + 2) tokens
# grid_patches = (grid/patch)^2
NCA_GRID_PATCHES = (NCA_GRID // NCA_PATCH) ** 2  # 36 for 12x12 grid, patch=2
TOKENS_PER_GRID = NCA_GRID_PATCHES + 2  # 38 (BOS, 36 patch tokens, EOS)
NCA_EXAMPLES = (
    args.nca_examples
    if args.nca_examples > 0
    else math.ceil(args.ppt_seq_len / TOKENS_PER_GRID)
)
SEQ_LEN = TOKENS_PER_GRID * NCA_EXAMPLES

DEPTH = args.n_layer
N_EMBD = args.n_embd
N_HEAD = args.n_head
HEAD_DIM = N_EMBD // N_HEAD
WINDOW_PATTERN = "SSSL"
MAX_SEQ_LEN = max(args.ppt_seq_len, 512)


# =============================================================================
# Utilities
# =============================================================================


def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return (
            True,
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
    return False, 0, 0, 1


def print0(s="", **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kwargs)


class DummyWandb:
    def __init__(self):
        self.summary = {}

    def log(self, *a, **kw):
        pass

    def finish(self):
        pass

    def log_code(self, *a, **kw):
        pass


# =============================================================================
# Flash Attention (FA3 on Hopper, SDPA fallback elsewhere)
# =============================================================================


def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel

        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


_fa3 = _load_fa3()


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=enable_gqa
        )
    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k, v = k[:, :, start:, :], v[:, :, start:, :]
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=enable_gqa
        )
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=enable_gqa
    )


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# =============================================================================
# GPT Model — identical to tiny/train.py but with NCA vocab
# =============================================================================


@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = NCA_VOCAB_TOTAL
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.0  # no dropout during PPT


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)
        pattern = config.window_pattern.upper()
        char = pattern[layer_idx % len(pattern)]
        self.use_key_offset = (char == "L") or (layer_idx == config.n_layer - 1)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        if self.use_key_offset and T > 1:
            k[:, 1:, :, self.head_dim // 2 :] = k[
                :, :-1, :, self.head_dim // 2 :
            ].clone()
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y * torch.sigmoid(
            self.attn_gate(x[..., : self.attn_gate_channels])
        ).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict(
            {
                str(i): nn.Linear(config.n_embd, kv_dim, bias=False)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.1)
        self.x0_lambdas.fill_(0.1)
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
        self.skip_weights.fill_(1.0)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        half = head_dim // 4
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, half * 2, 2, dtype=torch.float32, device=device)
                / (half * 2)
            )
        )
        inv_freq = torch.cat(
            [
                inv_freq,
                torch.zeros(head_dim // 2 - half, dtype=torch.float32, device=device),
            ]
        )
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        skip_connections = []
        for i, block in enumerate(self.transformer.h):
            if i >= self.encoder_layers and skip_connections:
                skip = skip_connections.pop()
                x = x + self.skip_weights[i - self.encoder_layers] * skip
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i < self.encoder_layers:
                skip_connections.append(x)
        x = norm(x)
        logits = self.lm_head(x)[..., : self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # softcap (same as tiny/train.py)
        if targets is not None:
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
        return logits


# =============================================================================
# Pure-PyTorch NCA Data Generator
# =============================================================================


class NCADataGenerator:
    """
    Discrete Neural Cellular Automaton implemented in pure PyTorch.

    Rules are random CNNs; each rule defines unique spatiotemporal dynamics.
    Trajectories are tokenized into patch-level tokens for causal LM training.

    Key design:
    - Runs entirely on GPU → no data loading bottleneck
    - Each DDP rank uses independent seeds → no communication for data gen
    - Complexity filtering: skip boring/trivial rules (optional)
    """

    def __init__(
        self,
        grid=12,
        patch=2,
        num_colors=10,
        identity_bias=2.0,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        assert grid % patch == 0, "grid must be divisible by patch"
        self.grid = grid
        self.patch = patch
        self.P = patch * patch  # pixels per patch
        self.num_colors = num_colors
        self.identity_bias = identity_bias
        self.device = device
        self.dtype = dtype
        self.n_patches = (grid // patch) ** 2

        # Vocabulary
        self.vocab = num_colors ** (patch * patch)  # e.g. 10^4 = 10000
        self.bos = self.vocab
        self.eos = self.vocab + 1

        # Precompute patch-to-token powers: token = sum(color[i] * num_colors^i)
        powers = (num_colors ** torch.arange(self.P, device=device)).long()
        self.register_powers(powers)

    def register_powers(self, powers):
        self.powers = powers  # (P,)

    def sample_rule(self, seed: int):
        """Sample a reference-like 3-layer NCA rule from a seed."""
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        def sample_weight(*shape):
            fan_in = 1
            for dim in shape[1:]:
                fan_in *= dim
            std = fan_in ** -0.5
            return torch.empty(*shape, device=self.device, dtype=self.dtype).normal_(
                mean=0.0, std=std, generator=rng
            )

        w1 = sample_weight(
            4,
            self.num_colors,
            3,
            3,
        )
        b1 = torch.zeros(4, device=self.device, dtype=self.dtype)
        w2 = sample_weight(
            16,
            4,
            1,
            1,
        )
        b2 = torch.zeros(16, device=self.device, dtype=self.dtype)
        w3 = sample_weight(
            self.num_colors,
            16,
            1,
            1,
        )
        b3 = torch.zeros(self.num_colors, device=self.device, dtype=self.dtype)
        return (w1, b1, w2, b2, w3, b3)

    def nca_step(
        self, state: torch.Tensor, rule, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        One NCA step.
        state: (B, H, W) integer tensor of cell colors
        Returns: (B, H, W) next state
        """
        w1, b1, w2, b2, w3, b3 = rule
        B, H, W = state.shape

        # One-hot encode: (B, num_colors, H, W)
        state_oh = (
            F.one_hot(state.long(), self.num_colors).permute(0, 3, 1, 2).to(self.dtype)
        )

        # Circular (toroidal) padding
        state_pad = F.pad(state_oh, (1, 1, 1, 1), mode="circular")

        # Apply CNN
        h = F.conv2d(state_pad, w1, b1)  # (B, 4, H, W)
        h = F.conv2d(h, w2, b2)  # (B, 16, H, W)
        h = F.relu(h)
        logits = F.conv2d(h, w3, b3)  # (B, num_colors, H, W)

        # Identity bias: bias toward staying in current state
        logits = logits + self.identity_bias * state_oh

        # Stochastic update (Gumbel softmax argmax = categorical sampling)
        # Temperature close to 0 → deterministic; temperature = 1 → random
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8)))
        next_state = (
            logits / max(temperature, 1e-6) + gumbel
        ).argmax(dim=1)  # (B, H, W)
        return next_state

    def rollout(
        self, rule, n_steps: int, batch_size: int, seed: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Roll out NCA for n_steps, collecting states.
        Returns: (batch_size, n_steps, H, W) integer tensor
        """
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        H = W = self.grid

        # Reference implementation samples a rollout-specific categorical prior.
        init_logits = torch.randn(
            batch_size, self.num_colors, generator=rng, device=self.device, dtype=self.dtype
        )
        init_probs = torch.softmax(init_logits, dim=-1)
        state = torch.multinomial(
            init_probs,
            H * W,
            replacement=True,
            generator=rng,
        ).view(batch_size, H, W)

        # Burn-in steps (not recorded)
        for _ in range(args.init_rollout_steps):
            state = self.nca_step(state, rule, temperature)

        states = []
        for _ in range(n_steps):
            states.append(state.clone())
            state = self.nca_step(state, rule, temperature)

        return torch.stack(states, dim=1)  # (B, T, H, W)

    def tokenize(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize NCA states into sequences with BOS/EOS per grid.

        states: (B, T, H, W) integer
        Returns: inputs (B, seq_len), targets (B, seq_len) with -1 for masked positions
        """
        B, T, _, _ = states.shape
        tokens = self.patch_tokens(states)
        n_patches = tokens.size(-1)

        # Build sequence with BOS/EOS wrapping each time step
        # Layout per time step: [BOS, patch_0, ..., patch_{n-1}, EOS]
        # Total per step: n_patches + 2
        step_len = n_patches + 2
        bos_col = torch.full((B, T, 1), self.bos, dtype=torch.long, device=self.device)
        eos_col = torch.full((B, T, 1), self.eos, dtype=torch.long, device=self.device)
        seq_3d = torch.cat([bos_col, tokens, eos_col], dim=2)  # (B, T, step_len)
        seq = seq_3d.view(B, T * step_len)  # (B, total_len)

        # Reference objective:
        # - causal next-token prediction over the flattened stream
        # - mask the entire first grid so the model conditions on it instead
        #   of spending capacity reconstructing the prompt grid
        inp = seq[:, :-1]  # (B, total_len-1)
        tgt = seq[:, 1:].clone()  # (B, total_len-1)
        tgt[:, : step_len - 1] = -1

        return inp, tgt

    def patch_tokens(self, states: torch.Tensor) -> torch.Tensor:
        B, T, H, W = states.shape
        P = self.patch
        Np = H // P
        n_patches = Np * Np
        s = states.view(B, T, Np, P, Np, P)
        s = s.permute(0, 1, 2, 4, 3, 5)
        s = s.reshape(B, T, n_patches, self.P)
        return (s.long() * self.powers.view(1, 1, 1, -1)).sum(-1)

    @torch.no_grad()
    def generate_batch(
        self,
        batch_size: int,
        n_steps: int,
        seed: int,
        n_rules: int = 1,
        temperature: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of NCA training sequences.
        Uses n_rules distinct rules, rolling out batch_size // n_rules trajectories per rule.

        Returns: (inputs, targets) each (batch_size, seq_len)
        """
        assert batch_size % n_rules == 0 or n_rules == 1
        per_rule = max(1, batch_size // n_rules)

        all_x, all_y = [], []
        for r in range(n_rules):
            rule_seed = seed * 10007 + r  # deterministic but diverse rule seeds
            rule = self.sample_rule(rule_seed)
            states = self.rollout(rule, n_steps, per_rule, seed + r + 1, temperature)
            x, y = self.tokenize(states)
            all_x.append(x)
            all_y.append(y)

        if len(all_x) == 1:
            return all_x[0], all_y[0]

        # Pad/truncate to the same length before stacking
        min_len = min(t.shape[1] for t in all_x)
        all_x = [t[:, :min_len] for t in all_x]
        all_y = [t[:, :min_len] for t in all_y]
        return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)

    @torch.no_grad()
    def generate_batch_from_rules(
        self,
        batch_size: int,
        n_steps: int,
        rules: list,
        rule_start_idx: int = 0,
        sample_seed: int = 0,
        temperature: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of NCA training sequences using pre-generated rules.
        Uses rules cyclically if batch_size > len(rules).

        Returns: (inputs, targets) each (batch_size, seq_len)
        """
        per_rule = max(1, batch_size // len(rules))

        all_x, all_y = [], []
        for r_idx, rule_seed in enumerate(rules):
            rule = self.sample_rule(rule_seed)
            rollout_seed = (
                sample_seed * 1_000_003 + rule_seed * 97 + rule_start_idx + r_idx + 1
            )
            states = self.rollout(
                rule, n_steps, per_rule, rollout_seed, temperature
            )
            x, y = self.tokenize(states)
            all_x.append(x)
            all_y.append(y)

        if len(all_x) == 1:
            return all_x[0], all_y[0]

        # Pad/truncate to the same length before stacking
        min_len = min(t.shape[1] for t in all_x)
        all_x = [t[:, :min_len] for t in all_x]
        all_y = [t[:, :min_len] for t in all_y]
        return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)


# =============================================================================
# Gzip Complexity Filtering (KEY from paper)
# =============================================================================


def compute_gzip_complexity(tokens: torch.Tensor) -> float:
    """
    Compute gzip compression ratio for a token sequence.
    Higher ratio = more complex/incompressible = more "interesting" dynamics.
    """
    byte_data = tokens.to(dtype=torch.int32).cpu().numpy().tobytes()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
        f.write(byte_data)
    compressed_size = len(buf.getvalue())
    original_size = len(byte_data)
    return compressed_size / original_size if original_size > 0 else 0.0


def generate_filtered_rules(
    nca_gen: NCADataGenerator,
    num_rules: int,
    threshold: float,
    upper_bound: float,
    n_steps: int = 10,
    batch_size: int = 1,
    max_attempts: int = 1000,
    seed: int = 42,
):
    """
    Generate NCA rules that pass the gzip complexity filter.

    The paper selects rules with 0.5 < gzip_ratio < 1.0 to get interesting dynamics.
    Too simple (gzip < 0.5) = boring repetitive patterns
    Too complex (gzip > 1.0) = random noise
    """
    filtered_rules = []
    attempts = 0
    found = 0
    rng = torch.Generator(device=nca_gen.device)

    while found < num_rules and attempts < max_attempts * num_rules:
        attempts += 1
        seed_val = seed * 1000007 + attempts

        # Sample a rule
        rule = nca_gen.sample_rule(seed_val)

        # Roll out to get trajectory
        states = nca_gen.rollout(
            rule, n_steps, batch_size, seed_val + 12345, temperature=args.temperature
        )

        # Reference code filters on the flattened patch-token stream, excluding BOS/EOS.
        flat_tokens = nca_gen.patch_tokens(states).reshape(states.size(0), -1)
        gzip_ratio = compute_gzip_complexity(flat_tokens[0])

        # Filter: keep rules with complexity in the desired range
        if threshold < gzip_ratio < upper_bound:
            filtered_rules.append((seed_val, rule, gzip_ratio))
            found += 1

            if found % 1000 == 0 and found > 0:
                print0(f"  Filtered rules found: {found}/{num_rules}")

    if found < num_rules:
        print0(f"  Warning: only found {found}/{num_rules} rules passing filter")

    return filtered_rules


def save_stage1_checkpoint(save_path: str, model: nn.Module):
    ckpt = {n: p.data.float().cpu() for n, p in model.named_parameters()}
    torch.save(ckpt, save_path)


def build_filtered_rule_bank(base_seed: int, num_rules: int):
    filtered_rules = generate_filtered_rules(
        nca_gen=nca_gen,
        num_rules=num_rules,
        threshold=args.complexity_threshold,
        upper_bound=args.complexity_upper_bound,
        n_steps=10,
        batch_size=1,
        seed=base_seed,
    )
    return [(r[0], r[2]) for r in filtered_rules]  # (seed, gzip_ratio)


# =============================================================================
# DDP + Device Init
# =============================================================================

ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(42 + ddp_rank)  # rank-specific seed for data diversity

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42 + ddp_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_type = device.type
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# FA3 status
if _fa3 is not None:
    print0("Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("Using PyTorch SDPA fallback (no FA3)")

# =============================================================================
# Wandb
# =============================================================================

enable_filtering = not args.no_filter_rules

run_name = args.run if args.run else f"nca_ppt_tiny_{time.strftime('%Y%m%d_%H%M%S')}"
wandb_run = DummyWandb()
if master_process:
    wandb_run = wandb.init(
        project="slowrun",
        name=run_name,
        group=args.wandb_group,
        config={
            "nca_grid": NCA_GRID,
            "nca_patch": NCA_PATCH,
            "nca_colors": NCA_COLORS,
            "nca_vocab": NCA_VOCAB_TOTAL,
            "ppt_model_vocab": PPT_MODEL_VOCAB,
            "seq_len": SEQ_LEN,
            "nca_tokens": args.nca_tokens,
            "n_layer": DEPTH,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "total_batch_size": args.total_batch_size,
            "device_batch_size": args.device_batch_size,
            "complexity_threshold": args.complexity_threshold,
            "complexity_upper_bound": args.complexity_upper_bound,
            "num_rules": args.num_rules,
            "filter_enabled": enable_filtering,
        },
    )
    wandb_run.log_code(".")

# =============================================================================
# Model
# =============================================================================

print0(f"\n--- NCA Pre-Pre-Training (Stage 1) ---")
print0(f"NCA token vocab: {NCA_VOCAB_TOTAL} | stage-1 model vocab: {PPT_MODEL_VOCAB} | seq_len: {SEQ_LEN}")
print0(f"NCA tokens budget: {args.nca_tokens:,}")
print0(f"n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}")
print0(f"Model: same GPT arch as tiny/train.py, NCA vocab")
print0(
    f"NCA dynamics: temperature={args.temperature}, identity_bias={args.identity_bias}, init_rollout_steps={args.init_rollout_steps}"
)
print0(
    f"Synthetic context: ppt_seq_len={args.ppt_seq_len}, grids_per_seq={NCA_EXAMPLES}, raw_seq_len={SEQ_LEN}"
)
print0(f"Complexity filtering: {'enabled' if enable_filtering else 'disabled'}")
if enable_filtering:
    print0(
        f"  threshold={args.complexity_threshold}, upper_bound={args.complexity_upper_bound}"
    )
    print0(f"  num_rules={args.num_rules}")

config = GPTConfig(
    vocab_size=PPT_MODEL_VOCAB,
    sequence_len=MAX_SEQ_LEN,
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
orig_model = model

param_count = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {param_count:,}")

model = torch.compile(model, dynamic=False)

# =============================================================================
# Optimizer — stage-1 reference uses Adam with warmup+cosine, no weight decay
# =============================================================================

optimizer = torch.optim.Adam(
    orig_model.parameters(),
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    fused=(device_type == "cuda"),
)

# =============================================================================
# NCA Data Generator
# =============================================================================

nca_gen = NCADataGenerator(
    grid=NCA_GRID,
    patch=NCA_PATCH,
    num_colors=NCA_COLORS,
    identity_bias=args.identity_bias,
    device=device,
)

SEQ_LEN_ACTUAL = nca_gen.n_patches + 2  # tokens per single-step grid
FULL_SEQ_LEN = args.ppt_seq_len

# =============================================================================
# Rule Generation (with complexity filtering)
# =============================================================================

if enable_filtering:
    print0(
        f"\nGenerating {args.num_rules + args.val_rules} filtered rules (threshold={args.complexity_threshold}, bound={args.complexity_upper_bound})..."
    )
    all_rule_data = build_filtered_rule_bank(
        base_seed=42,
        num_rules=args.num_rules + args.val_rules,
    )
    print0(f"Generated {len(all_rule_data)} filtered rules")
    rule_data = all_rule_data[: args.num_rules]
    val_rule_data = all_rule_data[args.num_rules :]
    print0(f"  train rules: {len(rule_data)} | val rules: {len(val_rule_data)}")
else:
    print0("\nUsing unfiltered random rules (baseline)")
    rule_data = None
    val_rule_data = None

# =============================================================================
# Training config
# =============================================================================

tokens_per_fwdbwd = args.device_batch_size * FULL_SEQ_LEN * ddp_world_size
grad_accum_steps = max(1, args.total_batch_size // tokens_per_fwdbwd)
tokens_per_step = tokens_per_fwdbwd * grad_accum_steps
total_steps = math.ceil(args.nca_tokens / tokens_per_step)
warmup_steps = max(1, round(args.warmup_frac * total_steps))
global_trajectories_per_step = args.device_batch_size * ddp_world_size * grad_accum_steps

print0(f"Seq len: {FULL_SEQ_LEN} tokens")
print0(f"Device batch size: {args.device_batch_size} | grad accum: {grad_accum_steps}")
print0(f"Effective batch size: {tokens_per_step:,} tokens")
print0(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
if enable_filtering and rule_data is not None and args.refresh_rules_every > 0:
    print0(f"Rule-bank regeneration: every {args.refresh_rules_every} optimizer steps")


def get_lr(step):
    if step < warmup_steps:
        return args.lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return args.lr * (0.5 * (1 + math.cos(math.pi * progress)))  # cosine decay to 0


# =============================================================================
# Training loop
# =============================================================================

step = 0
tokens_trained = 0
smooth_loss = 0.0
best_val_loss = float("inf")
best_step = 0
total_training_time = 0.0
timed_steps_count = 0
timing_start_step = 4

print0(f"\nStarting NCA pre-pre-training...")

while tokens_trained < args.nca_tokens:
    synchronize()
    t0 = time.time()

    if (
        enable_filtering
        and rule_data is not None
        and args.refresh_rules_every > 0
        and step > 0
        and step % args.refresh_rules_every == 0
    ):
        refresh_seed = 42 + step * 1_000_003
        print0(f"\nRefreshing filtered training rules at step {step}...")
        rule_data = build_filtered_rule_bank(refresh_seed, args.num_rules)
        print0(f"Refreshed {len(rule_data)} training rules")

    # Gradient accumulation
    total_loss = 0.0
    model.train()
    for micro_step in range(grad_accum_steps):
        if enable_filtering and rule_data is not None:
            # Use filtered rules (cyclically), offset by rank for diversity.
            n_rules = min(len(rule_data), args.device_batch_size)
            start_idx = (step * grad_accum_steps + micro_step + ddp_rank * 1000) % len(
                rule_data
            )
            rule_seeds = [
                rule_data[(start_idx + r) % len(rule_data)][0] for r in range(n_rules)
            ]
            sample_seed = (
                (step * grad_accum_steps + micro_step) * ddp_world_size + ddp_rank + 1
            )
            x, y = nca_gen.generate_batch_from_rules(
                batch_size=args.device_batch_size,
                n_steps=NCA_EXAMPLES,
                rules=rule_seeds,
                rule_start_idx=start_idx,
                sample_seed=sample_seed,
                temperature=args.temperature,
            )
        else:
            # Fallback to random rules
            batch_seed = step * ddp_world_size + ddp_rank + 1
            micro_seed = batch_seed * grad_accum_steps + micro_step
            x, y = nca_gen.generate_batch(
                batch_size=args.device_batch_size,
                n_steps=NCA_EXAMPLES,
                seed=micro_seed,
                n_rules=max(1, args.device_batch_size // 4),
                temperature=args.temperature,
            )

        # Truncate/pad to exact FULL_SEQ_LEN
        x = x[:, :FULL_SEQ_LEN]
        y = y[:, :FULL_SEQ_LEN]

        with autocast_ctx:
            loss = model(x, y)

        total_loss += loss.detach().float().item()
        (loss / grad_accum_steps).backward()

    # Average gradients across DDP ranks
    if ddp:
        for p in orig_model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    # Update LR
    current_lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = current_lr

    if args.grad_clip_enable:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), args.grad_clip)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    step_tokens = tokens_per_step
    tokens_trained += step_tokens

    synchronize()
    dt = time.time() - t0

    step += 1
    if step >= timing_start_step:
        total_training_time += dt
        timed_steps_count += 1

    # Logging
    avg_loss = total_loss / grad_accum_steps
    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * avg_loss
    debiased = smooth_loss / (1 - ema_beta**step)

    if step % args.log_interval == 0:
        pct = 100 * min(tokens_trained, args.nca_tokens) / args.nca_tokens
        tok_per_sec = int(step_tokens / dt) if dt > 0 else 0
        eta_m = (
            (
                (args.nca_tokens - tokens_trained)
                / step_tokens
                * dt
                / 60
            )
            if dt > 0
            else float("inf")
        )
        print0(
            f"step {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | "
            f"lr: {current_lr:.2e} | tok/s: {tok_per_sec:,} | eta: {eta_m:.1f}m"
        )
        wandb_run.log(
            {
                "nca_ppt/step": step,
                "nca_ppt/train_loss": debiased,
                "nca_ppt/tokens_trained": tokens_trained,
                "nca_ppt/lr": current_lr,
            }
        )

    # Validation: generate a fresh batch and compute loss
    if step % args.eval_interval == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vs in range(args.eval_steps):
                if enable_filtering and val_rule_data:
                    n_rules = min(len(val_rule_data), args.device_batch_size)
                    base_idx = (
                        step * args.eval_steps
                        + vs * n_rules
                        + ddp_rank * 1000
                    ) % len(val_rule_data)
                    eval_rule_seeds = [
                        val_rule_data[(base_idx + r) % len(val_rule_data)][0]
                        for r in range(n_rules)
                    ]
                    eval_sample_seed = (
                        999_999
                        + step * args.eval_steps * ddp_world_size
                        + vs * ddp_world_size
                        + ddp_rank
                    )
                    vx, vy = nca_gen.generate_batch_from_rules(
                        batch_size=args.device_batch_size,
                        n_steps=NCA_EXAMPLES,
                        rules=eval_rule_seeds,
                        rule_start_idx=base_idx,
                        sample_seed=eval_sample_seed,
                        temperature=args.temperature,
                    )
                else:
                    vx, vy = nca_gen.generate_batch(
                        batch_size=args.device_batch_size,
                        n_steps=NCA_EXAMPLES,
                        seed=999_999 + vs * ddp_world_size + ddp_rank,
                        n_rules=max(1, args.device_batch_size // 4),
                        temperature=args.temperature,
                    )
                vx = vx[:, :FULL_SEQ_LEN]
                vy = vy[:, :FULL_SEQ_LEN]
                with autocast_ctx:
                    vl = model(vx, vy)
                val_losses.append(vl.float().item())

        val_loss = sum(val_losses) / len(val_losses)

        # Reduce across ranks
        if ddp:
            val_loss_t = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
            val_loss = val_loss_t.item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            if master_process:
                save_stage1_checkpoint(args.save_checkpoint, orig_model)
                print0(f"  Saved new best checkpoint to: {args.save_checkpoint}")

        print0(
            f"  [Val] step {step:05d} | Val loss: {val_loss:.4f} | Best: {best_val_loss:.4f}"
        )
        wandb_run.log(
            {
                "nca_ppt/step": step,
                "nca_ppt/val_loss": val_loss,
            }
        )
        model.train()

    # GC management
    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()

# =============================================================================
# Save checkpoint
# =============================================================================

print0(f"\nNCA pre-pretraining complete.")
print0(f"Total steps: {step} | Tokens trained: {tokens_trained:,}")
print0(f"Best val loss: {best_val_loss:.4f}")
if timed_steps_count > 0:
    avg_step_time = total_training_time / timed_steps_count
    print0(f"Training time: {total_training_time / 60:.2f}m")

if master_process:
    save_path = args.save_checkpoint
    if not os.path.exists(save_path):
        save_stage1_checkpoint(save_path, orig_model)
        print0(f"Checkpoint saved to: {save_path}")
    else:
        print0(f"Best checkpoint kept at: {save_path}")

    result = {
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "tokens_trained": tokens_trained,
        "checkpoint_path": save_path,
        "stage_name": "nca_prepretrain",
        "final_train_loss": smooth_loss / (1 - 0.9**step) if step > 0 else float("inf"),
        "total_stage_time_sec": total_training_time,
        "num_steps": step,
        "world_size": ddp_world_size,
        "rank": ddp_rank,
        "nca_vocab": NCA_VOCAB_TOTAL,
        "n_layer": DEPTH,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
    }
    if args.save_result:
        with open(args.save_result, "w") as f:
            json.dump(result, f, indent=2)
        print0(f"Result saved to: {args.save_result}")

wandb_run.finish()
if dist.is_initialized():
    dist.destroy_process_group()
