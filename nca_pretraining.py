"""
NCA Pre-Pre-Training — Stage 1 (Main Track)
============================================================
Implements the pre-pre-training stage from:
  "Training Language Models via Neural Cellular Automata"
  https://arxiv.org/abs/2603.10055

Trains the slowrun GPT architecture (main track: n_layer=30, n_head=14, n_embd=1792)
on synthetic NCA token sequences generated entirely in PyTorch (no JAX), 
then saves a checkpoint for Stage 2 (language pre-training via train.py --nca-checkpoint).

Usage:
    torchrun --standalone --nproc_per_node=8 nca_pretraining.py
    torchrun --standalone --nproc_per_node=8 nca_pretraining.py \\
        --nca-tokens 164000000 \\
        --save-checkpoint nca_ppt.pt \\
        --run nca_ppt_main
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import argparse
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
parser.add_argument("--nca-grid", type=int, default=12,
                    help="NCA grid size H=W (paper: 12)")
parser.add_argument("--nca-patch", type=int, default=2,
                    help="Patch size for tokenization (paper: 2, gives 10^4 vocab)")
parser.add_argument("--nca-colors", type=int, default=10,
                    help="Number of NCA states/colors (paper: 10)")
parser.add_argument("--nca-steps", type=int, default=16,
                    help="Number of NCA simulation steps per example")
parser.add_argument("--nca-examples", type=int, default=8,
                    help="Number of steps shown per trajectory (in-context examples)")
parser.add_argument("--identity-bias", type=float, default=2.0,
                    help="Identity bias for NCA (larger = slower dynamics)")
parser.add_argument("--complexity-threshold", type=float, default=0.0,
                    help="Minimum gzip-complexity ratio for rule filtering (0=off)")

# Token budget
parser.add_argument("--nca-tokens", type=int, default=164_000_000,
                    help="Total NCA tokens to train on (default: 164M as in paper)")
parser.add_argument("--device-batch-size", type=int, default=4,
                    help="Sequences per GPU per micro-step")
parser.add_argument("--total-batch-size", type=int, default=524288,
                    help="Total tokens per optimizer step (matches main train.py)")

# Model (must match train.py defaults for main track)
parser.add_argument("--n-layer", type=int, default=30)
parser.add_argument("--n-head", type=int, default=14)
parser.add_argument("--n-embd", type=int, default=1792)

# Optimizer
parser.add_argument("--lr", type=float, default=3e-4,
                    help="Peak learning rate for AdamW")
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--warmup-frac", type=float, default=0.05)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)

# Output
parser.add_argument("--save-checkpoint", type=str, default="nca_ppt.pt",
                    help="Path to save NCA pre-pretrained checkpoint")
parser.add_argument("--save-result", type=str, default="nca_ppt_result.json")
parser.add_argument("--run", type=str, default=None,
                    help="Wandb run name")
parser.add_argument("--wandb-group", type=str, default="nca_ppt_main",
                    help="Wandb group")
parser.add_argument("--log-interval", type=int, default=50)
parser.add_argument("--eval-interval", type=int, default=200)
parser.add_argument("--eval-steps", type=int, default=20)

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
NCA_BOS = NCA_VOCAB          # special start-of-grid token
NCA_EOS = NCA_VOCAB + 1      # special end-of-grid token
NCA_VOCAB_TOTAL = NCA_VOCAB + 2  # 10002

# Sequence layout: each NCA "grid" becomes (grid_patches + 2) tokens
# grid_patches = (grid/patch)^2
NCA_GRID_PATCHES = (NCA_GRID // NCA_PATCH) ** 2  # 36 for 12x12 grid, patch=2
TOKENS_PER_GRID = NCA_GRID_PATCHES + 2           # 38 (BOS, 36 patch tokens, EOS)
SEQ_LEN = TOKENS_PER_GRID * args.nca_examples    # e.g. 38 * 8 = 304

DEPTH = args.n_layer
N_EMBD = args.n_embd
N_HEAD = args.n_head
HEAD_DIM = N_EMBD // N_HEAD
WINDOW_PATTERN = "SSSL"
MAX_SEQ_LEN = max(SEQ_LEN, 512)  # ensure power-of-2-friendly


# =============================================================================
# Utilities
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs)

class DummyWandb:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass
    def log_code(self, *a, **kw): pass

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
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None

_fa3 = _load_fa3()

def _sdpa_attention(q, k, v, window_size, enable_gqa):
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k, v = k[:, :, start:, :], v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

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
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)
        pattern = config.window_pattern.upper()
        char = pattern[layer_idx % len(pattern)]
        self.use_key_offset = (char == 'L') or (layer_idx == config.n_layer - 1)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        if self.use_key_offset and T > 1:
            k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:].clone()
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_channels])).unsqueeze(-1)
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
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({
            str(i): nn.Linear(config.n_embd, kv_dim, bias=False)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
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
        inv_freq = 1.0 / (base ** (torch.arange(0, half * 2, 2, dtype=torch.float32, device=device) / (half * 2)))
        inv_freq = torch.cat([inv_freq, torch.zeros(head_dim // 2 - half, dtype=torch.float32, device=device)])
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

    def forward(self, idx, targets=None, loss_reduction='mean'):
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
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # softcap (same as tiny/train.py)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction
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

    def __init__(self, grid=12, patch=2, num_colors=10, identity_bias=2.0,
                 device="cuda", dtype=torch.bfloat16):
        assert grid % patch == 0, "grid must be divisible by patch"
        self.grid = grid
        self.patch = patch
        self.P = patch * patch          # pixels per patch
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
        """Sample random NCA rule parameters (a tiny CNN) from a seed."""
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        # Small CNN: conv3x3(in=num_colors, out=16) → conv1x1(in=16, out=num_colors)
        # We use 1×1 final conv for efficiency
        w1 = torch.randn(16, self.num_colors, 3, 3, generator=rng, device=self.device, dtype=self.dtype)
        b1 = torch.randn(16, generator=rng, device=self.device, dtype=self.dtype)
        w2 = torch.randn(self.num_colors, 16, 1, 1, generator=rng, device=self.device, dtype=self.dtype)
        b2 = torch.randn(self.num_colors, generator=rng, device=self.device, dtype=self.dtype)
        return (w1, b1, w2, b2)

    def nca_step(self, state: torch.Tensor, rule, temperature: float = 1.0) -> torch.Tensor:
        """
        One NCA step.
        state: (B, H, W) integer tensor of cell colors
        Returns: (B, H, W) next state
        """
        w1, b1, w2, b2 = rule
        B, H, W = state.shape

        # One-hot encode: (B, num_colors, H, W)
        state_oh = F.one_hot(state.long(), self.num_colors).permute(0, 3, 1, 2).to(self.dtype)

        # Circular (toroidal) padding 
        state_pad = F.pad(state_oh, (1, 1, 1, 1), mode='circular')

        # Apply CNN
        h = F.conv2d(state_pad, w1, b1)    # (B, 16, H, W)
        h = F.relu(h)
        logits = F.conv2d(h, w2, b2)       # (B, num_colors, H, W)

        # Identity bias: bias toward staying in current state
        logits = logits + self.identity_bias * state_oh

        # Stochastic update (Gumbel softmax argmax = categorical sampling)
        # Temperature close to 0 → deterministic; temperature = 1 → random
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8)))
        next_state = (logits / (temperature + 1e-8) + gumbel).argmax(dim=1)  # (B, H, W)
        return next_state

    def rollout(self, rule, n_steps: int, batch_size: int, seed: int,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Roll out NCA for n_steps, collecting states.
        Returns: (batch_size, n_steps, H, W) integer tensor
        """
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        H = W = self.grid

        # Random initial states
        state = torch.randint(0, self.num_colors, (batch_size, H, W),
                              generator=rng, device=self.device)

        # Burn-in steps (not recorded)
        burn_in = 10
        for _ in range(burn_in):
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
        B, T, H, W = states.shape
        P = self.patch
        Np = H // P  # patches per side
        n_patches = Np * Np

        # Reshape into patches: (B, T, Np, P, Np, P) → (B, T, Np, Np, P, P)
        s = states.view(B, T, Np, P, Np, P)
        s = s.permute(0, 1, 2, 4, 3, 5)      # (B, T, Np, Np, P, P)
        s = s.reshape(B, T, n_patches, self.P)  # (B, T, n_patches, P)

        # Encode each patch as a single integer token
        # token = sum(color[i] * num_colors^i for i in range(P))
        tokens = (s.long() * self.powers.view(1, 1, 1, -1)).sum(-1)  # (B, T, n_patches)

        # Build sequence with BOS/EOS wrapping each time step
        # Layout per time step: [BOS, patch_0, ..., patch_{n-1}, EOS]
        # Total per step: n_patches + 2
        step_len = n_patches + 2
        bos_col = torch.full((B, T, 1), self.bos, dtype=torch.long, device=self.device)
        eos_col = torch.full((B, T, 1), self.eos, dtype=torch.long, device=self.device)
        seq_3d = torch.cat([bos_col, tokens, eos_col], dim=2)  # (B, T, step_len)
        seq = seq_3d.view(B, T * step_len)  # (B, total_len)

        # Targets: predict patch tokens; mask BOS/EOS positions with -1
        # For language modeling: target[t] = token[t+1]
        inp = seq[:, :-1]       # (B, total_len-1)
        tgt = seq[:, 1:].clone()  # (B, total_len-1)

        # Mask target positions that correspond to BOS tokens (position 0 of each step)
        total_len = T * step_len
        positions = torch.arange(total_len - 1, device=self.device)
        is_bos_target = ((positions + 1) % step_len == 0)  # EOS→BOS transition
        tgt[:, is_bos_target] = -1

        return inp, tgt

    @torch.no_grad()
    def generate_batch(self, batch_size: int, n_steps: int, seed: int,
                       n_rules: int = 1, temperature: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
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
autocast_ctx = (torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
                if device_type == "cuda" else nullcontext())
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# FA3 status
if _fa3 is not None:
    print0("Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("Using PyTorch SDPA fallback (no FA3)")

# =============================================================================
# Wandb
# =============================================================================

run_name = args.run if args.run else f"nca_ppt_tiny_{time.strftime('%Y%m%d_%H%M%S')}"
wandb_run = DummyWandb()
if master_process:
    wandb_run = wandb.init(
        project="slowrun",
        name=run_name,
        group=args.wandb_group,
        config={
            "nca_grid": NCA_GRID, "nca_patch": NCA_PATCH, "nca_colors": NCA_COLORS,
            "nca_vocab": NCA_VOCAB_TOTAL, "seq_len": SEQ_LEN,
            "nca_tokens": args.nca_tokens,
            "n_layer": DEPTH, "n_embd": N_EMBD, "n_head": N_HEAD,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "total_batch_size": args.total_batch_size,
            "device_batch_size": args.device_batch_size,
        }
    )
    wandb_run.log_code(".")

# =============================================================================
# Model
# =============================================================================

print0(f"\n--- NCA Pre-Pre-Training (Stage 1) ---")
print0(f"NCA vocab: {NCA_VOCAB_TOTAL} | seq_len: {SEQ_LEN}")
print0(f"NCA tokens budget: {args.nca_tokens:,}")
print0(f"n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}")
print0(f"Model: same GPT arch as tiny/train.py, NCA vocab")

config = GPTConfig(
    vocab_size=NCA_VOCAB_TOTAL,
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
# Optimizer — AdamW (matches paper; Muon designed for language scale)
# =============================================================================

optimizer = torch.optim.AdamW(
    orig_model.parameters(),
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
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
FULL_SEQ_LEN = SEQ_LEN_ACTUAL * args.nca_examples  # full trajectory seq len

# =============================================================================
# Training config
# =============================================================================

tokens_per_fwdbwd = args.device_batch_size * FULL_SEQ_LEN * ddp_world_size
grad_accum_steps = max(1, args.total_batch_size // tokens_per_fwdbwd)
total_steps = math.ceil(args.nca_tokens / (args.total_batch_size))
warmup_steps = max(1, round(args.warmup_frac * total_steps))

print0(f"Seq len: {FULL_SEQ_LEN} tokens")
print0(f"Device batch size: {args.device_batch_size} | grad accum: {grad_accum_steps}")
print0(f"Effective batch size: {tokens_per_fwdbwd * grad_accum_steps:,} tokens")
print0(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

def get_lr(step):
    if step < warmup_steps:
        return args.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return args.lr * (0.5 * (1 + math.cos(math.pi * progress)))  # cosine decay to 0

# =============================================================================
# Training loop
# =============================================================================

step = 0
tokens_trained = 0
smooth_loss = 0.0
best_val_loss = float("inf")
total_training_time = 0.0
timed_steps_count = 0
timing_start_step = 4

print0(f"\nStarting NCA pre-pre-training...")

while tokens_trained < args.nca_tokens:
    synchronize()
    t0 = time.time()

    # Each rank generates its own NCA batch (independent seeds from rank offset)
    batch_seed = step * ddp_world_size + ddp_rank + 1

    # Gradient accumulation
    total_loss = 0.0
    model.train()
    for micro_step in range(grad_accum_steps):
        micro_seed = batch_seed * grad_accum_steps + micro_step
        x, y = nca_gen.generate_batch(
            batch_size=args.device_batch_size,
            n_steps=args.nca_examples,
            seed=micro_seed,
            n_rules=max(1, args.device_batch_size // 4),  # diversity: multiple rules per batch
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

    # Gradient clip
    torch.nn.utils.clip_grad_norm_(orig_model.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    step_tokens = tokens_per_fwdbwd * grad_accum_steps
    tokens_trained += step_tokens * ddp_world_size  # total across all ranks

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
    debiased = smooth_loss / (1 - ema_beta ** step)

    if step % args.log_interval == 0:
        pct = 100 * tokens_trained / args.nca_tokens
        tok_per_sec = int(step_tokens * ddp_world_size / dt) if dt > 0 else 0
        eta_m = ((args.nca_tokens - tokens_trained) / (step_tokens * ddp_world_size) *
                 dt / 60) if dt > 0 else float('inf')
        print0(f"step {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | "
               f"lr: {current_lr:.2e} | tok/s: {tok_per_sec:,} | eta: {eta_m:.1f}m")
        wandb_run.log({
            "nca_ppt/step": step,
            "nca_ppt/train_loss": debiased,
            "nca_ppt/tokens_trained": tokens_trained,
            "nca_ppt/lr": current_lr,
        })

    # Validation: generate a fresh batch and compute loss
    if step % args.eval_interval == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vs in range(args.eval_steps):
                vx, vy = nca_gen.generate_batch(
                    batch_size=args.device_batch_size,
                    n_steps=args.nca_examples,
                    seed=999_999 + vs * ddp_world_size + ddp_rank,
                    n_rules=max(1, args.device_batch_size // 4),
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

        print0(f"  [Val] step {step:05d} | Val loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        wandb_run.log({
            "nca_ppt/step": step,
            "nca_ppt/val_loss": val_loss,
        })
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
    print0(f"Training time: {total_training_time/60:.2f}m")

if master_process:
    # Save the FULL state dict (including wte & lm_head)
    # Stage 2 will re-init wte/lm_head and keep everything else
    ckpt = {n: p.data.float().cpu() for n, p in orig_model.named_parameters()}
    save_path = args.save_checkpoint
    torch.save(ckpt, save_path)
    print0(f"Checkpoint saved to: {save_path}")

    result = {
        "best_val_loss": best_val_loss,
        "best_step": step,
        "tokens_trained": tokens_trained,
        "checkpoint_path": save_path,
        "stage_name": "nca_prepretrain",
        "final_train_loss": smooth_loss / (1 - 0.9 ** step) if step > 0 else float('inf'),
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
