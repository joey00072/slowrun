# Nexus as a Small Muon Correction in Slowrun Tiny

## Abstract

We implement Nexus-style inner normalized SGD for Slowrun and evaluate it on the tiny protocol. A faithful AdamW reproduction matches the paper's qualitative claim: Nexus improves over a matched AdamW baseline. However, directly replacing Muon's gradient with the Nexus pseudo-gradient hurts the tuned Slowrun Muon baseline. We find that Nexus becomes useful with Muon only as a small additive correction to the normal accumulated gradient. The best tiny result improves validation loss from `3.665361` to `3.663149`, a gain of `0.002212`. This result is positive but small, and should be treated as a candidate until repeated on main or across seeds.

## 1. Background

Nexus computes an outer pseudo-gradient by running a short inner optimization trajectory. For parameters `theta_0`, an inner optimizer runs `K` normalized SGD steps to produce `theta_K`. The outer optimizer then receives a pseudo-gradient proportional to `theta_0 - theta_K`.

The paper describes a generic outer optimizer and explicitly allows AdamW or Muon. In our Slowrun setting, the important practical question is not whether Nexus beats AdamW, but whether it improves the already strong Muon baseline used by the track.

## 2. Implementation

We added Nexus flags to both `tiny/train.py` and `train.py`.

Core flags:

- `--nexus`: enable inner normalized-SGD trajectory.
- `--nexus-inner-steps`: number of inner steps; `0` means use `grad_accum_steps`.
- `--nexus-inner-lr`: normalized-SGD inner step size.
- `--nexus-scale-by-inner-lr`: use `(theta_0 - theta_K) / inner_lr` as pseudo-gradient.
- `--nexus-average-pseudo-grad`: divide pseudo-gradient by inner steps.
- `--nexus-add-baseline-grad`: add the ordinary accumulated gradient before the outer step.
- `--nexus-correction-weight`: scale Nexus pseudo-gradient before adding the baseline gradient.
- `--outer-optimizer {muon,adamw}`: select Muon or AdamW outer optimizer.

The pure paper-style path uses only the Nexus pseudo-gradient. The additive Muon path accumulates the usual baseline gradient over the same microbatches, computes the Nexus displacement, restores the original parameters, scales the displacement pseudo-gradient, then adds the baseline gradient before the Muon step.

This corrected additive implementation matters: recomputing a single baseline batch after the inner loop is not equivalent to the standard accumulated Muon baseline.

## 3. Experimental Setup

All experiments use strict step-0 validation. The step-0 validation loss is `10.826474` in all reported tiny runs.

Model and data:

- Model: Slowrun tiny GPT.
- Layers: `16`.
- Embedding width: `1024`.
- Attention heads: `8`.
- Head dimension: `128`.
- Sequence length: `2048`.
- Window pattern: `SSSL`.
- Parameters: `316,935,720`.
- Training tokens per global step: `524,288`.
- Device batch size: `8`.
- World size: `8`.
- Gradient accumulation: `4`.
- Epochs: `4`.

Muon baseline hyperparameters:

- Matrix LR: `0.032` at runtime for this branch.
- Scalar LR: `0.2`.
- Embedding LR: `0.12`.
- Unembedding LR: `0.0008`.
- Weight decay: `0.8`.
- Adam betas for scalar/small params: `(0.8, 0.95)`.
- Warmdown ratio: `0.6`.
- Dropout: `0.1`.

AdamW paper reproduction hyperparameters:

- Outer optimizer: AdamW.
- LR: `1e-3`.
- Betas: `(0.9, 0.95)`.
- Epsilon: `1e-10`.
- Weight decay: `0.2`.
- Gradient clipping: `1.0`.
- Nexus inner LR: `0.01`.
- Nexus inner steps: `4`.

Best Muon correction hyperparameters:

- Outer optimizer: Muon.
- Nexus inner LR: `0.01`.
- Pseudo-gradient scaling: enabled.
- Add baseline gradient: enabled.
- Nexus correction weight: `0.003`.

## 4. Results

### 4.1 AdamW Reproduction

| Run | Epochs | Best Val Loss | Final/EMA Val Loss |
| --- | ---: | ---: | ---: |
| AdamW baseline | 1 | 5.423005 | 5.809904 |
| AdamW+Nexus | 1 | 5.147488 | 5.502454 |
| AdamW baseline | 4 | 4.379319 | 4.443738 |
| AdamW+Nexus | 4 | 4.193290 | 4.240329 |

Nexus improves over matched AdamW by `0.275517` after one epoch and `0.186029` after four epochs.

### 4.2 Muon Reproduction

| Run | Epochs | Best Val Loss | Final/EMA Val Loss | Delta vs Muon |
| --- | ---: | ---: | ---: | ---: |
| Muon baseline | 4 | 3.665361 | 3.706752 | 0.000000 |
| Pure Muon+Nexus | 4 | 3.843224 | 3.942633 | +0.177864 |
| Pure Muon+Nexus, scaled | 4 | 3.840637 | 3.936460 | +0.175276 |
| Additive Muon+Nexus, weight `0.01` | 4 | 3.663345 | 3.704972 | -0.002016 |
| Additive Muon+Nexus, weight `0.003` | 4 | 3.663149 | 3.704232 | -0.002212 |

Pure Muon+Nexus is substantially worse than the tuned Muon baseline. The additive variant produces the best observed tiny result, but the gain is small.

## 5. Discussion

The AdamW result reproduces the paper's central optimizer-level phenomenon in Slowrun tiny: replacing AdamW's gradient with a Nexus pseudo-gradient improves validation loss under matched hyperparameters.

The Muon result is different. Muon is already a highly tuned matrix optimizer, and the pure Nexus pseudo-gradient appears to discard useful gradient structure that Muon relies on. In this setting, Nexus is not a replacement for the Muon gradient.

The additive correction behaves more like a weak regularized lookahead signal. It preserves the baseline Muon update while adding a small displacement-derived direction. This is the only Muon-compatible variant that beat the baseline in our tiny runs.

## 6. Limitations

The best Muon improvement is only `0.002212` validation loss. This is enough to justify a main run or repeated tiny runs, but not enough to claim a robust new result by itself.

The additive Muon variant is not the pure paper Algorithm 3 path. It is a practical adaptation for a tuned Muon baseline.

We have not completed the full main Slowrun protocol for this variant in this report.

## 7. Reproduction Commands

Best additive Muon+Nexus tiny command:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_muon_nexus_addgrad_w0003_epoch4_db8_2604 \
  --outer-optimizer muon --device-batch-size 8 \
  --nexus --nexus-inner-lr 0.01 --nexus-scale-by-inner-lr \
  --nexus-add-baseline-grad --nexus-correction-weight 0.003 \
  --num-epochs 4 \
  --save-result reports/tiny_muon_nexus_addgrad_w0003_epoch4_db8_2604.json
```

Matched AdamW+Nexus tiny command:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_adamw_nexus_paper_epoch4_2604 \
  --outer-optimizer adamw --adamw-lr 1e-3 --adamw-weight-decay 0.2 --grad-clip 1.0 \
  --device-batch-size 8 --nexus --nexus-inner-lr 0.01 --num-epochs 4 \
  --save-result reports/tiny_adamw_nexus_paper_epoch4_2604.json
```
