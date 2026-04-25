# Nexus 2604.09258 Reproduction Notes

Date: 2026-04-24

Paper target: https://arxiv.org/abs/2604.09258

## Implementation

Implemented Nexus for Slowrun as an optional training mode in `tiny/train.py` and `train.py`.

The paper-mode path uses:

- Inner optimizer: normalized SGD over the gradient accumulation window.
- Pseudo-gradient: raw displacement `theta_start - theta_inner`.
- Outer optimizer: AdamW, not Muon.
- AdamW betas: `(0.9, 0.95)`.
- AdamW epsilon: `1e-10`.
- AdamW LR: `1e-3`.
- AdamW weight decay: `0.2`.
- Gradient clipping: `1.0`.
- Global batch: `524288` tokens.
- Tiny run device batch: `8`, giving `grad_accum_steps=4`.
- Nexus inner steps: default `grad_accum_steps`, so `4` for the tiny paper runs.
- Nexus inner LR: `0.01`.

Important paper/protocol note: the clean paper reproduction is evaluated against a matched AdamW baseline. The paper describes the outer optimizer as generic and names AdamW and Muon, but the clear table result we found is Adam+Nexus versus AdamW/Muon, not Muon+Nexus beating tuned Muon.

For Muon, a direct Algorithm 3-style replacement of the outer gradient with the Nexus displacement underperformed the tuned Slowrun baseline. A separate additive variant was therefore tested:

- Accumulate the normal averaged baseline gradient over the same inner microbatches.
- Run Nexus inner normalized-SGD steps on those microbatches.
- Restore the start weights and set the pseudo-gradient from the inner displacement.
- Multiply the pseudo-gradient by `--nexus-correction-weight`.
- Add the normal baseline gradient back before the Muon outer step.

This additive path is implemented in both `tiny/train.py` and `train.py`, but it is not the pure Algorithm 3 path from the paper.

## Tiny Commands

Matched AdamW baseline, 1 epoch:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_adamw_paper_baseline_epoch1_2604 \
  --outer-optimizer adamw --adamw-lr 1e-3 --adamw-weight-decay 0.2 --grad-clip 1.0 \
  --device-batch-size 8 --num-epochs 1 \
  --save-result reports/tiny_adamw_paper_baseline_epoch1_2604.json
```

Matched Nexus, 1 epoch:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_adamw_nexus_paper_epoch1_2604 \
  --outer-optimizer adamw --adamw-lr 1e-3 --adamw-weight-decay 0.2 --grad-clip 1.0 \
  --device-batch-size 8 --nexus --nexus-inner-lr 0.01 --num-epochs 1 \
  --save-result reports/tiny_adamw_nexus_paper_epoch1_2604.json
```

Matched AdamW baseline, 4 epochs:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_adamw_paper_baseline_epoch4_2604 \
  --outer-optimizer adamw --adamw-lr 1e-3 --adamw-weight-decay 0.2 --grad-clip 1.0 \
  --device-batch-size 8 --num-epochs 4 \
  --save-result reports/tiny_adamw_paper_baseline_epoch4_2604.json
```

Matched Nexus, 4 epochs:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_adamw_nexus_paper_epoch4_2604 \
  --outer-optimizer adamw --adamw-lr 1e-3 --adamw-weight-decay 0.2 --grad-clip 1.0 \
  --device-batch-size 8 --nexus --nexus-inner-lr 0.01 --num-epochs 4 \
  --save-result reports/tiny_adamw_nexus_paper_epoch4_2604.json
```

Muon baseline, 4 epochs:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_muon_baseline_epoch4_db8_2604 \
  --outer-optimizer muon --device-batch-size 8 --num-epochs 4 \
  --save-result reports/tiny_muon_baseline_epoch4_db8_2604.json
```

Pure Muon+Nexus, 4 epochs:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_muon_nexus_gamma001_epoch4_db8_2604 \
  --outer-optimizer muon --device-batch-size 8 \
  --nexus --nexus-inner-lr 0.01 --num-epochs 4 \
  --save-result reports/tiny_muon_nexus_gamma001_epoch4_db8_2604.json
```

Best additive Muon+Nexus correction, 4 epochs:

```bash
WANDB_MODE=offline ./.venv/bin/torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run tiny_muon_nexus_addgrad_w0003_epoch4_db8_2604 \
  --outer-optimizer muon --device-batch-size 8 \
  --nexus --nexus-inner-lr 0.01 --nexus-scale-by-inner-lr \
  --nexus-add-baseline-grad --nexus-correction-weight 0.003 \
  --num-epochs 4 \
  --save-result reports/tiny_muon_nexus_addgrad_w0003_epoch4_db8_2604.json
```

## Tiny Results

All runs kept strict step-0 validation. Step-0 validation loss matched at `10.826474`.

| Run | Epochs | Best val loss | Final/EMA val loss | Wall time |
| --- | ---: | ---: | ---: | ---: |
| AdamW baseline | 1 | 5.423005 | 5.809904 | 1.72m |
| Nexus | 1 | 5.147488 | 5.502454 | 1.99m |
| AdamW baseline | 4 | 4.379319 | 4.443738 | 4.97m |
| Nexus | 4 | 4.193290 | 4.240329 | 6.06m |

Relative to the matched AdamW baseline, Nexus improved:

- 1 epoch: `0.275517` lower best validation loss.
- 4 epochs: `0.186029` lower best validation loss.

Muon tiny results:

| Run | Epochs | Best val loss | Final/EMA val loss | Notes |
| --- | ---: | ---: | ---: | --- |
| Muon baseline | 4 | 3.665361 | 3.706752 | Tuned Slowrun baseline |
| Pure Muon+Nexus | 4 | 3.843224 | 3.942633 | Worse by `0.177864` |
| Pure Muon+Nexus, scaled displacement | 4 | 3.840637 | 3.936460 | Worse by `0.175276` |
| Additive Muon+Nexus, weight `0.01` | 4 | 3.663345 | 3.704972 | Better by `0.002016` |
| Additive Muon+Nexus, weight `0.003` | 4 | 3.663149 | 3.704232 | Better by `0.002212` |

The additive result is a real tiny-run improvement over the saved baseline, but the margin is small. It should be treated as a candidate rather than a robust new result until rerun across seeds or on the full main protocol.

## Interpretation

This reproduces a positive Nexus effect on Slowrun tiny when using the paper-compatible AdamW outer optimizer. It does not support pure Nexus-as-gradient-replacement on top of the tuned Muon baseline; that path clearly hurts.

The best result found for Muon is the additive correction variant with `--nexus-correction-weight 0.003`. That variant narrowly beats the saved tiny Muon baseline, but it is not pure paper Algorithm 3 and the margin is only about `0.06%` relative on validation loss.

Main `train.py` now has the same flags and optimizer path, but a full main run has not been completed in this pass.
