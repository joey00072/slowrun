# Newton-Muon Tiny Report

## Summary

This branch adds a `newton_muon` optimizer option to the tiny track and evaluates it against the existing Muon baseline.

Result: Newton-Muon was competitive, but it did not beat the tiny Muon baseline in this repository.

Best final numbers:

| Run | Optimizer | Best val loss |
| --- | --- | ---: |
| `tiny_muon_baseline_lp` | Muon | `3.347561886436061` |
| `tiny_newton_muon_lp` | Newton-Muon (`refresh=32`, `ridge=0.2`) | `3.3477522197522616` |
| `tiny_newton_muon_lp_r32_g10` | Newton-Muon (`refresh=32`, `ridge=0.1`) | `3.347883525647615` |

The best Newton run missed the Muon baseline by `0.00019033331620077476`. The tuned Newton run with lower ridge missed by `0.00032163921155402257`.

## What Was Added

Implementation lives in `tiny/train.py`.

Main additions:

- `--optimizer {muon,newton_muon}`
- Newton-specific knobs:
  - `--newton-refresh-interval`
  - `--newton-ewma`
  - `--newton-ridge-mult`
  - `--newton-init-diag`
  - `--newton-eps`
  - `--newton-block-size`
- Activation second-moment accumulation for linear layers used by Newton preconditioning
- Newton-style right preconditioning of matrix gradients before the usual Muon-style update

The implementation was based on the idea described in the Newton-Muon reference repository:

- <https://github.com/zhehangdu/Newton-Muon>

## Tiny Setup

All full tiny comparisons used the same tiny architecture and training recipe from `tiny/train.py`, except for the optimizer-specific Newton knobs.

Shared tiny configuration:

- `n_layer=16`
- `n_embd=1024`
- `n_head=8`
- `seq_len=2048`
- `window_pattern=SSSL`
- `total_batch_size=524288`
- `device_batch_size=32`
- `num_epochs=16`
- `dropout=0.1`
- base learning rates before `lr_multiplier=0.8`:
  - `matrix_lr=0.04`
  - `scalar_lr=0.25`
  - `embedding_lr=0.15`
  - `unembedding_lr=0.001`
- effective learning rates during the runs:
  - `matrix_lr=0.032`
  - `scalar_lr=0.2`
  - `embedding_lr=0.12`
  - `unembedding_lr=0.0008`
- `weight_decay=0.8`
- `adam_betas=(0.8, 0.95)`
- `warmdown_ratio=0.6`
- `swa_last_epochs=4`

Newton-specific settings in the first full run:

- `newton_refresh_interval=32`
- `newton_ewma=0.95`
- `newton_ridge_mult=0.2`
- `newton_init_diag=1e-3`
- `newton_eps=1e-8`
- `newton_block_size=256`

Newton-specific settings in the second full run:

- same as above, except `newton_ridge_mult=0.1`

## Runs

Full runs committed in this branch:

- Baseline JSON: `newton_runs/tiny_muon_baseline_lp.json`
- Baseline log: `newton_runs/tiny_muon_baseline_lp.log`
- Newton JSON: `newton_runs/tiny_newton_muon_lp.json`
- Newton log: `newton_runs/tiny_newton_muon_lp.log`
- Tuned Newton JSON: `newton_runs/tiny_newton_muon_lp_r32_g10.json`
- Tuned Newton log: `newton_runs/tiny_newton_muon_lp_r32_g10.log`

Key checkpoints:

| Run | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 13 | Epoch 14 | Ckpt avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Muon baseline | `3.635608` | `3.548180` | `3.531317` | `3.383306` | `3.381803` | `3.347562` |
| Newton (`ridge=0.2`) | `3.638733` | `3.549251` | `3.533313` | `3.385433` | `3.381928` | `3.347752` |
| Newton (`ridge=0.1`) | `3.644421` | `3.552593` | `3.533580` | `3.386817` | `3.382904` | `3.347884` |

## Tuning Notes

I also ran shorter 4-epoch Newton probes to choose which Newton configuration deserved a full 16-epoch rerun.

Important caveat: those 4-epoch probes are only useful for ranking Newton settings against each other. They are not directly comparable to the 16-epoch baseline because changing `--num-epochs` changes the LR and weight-decay schedules.

The most useful outcome from those probes was that `refresh=32, ridge=0.1` looked like the best Newton candidate to promote into a real full run. After the full rerun, it still finished behind Muon.

## Interpretation

The conclusion from this branch is narrow and concrete:

- Newton-Muon was integrated successfully into the tiny track
- it trains stably
- it is very close to the Muon baseline
- but it does not beat the Muon baseline on tiny in this codebase

This is not a collapse or obvious regression. It is a near miss. But for the benchmark question that matters here, near miss is still a miss.

## Recommendation

I would not spend main-track compute on Newton-Muon yet.

The tiny result says Newton is competitive, but there is no evidence in this repository that it improves over the existing well-tuned Muon baseline. If we revisit it, the next work should be a tighter optimizer sweep or a different optimizer idea, not an immediate main-track reproduction attempt.
