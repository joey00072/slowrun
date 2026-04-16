# Reproduction Report: NCA Pre-Pretraining in Slowrun

Date: 2026-04-15

Paper under test: [arXiv:2603.10055](https://arxiv.org/abs/2603.10055)

## Summary

We implemented the paper's two-stage recipe in Slowrun:

1. Stage 1: NCA-style synthetic pre-pretraining
2. Stage 2: FineWeb language-model training with transferred trunk weights

We were able to run the method end-to-end on the Slowrun tiny track, including separate logs for stage 1 and stage 2. However, after multiple implementation fixes, reference-code alignment passes, longer stage-1 budgets, and several transfer ablations, we were **not able to beat the tiny baseline**.

The strongest result we obtained was:

- Baseline: `3.3469593650416325`
- Best NCA-based run: `3.355248501426295`
- Gap to baseline: `+0.008289136384662665`

This is enough to say that **we could not reproduce an improvement from the paper in the Slowrun tiny setting**. It is **not** enough, by itself, to prove that the paper is dishonest or fabricated.

## Scope

This report covers the Slowrun tiny track in this repository.

- Benchmark context: [README.md](/mnt/vast/joey/slowrun/README.md:54)
- Tiny-track record in this repo: `3.345`
- Our local reproduced baseline: `3.3469593650416325`

We also wired the method into the main run, but we did not spend the full main-run budget after failing to show a tiny-track win first.

## What We Implemented

We added a full NCA pre-pretraining pipeline and transfer path:

- Stage 1 synthetic training: [tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:1)
- Stage 2 transfer into Slowrun tiny: [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:72)

We used the paper authors' released code in `ref/` as a read-only reference and repeatedly compared our implementation against it.

## Exact Configuration We Tested

### Slowrun Tiny Baseline / Stage 2 Model

The tiny-track language model we used is the current Slowrun tiny architecture in [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:38).

Core architecture:

- Depth: `16` layers ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:59))
- Width: `1024` hidden size ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:61))
- Heads: `8` attention heads, `128` head dim ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:60), [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:100))
- Context length: `2048` tokens ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:101))
- Window pattern: `SSSL` ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:102))
- Vocab: GPT-2 tokenizer, padded to a multiple of 64 ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:226), [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:321))
- Nonstandard trunk features:
  - alternating value projections / value residual path ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:240), [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:278))
  - per-head attention gate ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:265), [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:290))
  - partial key offset on long-window layers ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:268), [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:286))
  - SwiGLU MLP ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:296))
  - U-Net style skip connections plus `resid_lambdas` and `x0_lambdas` ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:333), [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:338))

Optimizer and regularization:

- Matrix parameters: Muon ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:416))
- Small/scalar/embed/output groups: AdamW-style groups ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:429))
- Effective LRs after `lr_multiplier=0.8`:
  - `matrix_lr=0.032`
  - `scalar_lr=0.2`
  - `embedding_lr=0.12`
  - `unembedding_lr=0.0008`
  - logged in [tiny_crazy_tophalf/tiny_tophalf_1312m_main.log](/mnt/vast/joey/slowrun/tiny_crazy_tophalf/tiny_tophalf_1312m_main.log:10)
- Weight decay: `0.8` with 3-phase schedule, `wd_mid=0.1`, `wd_end=1.25` ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:47), [tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:52))
- Dropout: `0.1` ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:67))
- EMA: update every `10` steps, decay-per-epoch `0.15` ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:68))
- SWA-style last-epoch checkpoint averaging: enabled for last `4` epochs ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:70))

Training budget:

- Total batch size: `524,288` tokens/step ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:57))
- Device batch size: `32` per GPU ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:39))
- Epochs: `16` ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:40))
- Eval set size: `10,000,000` tokens ([tiny/train.py](/mnt/vast/joey/slowrun/tiny/train.py:104))
- Parameter count in our run: `316,935,720` ([tiny_crazy_tophalf/tiny_tophalf_1312m_main.log](/mnt/vast/joey/slowrun/tiny_crazy_tophalf/tiny_tophalf_1312m_main.log:28))

### Stage 1 NCA Pre-Pretraining Setup

The synthetic stage lives in [tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:1) and intentionally mirrors the tiny model shape as closely as possible.

NCA data-generation setup:

- Grid size: `12x12` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:50))
- Patch size: `2` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:54))
- Number of colors/states: `10` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:60))
- NCA steps per example: `16` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:63))
- Burn-in rollout steps: `10` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:87))
- Sampling temperature: `1e-4` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:81))
- Complexity filtering enabled by default ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:141))
- Compression-ratio band: `[0.5, 1.0]` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:117))
- Training rules: `16000` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:129))
- Held-out validation rules: `2000` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:135))
- Rule-bank refresh cadence: every `500` optimizer steps ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:187), [tiny_fixobj1312/tiny_fixobj_1312m_ppt.log](/mnt/vast/joey/slowrun/tiny_fixobj1312/tiny_fixobj_1312m_ppt.log:59))

Stage-1 model and optimizer:

- Same transformer shape as tiny: `16` layers, `8` heads, `1024` hidden ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:110), [tiny_fixobj1312/tiny_fixobj_1312m_ppt.log](/mnt/vast/joey/slowrun/tiny_fixobj1312/tiny_fixobj_1312m_ppt.log:10))
- Synthetic model vocab size: `64000` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:160))
- Synthetic context target: `1024`, realized raw sequence length `1026` in our run ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:167), [tiny_fixobj1312/tiny_fixobj_1312m_ppt.log](/mnt/vast/joey/slowrun/tiny_fixobj1312/tiny_fixobj_1312m_ppt.log:8))
- Stage-1 dropout: `0.0` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:337))
- Stage-1 optimizer: Adam, not Muon ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:152))
- Stage-1 learning rate: `1e-4` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:152))
- Stage-1 weight decay: `0.0` ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:154))
- Stage-1 warmup fraction: `0.1` with cosine decay ([tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:155), [tiny/nca_pretraining.py](/mnt/vast/joey/slowrun/tiny/nca_pretraining.py:1080))

For the strongest full-transfer NCA run, we pushed stage-1 budget to roughly `1.312B` synthetic tokens:

- Stage-1 result: [tiny_fixobj1312/tiny_fixobj_1312m_ppt.json](/mnt/vast/joey/slowrun/tiny_fixobj1312/tiny_fixobj_1312m_ppt.json:1)
- Stage-1 log: [tiny_fixobj1312/tiny_fixobj_1312m_ppt.log](/mnt/vast/joey/slowrun/tiny_fixobj1312/tiny_fixobj_1312m_ppt.log:45)

## Reference-Alignment Fixes We Made

Before running the final experiments, we fixed several concrete mismatches relative to the reference implementation:

- Corrected the stage-1 objective to mask the full first grid, matching the intended predictive setup.
- Switched reused rules to sample fresh rollouts instead of replaying identical trajectories.
- Separated held-out validation rules from training rules.
- Saved the best stage-1 checkpoint rather than just the last checkpoint.
- Extended stage-1 duration substantially beyond the original first attempt.
- Added transfer ablations so we could test whether only part of the NCA trunk helps.

These changes improved the method relative to the earliest NCA attempt, but they still did not produce a baseline win.

## Tiny-Track Results

### Baseline

- Baseline result: [tiny_baseline_repro.json](/mnt/vast/joey/slowrun/tiny_baseline_repro.json:1)
- Best val loss: `3.3469593650416325`

### NCA Runs

| Run | Artifact | Best val loss | Delta vs baseline |
| - | - | - | - |
| First NCA reproduction | [tiny_nca_main_repro.json](/mnt/vast/joey/slowrun/tiny_nca_main_repro.json:1) | `3.368551956979852` | `+0.02159259193821944` |
| Sweep A | [sweeps/tiny_nca_sweep_a_main.json](/mnt/vast/joey/slowrun/sweeps/tiny_nca_sweep_a_main.json:1) | `3.3568757709703947` | `+0.009916405928762213` |
| Longer stage 1, 328M tokens | [tiny_long/tiny_ref_long_main.json](/mnt/vast/joey/slowrun/tiny_long/tiny_ref_long_main.json:1) | `3.3582121196546053` | `+0.01125275461297282` |
| Longer stage 1, 656M tokens | [tiny_longer2/tiny_ref_656m_main.json](/mnt/vast/joey/slowrun/tiny_longer2/tiny_ref_656m_main.json:1) | `3.35636379844264` | `+0.009404433401007668` |
| Fixed objective, 656M tokens | [tiny_fixobj656/tiny_fixobj_656m_main.json](/mnt/vast/joey/slowrun/tiny_fixobj656/tiny_fixobj_656m_main.json:1) | `3.3641536110325863` | `+0.01719424599095375` |
| Fixed objective, 1.312B tokens | [tiny_fixobj1312/tiny_fixobj_1312m_main.json](/mnt/vast/joey/slowrun/tiny_fixobj1312/tiny_fixobj_1312m_main.json:1) | `3.3569512618215462` | `+0.0099918967799137` |
| Embedding/lm_head bridge | [tiny_bridge/tiny_bridge_200_main.json](/mnt/vast/joey/slowrun/tiny_bridge/tiny_bridge_200_main.json:1) | `3.3576102005807975` | `+0.010650835539165027` |
| Top-half-only transfer | [tiny_crazy_tophalf/tiny_tophalf_1312m_main.json](/mnt/vast/joey/slowrun/tiny_crazy_tophalf/tiny_tophalf_1312m_main.json:1) | `3.355248501426295` | `+0.008289136384662665` |

### Best NCA Result

The best NCA-based run was the top-half-only transfer:

- Result: [tiny_crazy_tophalf/tiny_tophalf_1312m_main.json](/mnt/vast/joey/slowrun/tiny_crazy_tophalf/tiny_tophalf_1312m_main.json:1)
- Final checkpoint-averaged val loss: `3.355249`
- It improved over the previous best NCA run by `0.0017027603952510795`
- It still failed to beat baseline by `0.008289136384662665`

Late-stage validation for the previous best full-transfer run:

- [tiny_fixobj1312/tiny_fixobj_1312m_main.log](/mnt/vast/joey/slowrun/tiny_fixobj1312/tiny_fixobj_1312m_main.log:2530)
- [tiny_fixobj1312/tiny_fixobj_1312m_main.log](/mnt/vast/joey/slowrun/tiny_fixobj1312/tiny_fixobj_1312m_main.log:3108)

Late-stage validation for the best filtered-transfer run:

- [tiny_crazy_tophalf/tiny_tophalf_1312m_main.log](/mnt/vast/joey/slowrun/tiny_crazy_tophalf/tiny_tophalf_1312m_main.log:2529)
- [tiny_crazy_tophalf/tiny_tophalf_1312m_main.log](/mnt/vast/joey/slowrun/tiny_crazy_tophalf/tiny_tophalf_1312m_main.log:3107)

## What We Learned

### 1. More Stage-1 Compute Alone Did Not Solve It

We increased stage-1 training substantially, up to roughly 1.312B synthetic tokens. Stage-1 synthetic validation improved, but the gain did not transfer into a language-model win on FineWeb.

### 2. Better Reference Fidelity Helped, But Not Enough

Fixing obvious mismatches with the reference code improved results compared with the earliest NCA attempt. That means the initial failures were not just due to a broken implementation. But even after those corrections, the method still remained behind baseline.

### 3. Full-Trunk Transfer Looks Slightly Worse Than Selective Transfer

The best result came from transferring only the top half of the trunk rather than all layers. This suggests the synthetic stage may be learning some useful late abstractions, while lower-layer features may be overspecialized to the NCA domain.

### 4. Narrower Attention-Only Variants Did Not Rescue It

We also tested narrower transfer ideas. `attn-only` and `top-half-attn-only` did not outperform `top-half`, and they did not show evidence of a path to a baseline win.

## Current Interpretation

Our current working interpretation is:

> In this setting, with a strong, well-tuned Slowrun tiny baseline and a Muon-based optimizer on the main language-training stage, NCA pre-pretraining does not appear to provide gains. In our runs it consistently hurt final validation loss relative to baseline.

That statement is a hypothesis grounded in our current evidence, not a theorem. But it is a fair reading of the results we actually obtained:

- Every completed NCA-based tiny run underperformed the reproduced baseline.
- Increasing stage-1 compute did not reverse that.
- Better fidelity to the reference implementation did not reverse that.
- Selective transfer reduced the damage slightly, but still did not turn the method into a win.

The most direct empirical takeaway from our experiments is not "pre-pretraining helps but we tuned it badly." The more natural reading is:

- the Slowrun tiny baseline is already very strong and well tuned
- Muon plus the existing Slowrun architecture may already be extracting most of the accessible gain from this 100M-token regime
- adding NCA pre-pretraining, at least in the form described by the paper and reference code, introduces a domain shift that hurts more than it helps

## Bottom Line

Our evidence supports the following claim:

> We implemented the method, aligned it against the reference code, ran multiple tiny-track reproductions and ablations, and still failed to reproduce a baseline-beating result in Slowrun tiny.

Our evidence does **not** support the stronger claim:

> The paper is a lie.

Non-reproduction in one benchmark setting can mean several things:

- the method is weaker than claimed
- the method is sensitive to setup details not fully captured in the paper
- the claimed gains do not transfer to Slowrun tiny
- the released code and the reported result are not fully aligned

Those are serious concerns. But they are not the same as proving misconduct.

## Suggested Wording

If the goal is to make a strong public statement that is still defensible, the wording below is safer and more accurate:

> We implemented the paper's method in Slowrun, compared carefully against the authors' reference code, and ran multiple tiny-track reproductions. Despite several rounds of fixes and ablations, we were unable to reproduce a baseline-beating result. In our experiments, the method consistently underperformed the Slowrun tiny baseline, with the best NCA-based run reaching 3.3552 versus a 3.3470 baseline. Our current working hypothesis is that, against a strong Muon-based baseline in this regime, NCA pre-pretraining does not help and may be net harmful. Based on our current evidence, we do not consider the paper's claimed improvement reproduced in this setting.

If you want sharper language without overclaiming:

> We spent substantial effort trying to reproduce the result and failed. At minimum, the result appears fragile or underspecified. The current paper and code were not enough for us to reproduce the claimed advantage in Slowrun tiny.
