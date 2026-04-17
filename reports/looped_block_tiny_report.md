# Looped Block Tiny Report

## Summary

This branch tests a looped residual block in the tiny trainer, based on repeating the attention and MLP updates inside each block with a linear ramp:

```python
for idx in range(N):
    x = x + self.attn(norm(x), ve, cos_sin, window_size) * (idx / N)
    x = x + self.mlp(norm(x)) * (idx / N)
```

The result is mixed:

- it does improve validation loss
- but it does not fit at the default tiny microbatch
- and the fitted run is much slower than the tiny-track time budget

## Code Change

Implementation is in `tiny/train.py`.

Added:

- `--block-loop-count`
- loop-aware block forward path
- result logging for `block_loop_count`

The default remains `--block-loop-count 1`, which preserves the baseline block behavior.

## Runs

### Baseline

Run:

- `loop_runs/tiny_baseline_looped_branch.json`
- `loop_runs/tiny_baseline_looped_branch.log`

Settings:

- `block_loop_count=1`
- `device_batch_size=32`
- `num_epochs=16`

Result:

- checkpoint-average best val loss: `3.349000428852282`
- wall clock: `16.03m`

### Exact Looped Variant at Default Microbatch

Run:

- `loop_runs/tiny_looped3.log`

Settings:

- `block_loop_count=3`
- `device_batch_size=32`

Result:

- failed with CUDA OOM before the first training step

This means the exact looped block is not a drop-in tiny-track replacement under the default memory budget.

### Fit Check

Run:

- `loop_runs/tiny_looped3_fitcheck.json`
- `loop_runs/tiny_looped3_fitcheck.log`

Settings:

- `block_loop_count=3`
- `device_batch_size=16`
- `num_epochs=1`

Purpose:

- verify that the looped architecture fits when the per-GPU microbatch is reduced
- measure rough throughput before spending the full run

Observed wall clock:

- `5.01m` for one epoch

This implies the full run would be around `5 * 16 / 2.1 ~= 38m`, which matched the eventual full run closely.

### Full Fitted Looped Run

Run:

- `loop_runs/tiny_looped3_dbs16.json`
- `loop_runs/tiny_looped3_dbs16.log`

Settings:

- `block_loop_count=3`
- `device_batch_size=16`
- `total_batch_size=524288`
- `num_epochs=16`

Result:

- checkpoint-average best val loss: `3.337236705579256`
- final logged val loss: `3.3631294652035364`
- wall clock: `38.17m`

## Comparison

| Run | Loop Count | Device Batch Size | Best Val Loss | Wall Clock |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `1` | `32` | `3.349000428852282` | `16.03m` |
| Looped fitted run | `3` | `16` | `3.337236705579256` | `38.17m` |

Absolute improvement:

- `0.011763723273026`

## Important Checkpoints

Looped fitted run:

- Epoch 8: `3.534778`
- Epoch 9: `3.523064`
- Epoch 13: `3.372278`
- Epoch 15: `3.383336`
- Checkpoint average: `3.337237`

Baseline:

- Checkpoint average: `3.349000`

## Interpretation

This looping idea works as a loss-improving architecture change on tiny.

But it does not work as a clean tiny-track result in its current form:

- with the default microbatch it OOMs
- the fitted version needs `device_batch_size=16`
- the fitted version takes about `38` minutes, far beyond the normal tiny-track runtime

So the correct conclusion is:

- yes, the looped block improves validation loss
- no, it is not currently a tiny-budget-compliant win

## Recommendation

If we want to keep pursuing this idea, the next step should be making the looped variant cheaper rather than just celebrating the loss number.

The two most obvious directions are:

- loop only a subset of layers
- enable looping only in later epochs, similar to other successful slowrun looping ideas
