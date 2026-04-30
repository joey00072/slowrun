# Exclusive MLP Experiment

## Summary

Exclusive MLP orthogonalizes the MLP residual update against the current residual stream before adding it:

```python
m = mlp(norm(h))
dot = (m * h).sum(dim=-1, keepdim=True)
h_norm2 = (h * h).sum(dim=-1, keepdim=True)
m = m - dot / h_norm2.clamp_min(eps) * h
h = h + m
```

This mirrors exclusive self-attention's projection idea, but applies it to the MLP update in residual-stream space. The projection is against `h`, not `norm(h)`, because the residual addition happens in unnormalized residual space.

## Implementation

The experiment adds these flags to `train.py`:

- `--exclusive-mlp`
- `--exclusive-mlp-eps`
- `--exclusive-mlp-stride`
- `--exclusive-mlp-offset`
- `--exclusive-mlp-skip-layers`

The final implementation uses vanilla PyTorch only. Earlier Triton experiments were removed because the custom kernel did not materially reduce the duplicate-layer phase runtime.

## Results

All runs used the current main-track setup with `matrix_lr=0.04`, `weight_decay=1.3`, and `num_epochs=11`.

| Run | Layers | Best val loss | Final val loss | Training time | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| Local baseline | none | 3.1993905415660455 | 3.237871471204256 | 59.35m | Reference |
| Exclusive MLP, all layers | all layers | 3.1942234517712342 | 3.2374086881938733 | 61.17m | Better loss, over 60m cap |
| Exclusive MLP, stride 8 offset 5 | 5, 13, 21, 29 | 3.1977059076491154 | 3.238688820286801 | 60.29m | Better than local baseline, over 60m cap |
| Exclusive MLP, stride 16 offset 5 | 5, 21 | 3.1996918120666553 | 3.2384362471731087 | 60.58m | Worse than local baseline, over 60m cap |
| Exclusive MLP, layer 15 only | 15 | 3.199834244619859 | 3.238018638209293 | 59.94m | Under cap, worse than local baseline |

The layer-15-only command was:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --exclusive-mlp \
  --exclusive-mlp-offset 15 \
  --exclusive-mlp-stride 1000 \
  --run-name main_exclusive_mlp_layer15_only \
  --save-result exclusive_mlp_runs/main_exclusive_mlp_layer15_only.json \
  --logit-avg-dir layer15_only_logit_avg_ckpts
```

## Conclusion

Exclusive MLP has a real loss signal when applied broadly, but it is too slow for the current 1 hour main-track constraint in this implementation. The only tested configuration that stayed under the cap, layer 15 only, did not beat the local baseline:

```text
3.199834244619859 - 3.1993905415660455 = +0.0004437030538135
```

This branch should be treated as a negative or inconclusive main-track experiment, not a new record submission.
