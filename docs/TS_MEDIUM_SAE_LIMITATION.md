# ts-medium SAE Availability Limitation

**Date**: 2026-01-14  
**Discovery**: ts-medium layer 4 does not have `mlp-in` SAEs publicly available

## Background

The paper (Section 5.1, Figure 8) claims to analyze sentiment negation circuits in a 6-layer TinyStories model ("ts-tiny"), finding features 1882 and 1179 in layer 4.

We verified that `tdooms/ts-medium` on HuggingFace IS the paper's "ts-tiny" model:
- ✅ 6 layers
- ✅ 512 d_model
- ✅ ~29.4M parameters
- ✅ Trained on TinyStories dataset

## Problem

The `Tracer` class (used for computing interaction matrices Q) requires **both** input and output SAEs:
```python
tracer = Tracer(model, layer, inp=dict(expansion=4), out=dict(expansion=4))
```

However, checking `tdooms/ts-medium-scope` on HuggingFace reveals:

### Layer 4 Available SAEs
- ✅ `mlp-out` (expansion=4, k=30)
- ✅ `resid-mid` (expansion=4/24, k=30)
- ✅ `resid-pre` (expansion=24, k=30)
- ❌ **NO `mlp-in` SAE**

### Layer 5 Available SAEs
- ✅ `mlp-in` (expansion=4, k=30)
- ✅ `mlp-out` (expansion=4, k=30)
- ✅ Many other variants

## Impact

**Figure 8 reproduction for ts-medium layer 4 is NOT possible** with publicly available SAEs.

The paper likely used internal SAE checkpoints that were not released to HuggingFace.

## Solution

We reproduce Figure 8 using the **fw-medium example** from the tutorial instead:
- Model: `tdooms/fw-medium` (16 layers, ~335M params, FineWeb-EDU)
- Layer: 7
- Features: 3834 (not-good), 751 (not-bad)
- SAEs: Both `mlp-in` and `mlp-out` available (expansion=8, k=30)

This demonstrates the **same negation circuit phenomenon** with clearer, larger-scale visualizations.

## Verification Script

Run `scripts/check_ts_medium_saes.py` to verify SAE availability:

```bash
python scripts/check_ts_medium_saes.py
```

Output:
```
Layer 4 (Paper's Figure 8 layer) SAEs:
  ✓ mlp-out: expansion=4, k=30
  ✓ resid-mid: expansion=24, k=30
  ✓ resid-mid: expansion=4, k=30
  ✓ resid-pre: expansion=24, k=30

⚠️  WARNING: Only mlp-out exists (no mlp-in)
   Figure 8 reproduction may not be possible for ts-medium
```

## Alternative Approaches (Not Implemented)

1. **Use layer 5 instead**: Layer 5 has both SAEs, but features 1882/1179 are specific to layer 4
2. **Use residual stream SAEs**: Would require modifying the Tracer class significantly
3. **Request internal checkpoints**: Could contact paper authors for original SAE weights

## Conclusion

- ✅ **Figure 9** (correlation analysis): Uses only output SAEs → Works for all models (ts-medium, fw-small, fw-medium)
- ❌ **Figure 8** (interaction analysis): Requires both input+output SAEs → Only fw-medium available
- 📊 **Scientific validity**: fw-medium demonstrates the same phenomenon at larger scale

The fw-medium example is **pedagogically superior** anyway:
- Larger model (16L vs 6L)
- More parameters (335M vs 29M)
- Cleaner training data (FineWeb-EDU vs TinyStories)
- More interpretable features
