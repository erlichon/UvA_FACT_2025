# Agent Prompt: Fix Discrepancies in FACT-AI Reproduction

**Purpose**: This prompt is for a subsequent agent to fix the identified discrepancies between our implementation and the original paper.

---

## Context

We are reproducing "Bilinear MLPs enable weight-based mechanistic interpretability" (arXiv:2410.08417). An investigation agent identified critical discrepancies that need fixing.

## Priority 1: Fix Effective Rank Formula (CRITICAL)

### Problem

Our `effective_rank()` function in `src/vision/spectral.py` uses the **wrong formula**.

**Current (WRONG) - Entropy-based**:
```python
def effective_rank(eigenvalues):
    abs_vals = eigenvalues.abs()
    p = abs_vals / abs_vals.sum(dim=-1, keepdim=True)
    p = p.clamp(min=1e-10)
    entropy = -(p * p.log()).sum(dim=-1)
    return entropy.exp()
```

**Required (CORRECT) - Ratio-based**:
```python
def effective_rank(eigenvalues):
    """
    Compute ratio-based effective rank: (||lambda||_1 / ||lambda||_2)^2

    This matches the original paper implementation in sae/functions.py
    """
    abs_vals = eigenvalues.abs()
    l1 = abs_vals.sum(dim=-1)
    l2 = abs_vals.pow(2).sum(dim=-1).sqrt()
    return (l1 / l2.clamp(min=1e-10)).pow(2)
```

### Tasks

1. **Read** the original implementation at `bilinear-decomposition-main/sae/functions.py` (lines 45-51)
2. **Update** `src/vision/spectral.py`:
   - Rename current `effective_rank` to `effective_rank_entropy` (keep for reference)
   - Add new `effective_rank` function with ratio-based formula
   - Update docstrings to explain the difference
3. **Create** a script `scripts/recompute_effective_rank.py` that:
   - Loads all checkpoints from `results/phase1/checkpoints/` and `results/phase1_fashion/checkpoints/`
   - Re-computes effective rank using the corrected formula
   - Updates the checkpoint files with new `metrics.effective_rank` values
   - Prints a comparison table (old vs new values)
4. **Update** documentation files (CONTEXT.md, WORKPLAN.md, CLAUDE.md) with corrected results

### Verification

After fixing, the effective rank values should be lower overall and the ratio (reg/no-reg) should be closer to expected.

---

## Priority 2: Update Documentation with Discrepancy Notes

### Problem

Our documentation doesn't clearly explain the known discrepancies.

### Tasks

1. **Add to CLAUDE.md** a new section "## Known Discrepancies" explaining:
   - Language experiments use `ts-medium` instead of `fw-medium` due to SAE availability
   - Language experiments use layer 5 instead of layer 7
   - SAE expansion is 4 instead of 8
   - These explain why interaction analysis shows 0% vs paper's 69%

2. **Update CONTEXT.md** Section 0 with:
   - Corrected effective rank results (after recomputation)
   - Clear note that noise affects eigenvector interpretability, not numerical rank

---

## Priority 3: Check fw-medium SAE Availability (Optional)

### Problem

The language discrepancies stem from using the wrong model. We should check if the correct model's SAEs are available.

### Tasks

1. **Check HuggingFace** for `tdooms/fw-medium-scope` repository
   - Run: `python -c "from sae.sae import SAE; SAE.from_pretrained('tdooms/fw-medium-scope', point=('mlp-out', 7), expansion=8, k=32)"`
2. **If available**:
   - Update `configs/language_interaction.yaml` with model=fw-medium, layer=7, expansion=8
   - Update `configs/language_negation.yaml` similarly
   - Re-run language experiments
3. **If not available**:
   - Document this limitation in the report
   - Explain that our results use a different model and are not directly comparable

---

## Priority 4: Re-run Analysis After Fixes

### Tasks

1. **Re-compute** effective rank for all 40 vision checkpoints
2. **Generate** updated results table:

```
| Config | Val Accuracy | Eff Rank (Old) | Eff Rank (New) |
|--------|-------------|----------------|----------------|
| mnist_none | 97.49% | 76.54 | ? |
| mnist_noise | 98.41% | 117.98 | ? |
| mnist_wd | 97.50% | 38.97 | ? |
| mnist_full | 98.30% | 61.35 | ? |
| fashion_none | 88.50% | 91.55 | ? |
| fashion_noise | 87.62% | 127.53 | ? |
| fashion_wd | 87.74% | 40.54 | ? |
| fashion_full | 87.18% | 62.06 | ? |
```

3. **Update** all documentation with new values
4. **Verify** the effective rank ratio (reg/no-reg) is now closer to paper expectations

---

## Files to Modify

| File | Action |
|------|--------|
| `src/vision/spectral.py` | Fix effective_rank formula |
| `scripts/recompute_effective_rank.py` | Create new script |
| `CLAUDE.md` | Add Known Discrepancies section |
| `CONTEXT.md` | Update results, add discrepancy notes |
| `WORKPLAN.md` | Update results tables |
| `configs/language_*.yaml` | Update model/layer if fw-medium available |

---

## Reference Files

| File | Purpose |
|------|---------|
| `bilinear-decomposition-main/sae/functions.py` | Correct effective rank implementation |
| `bilinear-decomposition-main/image/model.py` | Reference for decomposition |
| `bilinear-decomposition-main/tutorials/1_image.ipynb` | Paper's actual hyperparameters |

---

## Success Criteria

1. Effective rank formula matches paper implementation
2. All 40 checkpoints have recomputed effective rank values
3. Documentation clearly explains discrepancies
4. Language experiment limitations are documented
5. Results tables are updated with corrected values
