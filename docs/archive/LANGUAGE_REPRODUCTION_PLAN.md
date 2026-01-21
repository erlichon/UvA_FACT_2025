# Language Experiments: Dual Reproduction Plan

**Updated**: 2026-01-14

## Key Finding: ts-medium IS ts-tiny

✅ **VERIFIED** using `scripts/verify_ts_medium.py`:
- `tdooms/ts-medium` specifications: 6 layers, 512 d_model, 29.4M params, TinyStories
- Matches paper's "ts-tiny" specifications exactly
- Paper refers to it as "ts-tiny" but HuggingFace repo is "ts-medium"

---

## Dual Reproduction Strategy

We will reproduce **BOTH** negation circuit implementations to demonstrate:
1. **Exact paper reproduction** (ts-medium)
2. **Technique generalization** (fw-medium, larger model)
3. **Scalability** from 29M to 335M parameters

### Model Comparison

| Aspect | ts-medium (PAPER) | fw-medium (TUTORIAL) |
|--------|-------------------|---------------------|
| **Layers** | 6 | 16 |
| **d_model** | 512 | 1024 |
| **Parameters** | 29.4M | 335M |
| **Dataset** | TinyStories | FineWeb-EDU |
| **SAE Layer** | 4 | 7 |
| **SAE Expansion** | 4 | 8 |
| **Feature (not-good)** | **1882** | **3834** |
| **Feature (not-bad)** | **1179** | **751** |
| **Expected Cosine Sim** | -0.975 | < 0 (negative) |
| **Source** | Paper Figure 8 | Tutorial 2_language.ipynb |

---

## Updated Configurations

### 1. `configs/language_negation_ts.yaml` (PAPER)
- **Model**: tdooms/ts-medium (verified as ts-tiny)
- **Features**: 1882 (not-good), 1179 (not-bad)
- **Layer**: 4, Expansion: 4
- **Purpose**: Exact Figure 8 reproduction

### 2. `configs/language_negation_fw.yaml` (TUTORIAL)
- **Model**: tdooms/fw-medium
- **Features**: 3834 (not-good), 751 (not-bad)
- **Layer**: 7, Expansion: 8
- **Purpose**: Demonstrate technique on larger model

---

## Experiments to Run

### Full Sequential Pipeline (Recommended)

```bash
./scripts/run_language_sequential.sh mps full
```

**This will run:**
1. ✅ Correlation sweep (ts-medium, fw-small, fw-medium) for Figure 9
   - All active features analyzed
   - Ranks 1-60 for smooth progression curves
   - Scatter data saved for Figure 9C
2. ✅ Figure 8 (ts-medium): Paper reproduction, features 1882/1179
3. ✅ Figure 8 (fw-medium): Tutorial demonstration, features 3834/751
4. ✅ Generate all figures (Figure 9A, 9B, 9C, Figure 8 × 2)

**Estimated time**: ~10-14 hours on Apple Silicon MPS

### Individual Experiments

If memory issues persist, run separately:

```bash
# 1. Figure 9 correlation sweep (ts-medium)
python src/language/verify_correlation.py \
    --config configs/language_correlation_fw.yaml \
    --model tdooms/ts-medium --layer 4 --expansion 4 \
    --output results/language/correlation_ts-medium.json \
    --device mps --no-wandb \
    --n-features -1 --ranks 1-60 --save-scatter --max-scatter-samples 1000

# 2. Figure 9 correlation sweep (fw-small)
python src/language/verify_correlation.py \
    --config configs/language_correlation_fw.yaml \
    --model tdooms/fw-small --layer 8 --expansion 4 \
    --output results/language/correlation_fw-small.json \
    --device mps --no-wandb \
    --n-features -1 --ranks 1-60 --save-scatter --max-scatter-samples 1000

# 3. Figure 9 correlation sweep (fw-medium)
python src/language/verify_correlation.py \
    --config configs/language_correlation_fw.yaml \
    --model tdooms/fw-medium --layer 7 --expansion 8 \
    --output results/language/correlation_fw-medium.json \
    --device mps --no-wandb \
    --n-features -1 --ranks 1-60 --save-scatter --max-scatter-samples 1000

# 4. Figure 8 - Paper reproduction (ts-medium)
python src/language/negation_visualization.py \
    --config configs/language_negation_ts.yaml \
    --output results/language/figure_8_data_ts_medium.json \
    --feature 1882 --device mps

# 5. Figure 8 - Tutorial (fw-medium)
python src/language/negation_visualization.py \
    --config configs/language_negation_fw.yaml \
    --output results/language/figure_8_data_fw_medium.json \
    --feature 3834 --device mps

# 6. Generate all figures
python scripts/figures/generate_language_figures.py
```

---

## Expected Outputs

### Results Files
```
results/language/
├── README.md (documentation)
├── correlation_ts-medium.json      # Figure 9 data
├── correlation_fw-small.json       # Figure 9 data
├── correlation_fw-medium.json      # Figure 9 data (with scatter)
├── figure_8_data_ts_medium.json    # PAPER Figure 8
└── figure_8_data_fw_medium.json    # TUTORIAL Figure 8
```

### Generated Figures
```
Report/figures/
├── figure_9a_correlation_progression.pdf  # 3 models, ranks 1-60
├── figure_9b_correlation_histogram.pdf    # Rank-2 distribution
├── figure_9c_scatter_plots.pdf            # 3×3 scatter grid (fw-medium)
├── figure_8_ts_medium.pdf                 # PAPER (features 1882/1179)
└── figure_8_fw_medium.pdf                 # TUTORIAL (features 3834/751)
```

---

## Success Criteria

### Figure 9 (Section 5.2)
- [ ] ts-medium, fw-small, fw-medium correlation data collected
- [ ] Mean rank-2 correlation > 0.65 for all models
- [ ] >60% of features have rank-2 correlation >= 0.75
- [ ] Smooth progression curves from rank 1-60
- [ ] Scatter plots show high correlation at large activations

### Figure 8 - Paper (ts-medium)
- [ ] Features 1882 and 1179 identified
- [ ] Cosine similarity ≈ -0.975 (opposing directions)
- [ ] Interaction submatrix shows AND-gate pattern
- [ ] Top 2 eigenvalues are outliers (positive and negative)
- [ ] Rank-2 correlation > 0.6 for feature 1882

### Figure 8 - Tutorial (fw-medium)
- [ ] Features 3834 and 751 identified
- [ ] Negative cosine similarity (opposing directions)
- [ ] Interaction submatrix shows clear structure
- [ ] Low-rank eigenspectrum
- [ ] High rank-2 correlation

---

## Documentation Updates

All documentation has been updated with verified information:

✅ **CLAUDE.md**: Updated model configuration table
✅ **CONTEXT.md**: Updated Section 5 status and model comparison
✅ **.cursorrules**: Updated key paper claims
✅ **configs/language_negation_ts.yaml**: Updated with paper features (1882/1179)
✅ **configs/language_negation_fw.yaml**: Updated with tutorial features (3834/751)
✅ **results/language/README.md**: Created comprehensive guide
✅ **scripts/run_language_sequential.sh**: Updated to run both Figure 8 experiments

---

## Next Steps

1. **Run the sequential pipeline**:
   ```bash
   ./scripts/run_language_sequential.sh mps full
   ```

2. **Monitor progress**:
   - Check logs for each phase
   - Verify JSON outputs are created
   - Watch for OOM errors (should be resolved with sequential execution)

3. **Generate figures**:
   - Automatically done at end of pipeline
   - Or manually: `python scripts/figures/generate_language_figures.py`

4. **Verify results**:
   - Compare ts-medium results to paper claims
   - Verify fw-medium shows similar patterns
   - Check that both show opposing feature directions

5. **Update report**:
   - Include both reproductions in Section 5
   - Discuss scalability from 29M to 335M params
   - Compare interaction matrices side-by-side

---

## Why This Approach?

1. **Scientific rigor**: Exact paper reproduction (ts-medium)
2. **Validation**: Technique works on different models/datasets
3. **Scalability**: Demonstrates 11x parameter scaling
4. **Clarity**: Larger model often has clearer features
5. **Completeness**: Addresses both paper and tutorial code

This dual approach strengthens the reproduction by showing:
- The exact paper result is reproducible
- The technique generalizes beyond the specific model
- Interpretability emerges at multiple scales
