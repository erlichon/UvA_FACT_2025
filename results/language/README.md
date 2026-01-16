# Language Experiment Results (Section 5)

## Model Verification (2026-01-14)

✅ **VERIFIED**: `tdooms/ts-medium` IS the paper's "ts-tiny"
- Specifications match: 6 layers, 512 d_model, 29.4M parameters, TinyStories dataset
- Verified using `scripts/verify_ts_medium.py`

## Two Negation Circuit Implementations

### 1. Paper Figure 8 (Section 5.1) - EXACT REPRODUCTION
- **Model**: `tdooms/ts-medium` (paper's "ts-tiny")
- **Config**: `configs/language_negation_ts.yaml`
- **Features**:
  - **1882** ("not-good"): Activates on "not" + negative words
  - **1179** ("not-bad"): Activates on "not" + positive words
- **Expected**: Cosine similarity -0.975 (opposing directions)
- **Layer**: 4 (middle of 6-layer model)
- **Expansion**: 4

### 2. Tutorial Example - DEMONSTRATION
- **Model**: `tdooms/fw-medium`
- **Config**: `configs/language_negation_fw.yaml`
- **Features**:
  - **3834** ("not-good"): Activates on "not lost", "no interference"
  - **751** ("not-bad"): Activates on "not free", "little relief"
- **Expected**: Negative cosine similarity (opposing directions)
- **Layer**: 7 (middle of 16-layer model)
- **Expansion**: 8
- **Note**: Larger model (335M vs 29M params) provides clearer examples

## Figure 9: Low-Rank Correlation (Section 5.2)

All three models for comprehensive analysis:
- **ts-medium** (6L, layer 4, exp=4): Paper's baseline
- **fw-small** (12L, layer 8, exp=4): Medium-scale comparison
- **fw-medium** (16L, layer 7, exp=8): Large-scale comparison

**Paper Claim**: 69% of features have >0.75 rank-2 correlation

## Results Directory Structure

```
results/language/
├── README.md                           # This file
├── correlation_ts-medium.json          # Figure 9 data (ts-medium)
├── correlation_fw-small.json           # Figure 9 data (fw-small)
├── correlation_fw-medium.json          # Figure 9 data (fw-medium, includes scatter)
├── figure_8_data_ts_medium.json        # Figure 8 data (PAPER)
├── figure_8_data_fw_medium.json        # Figure 8 data (TUTORIAL)
└── figures/
    ├── figure_8_ts_medium.pdf          # Paper Figure 8 reproduction
    ├── figure_8_fw_medium.pdf          # Tutorial Figure 8 demonstration
    ├── figure_9a_correlation_progression.pdf
    ├── figure_9b_correlation_histogram.pdf
    └── figure_9c_scatter_plots.pdf
```

## Running Experiments

### Full Reproduction (Both Models)
```bash
# Sequential execution to avoid OOM
./scripts/run_language_sequential.sh mps full
```

### Individual Models
```bash
# Paper Figure 8 (ts-medium)
python src/language/negation_visualization.py \
    --config configs/language_negation_ts.yaml \
    --output results/language/figure_8_data_ts_medium.json \
    --feature 1882 --device mps

# Tutorial Figure 8 (fw-medium)
python src/language/negation_visualization.py \
    --config configs/language_negation_fw.yaml \
    --output results/language/figure_8_data_fw_medium.json \
    --feature 3834 --device mps
```

### Generate Figures
```bash
python scripts/figures/generate_language_figures.py
```

## Why Both Models?

1. **Paper Reproduction**: ts-medium (features 1882/1179) is what the paper actually used for Figure 8
2. **Tutorial Demonstration**: fw-medium (features 3834/751) shows the same technique scales to larger models
3. **Validation**: Both should show:
   - Opposing feature directions (negative cosine similarity)
   - Low-rank interaction matrices
   - Interpretable sentiment negation circuits

This dual reproduction demonstrates:
- Exact reproducibility of paper results (ts-medium)
- Generalization of the technique (fw-medium)
- Scalability from 29M to 335M parameters
