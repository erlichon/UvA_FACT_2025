# FACT-AI Project Work Plan
## Bilinear MLPs: Reproduction + Structure vs. Regularization

**Paper**: [arXiv:2410.08417](https://arxiv.org/pdf/2410.08417)
**Deadline**: 30 January 2026, 23:59
**Draft Feedback**: 22 January 2026, 23:59
**Result Freeze**: 17 January 2026

---

## Global Constraints (Apply to ALL Tasks)

### Environmental Tracking (MANDATORY)
Every experiment MUST track and report:

| Metric | Tool | Logged To |
|--------|------|-----------|
| GPU Model | `torch.cuda.get_device_name()` | wandb config |
| Wall-clock Time | `time.time()` | wandb log |
| GPU Hours | wall_time × gpu_count | wandb summary |
| CO2 Emissions (kg) | `codecarbon.EmissionsTracker` | wandb summary |

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="fact-bilinear")
tracker.start()
# ... training ...
emissions_kg = tracker.stop()
wandb.summary["co2_kg"] = emissions_kg
```

### Reproducibility
- All experiments run with 5 seeds: `[42, 43, 44, 45, 46]`
- Configs stored in `configs/` as YAML
- All metrics logged to wandb project `fact-bilinear`

---

## Core Hypothesis

> **Structural low-rank** (CP-Decomposition) produces equivalent or better interpretability than **emergent low-rank** (via regularization), with the advantage of explicit rank control.

---

## Phase 1: Reproduction (Mandatory Baseline)

**Goal**: Verify Pearce et al. Section 4 (Vision) findings.
**Timeline**: Days 1-7 (Jan 8-14)

### Task 1.1: Environment Setup
- [ ] Clone repo to Snellius `/home/scur0075/fact-project/`
- [ ] Create conda environment: `conda env create -f environment.yml`
- [ ] Verify GPU access: `srun --partition=gpu_a100 --gpus=1 --pty bash`
- [ ] Test: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Test original code tutorials run successfully

**Success Criterion**: `tutorials/1_image.ipynb` executes without errors.

### Task 1.2: Baseline Dense Model (No Regularization)
- [ ] Train `Bilinear(mode='dense')` on MNIST
- [ ] Config: `noise_std=0.0`, `weight_decay=0.0`
- [ ] Log: accuracy, eigenspectrum, effective rank
- [ ] Expected: Overfitting, high effective rank, non-interpretable eigenvectors

**Config**: `configs/mnist_dense_none.yaml`
```yaml
mode: dense
d_hidden: 256
epochs: 100
noise_std: 0.0
weight_decay: 0.0
seed: 42
```

### Task 1.3: Dense Model with Regularization
- [ ] Train `Bilinear(mode='dense')` with noise + weight decay
- [ ] Config: `noise_std=0.4`, `weight_decay=0.5` (paper defaults)
- [ ] Log: accuracy, eigenspectrum, effective rank
- [ ] Expected: Low effective rank, interpretable digit-like eigenvectors

**Config**: `configs/mnist_dense_reg.yaml`
```yaml
mode: dense
d_hidden: 256
epochs: 100
noise_std: 0.4
weight_decay: 0.5
seed: 42
```

### Task 1.4: Reproduce Figure 3/4 (Eigenspectrum Analysis)
- [ ] Generate eigenspectrum plots for both models
- [ ] Compute per-class eigenvalue distributions
- [ ] Visualize top eigenvectors as 28x28 images
- [ ] Compare effective rank: regularized vs. non-regularized

**Success Criterion**: Eigenspectrum plots show clear low-rank structure with regularization (matching paper Figure 3/4 trend).

### Task 1.5: Regularization Ablation
- [ ] Train 4 variants:
  | Config | Noise | Weight Decay |
  |--------|-------|--------------|
  | none | 0.0 | 0.0 |
  | noise_only | 0.4 | 0.0 |
  | wd_only | 0.0 | 0.5 |
  | full | 0.4 | 0.5 |
- [ ] Compare accuracy vs. effective rank trade-off
- [ ] Document which regularization contributes most to interpretability

### Phase 1 Deliverables
| Artifact | Location |
|----------|----------|
| Trained models | `results/phase1/checkpoints/` |
| Eigenspectrum plots | `results/phase1/figures/` |
| Experiment metrics | wandb project `fact-bilinear` |
| CO2 tracking | `codecarbon` via wandb summary |

### Phase 1 Gate Check (Jan 14)
- [ ] Effective rank of regularized model < 50% of non-regularized
- [ ] Top eigenvectors visually resemble digits
- [ ] Accuracy within 2% of paper reported values

**If FAIL**: Debug reproduction before proceeding. Do not start Phase 2.

---

## Phase 2: Extensions (Structure vs. Regularization)

**Goal**: Test hypothesis that structural low-rank matches emergent low-rank.
**Timeline**: Days 8-17 (Jan 15-24)

### Extension 1: Robustness Testing (Days 8-9)
**Question**: Do regularization-induced eigenvectors generalize?

- [ ] Take best model from Phase 1 (dense + regularization)
- [ ] Evaluate on Rotated MNIST (15, 30, 45, 60, 90 degrees)
- [ ] Evaluate on EMNIST (letters subset)
- [ ] Measure: accuracy drop, eigenvalue stability

**Datasets**:
```python
# Rotated MNIST
from torchvision.transforms import RandomRotation
rotations = [15, 30, 45, 60, 90]

# EMNIST Letters
from torchvision.datasets import EMNIST
emnist = EMNIST(root='./data', split='letters', download=True)
```

**Success Criterion**: Document how eigenvector interpretability degrades under distribution shift.

### Extension 2: CP-Decomposition Implementation (Days 9-11)
**Goal**: Implement structural low-rank bilinear layer.

- [ ] Implement `Bilinear(mode='cp')` in `src/models/bilinear_layer.py`
- [ ] Match API with dense mode for fair comparison
- [ ] Implement eigendecomposition for CP mode

**Implementation**:
```python
class Bilinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, mode: str = 'dense',
                 rank: int = None, bias: bool = False):
        super().__init__()
        self.mode = mode
        self.d_in = d_in
        self.d_out = d_out

        if mode == 'dense':
            self.W_l = nn.Linear(d_in, d_out, bias=bias)
            self.W_r = nn.Linear(d_in, d_out, bias=bias)
        elif mode == 'cp':
            assert rank is not None, "rank required for CP mode"
            self.rank = rank
            self.A = nn.Parameter(torch.randn(d_out, rank) * 0.02)
            self.B = nn.Parameter(torch.randn(d_in, rank) * 0.02)
            self.C = nn.Parameter(torch.randn(d_in, rank) * 0.02)
            self.lambdas = nn.Parameter(torch.ones(rank))

    def forward(self, x):
        if self.mode == 'dense':
            return self.W_l(x) * self.W_r(x)
        else:  # cp
            left = x @ self.B        # [batch, rank]
            right = x @ self.C       # [batch, rank]
            hidden = left * right * self.lambdas
            return hidden @ self.A.T  # [batch, d_out]

    @property
    def w_l(self):
        if self.mode == 'dense':
            return self.W_l.weight
        else:
            return (self.A @ torch.diag(self.lambdas) @ self.B.T).T

    @property
    def w_r(self):
        if self.mode == 'dense':
            return self.W_r.weight
        else:
            return (self.A @ torch.diag(self.lambdas) @ self.C.T).T
```

### Extension 2 Gate: XOR Sanity Check (Day 11)
- [ ] Train CP bilinear on XOR task (from `toy/model.py`)
- [ ] Test ranks: R=1, R=2, R=4, R=8
- [ ] Expected: R=1 fails, R>=2 succeeds (XOR is rank-2)

**If FAIL**: CP implementation is broken. Debug before proceeding.

### Extension 3: CP without Heavy Regularization (Days 12-14)
**Core Experiment**: Does structural rank eliminate need for regularization?

- [ ] Train `Bilinear(mode='cp', rank=R)` on MNIST
- [ ] Sweep ranks: R = 8, 16, 32, 64, 128, 256
- [ ] Config: `noise_std=0.0`, `weight_decay=0.1` (minimal)
- [ ] Compare to Phase 1 dense model with full regularization

**Key Metrics**:
| Model | Accuracy | Effective Rank | Top-5 Coverage |
|-------|----------|----------------|----------------|
| Dense (no reg) | ? | ? | ? |
| Dense (full reg) | ? | ? | ? |
| CP R=32 (no reg) | ? | ? | ? |
| CP R=64 (no reg) | ? | ? | ? |

**Success Criterion**: CP mode achieves comparable interpretability (low effective rank, interpretable eigenvectors) without noise augmentation.

### Extension 4: Structure vs. Regularization Comparison (Days 14-16)
**The Synthesis**: Direct comparison of both approaches.

- [ ] Generate comparison plots:
  - Accuracy vs. Effective Rank (Pareto frontier)
  - Accuracy vs. CP Rank
  - Eigenspectrum overlay: Dense(reg) vs. CP(no reg)
- [ ] Visualize eigenvectors side-by-side
- [ ] Statistical significance test (3 seeds each)

**Figure Spec**:
```
Figure 5: Structure vs. Regularization Trade-off
- Left: Accuracy vs. Effective Rank scatter (color by method)
- Right: Top-3 eigenvectors for each method
```

### Extension 5: ViT Formulation (Conceptual Only)
**Goal**: Demonstrate how CP-Bilinear integrates with Vision Transformers.

- [ ] Write mathematical formulation in report
- [ ] No code implementation (out of scope)
- [ ] Discuss: "BiLoRA" - combining LoRA with bilinear interpretability

**Report Section** (0.5 pages):
> The CP-decomposed bilinear layer naturally extends to Vision Transformers by replacing the MLP block:
> ```
> FFN(x) = W_p * (CPBilinear(LN(x)))
> ```
> This provides interpretable feature interactions at each layer while maintaining parameter efficiency through rank control.

---

## Experiment Matrix

### Phase 1: Reproduction (Regularization vs. Interpretability)
| ID | Dataset | Mode | Noise | WD | Seeds | GPU Hrs | CO2 (kg) | Figure Generated? |
|----|---------|------|-------|-----|-------|---------|----------|-------------------|
| P1.1 | MNIST | dense | 0.0 | 0.0 | 5 | 0.8 | 0.10 | [ ] eigenspectrum_noreg.pdf |
| P1.2 | MNIST | dense | 0.4 | 0.0 | 5 | 0.8 | 0.10 | [ ] (ablation table) |
| P1.3 | MNIST | dense | 0.0 | 0.5 | 5 | 0.8 | 0.10 | [ ] (ablation table) |
| P1.4 | MNIST | dense | 0.4 | 0.5 | 5 | 0.8 | 0.10 | [ ] eigenspectrum_reg.pdf, eigenvectors_reg.pdf |
| **P1 Total** | | | | | **20** | **3.2** | **0.42** | |

**Phase 1 Required Figures**:
- [ ] `eigenspectrum_comparison.pdf` - P1.1 vs P1.4 overlay
- [ ] `eigenvectors_noreg.pdf` - Top eigenvectors from P1.1
- [ ] `eigenvectors_reg.pdf` - Top eigenvectors from P1.4
- [ ] `accuracy_vs_effrank.pdf` - Scatter of all P1 configs

### Phase 2: Extensions (Robustness + CP-Bilinear + Synthesis)
| ID | Dataset | Mode | Rank | Noise | WD | Seeds | GPU Hrs | CO2 (kg) | Figure Generated? |
|----|---------|------|------|-------|-----|-------|---------|----------|-------------------|
| E1.1 | RotMNIST | dense | - | 0.4 | 0.5 | 5 | 0.8 | 0.10 | [ ] robustness_rotated.pdf |
| E1.2 | EMNIST | dense | - | 0.4 | 0.5 | 5 | 0.8 | 0.10 | [ ] robustness_emnist.pdf |
| E2.1 | XOR | cp | 1,2,4,8 | 0.0 | 0.0 | 5 | 0.2 | 0.03 | [ ] (sanity check) |
| E3.1 | MNIST | cp | 8 | 0.0 | 0.1 | 5 | 0.8 | 0.10 | [ ] cp_rank_sweep.pdf |
| E3.2 | MNIST | cp | 16 | 0.0 | 0.1 | 5 | 0.8 | 0.10 | [ ] cp_rank_sweep.pdf |
| E3.3 | MNIST | cp | 32 | 0.0 | 0.1 | 5 | 0.8 | 0.10 | [ ] cp_rank_sweep.pdf |
| E3.4 | MNIST | cp | 64 | 0.0 | 0.1 | 5 | 0.8 | 0.10 | [ ] cp_rank_sweep.pdf |
| E3.5 | MNIST | cp | 128 | 0.0 | 0.1 | 5 | 0.8 | 0.10 | [ ] cp_rank_sweep.pdf |
| E3.6 | MNIST | cp | 256 | 0.0 | 0.1 | 5 | 0.8 | 0.10 | [ ] cp_rank_sweep.pdf |
| **E Total** | | | | | **45** | **7.4** | **0.97** | |

**Phase 2 Required Figures**:
- [ ] `robustness_rotated.pdf` - Eigenvector stability under rotation
- [ ] `robustness_emnist.pdf` - Transfer to EMNIST letters
- [ ] `cp_rank_sweep.pdf` - Accuracy vs CP rank
- [ ] `structure_vs_reg.pdf` - Pareto frontier comparison (Dense+reg vs CP)

### Summary
| Phase | Focus | GPU Hours | CO2 (kg) |
|-------|-------|-----------|----------|
| P1 | Regularization vs. Interpretability | 3.2 | 0.42 |
| E1 | Robustness (RotMNIST, EMNIST) | 1.6 | 0.20 |
| E2 | CP-Bilinear Implementation (XOR) | 0.2 | 0.03 |
| E3 | CP Rank Sweep + Synthesis | 4.8 | 0.63 |
| **TOTAL** | | **9.8** | **1.28** |

*CO2 estimated via: A100 @ 400W TDP, Netherlands grid 0.328 kg CO2/kWh*

---

## Timeline

| Day | Date | Phase | Tasks | Gate |
|-----|------|-------|-------|------|
| 1 | Jan 8 | Setup | Environment, verify original code | |
| 2-3 | Jan 9-10 | P1 | Train dense baselines (P1.1-P1.4) | |
| 4-5 | Jan 11-12 | P1 | Eigenspectrum analysis, figures | |
| 6-7 | Jan 13-14 | P1 | Complete reproduction, ablation | **P1 GATE** |
| 8-9 | Jan 15-16 | E1 | Robustness testing (Rot/EMNIST) | |
| 9-11 | Jan 16-18 | E2 | Implement CPBilinear | **XOR GATE** |
| 12-14 | Jan 19-21 | E3 | CP rank sweep on MNIST | |
| 14-16 | Jan 21-23 | E4 | Comparison analysis, figures | |
| 17 | **Jan 24** | - | **RESULT FREEZE** | **HARD STOP** |
| 18-21 | Jan 25-28 | Report | Draft writing | |
| 22 | Jan 29 | Report | Final polish | |
| 23 | **Jan 30** | - | **SUBMISSION** | |

---

## Deliverables

### Code
```
src/
├── models/bilinear_layer.py   # Unified Dense + CP implementation
├── models/image_model.py      # MNIST classifier
├── analysis/spectral.py       # Effective rank, eigenvalue analysis
├── train.py                   # Main training script
└── evaluate.py                # Evaluation and figure generation
```

### Notebooks
```
notebooks/
├── 01_reproduction.ipynb      # Phase 1 results
├── 02_extensions.ipynb        # Phase 2 results
└── 03_all_results.ipynb       # Final combined notebook (required)
```

### Report Structure (10 pages max, TMLR template)
1. **Introduction** (1p) - Transparency motivation, hypothesis statement
2. **Background** (1p) - Bilinear layers, eigendecomposition, CP-decomposition
3. **Reproduction** (2p) - Phase 1 methodology and results
4. **Extensions** (3p) - Structure vs. Regularization experiments
5. **Discussion** (2p) - Findings, limitations, ViT formulation
6. **Conclusion** (0.5p)
7. **Environmental Impact** (0.5p) - GPU hours, CO2 estimate

### Presentation (10 minutes)
- Slides 1-2: Problem and hypothesis
- Slides 3-4: Reproduction results (eigenspectrum plots)
- Slides 5-7: Extension results (CP vs. Dense comparison)
- Slides 8-9: Conclusions and ViT outlook
- Slide 10: Environmental impact summary

---

## Risk Mitigation

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Phase 1 fails to reproduce | Extra debugging time in buffer | Contact TA |
| CP implementation broken | XOR sanity check catches early | Focus on regularization ablation |
| CP doesn't improve interpretability | Document negative result | Emphasize spectral analysis framework |
| Snellius queue delays | Submit jobs early, use preemptible | Local GPU backup |
| Time shortage | Result freeze gives 6 days buffer | Submit with Phase 1 + partial Phase 2 |

---

## Success Criteria

### Minimum (Pass)
- [ ] Phase 1 reproduction complete
- [ ] Eigenspectrum plots match paper trend
- [ ] One extension experiment (E1: Robustness)
- [ ] Report submitted on time

### Target (Good Grade)
- [ ] All Phase 1 + Phase 2 experiments complete
- [ ] Clear Structure vs. Regularization comparison
- [ ] Novel insight documented
- [ ] Environmental impact reported

### Stretch (Excellent)
- [ ] CP mode matches Dense(reg) interpretability without noise
- [ ] Quantitative framework for interpretability (effective rank correlation)
- [ ] ViT formulation section in report

---

## References

- Pearce et al. "Bilinear MLPs enable weight-based mechanistic interpretability" [arXiv:2410.08417](https://arxiv.org/pdf/2410.08417)
- Original code: [github.com/tdooms/bilinear-decomposition](https://github.com/tdooms/bilinear-decomposition)
- CP Decomposition: Kolda & Bader, "Tensor Decompositions and Applications" SIAM Review 2009
- Effective Rank: Roy & Bhattacharyya, "Effective Rank: Definition, Computation, and Applications" 2007
