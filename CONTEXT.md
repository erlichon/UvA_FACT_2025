# Project Context: Bilinear MLP Interpretability Reproduction

> **Purpose**: This document captures the complete context for the FACT-AI project. It serves as the single source of truth for all project knowledge accumulated during planning.

---

## 0. Current State (Updated: 2026-01-14)

### Section 4 (Vision) COMPLETE + ANALYZED + EXTENDED, Section 5 (Language) ~90% Complete

**Status Summary**:
| Person | Role | Status | Next Action |
|--------|------|--------|-------------|
| A | Infrastructure Lead (Section 4) | **COMPLETE** | All 40 vision experiments finished |
| B | Analysis Lead (Section 4) | **COMPLETE** | All figures (1-7, 9) generated |
| C | Robustness Testing (E1) | Ready | Phase 1 checkpoints available |
| D | CP Rank Sweep (E3) | Waiting | Needs CP implementation |
| E | Synthesis + ViT (E4+E5) | Waiting | Needs E1 + E3 results |
| F | CP Implementation (E2) | Ready | Can start (BilinearCP skeleton exists) |
| G | Language Infrastructure (Section 5) | **COMPLETE** | Run sweep to generate Figure 9 |

**Vision Experiments COMPLETE** (40 base + 36 extended = 76 runs finished):
- **MNIST Base**: 20 runs (4 configs x 5 seeds) - COMPLETE
- **Fashion-MNIST**: 20 runs (4 configs x 5 seeds) - COMPLETE
- **Model Size Sweep (Fig 5)**: 30 runs (6 sizes x 5 seeds) - COMPLETE
- **Noise σ=0.15 (Fig 7)**: 5 runs (5 seeds) - COMPLETE
- **Challenge Task (Fig 6)**: 1 run - COMPLETE
- **All Figures**: Figures 1-7, 9 generated and in `Report/figures/`
- **Tracking**: https://wandb.ai/itayerlich96-student/fact-bilinear

### Section 4 Results Summary (CORRECTED - 2026-01-12)

**Formula Fix**: Previously used entropy-based effective rank. Now using paper's ratio-based formula `(L1/L2)^2`.

**Paper Parameters (Table 1 in Appendix)**: `noise_std=0.5, weight_decay=1.0` (corresponds to our `mnist_dense_full` config)

| Config | Val Accuracy | Effective Rank | Notes |
|--------|-------------|----------------|-------|
| mnist_none | 97.49% +/- 0.05% | **38.50** +/- 0.66 | No regularization baseline |
| mnist_noise | 98.41% +/- 0.04% | 74.75 +/- 0.55 | Noise increases rank! |
| mnist_wd | 97.50% +/- 0.04% | **21.36** +/- 0.26 | WD most effective at reducing rank |
| mnist_full | 98.30% +/- 0.04% | 36.30 +/- 0.33 | **PAPER PARAMS** (σ=0.5, λ=1.0) |
| fashion_none | 88.50% +/- 0.07% | 47.20 +/- 0.83 | No regularization baseline |
| fashion_noise | 87.62% +/- 0.04% | 80.53 +/- 0.57 | Noise increases rank |
| fashion_wd | 87.74% +/- 0.14% | **19.61** +/- 0.75 | WD most effective |
| fashion_full | 87.18% +/- 0.09% | 33.16 +/- 0.26 | **PAPER PARAMS** (σ=0.5, λ=1.0) |

**Key Findings** (after formula correction):
1. **Effective rank ratio (wd/no-reg) = 0.55** (MNIST), close to paper's < 0.5 expectation
2. **Weight decay alone** achieves lowest effective rank (21.36 MNIST, 19.61 Fashion)
3. **Noise augmentation alone** INCREASES effective rank (expected - noise prevents overfitting but spreads eigenspectrum)
4. **Accuracy is higher than paper** (~98% vs expected 94-95%) due to 100 epochs vs paper's 20

### Section 5 Status

**IMPORTANT CLARIFICATION** (Verified 2026-01-14):

After verifying model specifications and SAE availability, we discovered:
- ✅ **`tdooms/ts-medium` IS the paper's "ts-tiny"** (6L, 512d, 29.4M params, TinyStories)
- ⚠️ **SAE LIMITATION**: ts-medium layer 4 does NOT have `mlp-in` SAEs publicly available
  - Only has: `mlp-out`, `resid-mid`, `resid-pre` (no `mlp-in`)
  - Figure 8 interaction analysis requires BOTH `mlp-in` and `mlp-out` (Tracer class requirement)
  - Paper's internal SAE checkpoints were not released to HuggingFace
- **Solution**: Reproduce Figure 8 using fw-medium only (has both SAEs available)
- **Figure 9**: All models work (correlation analysis only needs output SAEs)

#### Section 5.1: Negation Circuit Discovery (Figure 8)

**Paper's Intended Model (ts-medium)** - NOT REPRODUCIBLE:
- **Model**: `tdooms/ts-medium` (6 layers, 29.4M params, TinyStories)
- **SAE Layer**: 4 (middle of 6-layer model)
- **Expansion**: 4
- **Features**: 1882 (not-good), 1179 (not-bad)
- **Issue**: ❌ Layer 4 missing `mlp-in` SAEs on HuggingFace
- **Status**: SKIPPED (cannot reproduce without both SAEs)

**Our Reproduction (fw-medium)** - TUTORIAL EXAMPLE:
- **Model**: `tdooms/fw-medium` (16 layers, 335M params, FineWeb-EDU)
- **SAE Layer**: 7 (middle of 16-layer model)
- **Expansion**: 8
- **Features**: 3834 (not-good), 751 (not-bad)
- **Advantage**: ✅ Has both `mlp-in` and `mlp-out` SAEs available
- **Advantage**: ✅ Larger model → clearer visualizations
- **Config**: `configs/language_negation_fw.yaml`
- **Status**: READY to run

#### Section 5.2: Low-Rank Correlation (Figure 9)
- **Models**: ts-medium (layer 4), fw-small (layer 8), fw-medium (layer 7)
- **Expansion**: 4 for ts-medium/fw-small, 8 for fw-medium
- **Claim**: 69% of features have >0.75 rank-2 correlation
- **Config**: `configs/language_correlation_fw.yaml`

| Experiment | Status | Config | Notes |
|------------|--------|--------|-------|
| SAE Training | SKIPPED | - | Using pretrained SAEs from HuggingFace |
| Negation (ts-medium, L4) | SKIPPED | `language_negation_ts.yaml` | ❌ Missing mlp-in SAEs on HuggingFace |
| Negation (fw-medium, L7) | READY | `language_negation_fw.yaml` | ✅ **TUTORIAL** (features 3834/751) - Figure 8 |
| Figure 8 Generation | READY | - | Generate for fw-medium only |
| Correlation Sweep (all 3) | READY | `scripts/run_language_sweep.sh` | Full Figure 9 (all models work) |

**Model Configuration & SAE Availability (VERIFIED 2026-01-14)**:
| Model | Layers | d_model | Params | SAE Layer | Expansion | mlp-in | mlp-out | Use |
|-------|--------|---------|--------|-----------|-----------|--------|---------|-----|
| ts-medium | 6 | 512 | 29.4M | 4 | 4 | ❌ | ✅ | Figure 9 only |
| fw-small | 12 | - | 162M | 8 | 4 | ❌ | ✅ | Figure 9 |
| fw-medium | 16 | 1024 | 335M | 7 | 8 | ✅ | ✅ | **Figure 8 + Figure 9** |

**Notes**:
- ✅ ts-medium verified as paper's ts-tiny (same specs: 6L, 512d, ~29M params)
- ⚠️ **SAE Limitation**: ts-medium layer 4 lacks `mlp-in` SAEs (only `mlp-out`, `resid-mid`, `resid-pre`)
- Figure 8 (interaction analysis) requires BOTH `mlp-in` and `mlp-out` SAEs
- Figure 9 (correlation analysis) only requires `mlp-out` SAEs → all models work
- fw-medium chosen for Figure 8: clearer visualizations + all SAEs available
- All pretrained SAEs use k=30 (paper uses k=32, minor difference)

**Commands**:
- Run vision experiments: `./scripts/train/run_vision.sh train all`
- Run language experiments: `./scripts/train/run_language.sh all`
- Run correlation sweep (all 3 models): `./scripts/train/run_language.sh figure9`
- Generate all figures: `./scripts/train/run_vision.sh figures && ./scripts/train/run_language.sh figures`
- Run overnight (all experiments): `./scripts/train/run_overnight_mps.sh`
- Run on GPU: `sbatch jobs/language_fwmedium.job`

**Section 5 (Language) Components** (IMPLEMENTED):
- `src/language/context.py` - LanguageContext class for unified model/SAE loading (DRY)
- `src/language/run_sae_training.py` - SAE training wrapper (uses original paper code)
- `src/language/negation_discovery.py` - Negation circuit discovery
- `src/language/interaction_analysis.py` - Interaction matrix analysis
- `src/language/verify_correlation.py` - Correlation verification (CLI: `--model`, `--layer`, `--expansion`, `--k`)
- `src/plot_utils/language.py` - Centralized plotting for Figure 9 & 10 (progression, histogram, scatters)
- `scripts/train/run_language.sh` - Unified language experiment runner
- `scripts/figures/generate_language_figures.py` - Generate all language figures from results
- `configs/language_sae.yaml` - SAE training config
- `configs/language_negation.yaml` - Negation discovery config
- `configs/language_interaction.yaml` - Interaction analysis config
- `jobs/language_full_pipeline.job` - SLURM script for Snellius

**Directory Structure** (COMPLETE):
```
UvA_FACT_2025/
├── bilinear-decomposition-main/  # Original code (DO NOT MODIFY)
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Shared utilities (device, wandb, emissions) [COMPLETE]
│   ├── models/
│   │   ├── __init__.py
│   │   └── bilinear_layer.py     # BilinearDense + BilinearCP [COMPLETE]
│   ├── data/
│   │   ├── __init__.py
│   │   └── challenge_dataset.py  # Challenge dataset (Figure 6)
│   ├── vision/                   # Section 4 vision analysis (renamed from analysis/)
│   │   ├── __init__.py
│   │   ├── context.py            # VisionContext - unified experiment setup (DRY)
│   │   ├── spectral.py           # effective_rank, top_k_coverage, load_all_checkpoints
│   │   ├── truncation.py         # Truncation accuracy, similarity (Figure 5)
│   │   └── adversarial.py        # Adversarial mask generation (Figure 7)
│   ├── plot_utils/               # Reusable plotting functions
│   │   ├── __init__.py
│   │   ├── style.py              # Publication style, colors, constants
│   │   ├── eigenspectrum.py      # Eigenspectrum visualization
│   │   ├── eigenvectors.py       # Eigenvector visualization (with λ labels)
│   │   ├── ablation.py           # Ablation and trade-off plots
│   │   └── language.py           # Language Figure 9 & 10 plots
│   ├── language/                 # Section 5 language experiments
│   │   ├── __init__.py
│   │   ├── context.py            # LanguageContext - unified experiment setup (DRY)
│   │   ├── run_sae_training.py   # SAE training wrapper
│   │   ├── negation_discovery.py # Negation circuit analysis
│   │   ├── interaction_analysis.py # Interaction matrix analysis
│   │   └── verify_correlation.py # Correlation verification
│   └── train.py                  # Vision training script
├── configs/
│   ├── mnist_dense_{none,noise,wd,full,noise015}.yaml  # MNIST vision configs
│   ├── fashion_dense_{none,noise,wd,full}.yaml  # Fashion-MNIST configs
│   ├── mnist_challenge.yaml      # Challenge task config
│   ├── sweeps/
│   │   └── mnist_size_{30,50,100,300,500,1000}.yaml  # Model size sweep
│   ├── language_sae.yaml         # SAE training config (supports model/layer overrides)
│   ├── language_negation_{fw,ts}.yaml  # Negation discovery configs
│   └── language_interaction.yaml # Interaction analysis config
├── scripts/
│   ├── train/                    # Training & experiment runners
│   │   ├── run_vision.sh         # Unified vision experiments
│   │   ├── run_language.sh       # Unified language experiments
│   │   └── run_overnight_mps.sh  # Full overnight pipeline
│   └── figures/                  # Figure generation
│       ├── generate_vision_figures.py   # All vision figures (1-7)
│       ├── generate_language_figures.py # Figure 9 & 10
│       └── paper_hub.py          # Interactive figure viewer
├── tools/                        # Operational utilities
│   ├── sync_to_snellius.sh       # Upload code to cluster
│   ├── sync_from_snellius.sh     # Download results
│   └── monitor_memory.sh         # Memory monitoring
├── jobs/
│   ├── train_array.job           # MNIST 20 runs (Snellius)
│   ├── train_fashion_array.job   # Fashion-MNIST 20 runs
│   └── language_full_pipeline.job # Full Section 5 pipeline
├── docs/
│   └── WANDB_GUIDE.md            # wandb usage guide
├── tests/                        # Unit tests (82 tests)
├── results/
│   ├── phase1/
│   │   ├── checkpoints/          # MNIST vision checkpoints (26 files: 20 base + 5 noise015 + 1 20ep)
│   │   └── figures/              # Generated figures (16 PDFs + 2 CSVs)
│   ├── phase1_fashion/checkpoints/ # Fashion-MNIST checkpoints (20 files)
│   ├── sweeps/
│   │   ├── model_size/checkpoints/ # Model size sweep (30 files) [NEW]
│   │   └── noise_sweep/checkpoints/ # Noise sweep (6 files)
│   ├── challenge/checkpoints/    # Challenge task checkpoint [NEW]
│   ├── adversarial/              # Adversarial experiment results [NEW]
│   └── language/                 # Language results (JSON + figures)
│       ├── correlation_{ts-medium,fw-small,fw-medium}.json  # Sweep results
│       └── figures/              # Language figures (Figure 9A, 9B, etc.)
├── logs/                         # Experiment logs
├── notebooks/
│   └── 01_reproduction.ipynb     # Vision analysis notebook
├── Report/
│   ├── figures/                  # Publication figures (Figures 1-7, 9)
│   └── sections/                 # LaTeX sections
├── environment.yml               # Snellius GPU environment
└── environment_cpu.yml           # Local CPU/MPS environment
```

**What Has Been Delivered**:

*Section 4 (Vision) - COMPLETE + EXTENDED*:
1. `src/models/bilinear_layer.py` - BilinearDense (wraps original) + BilinearCP (extension)
2. `src/train.py` - Vision training with wandb + codecarbon tracking
3. `src/vision/context.py` - VisionContext class for unified experiment setup (DRY)
4. `src/vision/spectral.py` - effective_rank, top_k_coverage, spectral_summary, load_all_checkpoints
5. `src/vision/truncation.py` - Truncation accuracy, eigenvector similarity (Figure 5)
6. `src/vision/adversarial.py` - Adversarial mask generation (Figure 7)
7. `src/data/challenge_dataset.py` - Challenge dataset (Figure 6)
8. `src/plot_utils/` - Reusable plotting module (style, eigenspectrum, eigenvectors with λ labels, ablation)
9. `src/utils.py` - Shared utilities (device detection, wandb init, MPS fallbacks)
10. 9 vision config files (4 MNIST + 4 Fashion-MNIST + 1 noise015)
11. 6 model size sweep configs (30, 50, 100, 300, 500, 1000)
12. SLURM job scripts for Snellius
13. `scripts/train/run_vision.sh` - Unified vision experiment runner (train, figures, all)
14. `scripts/figures/generate_vision_figures.py` - All vision figures (1-7)
15. `notebooks/01_reproduction.ipynb` - Complete analysis notebook
16. **16 publication figures** in `Report/figures/` (Figures 1-7, 9, 10 complete)

*Section 5 (Language)*:
17. `src/language/context.py` - LanguageContext class for unified model/SAE loading (DRY)
18. `src/language/run_sae_training.py` - SAE training wrapper
19. `src/language/negation_discovery.py` - Negation circuit discovery
20. `src/language/interaction_analysis.py` - Interaction matrix analysis
21. `src/language/verify_correlation.py` - Correlation verification with CLI overrides
22. `src/plot_utils/language.py` - Centralized plotting for Figure 9 & 10 (DRY architecture)
23. `scripts/train/run_language.sh` - Unified language experiment runner (figure9, figure8, negation, interaction, figures, all)
24. `scripts/figures/generate_language_figures.py` - Generate all language figures
25. 3 language config files (supports model/layer/expansion overrides)

*Unified Infrastructure*:
26. `scripts/train/run_overnight_mps.sh` - Complete overnight pipeline (vision + language)
27. `scripts/figures/paper_hub.py` - Interactive figure viewer
28. `tools/` - Operational utilities (sync_to_snellius.sh, sync_from_snellius.sh, monitor_memory.sh)

*Testing & Tracking*:
29. 82 unit tests in `tests/`
30. wandb integration with single project: `itayerlich96-student/fact-bilinear`
31. CO2 tracking via codecarbon for all experiments
32. `docs/WANDB_GUIDE.md` - Comprehensive wandb usage guide

**Critical Agreements Standardized**:
1. **CP Factor Names**: A=[d_in,rank], B=[d_in,rank], C=[d_out,rank]
2. **Checkpoint Format**: Flat config dict (see Section 8)
3. **History Columns**: Handles both `train_acc` and `train/acc`
4. **Eigenvalues/Eigenvectors**: Saved directly in checkpoint
5. **Budget**: 25,000 SBUs total, Claude outputs job files only
6. **wandb Project**: Single project for all experiments with tags for filtering

**Next Actions**:
1. ~~Person B: Create visualization code and notebook~~ **DONE**
2. ~~Generate figures for report from checkpoints~~ **DONE** (16 figures in Report/figures/)
3. ~~Language correlation sweep infrastructure~~ **DONE** (scripts/train/run_language.sh + src/plot_utils/language.py)
4. ~~Figure 5: Model size sweep experiments~~ **DONE** (30 models trained, figures generated)
5. ~~Figure 6: Challenge task~~ **DONE** (similarity classification implemented)
6. ~~Figure 7: Adversarial masks~~ **DONE** (no-reg vs noise-reg with error bars)
7. **Run language correlation sweep**: `./scripts/train/run_language.sh figure9`
8. **Generate language figures (9 & 10)**: `./scripts/train/run_language.sh figures`
9. Run fw-medium interaction + negation experiments (Figure 8): `./scripts/train/run_language.sh figure8`
10. Update Report LaTeX with all figures and tables
11. Start Phase 2 extensions (CP implementation, robustness testing)
12. Review wandb results at https://wandb.ai/itayerlich96-student/fact-bilinear

---

## 1. Project Overview

### Course Information
- **Course**: FACT-AI (Fairness, Accountability, Confidentiality and Transparency in AI)
- **Institution**: UvA MSc AI, January 2026
- **Deadline**: 30 January 2026, 23:59
- **Draft Feedback**: 22 January 2026
- **Result Freeze**: 17 January 2026 (no new experiments after this)

### Paper Being Reproduced
- **Title**: "Bilinear MLPs enable weight-based mechanistic interpretability"
- **Authors**: Pearce, Oikarinen, Weng
- **arXiv**: [2410.08417](https://arxiv.org/pdf/2410.08417)
- **Original Code**: [github.com/tdooms/bilinear-decomposition](https://github.com/tdooms/bilinear-decomposition)

### Paper Scope (FULL REPRODUCTION)
We reproduce **both main experimental sections**:
- **Section 4 (Vision)**: MNIST/Fashion-MNIST eigendecomposition demonstrating low-rank interpretability
- **Section 5 (Language)**: Negation circuit discovery in bilinear transformers via Sparse Autoencoder analysis

### Core Hypothesis
> **Structural low-rank** (CP-Decomposition) produces equivalent or better interpretability than **emergent low-rank** (via regularization), with the advantage of explicit rank control.

### MLRC Requirements
Since open-source implementation exists, we must go beyond "paper is reproducible":
1. Extend experiments to new domains/datasets (Rotated MNIST, EMNIST)
2. Propose alternative approach (CP-Decomposition as structural low-rank)
3. Identify and analyze any discrepancies with paper results

---

## 2. Two-Phase Project Structure

### Phase 1: Reproduction (Days 1-7)

#### Section 4 (Vision): MNIST/Fashion-MNIST Eigendecomposition
**Goal**: Verify Pearce et al. Section 4 findings on image classification

**Experiments** (4 configs x 5 seeds = 20 runs):
| ID | Config | Noise | Weight Decay | Expected Outcome |
|----|--------|-------|--------------|------------------|
| P1.1 | none | 0.0 | 0.0 | High effective rank, overfitting |
| P1.2 | noise_only | 0.4 | 0.0 | Medium effective rank |
| P1.3 | wd_only | 0.0 | 0.5 | Medium effective rank |
| P1.4 | full | 0.4 | 0.5 | Low effective rank, interpretable |

**Gate Check Criteria** (Vision):
- [ ] Effective rank ratio (reg/no-reg) < 0.5
- [ ] Top eigenvectors visually resemble digits
- [ ] Accuracy within 2% of paper values

#### Section 5 (Language): Negation Circuit Discovery
**Goal**: Reproduce negation circuit analysis in bilinear transformers

**Section 5 Claims to Verify**:
| ID | Claim | Verification Method |
|----|-------|---------------------|
| L1 | SAE features reveal interpretable structure | Visual inspection of activations |
| L2 | Negation features form opposing directions | Cosine similarity < 0 |
| L3 | 69% features have >0.75 rank-2 correlation | Eigendecomposition analysis |
| L4 | Top 50 interactions form sparse submatrix | Interaction matrix visualization |
| L5 | Cross-model generalization | Test on TinyStories + FineWeb-EDU |

**Language Experiments**:
| ID | Experiment | Model | Dataset |
|----|------------|-------|---------|
| L5.1 | SAE Training | Bilinear Transformer | TinyStories |
| L5.2 | Negation Feature Discovery | Trained SAEs | Sentiment probes |
| L5.3 | Interaction Matrix Analysis | SAE features | Per-feature Q matrices |
| L5.4 | Low-Rank Verification | All features | Rank-2 correlation distribution |

**Gate Check Criteria** (Language):
- [ ] SAE reconstruction loss converges
- [ ] Negation features activate on expected patterns ("not + positive/negative")
- [ ] >60% features show >0.75 low-rank correlation

### Phase 2: Extensions (Days 8-17)
**Goal**: Test hypothesis that structural low-rank matches emergent low-rank

**Extension 1**: Robustness Testing
- Rotated MNIST (15, 30, 45, 60, 90 degrees)
- EMNIST letters subset

**Extension 2**: CP-Decomposition Implementation
- Implement `Bilinear(mode='cp', rank=R)`
- XOR sanity check (rank-2 problem)

**Extension 3**: CP Rank Sweep
- R = 8, 16, 32, 64, 128, 256 on MNIST
- Compare to dense+regularization

---

## 3. Team Structure

### Person A: Infrastructure Lead
**Responsibilities**:
- Project directory structure
- Training pipeline (`src/train.py`)
- SLURM job scripts
- wandb + codecarbon integration
- Checkpoint management
- Config files

**Key Deliverables**:
- `src/models/bilinear_layer.py` (dense wrapper + CP extension)
- `src/train.py` with full experiment tracking
- `configs/*.yaml` for all experiments
- `jobs/*.job` SLURM scripts

### Person B: Analysis Lead
**Responsibilities**:
- Paper deep-dive (Section 4 theory)
- Spectral analysis utilities
- Visualization code
- Analysis notebooks
- Figure generation

**Key Deliverables**:
- `src/vision/spectral.py` (effective_rank, eigenspectrum extraction)
- `src/vision/visualization.py` (publication-quality plots)
- `notebooks/01_reproduction.ipynb`
- All Phase 1 figures (eigenspectrum, eigenvectors)

### Person G: Language Infrastructure Lead (Section 5)
**Responsibilities**:
- TinyStories data loading and preprocessing
- Bilinear transformer model (or use pretrained)
- Sparse Autoencoder (SAE) training pipeline
- Negation circuit discovery code
- Interaction matrix analysis

**Key Deliverables**:
- `src/language/` module (data, model, sae, analysis)
- `configs/language_*.yaml` for all Section 5 experiments
- `jobs/train_sae.job`, `jobs/analyze_negation.job` SLURM scripts
- Checkpoint format for SAE features

### Coordination Points (Phase 1)
- **Checkpoint Format**: Agreed upon for Person A to save, Person B to load
- **Day 2 Sync**: Both have working code, test on same 2 runs
- **Day 4 Sync**: All 20 runs complete, analysis started
- **Day 7**: Joint gate check review

---

## 3b. Phase 2 Team Structure

### Person C: Robustness Testing Lead
**Extension**: E1 (Robustness Testing)
**Timeline**: Days 8-10 (Jan 15-17)
**Dependencies**: Phase 1 checkpoints from Person A

**Responsibilities**:
- Implement Rotated MNIST and EMNIST data loaders
- Evaluate Phase 1 regularized model on distribution shifts
- Generate robustness analysis figures

**Key Deliverables**:
- `src/data/rotated_mnist.py` - Rotated MNIST loader
- `src/data/emnist.py` - EMNIST letters loader
- `results/phase2/robustness/rotated_mnist_results.csv`
- `results/phase2/robustness/emnist_analysis.json`
- `notebooks/02a_robustness.ipynb` - E1 analysis
- Figures: `robustness_rotated.pdf`, `robustness_emnist.pdf`

### Person F: CP Implementation Lead
**Extension**: E2 (CP Implementation)
**Timeline**: Days 9-11 (Jan 16-18)
**Dependencies**: Person A's bilinear_layer.py skeleton

**Responsibilities**:
- Complete `BilinearCP` class with CP-decomposition
- Create `CPImageModel` for MNIST classification
- Validate CP implementation with XOR sanity check

**Key Deliverables**:
- `src/models/bilinear_layer.py` - Complete BilinearCP class
- `src/models/cp_model.py` - CP-compatible image model
- `src/test_xor_sanity.py` - XOR gate validation

**Gate Check (E2 - XOR Sanity)**:
- [ ] CP mode R=1 fails on XOR (accuracy ~50%)
- [ ] CP mode R>=2 succeeds on XOR (accuracy >95%)

### Person D: CP Rank Sweep Lead
**Extension**: E3 (CP Rank Sweep on MNIST)
**Timeline**: Days 12-14 (Jan 19-21)
**Dependencies**: CP implementation from Person F

**Responsibilities**:
- Create training configs for CP rank sweep (R=8,16,32,64,128,256)
- Run 30 experiments (6 ranks x 5 seeds)
- Extract eigenspectrum and metrics for all CP models
- Compare interpretability: CP (no reg) vs Dense (full reg)

**Key Deliverables**:
- `configs/mnist_cp_r*.yaml` - CP rank sweep configs
- `jobs/train_cp_array.job` - SLURM array job for sweep
- `results/phase2/checkpoints/` - CP model checkpoints
- `notebooks/02b_cp_sweep.ipynb` - E3 analysis
- Preliminary figure: `cp_rank_sweep.pdf`

**Key Metrics to Track**:
| Rank | Accuracy | Effective Rank | Top-5 Coverage |
|------|----------|----------------|----------------|
| 8 | ? | ? | ? |
| 16 | ? | ? | ? |
| ... | ... | ... | ... |

### Person E: Synthesis + ViT Formulation Lead
**Extensions**: E4 (Structure vs Reg Comparison) + E5 (ViT Formulation)
**Timeline**: Days 14-17 (Jan 21-24)
**Dependencies**: Results from Person C (E1) AND Person D (E3)

**Responsibilities**:
- Aggregate all results from Phase 1 and Phase 2
- Generate final comparison figures (Pareto frontier, eigenspectrum overlay)
- Perform statistical significance tests
- Write ViT formulation section (conceptual, no code)
- Finalize `notebooks/02_extensions.ipynb`

**Key Deliverables**:
- `notebooks/02_extensions.ipynb` - Complete Phase 2 notebook
- `notebooks/03_all_results.ipynb` - Final combined notebook (required)
- Figures: `structure_vs_reg.pdf`, `pareto_frontier.pdf`, `eigenvector_comparison.pdf`
- Report section: ViT formulation (Section 5 of report)

**Key Comparisons**:
1. Dense (no reg) vs Dense (full reg) vs CP R=32 (no reg)
2. Accuracy vs Effective Rank Pareto frontier
3. Eigenvector quality side-by-side

### Phase 2 Coordination Timeline

```
Day 8-10 (Jan 15-17): Person C completes E1 (Robustness)
Day 9-11 (Jan 16-18): Person F completes E2 (CP Implementation)
Day 11   (Jan 18):    *** E2 GATE CHECK (XOR) ***
Day 12-14 (Jan 19-21): Person D runs E3 (CP Sweep)
Day 14-16 (Jan 21-23): Person E starts E4 (Synthesis)
Day 16-17 (Jan 23-24): Person E completes E5 (ViT Formulation)
Day 17   (Jan 24):    *** RESULT FREEZE ***
```

### Phase 2 Handoff Protocol

**F → D Handoff (Day 11)**:
- Person F provides working `BilinearCP` class
- Person F verifies XOR sanity check passes
- Person D can import and use without modifications

**C,D → E Handoff (Day 14)**:
- Person C provides robustness analysis (rotated_mnist_results.csv)
- Person D provides CP sweep checkpoints and preliminary figures
- Person E has all data needed for synthesis

---

## 4. Technical Infrastructure

### Snellius HPC Cluster

**Connection**:
```bash
ssh scur0075@Snellius  # Case-sensitive hostname
```

**Directory Structure**:
```
/home/scur0075/fact-project/
├── bilinear-decomposition-main/   # UNTOUCHED original code
├── src/                           # Our implementation
├── configs/                       # Experiment configs
├── jobs/                          # SLURM scripts
├── results/                       # Outputs
├── notebooks/                     # Analysis notebooks
└── Report/                        # LaTeX report
```

**SLURM Configuration**:
```bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate fact
```

### Conda Environment
Based on DL course template (`environment.yml`):
```yaml
name: fact
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.12.7
  - pytorch-cuda=11.8
  - pytorch=2.5.0
  - torchvision=0.20.0
  - pip:
    - einops
    - jaxtyping
    - transformers
    - datasets
    - wandb
    - codecarbon
    - kornia
    - nnsight
```

### Environmental Tracking (MANDATORY)
Every experiment must log:
| Metric | Tool | Destination |
|--------|------|-------------|
| GPU Model | `torch.cuda.get_device_name()` | wandb config |
| Wall-clock Time | `time.time()` | wandb log |
| GPU Hours | wall_time x gpu_count | wandb summary |
| CO2 Emissions | `codecarbon.EmissionsTracker` | wandb summary |

---

## 5. Core Technical Concepts

### Bilinear Layer Mathematics

**Standard Bilinear Layer**:
```
y = (W_l @ x) ⊙ (W_r @ x)
```
Where:
- `W_l, W_r ∈ R^{d_out × d_in}` are learnable weight matrices
- `⊙` is Hadamard (element-wise) product

**Third-Order Interaction Tensor**:
```
y_i = Σ_{j,k} B_{ijk} x_j x_k
```
Where `B_{ijk} = (W_l)_{ij} (W_r)_{ik}`

**Eigendecomposition** (for interpretability):
For each output class c, the symmetric interaction matrix admits:
```
B_sym[c] = V[c] @ diag(λ[c]) @ V[c]^T
```

**Interpretation**:
- Large positive eigenvalues → features that INCREASE class probability
- Large negative eigenvalues → features that DECREASE class probability
- Near-zero eigenvalues → irrelevant directions
- **Low-rank structure** = few large eigenvalues = interpretable

### CP-Decomposition (Our Extension)

**CP Form**:
```
B_{ijk} = Σ_{r=1}^R λ_r A_{ir} B_{jr} C_{kr}
```
Where R is the explicit rank.

**Efficient Forward Pass**:
```python
left = x @ A        # [batch, rank]
right = x @ B       # [batch, rank]
hidden = left * right * lambdas  # [batch, rank]
output = hidden @ C.T  # [batch, d_out]
```

### Effective Rank Metric
Roy & Bhattacharyya (2007) entropy-based measure:
```
p_i = |λ_i| / Σ|λ|
EffRank = exp(-Σ p_i log(p_i))
```
- Lower effective rank = sharper eigenspectrum = more interpretable
- Range: [1, n] where n is dimension

### Sparse Autoencoders (Section 5)

**Purpose**: Discover interpretable features in MLP hidden states

**Architecture**:
```
z = ReLU(W_enc @ h + b_enc)      # Sparse features
h_hat = W_dec @ z + b_dec        # Reconstruction
```

**Training Loss**:
```
L = ||h - h_hat||^2 + λ * ||z||_1    # MSE + L1 sparsity
```

**Key Hyperparameters**:
- Expansion ratio: 8x (if MLP hidden=768, SAE features=6144)
- L1 coefficient: ~1e-3 to 1e-4
- Training tokens: ~10M

### Negation Circuit Analysis (Section 5)

**Negation Features**:
The paper identifies two key SAE output features:
- **Feature 3834**: Activates on "not + negative words" (e.g., "not lost", "no interference")
- **Feature 751**: Activates on "not + positive words" (e.g., "not free", "little relief")

These form opposing directions in feature space (negative cosine similarity).

**Interaction Matrix**:
For each SAE output feature o, compute:
```
Q[i,j]^(o) = Σ_samples z_i^in * z_j^in * z_o^out
```

Eigendecomposition of Q reveals which input feature pairs drive output feature o.

**Low-Rank Verification**:
- Compute rank-2 approximation: Q_hat = λ1*v1*v1^T + λ2*v2*v2^T
- Correlation between Q and Q_hat should be >0.75 for 69% of features

---

## 6. Original Code Structure

**Location**: `bilinear-decomposition-main/`

**Key Files**:
```
shared/components.py      # Bilinear layer class
image/model.py            # Image classifier with .decompose()
image/datasets.py         # GPU-resident MNIST loader
image/plotting.py         # Plotly visualization (reference only)
tutorials/1_image.ipynb   # Example usage
```

**Original Bilinear Layer** (`shared/components.py`):
```python
class Bilinear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None):
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {None: nn.Identity(), "relu": nn.ReLU(), ...}[gate]

    def forward(self, x):
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right

    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]
```

**Original Training Pattern**:
```python
from image.model import Model, Config
from image.datasets import MNIST
import kornia

model = Model(Config(epochs=100, d_hidden=256, wd=0.5))
train, test = MNIST(train=True), MNIST(train=False)
transform = kornia.augmentation.RandomGaussianNoise(mean=0, std=0.4, p=1.0)
history = model.fit(train, test, transform=transform)
vals, vecs = model.decompose()  # Eigendecomposition
```

**CRITICAL**: For Phase 1 reproduction, WRAP the original code (don't reimplement) to ensure exact reproduction fidelity.

---

## 7. Paper Baseline Values (Section 4)

### Experimental Setup (Paper)
- Dataset: MNIST
- Architecture: Single bilinear layer, d_hidden=256
- Training: 100 epochs, AdamW, lr=1e-3
- Regularization: Gaussian noise (std=0.4), weight decay (0.5)

### Expected Results
| Configuration | Test Accuracy | Effective Rank |
|---------------|---------------|----------------|
| No regularization | ~97-98% | ~150-200 (high) |
| With regularization | ~94-95% | ~20-40 (low) |

### Key Paper Claims
1. "Regularization induces low-rank structure" (Section 4.2)
2. "~10 eigenvalues per class capture digit structure" (Section 4.3)
3. "Eigenvectors resemble digit templates" (Figure 4)
4. "Noise augmentation is crucial" (Section 4.4)

---

## 8. Checkpoint Format Agreement

Person A saves checkpoints that Person B can load:
```python
checkpoint = {
    'config': {
        'mode': 'dense',
        'd_hidden': 256,
        'epochs': 100,
        'lr': 0.001,
        'noise_std': 0.4,
        'weight_decay': 0.5,
    },
    'model_state_dict': model.state_dict(),
    'metrics': {
        'train_acc': float,      # Final training accuracy
        'val_acc': float,        # Final validation accuracy
        'train_loss': float,     # Final training loss
        'val_loss': float,       # Final validation loss
        'effective_rank': float, # Mean effective rank across classes
    },
    'seed': int,
    'eigenvalues': Tensor,       # Shape: [10, 256] (n_classes, d_hidden)
    'eigenvectors': Tensor,      # Shape: [10, 256, 784] (n_classes, d_hidden, d_input)
}
# Saved to: results/phase1/checkpoints/{config_name}_seed{seed}.pt
```

**Person B Loading Example**:
```python
import torch
checkpoint = torch.load("results/phase1/checkpoints/mnist_dense_full_seed42.pt", map_location='cpu')
eigenvalues = checkpoint['eigenvalues']   # [10, 256]
eigenvectors = checkpoint['eigenvectors'] # [10, 256, 784]
config = checkpoint['config']
metrics = checkpoint['metrics']
```
```

---

## 9. Report Structure

**Location**: `Report/`

**Modular LaTeX**:
```
Report/
├── main.tex                    # Master document
├── sections/
│   ├── 01_introduction.tex     # Transparency motivation
│   ├── 02_scope.tex            # Claims to verify
│   ├── 03_method.tex           # BilinearLayer + CP math
│   ├── 04_reproduction.tex     # Phase 1 results
│   ├── 05_extension.tex        # Phase 2 results
│   └── 06_fact_discussion.tex  # Discussion, CO2, easy/difficult
├── figures/                    # Generated plots (PDF)
├── main.bib                    # References
└── REPORT_INSTRUCTIONS.md      # Continuous reporting guidelines
```

**Required Figures**:
- Phase 1: eigenspectrum_comparison.pdf, eigenvectors_noreg.pdf, eigenvectors_reg.pdf, accuracy_vs_effrank.pdf
- Phase 2: robustness_rotated.pdf, robustness_emnist.pdf, cp_rank_sweep.pdf, structure_vs_reg.pdf

**Key References** (in main.bib):
- pearce2024bilinear - The main paper
- kolda2009tensor - CP decomposition background
- roy2007effective - Effective rank definition

---

## 10. Experiment Budget

### Phase 1: Reproduction
| ID | Seeds | GPU Hrs | CO2 (kg) |
|----|-------|---------|----------|
| P1.1-P1.4 | 20 total | 3.2 | 0.42 |

### Phase 2: Extensions
| ID | Seeds | GPU Hrs | CO2 (kg) |
|----|-------|---------|----------|
| E1-E3 | 45 total | 6.6 | 0.86 |

**Total Budget**: ~9.8 GPU hours, ~1.28 kg CO2

*Estimated via: A100 @ 400W TDP, Netherlands grid 0.328 kg CO2/kWh*

---

## 11. Risk Mitigation

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Phase 1 fails to reproduce | Extra debugging time in buffer | Contact TA |
| CP implementation broken | XOR sanity check catches early | Focus on regularization ablation |
| CP doesn't improve interpretability | Document negative result | Emphasize spectral analysis |
| Snellius queue delays | Submit jobs early | Local GPU backup |
| Time shortage | Result freeze gives 6 days buffer | Submit with Phase 1 + partial Phase 2 |

---

## 12. Success Criteria

### Minimum (Pass)
- [ ] Section 4 (Vision) reproduction complete
- [ ] Eigenspectrum plots match paper trend
- [ ] One extension experiment (E1: Robustness)
- [ ] Report submitted on time

### Target (Good Grade)
- [ ] All Section 4 + Section 5 experiments complete
- [ ] Section 5: SAE training and negation feature discovery
- [ ] Section 5: Interaction matrix analysis with low-rank verification
- [ ] Clear Structure vs. Regularization comparison
- [ ] Environmental impact reported

### Stretch (Excellent)
- [ ] CP mode matches Dense(reg) interpretability without noise
- [ ] Section 5: Cross-model generalization (TinyStories + FineWeb-EDU)
- [ ] Quantitative framework for interpretability
- [ ] Novel extension beyond paper scope
