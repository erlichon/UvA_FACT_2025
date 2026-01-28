# FACT-AI: Bilinear MLP Interpretability (UvA MSc AI)

This repository reproduces and extends ["Bilinear MLPs enable weight-based mechanistic interpretability"](https://arxiv.org/pdf/2410.08417) (Pearce et al., ICLR 2025). It targets:
- **Section 4 (Vision)**: MNIST/Fashion-MNIST bilinear MLP eigendecomposition
- **Section 5 (Language)**: Negation circuit discovery via SAE analysis
- **Extensions**: CP decomposition and cross-dataset robustness

The goal of this README is to make the **codebase reproducible without extra effort**, and to document the **report** and **presentation** build steps.

## Quick Start (Reproducibility Smoke Tests)

These commands verify the environment and run minimal tests.

```bash
# Vision (2 epochs, MPS-friendly)
./scripts/train/run_vision.sh test

# Language (quick tests)
./scripts/train/run_language.sh test

# Extension 2 (2 epochs, MNIST only)
./scripts/train/run_extension2.sh test
```

## Environment Setup

```bash
# Snellius (GPU)
conda env create -f environment.yml && conda activate fact

# Local (CPU/MPS)
conda env create -f environment_cpu.yml && conda activate fact_cpu
```

## Full Reproduction Paths

### Vision (Section 4)

```bash
# Base configs (4 configs x 5 seeds x MNIST+Fashion)
./scripts/train/run_vision.sh train base

# Noise sweep (Figure 4)
./scripts/train/run_vision.sh train noise

# Model size sweep (Figure 5)
./scripts/train/run_vision.sh train size

# Generate all vision figures
./scripts/train/run_vision.sh figures
```

### Language (Section 5)

```bash
# Figure 9 correlation sweep (all 3 models)
./scripts/train/run_language.sh figure9

# Figure 8 negation circuit visualization
./scripts/train/run_language.sh figure8 --device mps

# Negation discovery
./scripts/train/run_language.sh negation

# Interaction analysis
./scripts/train/run_language.sh interaction

# Generate all language figures
./scripts/train/run_language.sh figures
```

### Extension 2: Cross-Dataset Robustness

```bash
# Train all extension2 models (MNIST + EMNIST, CoM enabled)
./scripts/train/run_extension2.sh train all

# Generate all extension2 figures
./scripts/train/run_extension2.sh figures
```

## Outputs and Expected Artifacts

- **Checkpoints**: `checkpoints/` (vision, extension2, CP)
- **Results JSONs**: `results/`
- **Figures for report**: `Report/figures/`
- **Interactive assets**: `Report/paper_hub_bundle/`, `results/interactive/`

## Pre-trained Checkpoints and Results

Due to file size constraints, trained model checkpoints and full result files are hosted on Google Drive:

**[Download Checkpoints and Results](https://drive.google.com/drive/folders/1et6EfHxvyEZKCZ1yXfA1EOHbcShCNWzt)**

Artifacts are **downloaded automatically** when running figure generation scripts via `src/artifact_loader.py`. To manually trigger download:

```python
from src.artifact_loader import ensure_artifacts
ensure_artifacts()  # Downloads checkpoints.zip and results.zip if not present
```

Or via CLI:
```bash
python -m src.artifact_loader
```

These checkpoints enable full reproduction of all figures without retraining (~40 GPU hours).

## Report Build

```bash
cd Report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Presentation Build

The final presentation lives in `presentation/`.

```bash
cd presentation
pdflatex main.tex
```

## Reproducibility Requirements (Must-Have)

These are enforced project conventions:

- **Original code**: `bilinear-decomposition-main/` contains the original paper code with **minor modifications for MPS compatibility** (Apple Silicon). Changes include CPU fallbacks for `torch.linalg.eigh()` which is unsupported on MPS.
- **Seeds**: experiments use `[42, 43, 44, 45, 46]`.
- **Logging**: every experiment must log to `wandb` and track CO2 via `codecarbon`.
- **Center-of-Mass (CoM)**: applied to raw tensors before normalization.
- **USPS upscaling**: 16x16 → 28x28 before CoM.
- **MPS caveat**: some `einsum` ops are forced to CPU on MPS (language).

## Known Constraints and Notes

- **Language on MPS is slow** (4–6 hours). Prefer GPU if available.
- **Figure 8 SAE limitation**: ts-medium lacks mlp-in SAE checkpoints. Use fw-medium (features 3834/751).
- **CO2 placeholder**: `scripts/figures/generate_extension_cp_figures.py` has a TODO for final CO2 values.

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v -k "test_effective_rank"
```

## Submission Checklist (Report + Presentation)

- Update author names in `Report/main.tex`.
- Ensure all required figures are present in `Report/figures/`.
- Include environmental impact table using `results/emissions_summary.json`.
- Build `Report/main.pdf` and `presentation/main.pdf`.

## Repository Structure (High-Level)

```
UvA_FACT_2025/
├── src/                 # Core code (vision, language, models)
├── scripts/             # Experiment runners and figure generation
├── configs/             # YAML experiment configs
├── results/             # Outputs and analysis JSONs
├── Report/              # Final report (LaTeX + figures)
├── presentation/        # Final slides
└── bilinear-decomposition-main/  # Original paper code (do not modify)
```
