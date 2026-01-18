# FACT-AI & MLRC Report Guidelines

## 0. Current Status (Updated: 2026-01-11)

**Experiments Running Overnight:**
- 40 vision runs (MNIST + Fashion-MNIST, 4 configs x 5 seeds each)
- 3 language runs (SAE training, negation discovery, interaction analysis)
- Tracking: https://wandb.ai/itayerlich96-student/fact-bilinear
- Estimated completion: ~6-8 hours

**After Experiments Complete:**
1. Download checkpoints from `results/` directories
2. Use `src/vision/spectral.py` to extract eigenspectrum data
3. Generate figures and save to `Report/figures/`
4. Uncomment figure/table blocks in LaTeX sections
5. Fill in actual values from wandb/checkpoints

---

## 1. Philosophy: Continuous Reporting
* **Never wait.** If a plot is generated in `src/`, immediately save a copy to `Report/figures/`.
* **Caption first.** When saving a figure, write the LaTeX figure block with a caption describing *what it shows* (not just what it is).

## 2. Structure & FACT-AI Requirements
The report uses the TMLR format but MUST address specific FACT-AI course requirements.

### Section 1: Introduction (Topic: Transparency)
* **Narrative:** Mechanistic Interpretability is usually "post-hoc" (SAEs). We are investigating "intrinsic" interpretability via Bilinear layers.
* **Link:** Connect this to **FACT-AI Topic 1.4 (Transparency)**. Explain that weight-based interpretability offers transparency without the massive compute cost of training SAEs on top of LLMs.

### Section 2: Scope of Reproducibility
* Explicitly state: "We reproduce Section 4 (Vision) of Pearce et al. (2024)."
* Define the Hyperparameters used (refer to `src/configs/`).

### Section 3: Methodology
* Define the `BilinearLayer` math: $y = (W_l x) \odot (W_r x)$.
* Define our Extension: The CP-Decomposition layer ($Rank-R$).

### Section 4: Reproduction Results (Phase 1)
* **Mandatory Figure:** The Eigenspectrum plot (Regularized vs. Unregularized).
* **Metric:** Effective Rank.

### Section 5: Extension Results (Phase 2)
* **Robustness:** Rotated MNIST/EMNIST eigenvector analysis.
* **Structure vs. Regularization:** The Pareto plot (Accuracy vs. Rank).

### Section 6: Discussion (Course Specifics)
* **What was easy:** (Fill this as we code - e.g., "The model architecture was simple...")
* **What was difficult:** (e.g., "Tuning the noise level to get emergence...")
* **Environmental Impact:**
    * **MUST** include a table of CO2 emissions from our `codecarbon` logs.
    * Discuss: "Low-Rank structures (CP) reduce memory footprint, potentially lowering inference energy."

## 3. LaTeX Conventions
* **Math:** Use `\input{math_commands.tex}`. Use `\E` for expectation, `\odot` for Hadamard.
* **Citations:** Update `main.bib` immediately. Key references: `pearce2024bilinear`, `kolda2009tensor`.
* **Compilation:** `pdflatex main.tex` -> `bibtex main` -> `pdflatex main` x2.

## 4. Figure Workflow

### When generating a figure in code:
```python
# In src/vision/visualization.py or notebook
fig.savefig('../Report/figures/eigenspectrum_comparison.pdf', bbox_inches='tight')
```

### Immediately add to LaTeX:
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/eigenspectrum_comparison.pdf}
    \caption{Eigenspectrum comparison showing that regularization (noise + weight decay)
    induces low-rank structure in the bilinear interaction tensor. Without regularization,
    the spectrum is flat (high effective rank ~150). With regularization, the spectrum
    shows sharp decay (low effective rank ~25), indicating interpretable low-rank structure.}
    \label{fig:eigenspectrum}
\end{figure}
```

## 5. Required Figures Checklist

### Phase 1 (Reproduction)
- [ ] `eigenspectrum_comparison.pdf` - Main result: reg vs no-reg decay
- [ ] `eigenvectors_noreg.pdf` - Overfitting patterns (noise-like)
- [ ] `eigenvectors_reg.pdf` - Interpretable digit-like patterns
- [ ] `accuracy_vs_effrank.pdf` - Trade-off scatter plot

### Phase 2 (Extensions)
- [ ] `robustness_rotated.pdf` - Eigenvector stability under rotation
- [ ] `robustness_emnist.pdf` - Transfer to EMNIST
- [ ] `cp_rank_sweep.pdf` - Accuracy vs CP rank
- [ ] `structure_vs_reg.pdf` - Pareto frontier comparison

## 6. Required Tables Checklist

- [ ] Table 1: Reproduction metrics (accuracy, effective rank) vs paper
- [ ] Table 2: Ablation results (none, noise, wd, full)
- [ ] Table 3: CP rank sweep results
- [ ] Table 4: Environmental impact (GPU hours, CO2 per experiment)

## 7. Section File Mapping

| Section | File | Content |
|---------|------|---------|
| 1 | `sections/01_introduction.tex` | Transparency motivation, contribution summary |
| 2 | `sections/02_scope.tex` | Claims to verify, paper context |
| 3 | `sections/03_method.tex` | BilinearLayer math, CP extension, datasets |
| 4 | `sections/04_reproduction.tex` | Phase 1 results, figures |
| 5 | `sections/05_extension.tex` | Phase 2 results, CP analysis |
| 6 | `sections/06_fact_discussion.tex` | Discussion, easy/difficult, environmental impact |

## 8. Key References (add to main.bib)

```bibtex
@article{pearce2024bilinear,
  title={Bilinear MLPs enable weight-based mechanistic interpretability},
  author={Pearce, Tim and others},
  journal={arXiv preprint arXiv:2410.08417},
  year={2024}
}

@article{kolda2009tensor,
  title={Tensor decompositions and applications},
  author={Kolda, Tamara G and Bader, Brett W},
  journal={SIAM review},
  volume={51},
  number={3},
  pages={455--500},
  year={2009}
}

@article{roy2007effective,
  title={The effective rank: A measure of effective dimensionality},
  author={Roy, Olivier and Bhattacharyya, Shyamashree},
  journal={European Signal Processing Conference},
  year={2007}
}
```

## 9. Post-Experiment Figure Generation

After overnight experiments complete, use these scripts to generate figures:

### Vision Figures (Section 4)

```python
# Load checkpoints and analyze
from pathlib import Path
import torch
from src.vision.spectral import load_checkpoint_eigenvalues, spectral_summary, effective_rank
import matplotlib.pyplot as plt

# 1. Eigenspectrum comparison
checkpoints = {
    'none': 'results/vision/checkpoints/mnist_dense_none_seed42.pt',
    'full': 'results/vision/checkpoints/mnist_dense_full_seed42.pt',
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (name, path) in enumerate(checkpoints.items()):
    eigenvalues, _ = load_checkpoint_eigenvalues(path)
    for cls in range(10):
        axes[i].plot(eigenvalues[cls].abs().numpy(), alpha=0.5)
    axes[i].set_title(f'{name.title()} Regularization')
    axes[i].set_xlabel('Eigenvalue Index')
    axes[i].set_ylabel('|Eigenvalue|')
fig.savefig('Report/figures/eigenspectrum_comparison.pdf', bbox_inches='tight')

# 2. Eigenvector visualization
eigenvalues, eigenvectors = load_checkpoint_eigenvalues(checkpoints['full'])
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for cls in range(10):
    ax = axes[cls // 5, cls % 5]
    img = eigenvectors[cls, 0].reshape(28, 28)  # Top eigenvector
    ax.imshow(img.numpy(), cmap='RdBu_r')
    ax.set_title(f'Class {cls}')
    ax.axis('off')
fig.savefig('Report/figures/eigenvectors_reg.pdf', bbox_inches='tight')
```

### Language Figures (Section 5)

```python
import json

# Negation results
with open('results/language/negation_analysis.json') as f:
    negation = json.load(f)
print(f"Top not+positive feature: {negation['not_positive_features'][0]}")
print(f"Top not+negative feature: {negation['not_negative_features'][0]}")
print(f"Cosine similarity: {negation['top_pair_analysis']['cosine_similarity']:.3f}")

# Interaction results
with open('results/language/interaction_analysis.json') as f:
    interaction = json.load(f)
print(f"Fraction above 0.75: {interaction['summary']['fraction_above_075']:.1%}")
print(f"Claim supported: {interaction['summary']['claim_supported']}")
```

### wandb Data Export

```python
import wandb
api = wandb.Api()
runs = api.runs("itayerlich96-student/fact-bilinear")

# Get all vision runs
vision_runs = [r for r in runs if 'vision' in r.tags]
for run in vision_runs:
    print(f"{run.name}: acc={run.summary.get('final_val_acc', 'N/A'):.4f}, "
          f"rank={run.summary.get('effective_rank', 'N/A'):.1f}")
```
