# FACT-AI & MLRC Report Guidelines

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
# In src/analysis/visualization.py or notebook
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
