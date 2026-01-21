# Prompt for Claude to Write FACT-AI Report

Copy this prompt and upload the `report_context.zip` file to Claude.

---

## PROMPT START

I need your help writing/improving my FACT-AI course report. I'm reproducing the paper "Bilinear MLPs enable weight-based mechanistic interpretability" (arXiv:2410.08417).

I've attached a zip file (`report_context.zip`) containing all the context you need:

### What's in the zip:

**Documentation (read these first for context):**
- `CLAUDE.md` - Project overview, architecture, commands
- `CONTEXT.md` - Current state, detailed results, what we found
- `.cursorrules` - Research questions and conventions
- `WORKPLAN.md` - Experiment matrix
- `docs/` - Additional analysis documents (SAE limitations, Figure 8 investigation, metrics analysis)

**Report template (current state):**
- `Report/main.tex` - Master document
- `Report/sections/` - All 6 section files (01_introduction through 06_fact_discussion)
- `Report/main.bib` - References
- `Report/REPORT_INSTRUCTIONS.md` - Course requirements

**Source code (for methodology understanding):**
- `src/models/bilinear_layer.py` - Core BilinearDense + BilinearCP implementation
- `src/vision/spectral.py` - Effective rank computation
- `src/language/` - Correlation verification, negation discovery, interaction analysis

**Results data:**
- `results/language/*.json` - All correlation/negation/interaction analysis results
- `results/extension2/*.json` - Extension 2 (cross-dataset robustness) results
- `results/vision/*.csv` - MNIST and Fashion-MNIST numerical results

**Original paper:**
- `arXiv-2410.08417v2/main.tex` - The paper we're reproducing (for reference)

---

### Project Summary:

**Course**: FACT-AI (Fairness, Accountability, Confidentiality and Transparency in AI), UvA MSc AI
**Deadline**: 30 January 2026
**Paper**: "Bilinear MLPs enable weight-based mechanistic interpretability"

**What we reproduced:**
1. **Section 4 (Vision)**: MNIST/Fashion-MNIST eigendecomposition - FULLY REPRODUCED
   - Trained bilinear MLPs under 4 regularization configs × 5 seeds
   - Verified that weight decay reduces effective rank (21.4 vs 38.5)
   - Reproduced Figures 4-7 from the paper

2. **Section 5 (Language)**: Negation circuit discovery - PARTIALLY REPRODUCED
   - ✅ Negation features with opposing directions (cosine similarity -0.16) - CONFIRMED
   - ❌ Low-rank interaction structure (69% > 0.75 correlation) - NOT REPRODUCED
   - We found mean rank-2 correlations of 0.37-0.55, far below the paper's claim
   - Root cause: HuggingFace SAEs appear under-trained vs paper's internal checkpoints

**Extensions:**
- Extension 2: Cross-dataset robustness (MNIST → EMNIST transfer, subspace overlap analysis)
- Ablation study disentangling noise augmentation vs weight decay

---

### What I need help with:

1. **Review the current report sections** in `Report/sections/` and improve them
2. **Ensure the methodology section** accurately describes our implementation
3. **Make the results section compelling** - we have interesting findings (partial reproduction with scientific explanation for discrepancies)
4. **Write a strong discussion section** that addresses:
   - What was easy/difficult to reproduce
   - The SAE training time discrepancy (this is scientifically interesting!)
   - Environmental impact (CO2 tracking)
   - FACT-AI transparency theme connection

5. **Check that all figures are properly referenced** and captions are informative
6. **Ensure the abstract accurately summarizes** our findings (partial success is still valuable)

---

### Key findings to emphasize:

**Vision (Section 4) - Success:**
- Effective rank ratio (WD/none) = 0.55, close to paper's <0.5 target
- Weight decay alone achieves lowest rank (21.4 MNIST, 19.6 Fashion)
- Eigenvectors show interpretable digit patterns with regularization
- All Figures 4-7 successfully reproduced

**Language (Section 5) - Partial success with scientific insight:**
- ✅ Negation circuit discovery works (features 5158/3223 with cosine sim -0.16)
- ❌ Low-rank correlation claim not reproduced (0.37-0.55 vs claimed 69% > 0.75)
- **Key insight**: We traced this to SAE training time (Figure 10 shows 2.5× improvement with longer training)
- This is actually a valuable finding - it explains WHY the discrepancy exists

---

### Report requirements (from course):

- TMLR format (already set up)
- Must connect to FACT-AI Topic 1.4 (Transparency)
- Must include environmental impact (CO2 emissions)
- Must discuss what was easy/difficult
- Should be honest about partial reproductions

---

Please start by reading the key documentation files, then review and improve the report sections. Focus on making the narrative clear: we successfully reproduced the vision experiments and discovered an important insight about why the language experiments partially failed (SAE training time).

## PROMPT END

---

## Tips for using this prompt:

1. Upload `report_context.zip` along with this prompt
2. Ask Claude to read `CONTEXT.md` and `CLAUDE.md` first for full context
3. Then ask it to review specific sections (e.g., "Please improve section 04_reproduction.tex")
4. You can ask for specific help like:
   - "Improve the abstract to better reflect our partial reproduction"
   - "Add more detail to the methodology section about the effective rank formula"
   - "Make the discussion section stronger regarding the SAE training insight"
   - "Check all figure references and improve captions"
