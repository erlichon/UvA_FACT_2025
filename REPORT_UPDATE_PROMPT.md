# Report Editing Prompt for Claude

**Context**: Attached is `report_context.zip` containing the full LaTeX codebase for a reproducibility report. The hard constraint is a **10-page limit** for the main content (Introduction through Conclusion). References and Appendices do NOT count toward this limit. The current draft is significantly over length.

---

## Role

You are a Senior Editor at TMLR (Transactions on Machine Learning Research) and an expert in academic writing for Machine Learning reproducibility studies. You understand the MLRC (ML Reproducibility Challenge) grading criteria and know what reviewers look for in an "Excellent" grade submission.

---

## Task

Aggressively edit and restructure the LaTeX code to fit the 10-page limit while preserving scientific depth and the criteria for an "Excellent" grade. You will output the complete, edited LaTeX files.

---

## File Structure

The report is structured as follows:
- `Report/main.tex` - Main document (includes sections, defines document class)
- `Report/sections/01_introduction.tex` - Introduction
- `Report/sections/02_scope.tex` - Scope of Reproducibility (claims table)
- `Report/sections/03_method.tex` - Methodology
- `Report/sections/04_reproduction.tex` - Main reproduction results (Vision + Language)
- `Report/sections/05_cross_dataset_structural_robustness_extension.tex` - Extension 1
- `Report/sections/06_cp_decomposition_extension.tex` - Extension 2
- `Report/sections/07_fact_discussion.tex` - Discussion + Conclusion

---

## Specific Editing Instructions

### 1. Condense Methodology (Section 3) — Target: ~50% reduction

**REMOVE** (assume expert reader):
- Textbook definitions of Transformers, Attention mechanisms
- Basic Linear Algebra explanations
- The detailed SAE equation explanation (Eq. 5)
- Verbose implementation notes

**KEEP** (these are novel/specific to this paper):
- Bilinear Layer definition (Eq. 1-2)
- Interaction tensor formulation (Eq. 3-4)
- Effective Rank formula (Eq. 6) — essential for understanding results
- Low-Rank Correlation Metric (Eq. 9-10) — needed for language results
- Table 1 (regularization configs) and Table 2 (language model configs)

**ACTION**: Combine Sections 3.1-3.3 into a single dense "Bilinear Interpretability Framework" subsection. Remove 3.4 (Language Model Analysis) preamble; jump directly to "SAE Integration" with a 2-sentence motivation.

---

### 2. Restructure Extension 1: Cross-Dataset Robustness (Section 5)

**MOVE TO APPENDIX**:
- The mathematical derivation of Quadratic Form Similarity (the full subsection 5.3 "Method: Quadratic Form Similarity")
- Table 4 (eigenvector comparison 0-O-X) — keep only the conclusion
- The verbose cosine similarity failure discussion

**KEEP IN MAIN TEXT**:
- 1-2 sentence motivation: "Cosine similarity fails to distinguish geometrically similar pairs (0.358 vs 0.339). We propose Quadratic Form Similarity..."
- The formula (inline, not as a numbered equation)
- Results heatmaps (Figures 8, 9, 10)
- Table 6 (ranking analysis) — this is the key result
- Conclusion paragraph

**TARGET**: Reduce from ~3 pages to ~1.5 pages.

---

### 3. Restructure Extension 2: CP-Decomposition (Section 6)

**MOVE TO APPENDIX**:
- Detailed methodology subsection (6.2) — keep only the bullet point summary
- The full rank sweep table (Table 7) — this is detailed but not essential for main claims
- The scalability analysis (6.5) — interesting but tangential

**KEEP IN MAIN TEXT**:
- Motivation paragraph (why CP instead of eigendecomposition)
- The Pareto Frontier figure (Figure 11) — this IS the key insight
- Key numerical finding: "Lambda CP at R=256 achieves 93.8% accuracy with eff. rank 17.5"
- Top-2 eigenvector visualization (Figure 12)
- Conclusion paragraph on disentanglement

**TARGET**: Reduce from ~2.5 pages to ~1 page.

---

### 4. Preserve Critical Content (DO NOT CUT)

These are the strongest contributions and must remain intact:

1. **Root Cause Analysis (Section 4.2.4)**: The SAE training time investigation is the paper's most significant finding. Keep:
   - Figure 6 (SAE training effect)
   - The 2.5× improvement finding (0.15 → 0.39)
   - The "Reproducible in Principle, Dependent on Artifact Quality" conclusion

2. **Claims Summary Table (Table 1 in Section 2)**: The ✓/×/~ reproduction status table is essential for reviewers.

3. **Vision vs. Language Distinction**: The clear narrative that Vision = fully reproduced, Language = partially reproduced with explained root cause.

4. **Environmental Impact (Section 7.3)**: FACT-AI course requirement. Keep Table 8 (CO2 emissions).

---

### 5. Formatting & Float Management

**Figures**:
- Resize multi-panel figures to `\textwidth` and use `[t]` or `[b]` placement
- For the eigenvector grids (Figures 3a, 3b), consider `[h!]` with `\small` captions
- Combine Figure 4a/4b/4c into a single `\begin{figure}[t]` with reduced panel sizes

**Tables**:
- Use `\small` or `\footnotesize` for large tables
- Consider `tabularx` for better width control
- Move verbose headers to table notes

**Text**:
- Replace verbose transitions ("We now turn to...", "Having established...") with direct statements
- Merge short paragraphs that cover the same point
- Use `\paragraph{}` for inline headings instead of `\subsubsection{}` where appropriate

---

### 6. Appendix Management

Create properly labeled appendix sections for moved content:

```latex
\appendix
\section{Quadratic Form Similarity Derivation}
\label{app:quadratic_form}
% Full mathematical derivation from Section 5.3

\section{CP-Decomposition Methodology Details}
\label{app:cp_methodology}
% Detailed regularization variants from Section 6.2

\section{Full Experimental Results}
\label{app:full_results}
% Table 4 (eigenvector similarity), Table 7 (CP rank sweep)

\section{Additional Figures}
\label{app:figures}
% Any figures that didn't fit

\section{Hyperparameter Details}
\label{app:hyperparams}
% Full configuration details

\section{CO2 Tracking Details}
\label{app:co2}
% Detailed codecarbon logs
```

Reference appendices from main text: "For the full derivation, see Appendix~\ref{app:quadratic_form}."

---

### 7. Specific Cuts by Section

| Section | Current Est. Pages | Target Pages | Action |
|---------|-------------------|--------------|--------|
| 1. Introduction | 1.0 | 0.75 | Tighten bullet list, remove last paragraph |
| 2. Scope | 1.5 | 1.25 | Keep claims table, compress success criteria |
| 3. Methodology | 2.5 | 1.25 | Major cuts as specified above |
| 4. Reproduction | 3.0 | 2.5 | Combine Figure 4 panels, tighter captions |
| 5. Extension 1 | 3.0 | 1.5 | Move derivation to appendix |
| 6. Extension 2 | 2.5 | 1.0 | Move table + methodology to appendix |
| 7. Discussion | 2.0 | 1.5 | Tighten, keep CO2 table |
| **Total** | **~15.5** | **~10** | |

---

### 8. Style Guidelines

- **Do not add emojis** or informal language
- **Preserve all \cite{} references** — do not remove citations
- **Keep the tmlr.sty formatting** — do not change document class or style
- **Preserve \label{} tags** — other sections may reference them
- **Keep the Author Contributions and Acknowledgments** at the end

---

### 9. Output Format

For each section file you edit, output the complete edited LaTeX code in a code block:

```latex
% Section X: [Title]
% [Any relevant comments]

\section{...}
...
```

Also output the updated `main.tex` if you modify it (e.g., to add appendix includes).

---

### 10. Quality Checklist

Before finalizing, verify:
- [ ] All numbered equations that are referenced elsewhere are preserved
- [ ] All figures referenced in text still exist
- [ ] Cross-references (\ref{}) point to valid labels
- [ ] No orphaned appendix references (if you cite Appendix X, it must exist)
- [ ] The narrative flow is preserved (Vision → Language → Extensions → Discussion)
- [ ] The SAE training time root cause analysis is complete and prominent
- [ ] The claims summary table is intact
- [ ] Environmental impact section is present

---

## Begin

Please proceed with editing the report. Start with the sections that need the most cuts (Sections 3, 5, 6), then move to lighter edits (Sections 1, 2, 4, 7). Output each edited section file in full.
