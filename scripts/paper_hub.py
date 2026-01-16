#!/usr/bin/env python3
"""
Generate the interactive PaperHub HTML bundle for the project.

Design goals:
- Separation of concerns: this script only *bundles already-generated figures* into a
  self-contained HTML+PDF bundle. It does not run vision/language analyses.
- No arXiv fallbacks: if a figure PDF is missing from our outputs, it appears as a
  disabled nav item marked "(missing)".
- Exclude Figure 8 (OOM-prone).

Outputs:
- results/interactive/paper_hub_bundle/paper_hub.html
- results/interactive/paper_hub.html (stable redirect)
- results/interactive/paper_hub_bundle.zip (shareable bundle)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class HubPaths:
    interactive_dir: Path
    bundle_dir: Path
    figures_dir: Path
    assets_dir: Path
    bundle_html: Path
    stable_html: Path
    bundle_zip: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _get_paths() -> HubPaths:
    interactive_dir = PROJECT_ROOT / "results/interactive"
    bundle_dir = interactive_dir / "paper_hub_bundle"
    figures_dir = bundle_dir / "figures"
    assets_dir = bundle_dir / "assets"
    return HubPaths(
        interactive_dir=interactive_dir,
        bundle_dir=bundle_dir,
        figures_dir=figures_dir,
        assets_dir=assets_dir,
        bundle_html=bundle_dir / "paper_hub.html",
        stable_html=interactive_dir / "paper_hub.html",
        bundle_zip=interactive_dir / "paper_hub_bundle.zip",
    )


def _figure_search_roots() -> List[Path]:
    """
    Ordered search roots for generated PDFs.

    IMPORTANT: no arXiv figure fallbacks here.
    """
    return [
        PROJECT_ROOT / "Report/figures",
        PROJECT_ROOT / "results/language/figures",
        PROJECT_ROOT / "results/vision/figures",
        PROJECT_ROOT / "results/phase1/figures",
    ]


def _find_figure_pdf(name: str) -> Optional[Path]:
    for root in _figure_search_roots():
        p = root / name
        if p.exists():
            return p
    return None


def _fig_rel_path(name: str) -> str:
    # Files are copied into bundle_dir/figures/<name>
    return f"figures/{name}"


def _build_sections() -> Dict[str, List[Dict[str, str]]]:
    """
    Declarative PaperHub nav.

    Note:
    - Missing PDFs will be disabled automatically.
    - Figure 8 is intentionally excluded.
    """
    sections: Dict[str, List[Dict[str, str]]] = {
        # -----------------------------
        # Vision (existing hub content)
        # -----------------------------
        "Vision / Regularization": [
            {"label": "Eigenspectrum comparison", "path": _fig_rel_path("eigenspectrum_comparison.pdf")},
            {"label": "Eigenvalue decay", "path": _fig_rel_path("eigenvalue_decay.pdf")},
            {"label": "Eigenvectors (no reg)", "path": _fig_rel_path("eigenvectors_noreg.pdf")},
            {"label": "Update: Eigenvectors (noise only, σ=0.5)", "path": _fig_rel_path("eigenvectors_noise.pdf")},
            {"label": "Eigenvectors (full reg)", "path": _fig_rel_path("eigenvectors_reg.pdf")},
            {"label": "Update: Fashion eigenvectors (noise only, σ=0.5)", "path": _fig_rel_path("fashion_eigenvectors_noise.pdf")},
            {"label": "Ablation (MNIST)", "path": _fig_rel_path("mnist_ablation.pdf")},
            {"label": "Tradeoff (MNIST)", "path": _fig_rel_path("accuracy_vs_effrank_mnist.pdf")},
        ],
        "Vision / Noise sweep (Figure 4)": [
            {"label": "Figure 4a: noise eigenvectors", "path": _fig_rel_path("figure_4_noise_eigenvectors.pdf")},
            {"label": "Figure 4b: noise vs rank", "path": _fig_rel_path("figure_4_noise_vs_rank.pdf")},
            {"label": "Figure 4c: noise vs accuracy", "path": _fig_rel_path("figure_4_noise_vs_accuracy.pdf")},
        ],
        "Vision / Truncation & similarity": [
            {"label": "Figure 5a: similarity", "path": _fig_rel_path("figure_5a_similarity.pdf")},
            {"label": "Figure 5b: truncation", "path": _fig_rel_path("figure_5b_truncation.pdf")},
        ],
        "Vision / Challenge task": [
            {"label": "Figure 6: challenge", "path": _fig_rel_path("figure_6_challenge.pdf")},
            {
                "label": "Update: Challenge decay by regularization (none/noise/wd/full)",
                "path": _fig_rel_path("figure_6_challenge_eigenvalue_decay_by_reg.pdf"),
            },
            {"label": "Figure 6 (no reg)", "path": _fig_rel_path("figure_6_challenge_none.pdf")},
            {"label": "Figure 6 (noise only σ=0.5)", "path": _fig_rel_path("figure_6_challenge_noise.pdf")},
            {"label": "Figure 6 (weight decay only λ=1.0)", "path": _fig_rel_path("figure_6_challenge_wd.pdf")},
            {"label": "Figure 6 (full reg σ=0.5, λ=1.0)", "path": _fig_rel_path("figure_6_challenge_full.pdf")},
        ],
        "Vision / Adversarial masks": [
            {"label": "Figure 7: adversarial masks", "path": _fig_rel_path("figure_7_adversarial.pdf")},
        ],
        "Vision / Explanations": [
            {"label": "Sample explanation", "path": _fig_rel_path("sample_explanation.pdf")},
        ],
        "Vision / Appendix": [
            {"label": "Appendix: eigenspectrum digit 2", "path": _fig_rel_path("appendix_mnist_eigenspectrum_digit2.pdf")},
            {"label": "Appendix: eigenspectrum digit 4", "path": _fig_rel_path("appendix_mnist_eigenspectrum_digit4.pdf")},
            {"label": "Appendix: eigenspectrum digit 6", "path": _fig_rel_path("appendix_mnist_eigenspectrum_digit6.pdf")},
            {"label": "Appendix: truncation acc drop", "path": _fig_rel_path("appendix_mnist_acc_drop.pdf")},
            {"label": "Appendix: inter-size similarity (ref=300, top-1)", "path": _fig_rel_path("appendix_mnist_inter_similarity.pdf")},
            {"label": "Appendix: inter-size similarity matrix", "path": _fig_rel_path("appendix_mnist_inter_size_similarity.pdf")},
            {"label": "Appendix: eigenvector sparsity", "path": _fig_rel_path("appendix_mnist_eigenvec_sparsity.pdf")},
            {"label": "Appendix: eigenvalue sparsity", "path": _fig_rel_path("appendix_mnist_eigenval_sparsity.pdf")},
            {"label": "Appendix: adversarial masks (more examples)", "path": _fig_rel_path("appendix_adversarial_encoders.pdf")},
        ],
        # -----------------------------
        # Language (Figure 9 only)
        # -----------------------------
        "Language / Correlation (Figure 9)": [
            {"label": "Figure 9A: correlation progression", "path": _fig_rel_path("figure_9a_correlation_progression.pdf")},
            {"label": "Figure 9B: correlation histogram", "path": _fig_rel_path("figure_9b_correlation_histogram.pdf")},
            {
                "label": "Figure 9C: scatter plots (fw-medium)",
                "path": _fig_rel_path("figure_9c_scatter_plots.pdf"),
            },
        ],
    }
    return sections


def generate_paper_hub() -> None:
    import shutil
    import zipfile

    hp = _get_paths()
    _ensure_dir(hp.figures_dir)
    _ensure_dir(hp.assets_dir)

    sections = _build_sections()

    # Copy PDFs into bundle figures/ and annotate missing status
    for sec in list(sections.keys()):
        kept: List[Dict[str, str]] = []
        for it in sections[sec]:
            name = Path(it["path"]).name

            # Explicit exclusion safeguard: never ship Figure 8 even if present.
            if name.startswith("figure_8"):
                continue

            src = _find_figure_pdf(name)
            if not src:
                it2 = dict(it)
                it2["missing"] = "1"
                kept.append(it2)
                continue

            dst = hp.figures_dir / name
            shutil.copy(src, dst)
            it2 = dict(it)
            it2["missing"] = "0"
            kept.append(it2)
        sections[sec] = kept

    # Build nav HTML
    nav_parts: List[str] = []
    first_path = ""
    for sec, items in sections.items():
        if not items:
            continue
        nav_parts.append(f'<div class="navSection">{sec}</div>')
        for it in items:
            if not first_path:
                first_path = it["path"]
            is_missing = it.get("missing") == "1"
            disabled = "disabled" if is_missing else ""
            label = it["label"] + (" (missing)" if is_missing else "")
            onclick = "" if is_missing else f"onclick=\"loadPdf({it['path']!r}, {it['label']!r})\""
            nav_parts.append(f'<button class="navItem" {onclick} {disabled}>{label}</button>')

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>FACT-AI Paper Hub</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 0; }}
    .layout {{ display: grid; grid-template-columns: 360px 1fr; height: 100vh; }}
    .sidebar {{ border-right: 1px solid #e5e7eb; padding: 14px; overflow:auto; }}
    .content {{ padding: 14px; overflow:hidden; display:flex; flex-direction:column; gap:12px; }}
    .title {{ font-size: 16px; font-weight: 700; margin-bottom: 10px; }}
    .navSection {{ margin-top: 14px; font-size: 12px; font-weight: 700; color: #374151; }}
    .navItem {{ width: 100%; text-align:left; padding: 8px 10px; margin-top: 6px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; cursor: pointer; }}
    .navItem:hover {{ background: #f9fafb; }}
    .navItem:disabled {{ opacity: 0.55; cursor: not-allowed; background: #f9fafb; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; margin-top: 12px; background: #fff; }}
    .pdfCard {{ flex: 1; min-height: 0; margin-top: 0; display:flex; flex-direction:column; gap:8px; }}
    iframe {{ width: 100%; flex: 1; min-height: 0; border: 1px solid #e5e7eb; border-radius: 12px; }}
    .muted {{ color: #6b7280; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="sidebar">
      <div class="title">Paper Hub</div>
      <div class="muted">Organized by paper sections/keywords</div>
      <div class="muted">Build: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
      {''.join(nav_parts)}
    </div>
    <div class="content">
      <div class="card pdfCard">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div id="pdfTitle" style="font-weight:700;">Figure</div>
          <div class="muted" id="pdfPath"></div>
        </div>
        <iframe id="pdfFrame" src=""></iframe>
      </div>
    </div>
  </div>

  <script>
    function loadPdf(path, title) {{
      const frame = document.getElementById('pdfFrame');
      const t = document.getElementById('pdfTitle');
      const p = document.getElementById('pdfPath');
      t.textContent = title;
      p.textContent = path;
      frame.src = path;
    }}

    // init iframe
    const initial = {first_path!r};
    if (initial) {{
      loadPdf(initial, "Overview");
    }}
  </script>
</body>
</html>
"""

    hp.bundle_html.write_text(html, encoding="utf-8")
    print(f"Saved: {hp.bundle_html}")

    # Stable redirect entrypoint
    stable_html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta http-equiv="refresh" content="0; url=paper_hub_bundle/paper_hub.html"/>
  <title>Paper Hub (redirect)</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding: 18px; }
  </style>
</head>
<body>
  <div>Redirecting to <code>paper_hub_bundle/paper_hub.html</code>…</div>
  <div>If you are not redirected, open: <code>paper_hub_bundle/paper_hub.html</code></div>
</body>
</html>
"""
    _ensure_dir(hp.stable_html.parent)
    hp.stable_html.write_text(stable_html, encoding="utf-8")
    print(f"Saved: {hp.stable_html}")

    # Zip bundle for sharing
    if hp.bundle_zip.exists():
        hp.bundle_zip.unlink()
    with zipfile.ZipFile(hp.bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in hp.bundle_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(hp.bundle_dir)))
    print(f"Saved: {hp.bundle_zip}")


def main() -> int:
    generate_paper_hub()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

