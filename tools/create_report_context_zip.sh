#!/bin/bash
# Create a zip file of the entire repo excluding results/ and checkpoints/ folders
# Usage: ./tools/create_report_context_zip.sh

set -e

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Output file
OUTPUT_ZIP="$PROJECT_ROOT/report_context.zip"

echo "Creating report context zip..."
echo "Project root: $PROJECT_ROOT"

# Remove existing zip if present
rm -f "$OUTPUT_ZIP"

# Create zip with ONLY what's needed for LaTeX editing
# Exclude everything large/unnecessary
zip -r "$OUTPUT_ZIP" . \
    -x "results/*" \
    -x "checkpoints/*" \
    -x "checkpoints.zip" \
    -x "results.zip" \
    -x "report_context.zip" \
    -x "data/*" \
    -x "FashionMNIST/*" \
    -x "MNIST/*" \
    -x "EMNIST/*" \
    -x "USPS/*" \
    -x "logs/*" \
    -x "wandb/*" \
    -x "Report/figures/*" \
    -x "Report/main.pdf" \
    -x "Report/paper_hub_bundle.zip" \
    -x "Report/paper_hub_bundle/*" \
    -x "Report/*.aux" \
    -x "Report/*.log" \
    -x "Report/*.out" \
    -x "Report/*.toc" \
    -x "Report/*.bbl" \
    -x "Report/*.blg" \
    -x "Report/*.fls" \
    -x "Report/*.fdb_latexmk" \
    -x "Report/*.synctex.gz" \
    -x "arXiv-2410.08417v2/*" \
    -x "bilinear-decomposition-main/*" \
    -x "notebooks/*" \
    -x "src/*" \
    -x "scripts/*" \
    -x "tests/*" \
    -x "jobs/*" \
    -x ".git/*" \
    -x ".vscode/*" \
    -x ".cursor/*" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x "__MACOSX/*" \
    -x "*.pyc" \
    -x "*.pdf" \
    -x "*.zip" \
    -x ".DS_Store" \
    -x "*.egg-info/*" \
    -x "environment*.yml" \
    -x "powermetrics_log.txt" \
    -x "texput.log"

# Show result
echo ""
echo "============================================"
echo "Report context zip created successfully!"
echo "Location: $OUTPUT_ZIP"
echo "Size: $(ls -lh "$OUTPUT_ZIP" | awk '{print $5}')"
echo "============================================"
echo ""
echo "Excluded (for minimal LaTeX editing context):"
echo "  - results/, checkpoints/, data/, logs/, wandb/"
echo "  - Report/figures/, Report/main.pdf, Report/paper_hub_bundle*"
echo "  - arXiv-2410.08417v2/, bilinear-decomposition-main/"
echo "  - notebooks/, src/, scripts/, tests/, jobs/"
echo "  - All *.pdf, *.zip files"
echo "  - .git/, .vscode/, .cursor/, __pycache__/"
echo ""
echo "Included:"
echo "  - Report/*.tex, Report/sections/*.tex (LaTeX source)"
echo "  - Report/*.sty, Report/*.bst (style files)"
echo "  - Report/main.bib (references)"
echo "  - CLAUDE.md, CONTEXT.md, .cursorrules (project context)"
echo "  - configs/*.yaml (experiment configs for reference)"
echo "  - docs/*.md (documentation)"
