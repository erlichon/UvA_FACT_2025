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

# Create zip excluding results/, checkpoints/, data folders, logs, and common non-essential files
zip -r "$OUTPUT_ZIP" . \
    -x "results/*" \
    -x "checkpoints/*" \
    -x "checkpoints.zip" \
    -x "report_context.zip" \
    -x "data/*" \
    -x "FashionMNIST/*" \
    -x "MNIST/*" \
    -x "EMNIST/*" \
    -x "logs/*" \
    -x "wandb/*" \
    -x "Report/*.bib" \
    -x "Report/*.aux" \
    -x "Report/*.log" \
    -x "Report/*.out" \
    -x "Report/*.toc" \
    -x "Report/*.bbl" \
    -x "Report/*.blg" \
    -x "Report/*.fls" \
    -x "Report/*.fdb_latexmk" \
    -x "Report/*.synctex.gz" \
    -x ".git/*" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc" \
    -x ".DS_Store" \
    -x "*.egg-info/*"

# Show result
echo ""
echo "============================================"
echo "Report context zip created successfully!"
echo "Location: $OUTPUT_ZIP"
echo "Size: $(ls -lh "$OUTPUT_ZIP" | awk '{print $5}')"
echo "============================================"
echo ""
echo "Excluded:"
echo "  - results/"
echo "  - checkpoints/"
echo "  - checkpoints.zip"
echo "  - data/, FashionMNIST/, MNIST/, EMNIST/"
echo "  - logs/, wandb/"
echo "  - Report/ LaTeX auxiliary files (.aux, .log, .bbl, etc.)"
echo "  - .git/"
echo "  - __pycache__/"
echo "  - *.pyc"
