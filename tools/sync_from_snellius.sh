#!/bin/bash
# Sync experiment results FROM Snellius HPC cluster
# Only syncs: checkpoints/, results/, Report/figures/
# Does NOT sync: code, Report/*.tex, configs, etc.
#
# Usage: ./tools/sync_from_snellius.sh [--dry-run]

set -e

# Configuration
SNELLIUS_USER="scur0075"
SNELLIUS_HOST="Snellius"
REMOTE_DIR="~/UvA_FACT_2025"

# Local directory (script location relative)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN MODE ==="
fi

echo "Syncing results FROM Snellius"
echo "Remote: $SNELLIUS_USER@$SNELLIUS_HOST:$REMOTE_DIR"
echo "Local:  $LOCAL_DIR"
echo ""

# Ensure local directories exist
mkdir -p "$LOCAL_DIR/checkpoints"
mkdir -p "$LOCAL_DIR/results"
mkdir -p "$LOCAL_DIR/Report/figures"

# Sync checkpoints (model weights with emissions)
echo ">>> Syncing checkpoints/"
rsync -avz --progress $DRY_RUN \
    "$SNELLIUS_USER@$SNELLIUS_HOST:$REMOTE_DIR/checkpoints/" \
    "$LOCAL_DIR/checkpoints/"

echo ""

# Sync results (JSON files with emissions)
echo ">>> Syncing results/"
rsync -avz --progress $DRY_RUN \
    "$SNELLIUS_USER@$SNELLIUS_HOST:$REMOTE_DIR/results/" \
    "$LOCAL_DIR/results/"

echo ""

# Sync generated figures only (not .tex files)
echo ">>> Syncing Report/figures/"
rsync -avz --progress $DRY_RUN \
    "$SNELLIUS_USER@$SNELLIUS_HOST:$REMOTE_DIR/Report/figures/" \
    "$LOCAL_DIR/Report/figures/"

echo ""

# Sync logs (optional, for debugging)
echo ">>> Syncing logs/"
rsync -avz --progress $DRY_RUN \
    "$SNELLIUS_USER@$SNELLIUS_HOST:$REMOTE_DIR/logs/" \
    "$LOCAL_DIR/logs/" 2>/dev/null || echo "    (No logs found)"

echo ""
echo "========================================"
echo "Sync complete!"
echo "========================================"
echo ""
echo "Synced:"
echo "  - checkpoints/ (model weights + emissions)"
echo "  - results/ (JSON files + emissions)"
echo "  - Report/figures/ (generated PDFs)"
echo "  - logs/ (job output)"
echo ""
echo "NOT synced (preserved locally):"
echo "  - Report/*.tex (your report)"
echo "  - src/ (code)"
echo "  - configs/ (configurations)"
echo ""
echo "Next steps:"
echo "  python scripts/figures/validate_emissions.py"
echo "  python scripts/figures/aggregate_emissions.py"
