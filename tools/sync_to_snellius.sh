#!/bin/bash
# Sync repository to Snellius HPC cluster
# Usage: ./scripts/sync_to_snellius.sh [--dry-run]

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

echo "Syncing: $LOCAL_DIR"
echo "     To: $SNELLIUS_USER@$SNELLIUS_HOST:$REMOTE_DIR"
echo ""

# Ensure remote directory exists
ssh "$SNELLIUS_USER@$SNELLIUS_HOST" "mkdir -p $REMOTE_DIR"

# Rsync with exclusions
# NOTE: No --delete flag, so remote files are preserved
rsync -avz --progress $DRY_RUN \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='/data/' \
    --exclude='/checkpoints/' \
    --exclude='/results/' \
    --exclude='/wandb/' \
    --exclude='.env' \
    --exclude='emissions.csv' \
    --exclude='emissions.csv.bak' \
    --exclude='powermetrics_log.txt' \
    --exclude='Report/*.pdf' \
    --exclude='Report/*.aux' \
    --exclude='Report/*.log' \
    --exclude='Report/*.out' \
    --exclude='.ipynb_checkpoints' \
    --exclude='node_modules' \
    "$LOCAL_DIR/" "$SNELLIUS_USER@$SNELLIUS_HOST:$REMOTE_DIR/"

echo ""
echo "Sync complete!"
echo ""
echo "Next steps on Snellius:"
echo "  ssh $SNELLIUS_USER@$SNELLIUS_HOST"
echo "  cd $REMOTE_DIR"
echo "  sbatch jobs/run_all.job"
