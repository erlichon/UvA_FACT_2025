#!/bin/bash
# Master script for ALL FACT-AI experiments with CodeCarbon tracking
#
# This script runs all experiments sequentially to ensure accurate per-experiment
# emissions measurements. It covers:
# - Vision experiments (MNIST, Fashion-MNIST, noise/size sweeps)
# - Extension 2 (Cross-dataset robustness with CoM)
# - Extension CP (CP decomposition rank sweep)
# - Language experiments (Figure 8, 9, 10, negation, interaction)
# - Figure generation for all sections
# - Emissions validation and aggregation
#
# Usage:
#   ./scripts/train/run_all.sh          # Full production run (~6-7h on A100)
#   ./scripts/train/run_all.sh --test   # Quick validation (~15-20 min)
#
# The --test flag runs minimal configurations to verify emissions tracking works:
# - 1 seed, 2 epochs for vision
# - 100 features for figure8 search
# - Single model for figure9
#
# After full run, results are available at:
# - checkpoints/vision/**/*.pt (with emissions field)
# - checkpoints/extension2/*.pt (with emissions field)
# - checkpoints/extension_cp/*.pt (with emissions field)
# - results/language/*.json (with co2_kg field)
# - results/emissions_summary.json (aggregated)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Parse arguments
TEST_MODE=false
SKIP_VISION=false
SKIP_EXTENSION2=false
SKIP_EXTENSION_CP=false
SKIP_LANGUAGE=false
SKIP_FIGURES=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --skip-vision)
            SKIP_VISION=true
            shift
            ;;
        --skip-extension2)
            SKIP_EXTENSION2=true
            shift
            ;;
        --skip-extension-cp)
            SKIP_EXTENSION_CP=true
            shift
            ;;
        --skip-language)
            SKIP_LANGUAGE=true
            shift
            ;;
        --skip-figures)
            SKIP_FIGURES=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test              Quick validation mode (~15-20 min)"
            echo "  --skip-vision       Skip vision experiments"
            echo "  --skip-extension2   Skip extension 2 experiments"
            echo "  --skip-extension-cp Skip extension CP experiments"
            echo "  --skip-language     Skip language experiments"
            echo "  --skip-figures      Skip figure generation"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Full production run"
            echo "  $0 --test           # Quick validation"
            echo "  $0 --skip-vision    # Skip vision, run everything else"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Header
echo "========================================================================"
echo "FACT-AI COMPLETE EXPERIMENT PIPELINE"
echo "========================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Test mode: $TEST_MODE"
echo "Start time: $(date)"
echo ""

if $TEST_MODE; then
    echo ">>> TEST MODE: Running minimal configurations for validation"
    echo "    - 1 seed, 2 epochs for vision"
    echo "    - 100 features for figure8 search"
    echo "    - Single model for figure9"
    echo ""
fi

# Create log directory
mkdir -p logs

# ============================================================================
# PHASE 1: VISION EXPERIMENTS
# ============================================================================
if ! $SKIP_VISION; then
    echo ""
    echo "========================================================================"
    echo "PHASE 1: VISION EXPERIMENTS"
    echo "========================================================================"
    
    if $TEST_MODE; then
        echo ">>> Running vision in test mode (1 seed, 2 epochs, base configs only)"
        ./scripts/train/run_vision.sh train base --quick --no-wandb
    else
        echo ">>> Running all vision experiments (base, noise, size, challenge, adversarial)"
        ./scripts/train/run_vision.sh train all
    fi
    
    echo ">>> Vision experiments complete"
else
    echo ""
    echo ">>> Skipping vision experiments (--skip-vision)"
fi

# ============================================================================
# PHASE 2: EXTENSION 2 (Cross-Dataset Robustness)
# ============================================================================
if ! $SKIP_EXTENSION2; then
    echo ""
    echo "========================================================================"
    echo "PHASE 2: EXTENSION 2 (Cross-Dataset Robustness)"
    echo "========================================================================"
    
    if $TEST_MODE; then
        echo ">>> Running extension 2 in test mode (MNIST only, 1 seed)"
        ./scripts/train/run_extension2.sh train mnist --seeds 42
    else
        echo ">>> Running all extension 2 experiments"
        ./scripts/train/run_extension2.sh train all
    fi
    
    echo ">>> Extension 2 experiments complete"
else
    echo ""
    echo ">>> Skipping extension 2 experiments (--skip-extension2)"
fi

# ============================================================================
# PHASE 3: EXTENSION CP (CP Decomposition)
# ============================================================================
if ! $SKIP_EXTENSION_CP; then
    echo ""
    echo "========================================================================"
    echo "PHASE 3: EXTENSION CP (CP Decomposition)"
    echo "========================================================================"
    
    if $TEST_MODE; then
        echo ">>> Running extension CP in test mode (2 epochs)"
        ./scripts/train/run_extension_cp.sh test
    else
        echo ">>> Running all extension CP experiments"
        ./scripts/train/run_extension_cp.sh train all
    fi
    
    echo ">>> Extension CP experiments complete"
else
    echo ""
    echo ">>> Skipping extension CP experiments (--skip-extension-cp)"
fi

# ============================================================================
# PHASE 4: LANGUAGE EXPERIMENTS
# ============================================================================
if ! $SKIP_LANGUAGE; then
    echo ""
    echo "========================================================================"
    echo "PHASE 4: LANGUAGE EXPERIMENTS"
    echo "========================================================================"
    
    if $TEST_MODE; then
        echo ">>> Running language experiments in test mode"
        
        # Figure 9 (correlation sweep) - single model, quick mode
        echo ""
        echo ">>> Figure 9: Correlation sweep (fw-medium only, quick mode)"
        ./scripts/train/run_language.sh figure9 --quick --model fw-medium
        
        # Figure 8 (circuit search) - 100 features only
        echo ""
        echo ">>> Figure 8: Circuit search (100 features)"
        ./scripts/train/run_language.sh figure8 search --quick
        
        # Negation discovery - quick mode
        echo ""
        echo ">>> Negation discovery (quick mode)"
        ./scripts/train/run_language.sh negation --quick
        
    else
        echo ">>> Running all language experiments"
        
        # Figure 9 (correlation sweep) - all 3 models
        echo ""
        echo ">>> Figure 9: Correlation sweep (all 3 models)"
        ./scripts/train/run_language.sh figure9
        
        # Figure 8 (full circuit search) - all 8192 features
        echo ""
        echo ">>> Figure 8: Full circuit search (8192 features, ~2h on A100)"
        ./scripts/train/run_language.sh figure8 all
        
        # Figure 10 (SAE training time analysis)
        echo ""
        echo ">>> Figure 10: SAE training time analysis"
        ./scripts/train/run_language.sh figure10
        
        # Negation discovery
        echo ""
        echo ">>> Negation discovery"
        ./scripts/train/run_language.sh negation
        
        # Interaction analysis
        echo ""
        echo ">>> Interaction analysis"
        ./scripts/train/run_language.sh interaction
    fi
    
    echo ">>> Language experiments complete"
else
    echo ""
    echo ">>> Skipping language experiments (--skip-language)"
fi

# ============================================================================
# PHASE 5: GENERATE ALL FIGURES
# ============================================================================
if ! $SKIP_FIGURES; then
    echo ""
    echo "========================================================================"
    echo "PHASE 5: GENERATE ALL FIGURES"
    echo "========================================================================"
    
    if ! $SKIP_VISION; then
        echo ">>> Generating vision figures"
        ./scripts/train/run_vision.sh figures
    fi
    
    if ! $SKIP_LANGUAGE; then
        echo ">>> Generating language figures"
        ./scripts/train/run_language.sh figures
    fi
    
    if ! $SKIP_EXTENSION2; then
        echo ">>> Generating extension 2 figures"
        ./scripts/train/run_extension2.sh figures
    fi
    
    if ! $SKIP_EXTENSION_CP; then
        echo ">>> Generating extension CP figures"
        # Check if script exists
        if [ -f "./scripts/train/run_extension_cp.sh" ]; then
            ./scripts/train/run_extension_cp.sh figures 2>/dev/null || echo "    (No extension CP figures script)"
        fi
    fi
    
    echo ">>> Figure generation complete"
else
    echo ""
    echo ">>> Skipping figure generation (--skip-figures)"
fi

# ============================================================================
# PHASE 6: VALIDATE & AGGREGATE EMISSIONS
# ============================================================================
echo ""
echo "========================================================================"
echo "PHASE 6: VALIDATE & AGGREGATE EMISSIONS"
echo "========================================================================"

echo ">>> Validating emissions data in all outputs"
python scripts/figures/validate_emissions.py

echo ""
echo ">>> Aggregating emissions data"
python scripts/figures/aggregate_emissions.py

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "========================================================================"
echo "ALL COMPLETE"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Vision checkpoints: checkpoints/vision/**/*.pt"
echo "  - Extension 2 checkpoints: checkpoints/extension2/*.pt"
echo "  - Extension CP checkpoints: checkpoints/extension_cp/*.pt"
echo "  - Language results: results/language/*.json"
echo "  - Emissions summary: results/emissions_summary.json"
echo "  - Figures: Report/figures/"
echo ""
echo "Next steps:"
echo "  1. Review results/emissions_summary.json for report Table 1"
echo "  2. Check Report/figures/ for generated figures"
echo "  3. Update Report/sections/06_fact_discussion.tex with emissions data"
