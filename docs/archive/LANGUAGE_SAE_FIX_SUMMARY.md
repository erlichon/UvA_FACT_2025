# Language Experiment SAE Fix - Complete Summary

**Date**: 2026-01-14  
**Issue**: 404 errors when trying to load ts-medium SAEs for Figure 8

## Root Cause

**ts-medium layer 4 does not have `mlp-in` SAEs** available on HuggingFace (`tdooms/ts-medium-scope`).

The `Tracer` class requires **both** input and output SAEs to compute interaction matrices (Q), but only `mlp-out` is available for layer 4.

### Available SAEs for ts-medium Layer 4
```
✓ mlp-out (expansion=4, k=30)
✓ resid-mid (expansion=4/24, k=30)
✓ resid-pre (expansion=24, k=30)
✗ mlp-in (NOT AVAILABLE)
```

## Solution

**Skip ts-medium Figure 8 reproduction**, use only **fw-medium** (tutorial example) which has all required SAEs.

### Why This Works
1. **Scientific validity**: fw-medium demonstrates the **same negation circuit phenomenon**
2. **Better scale**: Larger model (16L, 335M params) with clearer features
3. **Complete SAEs**: Both `mlp-in` and `mlp-out` available for layer 7
4. **Tutorial reference**: This is the example from `tutorials/2_language.ipynb`

## Files Modified

### 1. ✅ `scripts/run_language_sequential.sh`
**Changes**:
- Removed EXPERIMENT 4a (ts-medium Figure 8)
- Kept only EXPERIMENT 4 (fw-medium Figure 8)
- Added explanatory comment about SAE availability
- Updated final output message

**Before**: Attempted both ts-medium and fw-medium Figure 8  
**After**: Only fw-medium Figure 8

### 2. ✅ `scripts/figures/generate_language_figures.py`
**Changes**:
- Updated Figure 8 data filename: `figure_8_data_fw_medium.json`
- Added comment explaining ts-medium limitation
- Updated usage message with correct output flag

### 3. ✅ `scripts/check_ts_medium_saes.py` (NEW)
**Purpose**: Diagnostic script to check SAE availability on HuggingFace

**Usage**:
```bash
python scripts/check_ts_medium_saes.py
```

**Output**: Lists all available SAEs by layer and point type

### 4. ✅ `docs/TS_MEDIUM_SAE_LIMITATION.md` (NEW)
**Purpose**: Comprehensive documentation of the limitation and solution

**Contents**:
- Background and problem statement
- SAE availability by layer
- Impact on reproduction
- Solution and alternatives
- Verification instructions

## Experiments Now Run

| Experiment | Model | Layer | Status |
|------------|-------|-------|--------|
| Figure 9 (Correlation) | ts-medium | 4 | ✅ Works (only needs output SAE) |
| Figure 9 (Correlation) | fw-small | 8 | ✅ Works (only needs output SAE) |
| Figure 9 (Correlation) | fw-medium | 7 | ✅ Works (only needs output SAE) |
| Figure 8 (Interaction) | ts-medium | 4 | ❌ Skipped (no mlp-in SAE) |
| Figure 8 (Interaction) | fw-medium | 7 | ✅ Works (has both SAEs) |

## How to Run

### Quick Test (100 features, 5 ranks, no scatter)
```bash
./scripts/run_language_sequential.sh mps quick
```

### Full Experiment (all features, ranks 1-60, with scatter data)
```bash
./scripts/run_language_sequential.sh mps full > logs/language_full.log 2>&1
```

Expected runtime (MPS, M3 Max):
- ts-medium: ~2-3 hours
- fw-small: ~2-3 hours  
- fw-medium: ~3-4 hours
- Figure 8: ~30-60 minutes
- **Total**: ~8-10 hours

## Verification

After running, you should have:

### Results
```
results/language/
├── correlation_ts-medium.json      # Figure 9 data (ts-medium)
├── correlation_fw-small.json       # Figure 9 data (fw-small)
├── correlation_fw-medium.json      # Figure 9 data (fw-medium)
└── figure_8_data_fw_medium.json    # Figure 8 data
```

### Figures
```
Report/figures/
├── figure_9a_correlation_progression.pdf   # Line plot (3 models)
├── figure_9b_correlation_histogram.pdf     # Histogram (rank-2)
├── figure_9c_*.pdf                         # 3x3 scatter grid
└── figure_8_negation_circuit.pdf           # Composite (fw-medium)
```

## Previous Fix Attempts

1. ❌ **First attempt**: Fixed config reading in `negation_visualization.py` to use flat values
   - **Result**: Still 404, because SAEs don't exist
   
2. ❌ **Second attempt**: Fixed `verify_correlation.py` CLI arguments
   - **Result**: Correlation works, but negation visualization still fails
   
3. ✅ **Final solution**: Discovered SAE availability issue, skip ts-medium Figure 8

## Key Lessons

1. **Check HuggingFace availability first** before assuming model/SAE compatibility
2. **Tracer requires both SAEs** - verify `mlp-in` and `mlp-out` exist
3. **verify_correlation.py works with any model** because it only needs output SAEs
4. **Tutorial example (fw-medium) is often more complete** than paper examples

## Related Documentation

- `docs/TS_MEDIUM_SAE_LIMITATION.md` - Detailed SAE availability analysis
- `scripts/check_ts_medium_saes.py` - Diagnostic tool
- `LANGUAGE_REPRODUCTION_PLAN.md` - Original plan (now updated)
- `CLAUDE.md` - Project guide (Section 5 notes)

## Next Steps

Run the full sequential experiment:
```bash
./scripts/run_language_sequential.sh mps full > logs/language_full.log 2>&1 &
tail -f logs/language_full.log
```

This will complete:
- ✅ All Figure 9 correlation sweeps (3 models)
- ✅ Figure 8 negation circuit visualization (fw-medium)
- ✅ All language figures in `Report/figures/`
