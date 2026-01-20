# Agent Prompt: Fix Language Experiment Configuration

**Purpose**: This prompt is for a subsequent agent to fix the language experiment configuration to match the paper's setup.

---

## Context

Our language experiments (Section 5) produce results that differ dramatically from the paper:
- Negation discovery finds SAME feature for both patterns (should find different, opposing features)
- Interaction analysis shows 0% >0.75 correlation (paper: 69%)

**Root cause**: We use the wrong model (`ts-medium` instead of `fw-medium`), wrong layer (2/5 instead of 7), and wrong SAE config (expansion=4 instead of 8).

---

## Priority 1: Update Configuration Files

### Task 1.1: Update Negation Config

**File**: `/Users/itay/Documents/repos/MSc/FACT/UvA_FACT_2025/configs/language_negation.yaml`

Change from:
```yaml
model:
  pretrained: tdooms/ts-medium
sae:
  use_pretrained: true
  point: mlp-out
  layer: 2
  expansion: 4
  k: 30
```

To:
```yaml
model:
  pretrained: tdooms/fw-medium
sae:
  use_pretrained: true
  point: mlp-out
  layer: 7
  expansion: 8
  k: 32
```

### Task 1.2: Update Interaction Config

**File**: `/Users/itay/Documents/repos/MSc/FACT/UvA_FACT_2025/configs/language_interaction.yaml`

Apply same changes:
- `model.pretrained`: `tdooms/fw-medium`
- `sae.layer`: 7
- `sae.expansion`: 8
- `sae.k`: 32

### Task 1.3: Update SAE Training Config (if used)

**File**: `/Users/itay/Documents/repos/MSc/FACT/UvA_FACT_2025/configs/language_sae.yaml`

Apply same changes for consistency.

---

## Priority 2: Verify fw-medium SAE Availability

### Task 2.1: Test SAE Loading

Run this test to verify fw-medium SAEs are available:

```python
import sys
sys.path.insert(0, "bilinear-decomposition-main")
from sae.sae import SAE

# Test loading fw-medium SAE at layer 7 with expansion 8
try:
    sae = SAE.from_pretrained(
        "tdooms/fw-medium-scope",
        point=("mlp-out", 7),
        expansion=8,
        k=32
    )
    print(f"Success! SAE loaded with {sae.w_enc.out_features} features")
except Exception as e:
    print(f"Failed: {e}")
```

### Task 2.2: Handle Unavailability

If fw-medium SAEs are not available with expansion=8:
1. Check what expansions ARE available: `from huggingface_hub import list_repo_files; list_repo_files("tdooms/fw-medium-scope")`
2. Use the closest available configuration
3. Document the limitation in the report

---

## Priority 3: Update SLURM Job for GPU

### Task 3.1: Create fw-medium Job Script

**File**: `/Users/itay/Documents/repos/MSc/FACT/UvA_FACT_2025/jobs/language_fwmedium.job`

fw-medium (335M params) needs GPU and more memory:

```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/language_fwmedium_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate fact

echo "Running Language experiments with fw-medium..."

# Negation discovery
python src/language/negation_discovery.py \
    --config configs/language_negation.yaml \
    --output results/language/negation_analysis_fwmedium.json

# Interaction analysis
python src/language/interaction_analysis.py \
    --config configs/language_interaction.yaml \
    --output results/language/interaction_analysis_fwmedium.json

echo "Done!"
```

---

## Priority 4: Update Documentation

### Task 4.1: Add fw-medium Note to CLAUDE.md

Add to the Language section:

```markdown
> **NOTE**: Language experiments require `fw-medium` model (335M params) for accurate reproduction.
> The `ts-medium` model (30M params) is too small to develop the same feature structure.
> Run on Snellius GPU cluster for best results.
```

### Task 4.2: Update CONTEXT.md

Update Section 5 status to reflect:
- ts-medium results are "smaller model comparison"
- fw-medium experiments pending on GPU cluster

---

## Expected Results After Fix

With fw-medium at layer 7 with expansion=8 SAEs:

| Metric | ts-medium (Current) | fw-medium (Expected) |
|--------|--------------------|--------------------|
| not_positive_feature | 1875 | ~751 |
| not_negative_feature | 1875 | ~3834 |
| cosine_similarity | 1.0 | < 0 (negative) |
| fraction_above_075 | 0% | ~69% |
| mean_effective_rank | 305 | < 50 |

---

## Fallback Plan

If fw-medium cannot be used (memory constraints, SAE unavailability):

1. **Document limitation**: "Paper's features specific to fw-medium architecture"
2. **Report ts-medium results**: As "smaller model ablation study"
3. **Focus on methodology**: Show that eigendecomposition approach is sound
4. **Qualitative analysis**: Find ts-medium's own negation features (even if different from paper)

---

## Files to Modify

| File | Action |
|------|--------|
| `configs/language_negation.yaml` | Update model/layer/expansion |
| `configs/language_interaction.yaml` | Update model/layer/expansion |
| `configs/language_sae.yaml` | Update for consistency |
| `jobs/language_fwmedium.job` | Create new SLURM job |
| `CLAUDE.md` | Add fw-medium requirement note |
| `CONTEXT.md` | Update Section 5 status |

---

## Verification Checklist

After making changes, verify:

- [ ] fw-medium SAEs load successfully
- [ ] Negation discovery finds DIFFERENT features for each pattern
- [ ] Cosine similarity is NEGATIVE (opposing directions)
- [ ] Interaction analysis shows >50% features with >0.75 correlation
- [ ] Mean effective rank is lower (< 100)
- [ ] Results saved to separate files (*_fwmedium.json)
