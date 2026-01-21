# Language Sequential Script Fix

**Date**: 2026-01-14  
**Issue**: Script failing to load ts-medium SAEs with 404 error

## Problem

The `negation_visualization.py` script was trying to load SAEs with `expansion=8` for ts-medium, but ts-medium only has SAEs with `expansion=4` available on HuggingFace.

**Error message**:
```
404 Client Error: Not Found for url: 
https://huggingface.co/tdooms/ts-medium-scope/resolve/main/4-mlp-in-x8-k30/config.json
```

## Root Cause

The script had hardcoded defaults for SAE configuration:
```python
inp_config = sae_config.get("input", {"expansion": 8, "k": 30})  # Wrong default!
out_config = sae_config.get("output", {"expansion": 8, "k": 30})  # Wrong default!
```

The configs use a **flat structure**:
```yaml
sae:
  layer: 4
  expansion: 4  # Flat, not nested under input/output
  k: 30
```

But the script expected a **nested structure**:
```yaml
sae:
  layer: 4
  input:
    expansion: 4
    k: 30
  output:
    expansion: 4
    k: 30
```

When flat values were present, they were ignored and the hardcoded defaults (expansion=8) were used.

## Solution

Updated `src/language/negation_visualization.py` to:

1. **Read flat values first** as defaults:
```python
default_expansion = sae_config.get("expansion", 8)  # Read from flat config
default_k = sae_config.get("k", 30)
point_name = sae_config.get("point", "mlp-out")
```

2. **Use flat values for nested configs** if not explicitly set:
```python
inp_config = sae_config.get("input", {
    "name": "mlp-in",
    "expansion": default_expansion,  # Use flat value
    "k": default_k
})
```

3. **Ensure all required keys** are present:
```python
inp_config.setdefault("expansion", default_expansion)
out_config.setdefault("expansion", default_expansion)
```

4. **Added debug logging**:
```python
print(f"  Input SAE: {inp_config['name']}, expansion={inp_config['expansion']}, k={inp_config['k']}")
print(f"  Output SAE: {out_config['name']}, expansion={out_config['expansion']}, k={out_config['k']}")
```

## Verification

```bash
# Config values confirmed:
ts-medium:  layer=4, expansion=4, k=30  ✅
fw-medium:  layer=7, expansion=8, k=30  ✅
```

## Files Modified

1. ✅ `src/language/negation_visualization.py` - Fixed config reading logic
2. ✅ `src/language/verify_correlation.py` - Already fixed in previous update

## Ready to Run

```bash
./scripts/run_language_sequential.sh mps full
```

This should now:
- ✅ Load ts-medium with expansion=4 (correct)
- ✅ Load fw-medium with expansion=8 (correct)
- ✅ Run both Figure 8 experiments successfully
- ✅ Complete full correlation sweep for Figure 9
