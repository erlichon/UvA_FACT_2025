#!/usr/bin/env python3
"""Check what SAEs are available for ts-medium on HuggingFace."""

from huggingface_hub import list_repo_files, hf_hub_url
import sys
from pathlib import Path

# Add original code to path
_ORIG_PATH = Path(__file__).parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

def main():
    repo_id = "tdooms/ts-medium-scope"
    
    print(f"Checking SAEs available in {repo_id}...")
    print("=" * 60)
    
    try:
        files = list_repo_files(repo_id)
        
        # Filter for config.json files
        sae_configs = [f for f in files if f.endswith("config.json")]
        
        print(f"\nFound {len(sae_configs)} SAE configurations:")
        print("")
        
        # Parse and organize by layer
        saes_by_layer = {}
        for config_path in sorted(sae_configs):
            parts = config_path.split("/")
            if len(parts) >= 2:
                sae_name = parts[0]
                # Parse SAE name (e.g., "4-mlp-out-x4-k30")
                try:
                    layer = int(sae_name.split("-")[0])
                    point_name = "-".join(sae_name.split("-")[1:-2])  # e.g., "mlp-out"
                    expansion_str = sae_name.split("-")[-2]  # e.g., "x4"
                    k_str = sae_name.split("-")[-1]  # e.g., "k30"
                    
                    expansion = int(expansion_str.replace("x", ""))
                    k = int(k_str.replace("k", ""))
                    
                    if layer not in saes_by_layer:
                        saes_by_layer[layer] = []
                    saes_by_layer[layer].append({
                        "point": point_name,
                        "expansion": expansion,
                        "k": k,
                        "name": sae_name,
                    })
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not parse {config_path}: {e}")
        
        # Display organized results
        for layer in sorted(saes_by_layer.keys()):
            print(f"Layer {layer}:")
            for sae in saes_by_layer[layer]:
                print(f"  - {sae['point']}: expansion={sae['expansion']}, k={sae['k']}")
            print("")
        
        # Check specifically for layer 4 (paper's layer)
        print("=" * 60)
        print("Layer 4 (Paper's Figure 8 layer) SAEs:")
        if 4 in saes_by_layer:
            for sae in saes_by_layer[4]:
                print(f"  ✓ {sae['point']}: expansion={sae['expansion']}, k={sae['k']}")
            
            # Check if both mlp-in and mlp-out exist
            points = [sae['point'] for sae in saes_by_layer[4]]
            if "mlp-in" in points and "mlp-out" in points:
                print("\n✅ GOOD: Both mlp-in and mlp-out SAEs exist for layer 4!")
            elif "mlp-out" in points:
                print("\n⚠️  WARNING: Only mlp-out exists (no mlp-in)")
                print("   Figure 8 reproduction may not be possible for ts-medium")
            else:
                print("\n❌ ERROR: No standard SAEs found for layer 4")
        else:
            print("  ❌ No SAEs found for layer 4")
        
    except Exception as e:
        print(f"Error accessing repository: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
