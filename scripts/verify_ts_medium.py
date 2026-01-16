"""
Verify if tdooms/ts-medium matches the paper's ts-tiny specifications.

Paper's ts-tiny specs (from Section 5.1 and Appendix):
- 6 layers
- ~33M parameters
- d_model = 512
- d_hidden = 2048
- Trained on TinyStories (cleaned)
"""

import sys
from pathlib import Path

# Add original code to path
_ORIG_PATH = Path(__file__).parent.parent / "bilinear-decomposition-main"
sys.path.insert(0, str(_ORIG_PATH))

from language.transformer import Transformer

def main():
    print("="*70)
    print("Verifying tdooms/ts-medium Model Specifications")
    print("="*70)
    
    # Load the model
    print("\nLoading tdooms/ts-medium from HuggingFace...")
    model = Transformer.from_pretrained("tdooms/ts-medium", device='cpu')
    
    # Extract model specs
    config = model.config
    
    # Print all config attributes for debugging
    print(f"\nConfig type: {type(config)}")
    print(f"Available config attributes:")
    for k in sorted(dir(config)):
        if not k.startswith('_'):
            try:
                val = getattr(config, k)
                if not callable(val):
                    print(f"  {k}: {val}")
            except:
                pass
    
    # Extract specs safely
    n_layer = getattr(config, 'n_layer', None)
    d_model = getattr(config, 'd_model', None)
    
    # Try different possible names for MLP dimension
    d_mlp = None
    for attr in ['d_mlp', 'd_hidden', 'n_embd', 'd_ff', 'intermediate_size']:
        if hasattr(config, attr):
            d_mlp = getattr(config, attr)
            break
    
    n_head = getattr(config, 'n_head', None)
    vocab_size = getattr(config, 'vocab_size', None)
    dataset = getattr(config, 'dataset', 'unknown')
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1e6
    
    # Print actual specs
    print("\n" + "="*70)
    print("ACTUAL: tdooms/ts-medium Specifications")
    print("="*70)
    print(f"  Layers (n_layer):        {n_layer}")
    print(f"  Hidden dim (d_model):    {d_model}")
    print(f"  MLP dim (d_mlp):         {d_mlp}")
    print(f"  Attention heads:         {n_head}")
    print(f"  Vocab size:              {vocab_size}")
    print(f"  Dataset:                 {dataset}")
    print(f"  Total parameters:        {total_params_m:.1f}M")
    
    # Print paper's expected specs
    print("\n" + "="*70)
    print("EXPECTED: Paper's ts-tiny Specifications")
    print("="*70)
    print(f"  Layers (n_layer):        6")
    print(f"  Hidden dim (d_model):    512")
    print(f"  MLP dim (d_mlp):         2048")
    print(f"  Total parameters:        ~33M")
    print(f"  Dataset:                 TinyStories (cleaned)")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    match_layers = (n_layer == 6)
    match_d_model = (d_model == 512)
    match_d_mlp = (d_mlp == 2048)
    match_params = (abs(total_params_m - 33) < 5)  # Within 5M tolerance
    match_dataset = ('tinystories' in dataset.lower() or 'ts' in dataset.lower())
    
    print(f"  Layers match (6):            {'✅ YES' if match_layers else '❌ NO'} ({n_layer})")
    print(f"  d_model match (512):         {'✅ YES' if match_d_model else '❌ NO'} ({d_model})")
    print(f"  d_mlp match (2048):          {'✅ YES' if match_d_mlp else '❌ NO'} ({d_mlp})")
    print(f"  Params match (~33M):         {'✅ YES' if match_params else '❌ NO'} ({total_params_m:.1f}M)")
    print(f"  Dataset match (TinyStories): {'✅ YES' if match_dataset else '❌ NO'} ({dataset})")
    
    # Final verdict
    all_match = match_layers and match_d_model and match_d_mlp and match_params and match_dataset
    
    print("\n" + "="*70)
    if all_match:
        print("✅ CONCLUSION: tdooms/ts-medium IS the paper's ts-tiny model!")
        print("   The naming is confusing, but specs match perfectly.")
    else:
        print("❌ CONCLUSION: tdooms/ts-medium is NOT the same as ts-tiny!")
        print("   These are different models.")
    print("="*70)
    
    return all_match

if __name__ == "__main__":
    main()
