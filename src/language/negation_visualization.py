"""
Figure 8 Reproduction: Sentiment Negation Circuit Visualization.

This module implements the three panels of Figure 8 from the paper:
- 8A: Interaction submatrix (top 15 interactions)
- 8B: Feature projections onto eigenvectors + meaningful directions
- 8C: Activation vs rank-2 approximation scatter

Paper Description:
"The sentiment negation circuit that computes the not-good and not-bad output features.
A) The interaction submatrix containing the top 15 interactions.
B) The projection of top interacting features onto the top eigenvectors using cosine similarity.
   Clusters coincide with the projection of meaningful directions such as the difference in
   'bad' vs 'good' token unembeddings and the MLP's input activations for '[BOS] not'.
C) The not-good feature activation compared to its approximation by the top two eigenvectors."

Key Features (fw-medium, layer 7, expansion 8):
- Feature 3834: "not-good" (fires on "not lost", "no interference")
- Feature 751: "not-bad" (fires on "not free", "little relief")

Usage:
    python src/language/negation_visualization.py --config configs/language_negation_fw.yaml
"""

import sys
from pathlib import Path
import argparse
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Suppress warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from language.transformer import Transformer
from language.utils import Sight
from sae.sae import SAE
from sae.tracer import Tracer

from src.utils import (
    get_device,
    load_config,
    track_emissions,
    safe_eigh,
    setup_mps_fallbacks,
    is_mps_device,
)
from src.language.interaction_utils import (
    get_interaction_eigenpairs,
    predict_activation_from_eigenpairs,
)
from src.language.verify_correlation import (
    create_validation_dataloader,
    pearson_correlation,
)


@dataclass
class Figure8Data:
    """Container for all Figure 8 data."""
    # Panel A: Interaction submatrix
    Q_submatrix: torch.Tensor
    submatrix_feature_indices: List[int]
    
    # Panel B: Eigenvector projections
    feature_projections: Dict[int, Tuple[float, float]]
    meaningful_directions: Dict[str, Tuple[float, float]]
    eigenvector_1: torch.Tensor
    eigenvector_2: torch.Tensor
    
    # Panel C: Scatter data
    z_true: torch.Tensor
    z_pred_rank2: torch.Tensor
    correlation: float
    
    # Metadata
    output_feature_idx: int
    layer: int
    model_name: str


def compute_figure_8a_submatrix(
    tracer: Tracer,
    feat_idx: int,
    top_k: int = 15,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute the interaction submatrix for Figure 8A.
    
    Finds the top-k strongest interactions in the Q matrix and returns
    the submatrix containing those features.
    
    Memory-efficient approach: Compute Q in d_model space, then compute
    interaction strengths with SAE features to avoid OOM.
    
    Args:
        tracer: Tracer object with loaded SAEs
        feat_idx: Output feature index (e.g., 3834 for "not-good")
        top_k: Number of top interacting features to include
    
    Returns:
        (Q_submatrix, feature_indices) - The submatrix and the feature indices it contains
    """
    # Get Q matrix in d_model space (NOT projected to avoid OOM)
    Q = tracer.q(feat_idx, project=False).float().cpu()  # [d_model, d_model]
    
    # Get SAE input decoder directions
    inp_latents = tracer.inp_latents.float().cpu()  # [d_model, n_features]
    
    # Compute interaction strength for each SAE feature pair efficiently
    # For feature i and j: strength = v_i^T Q v_j where v_i, v_j are SAE decoder directions
    # We compute this for all pairs by doing: V^T Q V where V = [v_1, v_2, ..., v_n]
    # But this is still expensive, so we do it in chunks
    
    n_features = inp_latents.shape[1]
    chunk_size = 256  # Process features in chunks to avoid OOM
    
    print(f"  Computing interaction strengths for {n_features} features (chunked)...")
    
    # Compute Q @ V first (can be done efficiently)
    Q_V = torch.mm(Q, inp_latents)  # [d_model, n_features]
    
    # Now compute V^T @ (Q @ V) in chunks
    interaction_strengths = []
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        chunk_latents = inp_latents[:, start:end]  # [d_model, chunk_size]
        chunk_interactions = torch.mm(chunk_latents.T, Q_V)  # [chunk_size, n_features]
        interaction_strengths.append(chunk_interactions)
    
    # Concatenate all chunks
    interaction_matrix = torch.cat(interaction_strengths, dim=0)  # [n_features, n_features]
    
    # Symmetrize
    interaction_matrix = 0.5 * (interaction_matrix + interaction_matrix.T)
    
    # Find top-k interactions by absolute value
    # Look at upper triangle to avoid counting symmetric pairs twice
    n = interaction_matrix.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1)
    triu_values = interaction_matrix[triu_indices[0], triu_indices[1]]
    
    # Get top interactions
    top_interaction_indices = triu_values.abs().topk(min(top_k * 3, len(triu_values))).indices
    
    # Collect unique features from top interactions
    unique_features = set()
    for idx in top_interaction_indices:
        i = triu_indices[0][idx].item()
        j = triu_indices[1][idx].item()
        unique_features.add(i)
        unique_features.add(j)
        if len(unique_features) >= top_k:
            break
    
    # Sort and limit to top_k
    feature_indices = sorted(list(unique_features))[:top_k]
    
    # Extract submatrix
    idx_tensor = torch.tensor(feature_indices)
    Q_submatrix = interaction_matrix[idx_tensor][:, idx_tensor]
    
    print(f"  Selected features: {feature_indices}")
    
    return Q_submatrix, feature_indices


def compute_figure_8b_projections(
    model: Transformer,
    tracer: Tracer,
    feat_idx: int,
    top_features: List[int],
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, Tuple[float, float]], torch.Tensor, torch.Tensor]:
    """
    Compute projections for Figure 8B.
    
    Projects SAE input features onto top 2 eigenvectors, plus meaningful directions:
    - "bad" - "good" token unembedding
    - "[BOS] not" MLP input activation
    
    Mathematically exact: Computes full projected Q matrix in CPU memory (~268MB for fw-medium).
    This is acceptable for modern systems.
    
    Args:
        model: Transformer model
        tracer: Tracer with loaded SAEs
        feat_idx: Output feature index
        top_features: List of SAE input feature indices to project
    
    Returns:
        (feature_projections, meaningful_directions, v1, v2)
    """
    # Get Q matrix in d_model space
    Q = tracer.q(feat_idx, project=False).float().cpu()  # [d_model, d_model]
    
    # Get SAE input latents (decoder directions)
    inp_latents = tracer.inp_latents.float().cpu()  # [d_model, n_features]
    
    n_features = inp_latents.shape[1]
    print(f"  Computing projected Q matrix ({n_features}×{n_features}, ~{n_features**2*4/(1024**2):.0f}MB)...")
    print(f"  This may take 1-2 minutes but is mathematically exact...")
    
    # Compute V^T @ Q @ V in chunks to manage memory efficiently
    # Result will be exact, we're just chunking the computation
    chunk_size = 512  # Process 512 features at a time
    
    # First compute Q @ V (can do all at once, result is [d_model, n_features])
    Q_V = torch.mm(Q, inp_latents)  # [d_model, n_features]
    
    # Now compute V^T @ Q_V in chunks
    Q_projected_chunks = []
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        chunk_latents = inp_latents[:, start:end]  # [d_model, chunk_size]
        chunk_result = torch.mm(chunk_latents.T, Q_V)  # [chunk_size, n_features]
        Q_projected_chunks.append(chunk_result)
        
        if (start // chunk_size) % 4 == 0:
            print(f"    Processed {end}/{n_features} features...")
    
    # Concatenate chunks to get full projected Q
    Q_projected = torch.cat(Q_projected_chunks, dim=0)  # [n_features, n_features]
    Q_projected = 0.5 * (Q_projected + Q_projected.T)  # Symmetrize
    
    del Q_projected_chunks, Q_V  # Free memory
    
    print(f"  Computing eigendecomposition...")
    # Compute top 2 eigenvectors using torch.lobpcg (for large sparse-ish matrices)
    # Or just use eigh since we have the full matrix now
    eigenvalues, eigenvectors = safe_eigh(Q_projected)
    
    # Sort by magnitude (largest first)
    sort_indices = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvalues_sorted = eigenvalues[sort_indices]
    eigenvectors_sorted = eigenvectors[:, sort_indices]
    
    v1 = eigenvectors_sorted[:, 0]  # Top eigenvector in SAE latent space
    v2 = eigenvectors_sorted[:, 1]  # Second eigenvector
    
    print(f"  Top 2 eigenvalues: λ1={eigenvalues_sorted[0]:.4f}, λ2={eigenvalues_sorted[1]:.4f}")
    
    # v1, v2 are in SAE latent space [n_features]
    # Project features onto these eigenvectors
    feature_projections = {}
    for feat in top_features:
        proj_v1 = v1[feat].item()
        proj_v2 = v2[feat].item()
        feature_projections[feat] = (proj_v1, proj_v2)
    
    # Project eigenvectors back to d_model space for comparison with meaningful directions
    v1_dmodel = torch.mv(inp_latents, v1)  # [d_model]
    v2_dmodel = torch.mv(inp_latents, v2)  # [d_model]
    v1_dmodel = v1_dmodel / v1_dmodel.norm()
    v2_dmodel = v2_dmodel / v2_dmodel.norm()
    
    # Compute meaningful directions (project onto v1, v2 in d_model space)
    meaningful_directions = {}
    
    # 1. "bad" - "good" token unembedding direction
    try:
        tokenizer = model.tokenizer
        good_tokens = tokenizer("good", add_special_tokens=False).input_ids
        bad_tokens = tokenizer("bad", add_special_tokens=False).input_ids
        
        if good_tokens and bad_tokens:
            good_id = good_tokens[0]
            bad_id = bad_tokens[0]
            
            # Get unembedding vectors (in d_model space)
            good_unembed = model.unembed.weight[:, good_id].float().cpu()
            bad_unembed = model.unembed.weight[:, bad_id].float().cpu()
            sentiment_dir = bad_unembed - good_unembed
            sentiment_dir = sentiment_dir / sentiment_dir.norm()
            
            # Project onto v1_dmodel, v2_dmodel (which are in d_model space)
            proj_v1 = F.cosine_similarity(sentiment_dir.unsqueeze(0), v1_dmodel.unsqueeze(0)).item()
            proj_v2 = F.cosine_similarity(sentiment_dir.unsqueeze(0), v2_dmodel.unsqueeze(0)).item()
            meaningful_directions["bad-good"] = (proj_v1, proj_v2)
    except Exception as e:
        print(f"  Warning: Could not compute 'bad-good' direction: {e}")
    
    # 2. "[BOS] not" MLP input activation
    try:
        sight = Sight(model)
        device = next(model.parameters()).device
        
        # Tokenize "[BOS] not" 
        tokens = tokenizer("[BOS] not", return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            with sight.trace(tokens, validate=False, scan=False):
                not_activation = sight["mlp-in", tracer.layer].save()
        
        # Get activation at the "not" position (last token)
        not_dir = not_activation[0, -1].float().cpu()
        not_dir = not_dir / not_dir.norm()
        
        # Project onto v1_dmodel, v2_dmodel (which are in d_model space)
        proj_v1 = F.cosine_similarity(not_dir.unsqueeze(0), v1_dmodel.unsqueeze(0)).item()
        proj_v2 = F.cosine_similarity(not_dir.unsqueeze(0), v2_dmodel.unsqueeze(0)).item()
        meaningful_directions["[BOS] not"] = (proj_v1, proj_v2)
    except Exception as e:
        print(f"  Warning: Could not compute '[BOS] not' direction: {e}")
    
    return feature_projections, meaningful_directions, v1_dmodel, v2_dmodel


def compute_figure_8c_scatter(
    model: Transformer,
    sae_out: SAE,
    layer: int,
    feat_idx: int,
    dataloader,
    device: str,
    max_batches: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute true vs predicted activations for Figure 8C.
    
    Reuses existing functions from interaction_utils.py for DRY compliance.
    
    Args:
        model: Transformer model
        sae_out: Output SAE
        layer: Layer index
        feat_idx: Feature index (3834 for "not-good")
        dataloader: Validation dataloader
        device: Device for computation
        max_batches: Maximum batches to process
    
    Returns:
        (z_true, z_pred_rank2, correlation)
    """
    sight = Sight(model)
    sae_out_device = sae_out.to(device)
    
    # Accumulate activations
    all_mlp_in = []
    all_z_true = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Figure 8C", total=min(max_batches, len(dataloader)))):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with sight.trace(batch, validate=False, scan=False):
                mlp_in = sight["mlp-in", layer].save()
                mlp_out = sight["mlp-out", layer].save()
            
            mlp_in_flat = mlp_in.flatten(0, 1).float()
            mlp_out_flat = mlp_out.flatten(0, 1).float()
            
            z_true = sae_out_device.encode(mlp_out_flat)[:, feat_idx]
            
            all_mlp_in.append(mlp_in_flat.cpu())
            all_z_true.append(z_true.cpu())
    
    mlp_in_all = torch.cat(all_mlp_in, dim=0)
    z_true_all = torch.cat(all_z_true, dim=0)
    
    # Filter for active samples
    active_mask = z_true_all > 0
    x_active = mlp_in_all[active_mask]
    z_active = z_true_all[active_mask]
    
    print(f"  Found {active_mask.sum().item()} active samples for feature {feat_idx}")
    
    # Get eigenpairs (reuse existing function)
    # Use ENCODER direction as per paper's Tracer (use_encoder=True by default)
    out_direction = sae_out_device.w_enc.weight[feat_idx, :].cpu()
    eigenpairs = get_interaction_eigenpairs(
        model=model,
        layer=layer,
        feat_idx=feat_idx,
        out_direction=out_direction,
        device="cpu",
    )
    
    # Rank-2 prediction
    eigenvalues_2 = eigenpairs.eigenvalues[:2]
    eigenvectors_2 = eigenpairs.eigenvectors[:, :2]
    
    z_pred = predict_activation_from_eigenpairs(x_active, eigenvalues_2, eigenvectors_2)
    
    # Compute correlation
    corr = pearson_correlation(z_active, z_pred)
    
    return z_active, z_pred, corr


def generate_figure_8_data(
    model: Transformer,
    tracer: Tracer,
    sae_out: SAE,
    layer: int,
    feat_idx: int,
    dataloader,
    device: str,
    top_k_interactions: int = 15,
    max_batches: int = 50,
) -> Figure8Data:
    """
    Generate all data needed for Figure 8.
    
    Args:
        model: Transformer model
        tracer: Tracer with loaded SAEs
        sae_out: Output SAE
        layer: Layer index
        feat_idx: Output feature index (3834 for "not-good")
        dataloader: Validation dataloader
        device: Device for computation
        top_k_interactions: Number of top interactions for Panel A
        max_batches: Maximum batches for Panel C
    
    Returns:
        Figure8Data containing all panel data
    """
    print(f"\nGenerating Figure 8 data for feature {feat_idx}...")
    
    # Panel A: Interaction submatrix
    print("  Computing Panel A (interaction submatrix)...")
    Q_submatrix, submatrix_features = compute_figure_8a_submatrix(
        tracer, feat_idx, top_k=top_k_interactions
    )
    
    # Panel B: Eigenvector projections
    print("  Computing Panel B (eigenvector projections)...")
    feature_projections, meaningful_directions, v1, v2 = compute_figure_8b_projections(
        model, tracer, feat_idx, submatrix_features
    )
    
    # Panel C: Activation vs approximation
    print("  Computing Panel C (activation scatter)...")
    z_true, z_pred, corr = compute_figure_8c_scatter(
        model, sae_out, layer, feat_idx, dataloader, device, max_batches
    )
    
    return Figure8Data(
        Q_submatrix=Q_submatrix,
        submatrix_feature_indices=submatrix_features,
        feature_projections=feature_projections,
        meaningful_directions=meaningful_directions,
        eigenvector_1=v1,
        eigenvector_2=v2,
        z_true=z_true,
        z_pred_rank2=z_pred,
        correlation=corr,
        output_feature_idx=feat_idx,
        layer=layer,
        model_name=model.config.repo,
    )


def save_figure_8_data(data: Figure8Data, output_path: Path):
    """Save Figure 8 data to JSON."""
    results = {
        "output_feature_idx": data.output_feature_idx,
        "layer": data.layer,
        "model_name": data.model_name,
        "panel_a": {
            "Q_submatrix": data.Q_submatrix.tolist(),
            "feature_indices": data.submatrix_feature_indices,
        },
        "panel_b": {
            "feature_projections": {str(k): list(v) for k, v in data.feature_projections.items()},
            "meaningful_directions": {k: list(v) for k, v in data.meaningful_directions.items()},
        },
        "panel_c": {
            "z_true": data.z_true.tolist(),
            "z_pred_rank2": data.z_pred_rank2.tolist(),
            "correlation": data.correlation,
            "n_samples": len(data.z_true),
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFigure 8 data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 8 (Sentiment Negation Circuit)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/figure_8_data.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--feature", type=int, default=3834, 
                        help="Output feature index (default: 3834 'not-good')")
    parser.add_argument("--top-k", type=int, default=15, 
                        help="Number of top interactions for Panel A")
    parser.add_argument("--n-samples", type=int, default=2000, 
                        help="Number of validation samples")
    parser.add_argument("--max-batches", type=int, default=50, 
                        help="Maximum batches for Panel C")
    args = parser.parse_args()
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
    
    # Load config
    config = load_config(args.config)
    
    # Run with emissions tracking
    with track_emissions("fact-bilinear") as tracker:
        # Load model
        model_name = config.get("model", {}).get("pretrained", "tdooms/fw-medium")
        print(f"\nLoading model: {model_name}")
        model = Transformer.from_pretrained(model_name, device=device)
        
        # SAE configuration
        sae_config = config.get("sae", {})
        layer = sae_config.get("layer", 7)
        
        # Get expansion and k - support both flat and nested config formats
        default_expansion = sae_config.get("expansion", 8)  # Flat format
        default_k = sae_config.get("k", 30)  # Flat format
        point_name = sae_config.get("point", "mlp-out")  # Flat format point name
        
        inp_config = sae_config.get("input", {
            "name": "mlp-in",
            "expansion": default_expansion,
            "k": default_k
        })
        out_config = sae_config.get("output", {
            "name": point_name,
            "expansion": default_expansion,
            "k": default_k
        })
        
        # Ensure all required keys are set
        inp_config.setdefault("name", "mlp-in")
        inp_config.setdefault("expansion", default_expansion)
        inp_config.setdefault("k", default_k)
        
        out_config.setdefault("name", point_name)
        out_config.setdefault("expansion", default_expansion)
        out_config.setdefault("k", default_k)
        
        # Create Tracer (loads SAEs automatically)
        print(f"\nCreating Tracer for layer {layer}...")
        print(f"  Input SAE: {inp_config['name']}, expansion={inp_config['expansion']}, k={inp_config['k']}")
        print(f"  Output SAE: {out_config['name']}, expansion={out_config['expansion']}, k={out_config['k']}")
        tracer = Tracer(model, layer, inp=inp_config, out=out_config, device=device)
        
        # Load output SAE separately for Panel C
        repo = f"{model.config.repo}-scope"
        sae_out = SAE.from_pretrained(
            repo,
            point=(out_config["name"], layer),
            expansion=out_config["expansion"],
            k=out_config["k"],
        ).to(device)
        
        # Create validation dataloader
        dataloader = create_validation_dataloader(
            model.tokenizer, config, device,
            n_samples=args.n_samples,
            batch_size=32,
        )
        
        # Generate Figure 8 data
        figure_data = generate_figure_8_data(
            model=model,
            tracer=tracer,
            sae_out=sae_out,
            layer=layer,
            feat_idx=args.feature,
            dataloader=dataloader,
            device=device,
            top_k_interactions=args.top_k,
            max_batches=args.max_batches,
        )
        
        # Save results
        save_figure_8_data(figure_data, Path(args.output))
        
        # Print summary
        print(f"\n{'='*60}")
        print("FIGURE 8 GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Feature: {args.feature}")
        print(f"Panel A: {len(figure_data.submatrix_feature_indices)} features in submatrix")
        print(f"Panel B: {len(figure_data.feature_projections)} feature projections")
        print(f"         {len(figure_data.meaningful_directions)} meaningful directions")
        print(f"Panel C: {len(figure_data.z_true)} scatter points, correlation={figure_data.correlation:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
