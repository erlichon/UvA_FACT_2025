"""
Extension 2: Cross-Dataset Structural Robustness Evaluation

Objective: Prove that regularized Bilinear MLPs learn universal geometric shapes
(e.g., "circularity") rather than dataset-specific pixel artifacts.

This script implements a THREE-step validation:

Step 1: USPS Transfer Test (NEW)
- Train two models on MNIST: Baseline (no noise) and Regularized (noise σ=0.15)
- Evaluate both on USPS (different digit dataset, same classes 0-9)
- Success: High accuracy on USPS proves eigenvectors capture universal digit features,
  not MNIST-specific pixel artifacts

Step 2: Semantic Confusion Test
- Evaluate both MNIST-trained models on EMNIST letters 'O', 'I', 'Z', 'S', 'B'
- Success: Regularized model classifies 'O' as '0', etc., proving shape-based learning

Step 3: Subspace Geometry Metric
- Train a Bilinear MLP on EMNIST-Letters
- Extract top-10 eigenvectors for MNIST '0' and EMNIST 'O'
- Compute subspace overlap to quantify mechanism similarity
- Goal: High similarity proves universal shape learning

Usage:
    # Full pipeline
    python src/evaluate_extension2.py --full-pipeline
    
    # Just USPS transfer test with existing checkpoints
    python src/evaluate_extension2.py --usps-test --checkpoint-baseline path/to/baseline.pt
    
    # Just semantic confusion test
    python src/evaluate_extension2.py --semantic-test --checkpoint-baseline path/to/baseline.pt
    
    # Just subspace geometry
    python src/evaluate_extension2.py --subspace-test --checkpoint-regularized path/to/mnist.pt
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict, Tuple, Optional, List
import kornia

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from image.model import Model, Config

from src.data.cross_dataset import (
    load_mnist_normalized,
    load_emnist_letters_normalized,
    load_usps_normalized,
    extract_emnist_letters,
    get_emnist_letter_indices,
    LETTER_DIGIT_SIMILARITY,
    EMNIST_CLASS_NAMES,
)
from src.analysis.subspace import (
    compute_subspace_overlap,
    principal_angles,
    pairwise_class_similarity,
    semantic_similarity_score,
)
from src.analysis.spectral import effective_rank, spectral_summary
from src.utils import get_device, set_seed, setup_mps_fallbacks, is_mps_device


# ============================================================================
# Model Training Utilities
# ============================================================================

def train_mnist_model(
    device: str,
    noise_std: float = 0.0,
    weight_decay: float = 0.5,
    epochs: int = 100,
    d_hidden: int = 256,
    seed: int = 42,
    use_com_normalization: bool = True,
) -> Tuple[Model, torch.Tensor, torch.Tensor]:
    """
    Train a bilinear model on MNIST (with center-of-mass normalization).
    
    Args:
        device: Training device
        noise_std: Input noise standard deviation (0 = baseline, 0.15 = regularized)
        weight_decay: Weight decay for optimizer
        epochs: Number of training epochs
        d_hidden: Hidden dimension
        seed: Random seed
        use_com_normalization: Use center-of-mass normalized data
    
    Returns:
        (model, eigenvalues, eigenvectors)
    """
    set_seed(seed)
    
    # Load data
    if use_com_normalization:
        from src.data.cross_dataset import load_mnist_normalized
        train_data, test_data = load_mnist_normalized(device=device)
    else:
        from image.datasets import MNIST
        train_data = MNIST(train=True, device=device)
        test_data = MNIST(train=False, device=device)
    
    # Create model
    model_config = Config(
        epochs=epochs,
        d_hidden=d_hidden,
        wd=weight_decay,
        lr=1e-3,
        seed=seed,
    )
    model = Model(model_config).to(device)
    
    # Create noise transform
    transform = None
    if noise_std > 0:
        transform = kornia.augmentation.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=1.0
        )
    
    # Train
    print(f"Training MNIST model (noise_std={noise_std}, wd={weight_decay})...")
    history = model.fit(train_data, test_data, transform=transform)
    
    # Decompose
    print("Computing eigendecomposition...")
    if is_mps_device(device):
        setup_mps_fallbacks()
        eigenvalues, eigenvectors = decompose_model_mps_safe(model)
    else:
        eigenvalues, eigenvectors = model.decompose()
    
    return model, eigenvalues, eigenvectors


def train_emnist_model(
    device: str,
    noise_std: float = 0.15,
    weight_decay: float = 0.5,
    epochs: int = 100,
    d_hidden: int = 256,
    seed: int = 42,
) -> Tuple[Model, torch.Tensor, torch.Tensor]:
    """
    Train a bilinear model on EMNIST-Letters.
    
    Note: EMNIST has 26 classes (A-Z), so we need to modify the model config.
    
    Returns:
        (model, eigenvalues, eigenvectors)
    """
    set_seed(seed)
    
    # Load data
    train_data, test_data = load_emnist_letters_normalized(device=device)
    
    # Create model with 26 output classes
    model_config = Config(
        epochs=epochs,
        d_hidden=d_hidden,
        wd=weight_decay,
        lr=1e-3,
        seed=seed,
        d_output=26,  # EMNIST has 26 letter classes
    )
    model = Model(model_config).to(device)
    
    # Create noise transform
    transform = None
    if noise_std > 0:
        transform = kornia.augmentation.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=1.0
        )
    
    # Train
    print(f"Training EMNIST model (noise_std={noise_std}, wd={weight_decay})...")
    history = model.fit(train_data, test_data, transform=transform)
    
    # Decompose
    print("Computing eigendecomposition...")
    if is_mps_device(device):
        setup_mps_fallbacks()
        eigenvalues, eigenvectors = decompose_model_mps_safe(model)
    else:
        eigenvalues, eigenvectors = model.decompose()
    
    return model, eigenvalues, eigenvectors


def decompose_model_mps_safe(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """MPS-compatible eigendecomposition (same as in train.py)."""
    from einops import einsum
    
    device = next(model.parameters()).device
    
    w_u = model.w_u
    w_lr = model.w_lr[0]
    w_e = model.w_e
    
    l, r = w_lr.unbind(0)
    b = einsum(w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
    b = 0.5 * (b + b.mT)
    
    if device.type == "mps":
        b_cpu = b.cpu()
        vals, vecs = torch.linalg.eigh(b_cpu)
        vals = vals.to(device)
        vecs = vecs.to(device)
    else:
        vals, vecs = torch.linalg.eigh(b)
    
    vecs = einsum(vecs, w_e, "cls emb comp, emb inp -> cls comp inp")
    
    return vals, vecs


# ============================================================================
# Step 1: USPS Transfer Test (NEW)
# ============================================================================

def usps_transfer_test(
    model: Model,
    device: str,
    model_name: str = "model",
) -> Dict:
    """
    Test how an MNIST-trained model performs on USPS digits.
    
    USPS is a different digit dataset (US Postal Service) with the same
    10 classes (0-9). High accuracy on USPS proves the eigenvectors
    capture universal digit features, not MNIST-specific artifacts.
    
    Args:
        model: MNIST-trained bilinear model
        device: Device for computation
        model_name: Name for logging
    
    Returns:
        Dictionary with accuracy and per-class metrics
    """
    model.eval()
    
    # Load USPS test set (center-of-mass normalized)
    _, usps_test = load_usps_normalized(device=device)
    
    # Run inference
    with torch.no_grad():
        x_flat = usps_test.x.view(usps_test.x.size(0), -1)
        logits = model(x_flat)
        predictions = logits.argmax(dim=-1)
    
    # Overall accuracy
    correct = (predictions == usps_test.y).float()
    overall_accuracy = correct.mean().item()
    
    # Per-class accuracy
    per_class_accuracy = {}
    for digit in range(10):
        mask = usps_test.y == digit
        if mask.sum() > 0:
            class_acc = correct[mask].mean().item()
            per_class_accuracy[digit] = {
                'accuracy': class_acc,
                'n_samples': mask.sum().item(),
            }
    
    # Confusion matrix (simplified)
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for true_label in range(10):
        mask = usps_test.y == true_label
        preds_for_class = predictions[mask]
        for pred_label in range(10):
            confusion[true_label, pred_label] = (preds_for_class == pred_label).sum().item()
    
    results = {
        'model_name': model_name,
        'overall_accuracy': overall_accuracy,
        'n_samples': len(usps_test),
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': confusion.tolist(),
    }
    
    return results


def run_usps_transfer_comparison(
    model_baseline: Model,
    model_regularized: Model,
    device: str,
) -> Dict:
    """
    Compare USPS transfer performance between baseline and regularized models.
    
    This is Step 1 of Extension 2 validation:
    - If regularization helps learn universal digit features, the regularized
      model should perform BETTER on USPS than the baseline.
    
    Returns:
        Dictionary comparing both models' USPS performance
    """
    print("\n" + "=" * 60)
    print("STEP 1: USPS TRANSFER TEST")
    print("=" * 60)
    print("Testing if MNIST-trained models generalize to USPS digits...")
    print("(High accuracy = learned universal digit features, not MNIST artifacts)")
    
    print("\nEvaluating baseline model (no noise)...")
    baseline_results = usps_transfer_test(model_baseline, device, "baseline")
    
    print("Evaluating regularized model (with noise)...")
    regularized_results = usps_transfer_test(model_regularized, device, "regularized")
    
    # Summary
    comparison = {
        'baseline': baseline_results,
        'regularized': regularized_results,
    }
    
    print("\n" + "-" * 60)
    print("USPS TRANSFER RESULTS")
    print("-" * 60)
    print(f"{'Model':<15} {'USPS Accuracy':<15} {'Samples':<10}")
    print("-" * 60)
    print(f"{'Baseline':<15} {baseline_results['overall_accuracy']:.2%}{'':<8} {baseline_results['n_samples']}")
    print(f"{'Regularized':<15} {regularized_results['overall_accuracy']:.2%}{'':<8} {regularized_results['n_samples']}")
    
    improvement = regularized_results['overall_accuracy'] - baseline_results['overall_accuracy']
    print("-" * 60)
    print(f"Improvement: {improvement:+.2%}")
    
    # Per-class comparison
    print("\nPer-class accuracy:")
    print(f"{'Digit':<8} {'Baseline':<12} {'Regularized':<12} {'Δ':<10}")
    for digit in range(10):
        base_acc = baseline_results['per_class_accuracy'].get(digit, {}).get('accuracy', 0)
        reg_acc = regularized_results['per_class_accuracy'].get(digit, {}).get('accuracy', 0)
        delta = reg_acc - base_acc
        print(f"{digit:<8} {base_acc:.2%}{'':<5} {reg_acc:.2%}{'':<5} {delta:+.2%}")
    
    comparison['conclusion'] = {
        'baseline_accuracy': baseline_results['overall_accuracy'],
        'regularized_accuracy': regularized_results['overall_accuracy'],
        'improvement': improvement,
        'regularization_helps': improvement > 0,
    }
    
    return comparison


# ============================================================================
# Step 2: Semantic Confusion Test
# ============================================================================

def semantic_confusion_test(
    model: Model,
    device: str,
    letters: List[str] = ['O', 'I', 'Z', 'S', 'B'],
) -> Dict:
    """
    Test how an MNIST-trained model classifies EMNIST letters.
    
    The hypothesis: A regularized model that learned universal shapes
    will confidently classify 'O' as '0', 'I' as '1', etc.
    
    Args:
        model: MNIST-trained bilinear model
        device: Device for computation
        letters: Letters to test
    
    Returns:
        Dictionary with per-letter predictions and confidence scores
    """
    model.eval()
    
    # Load EMNIST and extract target letters
    _, emnist_test = load_emnist_letters_normalized(device=device)
    
    results = {}
    letter_indices = get_emnist_letter_indices(letters)
    
    for letter, letter_idx in zip(letters, letter_indices):
        # Get all samples of this letter
        mask = emnist_test.y == letter_idx
        letter_images = emnist_test.x[mask]  # [N, 1, 28, 28]
        
        if len(letter_images) == 0:
            print(f"Warning: No samples found for letter {letter}")
            continue
        
        # Run inference
        with torch.no_grad():
            # Flatten for model input
            x_flat = letter_images.view(letter_images.size(0), -1)
            logits = model(x_flat)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
        
        # Compute statistics
        expected_digit = LETTER_DIGIT_SIMILARITY.get(letter)
        
        # Count predictions per digit
        pred_counts = torch.bincount(predictions, minlength=10)
        pred_distribution = pred_counts.float() / pred_counts.sum()
        
        # Modal prediction (use CPU for mode operation on MPS)
        modal_prediction = predictions.cpu().mode().values.item()
        modal_confidence = (predictions == modal_prediction).float().mean().item()
        
        # Expected digit accuracy (if applicable)
        if expected_digit is not None:
            expected_accuracy = (predictions == expected_digit).float().mean().item()
            expected_confidence = probs[:, expected_digit].mean().item()
        else:
            expected_accuracy = None
            expected_confidence = None
        
        results[letter] = {
            'n_samples': len(letter_images),
            'expected_digit': expected_digit,
            'modal_prediction': modal_prediction,
            'modal_confidence': modal_confidence,
            'expected_accuracy': expected_accuracy,
            'expected_confidence': expected_confidence,
            'prediction_distribution': pred_distribution.cpu().tolist(),
        }
    
    # Compute aggregate score
    correct_mappings = 0
    total_mappings = 0
    for letter, info in results.items():
        if isinstance(info, dict) and info.get('expected_digit') is not None:
            total_mappings += 1
            if info['modal_prediction'] == info['expected_digit']:
                correct_mappings += 1
    
    results['aggregate'] = {
        'correct_mappings': correct_mappings,
        'total_mappings': total_mappings,
        'accuracy': correct_mappings / total_mappings if total_mappings > 0 else 0,
    }
    
    return results


def run_semantic_confusion_comparison(
    model_baseline: Model,
    model_regularized: Model,
    device: str,
    letters: List[str] = ['O', 'I', 'Z', 'S', 'B'],
) -> Dict:
    """
    Compare semantic confusion between baseline and regularized models.
    
    This is Step 2 of Extension 2 validation.
    
    Returns:
        Dictionary comparing both models' performance
    """
    print("\n" + "=" * 60)
    print("STEP 2: SEMANTIC CONFUSION TEST")
    print("=" * 60)
    print("Testing if MNIST-trained models classify EMNIST letters by shape...")
    print("(Letter 'O' should be classified as digit '0', etc.)")
    
    print("\nEvaluating baseline model (no noise)...")
    baseline_results = semantic_confusion_test(model_baseline, device, letters)
    
    print("Evaluating regularized model (with noise)...")
    regularized_results = semantic_confusion_test(model_regularized, device, letters)
    
    # Summary comparison
    comparison = {
        'baseline': baseline_results,
        'regularized': regularized_results,
        'comparison': {}
    }
    
    print("\n" + "-" * 60)
    print("SEMANTIC CONFUSION RESULTS")
    print("-" * 60)
    print(f"{'Letter':<8} {'Expected':<10} {'Baseline':<15} {'Regularized':<15}")
    print("-" * 60)
    
    for letter in letters:
        expected = LETTER_DIGIT_SIMILARITY.get(letter, '?')
        baseline_pred = baseline_results[letter]['modal_prediction']
        baseline_conf = baseline_results[letter]['modal_confidence']
        reg_pred = regularized_results[letter]['modal_prediction']
        reg_conf = regularized_results[letter]['modal_confidence']
        
        baseline_str = f"{baseline_pred} ({baseline_conf:.2f})"
        reg_str = f"{reg_pred} ({reg_conf:.2f})"
        
        baseline_correct = '✓' if baseline_pred == expected else '✗'
        reg_correct = '✓' if reg_pred == expected else '✗'
        
        print(f"{letter:<8} {expected:<10} {baseline_str:<12} {baseline_correct}  {reg_str:<12} {reg_correct}")
        
        comparison['comparison'][letter] = {
            'baseline_correct': baseline_pred == expected,
            'regularized_correct': reg_pred == expected,
        }
    
    print("-" * 60)
    print(f"Baseline accuracy: {baseline_results['aggregate']['accuracy']:.2%}")
    print(f"Regularized accuracy: {regularized_results['aggregate']['accuracy']:.2%}")
    
    comparison['conclusion'] = {
        'baseline_accuracy': baseline_results['aggregate']['accuracy'],
        'regularized_accuracy': regularized_results['aggregate']['accuracy'],
        'improvement': (regularized_results['aggregate']['accuracy'] - 
                       baseline_results['aggregate']['accuracy']),
    }
    
    return comparison


# ============================================================================
# Step 3: Subspace Geometry Metric
# ============================================================================

def run_subspace_geometry_test(
    mnist_eigenvalues: torch.Tensor,
    mnist_eigenvectors: torch.Tensor,
    emnist_eigenvalues: torch.Tensor,
    emnist_eigenvectors: torch.Tensor,
    k: int = 10,
) -> Dict:
    """
    Compare subspace geometry between MNIST and EMNIST models.
    
    This tests whether the learned mechanisms are truly universal
    by measuring eigenvector subspace overlap.
    
    Args:
        mnist_eigenvalues: MNIST model eigenvalues [10, n_components]
        mnist_eigenvectors: MNIST model eigenvectors [10, n_components, 784]
        emnist_eigenvalues: EMNIST model eigenvalues [26, n_components]
        emnist_eigenvectors: EMNIST model eigenvectors [26, n_components, 784]
        k: Number of top eigenvectors to compare
    
    Returns:
        Dictionary with subspace overlap metrics
    """
    print("\n" + "=" * 60)
    print("STEP 3: SUBSPACE GEOMETRY METRIC")
    print("=" * 60)
    print("Comparing eigenvector subspaces between MNIST and EMNIST models...")
    print("(High overlap for similar shapes = universal mechanism learning)")
    
    results = {}
    
    # Sort eigenvectors by eigenvalue magnitude
    def get_sorted_vecs(eigenvalues, eigenvectors):
        """Sort eigenvectors by eigenvalue magnitude."""
        n_classes = eigenvalues.shape[0]
        sorted_vecs = []
        for c in range(n_classes):
            _, indices = eigenvalues[c].abs().sort(descending=True)
            sorted_vecs.append(eigenvectors[c, indices])
        return torch.stack(sorted_vecs)
    
    mnist_vecs_sorted = get_sorted_vecs(mnist_eigenvalues, mnist_eigenvectors)
    emnist_vecs_sorted = get_sorted_vecs(emnist_eigenvalues, emnist_eigenvectors)
    
    # Test specific pairs (digit, letter, mnist_idx, emnist_idx)
    test_pairs = [
        ('0', 'O', 0, 14),   # Zero vs O
        ('1', 'I', 1, 8),    # One vs I
        ('2', 'Z', 2, 25),   # Two vs Z
        ('5', 'S', 5, 18),   # Five vs S
        ('8', 'B', 8, 1),    # Eight vs B
    ]
    
    print(f"\nComparing top-{k} eigenvector subspaces:")
    print("-" * 60)
    print(f"{'MNIST':<8} {'EMNIST':<8} {'Mean Cos':<12} {'Grassmann':<12} {'Projection':<12}")
    print("-" * 60)
    
    pair_results = {}
    for mnist_char, emnist_char, mnist_idx, emnist_idx in test_pairs:
        vecs_mnist = mnist_vecs_sorted[mnist_idx, :k]  # [k, 784]
        vecs_emnist = emnist_vecs_sorted[emnist_idx, :k]  # [k, 784]
        
        # Compute overlaps using different methods
        mean_cos = compute_subspace_overlap(vecs_mnist, vecs_emnist, k=k, method='mean_cos')
        grassmann = compute_subspace_overlap(vecs_mnist, vecs_emnist, k=k, method='grassmann')
        projection = compute_subspace_overlap(vecs_mnist, vecs_emnist, k=k, method='projection')
        
        pair_results[f"{mnist_char}-{emnist_char}"] = {
            'mean_cos': mean_cos,
            'grassmann': grassmann,
            'projection': projection,
        }
        
        print(f"{mnist_char:<8} {emnist_char:<8} {mean_cos:<12.4f} {grassmann:<12.4f} {projection:<12.4f}")
    
    results['expected_pairs'] = pair_results
    
    # Compute random baseline (non-matching pairs)
    print("\nRandom baseline (non-matching pairs):")
    random_overlaps = []
    for mnist_idx in range(10):
        for emnist_idx in range(26):
            # Skip expected pairs
            skip = False
            for _, _, m_idx, e_idx in test_pairs:
                if mnist_idx == m_idx and emnist_idx == e_idx:
                    skip = True
                    break
            if skip:
                continue
            
            vecs_mnist = mnist_vecs_sorted[mnist_idx, :k]
            vecs_emnist = emnist_vecs_sorted[emnist_idx, :k]
            overlap = compute_subspace_overlap(vecs_mnist, vecs_emnist, k=k, method='mean_cos')
            random_overlaps.append(overlap)
    
    random_mean = np.mean(random_overlaps)
    random_std = np.std(random_overlaps)
    print(f"Random pairs mean overlap: {random_mean:.4f} ± {random_std:.4f}")
    
    results['random_baseline'] = {
        'mean': random_mean,
        'std': random_std,
    }
    
    # Compute expected vs random ratio
    expected_mean = np.mean([v['mean_cos'] for v in pair_results.values()])
    ratio = expected_mean / (random_mean + 1e-10)
    
    print(f"\nExpected pairs mean: {expected_mean:.4f}")
    print(f"Ratio (expected/random): {ratio:.2f}x")
    
    results['summary'] = {
        'expected_mean': expected_mean,
        'random_mean': random_mean,
        'ratio': ratio,
        'conclusion': 'PASS' if ratio > 1.5 else 'INCONCLUSIVE' if ratio > 1.0 else 'FAIL'
    }
    
    return results


# ============================================================================
# Main Pipeline
# ============================================================================

def run_full_pipeline(
    device: str,
    output_dir: Path,
    epochs: int = 100,
    seed: int = 42,
):
    """
    Run the complete Extension 2 validation pipeline.
    
    1. Train baseline MNIST model (no noise)
    2. Train regularized MNIST model (noise σ=0.15)
    3. Run USPS Transfer Test (Step 1) - NEW
    4. Run Semantic Confusion Test (Step 2)
    5. Train EMNIST model
    6. Run Subspace Geometry Test (Step 3)
    7. Save all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("EXTENSION 2: CROSS-DATASET STRUCTURAL ROBUSTNESS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    # ---- Train MNIST models ----
    print("\n[1/6] Training baseline MNIST model (no noise)...")
    model_baseline, vals_baseline, vecs_baseline = train_mnist_model(
        device=device, noise_std=0.0, epochs=epochs, seed=seed
    )
    
    print("\n[2/6] Training regularized MNIST model (noise σ=0.15)...")
    model_regularized, vals_regularized, vecs_regularized = train_mnist_model(
        device=device, noise_std=0.15, epochs=epochs, seed=seed
    )
    
    # ---- Step 1: USPS Transfer Test ----
    print("\n[3/6] Running USPS Transfer Test...")
    usps_results = run_usps_transfer_comparison(
        model_baseline, model_regularized, device
    )
    
    # Save USPS results
    usps_path = output_dir / "usps_transfer_results.json"
    with open(usps_path, 'w') as f:
        json_results = json.dumps(usps_results, indent=2, default=str)
        f.write(json_results)
    print(f"USPS transfer results saved to {usps_path}")
    
    # ---- Step 2: Semantic Confusion Test ----
    print("\n[4/6] Running Semantic Confusion Test...")
    semantic_results = run_semantic_confusion_comparison(
        model_baseline, model_regularized, device
    )
    
    # Save semantic results
    semantic_path = output_dir / "semantic_confusion_results.json"
    with open(semantic_path, 'w') as f:
        json_results = json.dumps(semantic_results, indent=2, default=str)
        f.write(json_results)
    print(f"Semantic confusion results saved to {semantic_path}")
    
    # ---- Train EMNIST model ----
    print("\n[5/6] Training EMNIST model...")
    model_emnist, vals_emnist, vecs_emnist = train_emnist_model(
        device=device, noise_std=0.15, epochs=epochs, seed=seed
    )
    
    # ---- Step 3: Subspace Geometry Test ----
    print("\n[6/6] Running Subspace Geometry Test...")
    subspace_results = run_subspace_geometry_test(
        vals_regularized, vecs_regularized,
        vals_emnist, vecs_emnist,
        k=10
    )
    
    # Save subspace results
    subspace_path = output_dir / "subspace_geometry_results.json"
    with open(subspace_path, 'w') as f:
        json_results = json.dumps(subspace_results, indent=2, default=str)
        f.write(json_results)
    print(f"Subspace geometry results saved to {subspace_path}")
    
    # ---- Save checkpoints ----
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    for name, (model, vals, vecs) in [
        ('mnist_baseline', (model_baseline, vals_baseline, vecs_baseline)),
        ('mnist_regularized', (model_regularized, vals_regularized, vecs_regularized)),
        ('emnist_regularized', (model_emnist, vals_emnist, vecs_emnist)),
    ]:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'eigenvalues': vals.cpu(),
            'eigenvectors': vecs.cpu(),
            'seed': seed,
        }
        torch.save(checkpoint, checkpoint_dir / f"{name}_seed{seed}.pt")
    
    print(f"\nCheckpoints saved to {checkpoint_dir}")
    
    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("EXTENSION 2 COMPLETE - FINAL SUMMARY")
    print("=" * 60)
    
    print(f"\nStep 1 (USPS Transfer):")
    print(f"  Baseline accuracy: {usps_results['conclusion']['baseline_accuracy']:.2%}")
    print(f"  Regularized accuracy: {usps_results['conclusion']['regularized_accuracy']:.2%}")
    print(f"  Improvement: {usps_results['conclusion']['improvement']:+.2%}")
    print(f"  Regularization helps: {'YES' if usps_results['conclusion']['regularization_helps'] else 'NO'}")
    
    print(f"\nStep 2 (Semantic Confusion):")
    print(f"  Baseline accuracy: {semantic_results['conclusion']['baseline_accuracy']:.2%}")
    print(f"  Regularized accuracy: {semantic_results['conclusion']['regularized_accuracy']:.2%}")
    print(f"  Improvement: {semantic_results['conclusion']['improvement']:+.2%}")
    
    print(f"\nStep 3 (Subspace Geometry):")
    print(f"  Expected pairs overlap: {subspace_results['summary']['expected_mean']:.4f}")
    print(f"  Random pairs overlap: {subspace_results['summary']['random_mean']:.4f}")
    print(f"  Ratio: {subspace_results['summary']['ratio']:.2f}x")
    print(f"  Conclusion: {subspace_results['summary']['conclusion']}")
    
    return {
        'usps': usps_results,
        'semantic': semantic_results,
        'subspace': subspace_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Extension 2: Cross-Dataset Robustness")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full pipeline")
    parser.add_argument("--usps-test", action="store_true", help="Run only USPS transfer test")
    parser.add_argument("--semantic-test", action="store_true", help="Run only semantic confusion test")
    parser.add_argument("--subspace-test", action="store_true", help="Run only subspace geometry test")
    parser.add_argument("--checkpoint-baseline", type=str, help="Path to baseline checkpoint")
    parser.add_argument("--checkpoint-regularized", type=str, help="Path to regularized checkpoint")
    parser.add_argument("--checkpoint-emnist", type=str, help="Path to EMNIST checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/extension2", 
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS device detected - using CPU fallback for eigendecomposition")
    
    if args.full_pipeline:
        run_full_pipeline(
            device=device,
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            seed=args.seed,
        )
    elif args.usps_test:
        if not args.checkpoint_baseline or not args.checkpoint_regularized:
            print("Error: --checkpoint-baseline and --checkpoint-regularized required")
            return
        
        # Load models
        ckpt_baseline = torch.load(args.checkpoint_baseline, map_location=device)
        ckpt_regularized = torch.load(args.checkpoint_regularized, map_location=device)
        
        model_baseline = Model(Config()).to(device)
        model_baseline.load_state_dict(ckpt_baseline['model_state_dict'])
        
        model_regularized = Model(Config()).to(device)
        model_regularized.load_state_dict(ckpt_regularized['model_state_dict'])
        
        results = run_usps_transfer_comparison(model_baseline, model_regularized, device)
        
        output_path = Path(args.output_dir) / "usps_transfer_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")
        
    elif args.semantic_test:
        if not args.checkpoint_baseline or not args.checkpoint_regularized:
            print("Error: --checkpoint-baseline and --checkpoint-regularized required")
            return
        
        # Load models
        ckpt_baseline = torch.load(args.checkpoint_baseline, map_location=device)
        ckpt_regularized = torch.load(args.checkpoint_regularized, map_location=device)
        
        model_baseline = Model(Config()).to(device)
        model_baseline.load_state_dict(ckpt_baseline['model_state_dict'])
        
        model_regularized = Model(Config()).to(device)
        model_regularized.load_state_dict(ckpt_regularized['model_state_dict'])
        
        results = run_semantic_confusion_comparison(
            model_baseline, model_regularized, device
        )
        
        output_path = Path(args.output_dir) / "semantic_confusion_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")
        
    elif args.subspace_test:
        if not args.checkpoint_regularized or not args.checkpoint_emnist:
            print("Error: --checkpoint-regularized and --checkpoint-emnist required")
            return
        
        # Load eigenvalues/eigenvectors
        ckpt_mnist = torch.load(args.checkpoint_regularized, map_location='cpu')
        ckpt_emnist = torch.load(args.checkpoint_emnist, map_location='cpu')
        
        results = run_subspace_geometry_test(
            ckpt_mnist['eigenvalues'],
            ckpt_mnist['eigenvectors'],
            ckpt_emnist['eigenvalues'],
            ckpt_emnist['eigenvectors'],
        )
        
        output_path = Path(args.output_dir) / "subspace_geometry_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
