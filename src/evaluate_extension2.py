"""
Extension 2: Cross-Dataset Structural Robustness Evaluation

Objective: Prove that regularized Bilinear MLPs learn universal geometric shapes
(e.g., "circularity") rather than dataset-specific pixel artifacts.

This script implements a FOUR-step validation:

Step 1: Mechanism Stability Test
- Train MNIST (60k) and EMNIST-Digits (240k) models
- A. Functional: Cross-dataset accuracy (MNIST→EMNIST-Digits, EMNIST-Digits→MNIST)
- B. Representational: Eigenvector subspace overlap
- Success: High scores on both prove writer-independent mechanisms

Step 2: USPS Transfer Test
- Train two models on MNIST: Baseline (no noise) and Regularized (noise σ=0.15)
- Evaluate both on USPS (different digit dataset, same classes 0-9)
- Success: High accuracy on USPS proves eigenvectors capture universal digit features,
  not MNIST-specific pixel artifacts

Step 3: Semantic Confusion Test
- Evaluate both MNIST-trained models on EMNIST letters 'O', 'I', 'Z', 'S', 'B'
- Success: Regularized model classifies 'O' as '0', etc., proving shape-based learning

Step 4: Universal Geometry Proof
- Train a Bilinear MLP on EMNIST-Letters
- Extract top-3 eigenvectors for MNIST '0' and EMNIST 'O'
- Compute subspace overlap to quantify mechanism similarity
- Goal: High similarity proves universal shape learning

Usage:
    # Full pipeline (all 4 steps)
    python src/evaluate_extension2.py --full-pipeline
    
    # Step 1: Mechanism stability test
    python src/evaluate_extension2.py --mechanism-test \
        --checkpoint-regularized path/to/mnist.pt \
        --checkpoint-emnist-digits path/to/emnist_digits.pt
    
    # Step 2: USPS transfer test
    python src/evaluate_extension2.py --usps-test \
        --checkpoint-baseline path/to/baseline.pt \
        --checkpoint-regularized path/to/regularized.pt
    
    # Step 3: Semantic confusion test
    python src/evaluate_extension2.py --semantic-test \
        --checkpoint-baseline path/to/baseline.pt \
        --checkpoint-regularized path/to/regularized.pt
    
    # Step 4: Subspace geometry test
    python src/evaluate_extension2.py --subspace-test \
        --checkpoint-regularized path/to/mnist.pt \
        --checkpoint-emnist path/to/emnist_letters.pt
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


def train_emnist_digits_model(
    device: str,
    noise_std: float = 0.15,
    weight_decay: float = 0.5,
    epochs: int = 100,
    d_hidden: int = 256,
    seed: int = 42,
) -> Tuple[Model, torch.Tensor, torch.Tensor]:
    """
    Train a bilinear model on EMNIST-Digits.
    
    EMNIST-Digits: 240k samples, 10 classes (0-9), different writers than MNIST.
    Used for Step 1 (Mechanism Stability Test) to prove writer-independent learning.
    
    Returns:
        (model, eigenvalues, eigenvectors)
    """
    set_seed(seed)
    
    # Load data
    from src.data.cross_dataset import load_emnist_digits_normalized
    train_data, test_data = load_emnist_digits_normalized(device=device)
    
    # Create model with 10 output classes
    model_config = Config(
        epochs=epochs,
        d_hidden=d_hidden,
        wd=weight_decay,
        lr=1e-3,
        seed=seed,
        d_output=10,  # EMNIST-Digits has 10 digit classes (same as MNIST)
    )
    model = Model(model_config).to(device)
    
    # Create noise transform
    transform = None
    if noise_std > 0:
        transform = kornia.augmentation.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=1.0
        )
    
    # Train
    print(f"Training EMNIST-Digits model (noise_std={noise_std}, wd={weight_decay})...")
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
# Step 1: Mechanism Stability - Functional Similarity
# ============================================================================

def cross_dataset_accuracy_test(
    mnist_model: Model,
    emnist_digits_model: Model,
    device: str,
) -> Dict:
    """
    Evaluate cross-dataset accuracy between MNIST and EMNIST-Digits.
    
    Tests functional equivalence:
    - MNIST model → EMNIST-Digits test data
    - EMNIST-Digits model → MNIST test data
    
    High bidirectional accuracy proves models learn the same functional
    mechanisms, not just similar internal representations.
    
    Returns:
        Dictionary with bidirectional accuracy metrics
    """
    from src.data.cross_dataset import load_mnist_normalized, load_emnist_digits_normalized
    
    print("\nA. Functional Similarity (Cross-Dataset Accuracy):")
    print("-" * 60)
    
    # Load test datasets
    _, mnist_test = load_mnist_normalized(device=device)
    _, emnist_digits_test = load_emnist_digits_normalized(device=device)
    
    mnist_model.eval()
    emnist_digits_model.eval()
    
    # Test 1: MNIST model on EMNIST-Digits test
    with torch.no_grad():
        x_emnist = emnist_digits_test.x.view(emnist_digits_test.x.size(0), -1)
        logits = mnist_model(x_emnist)
        preds = logits.argmax(dim=-1)
        mnist_on_emnist_acc = (preds == emnist_digits_test.y).float().mean().item()
    
    # Test 2: EMNIST-Digits model on MNIST test
    with torch.no_grad():
        x_mnist = mnist_test.x.view(mnist_test.x.size(0), -1)
        logits = emnist_digits_model(x_mnist)
        preds = logits.argmax(dim=-1)
        emnist_on_mnist_acc = (preds == mnist_test.y).float().mean().item()
    
    # Bidirectional average
    avg_accuracy = (mnist_on_emnist_acc + emnist_on_mnist_acc) / 2
    
    print(f"  MNIST model → EMNIST-Digits test:     {mnist_on_emnist_acc:.2%}")
    print(f"  EMNIST-Digits model → MNIST test:     {emnist_on_mnist_acc:.2%}")
    print(f"  Bidirectional average:                 {avg_accuracy:.2%}")
    
    # Per-class accuracy for MNIST → EMNIST-Digits
    per_class = {}
    for digit in range(10):
        mask = emnist_digits_test.y == digit
        if mask.sum() > 0:
            with torch.no_grad():
                x_class = emnist_digits_test.x[mask].view(mask.sum(), -1)
                logits = mnist_model(x_class)
                preds = logits.argmax(dim=-1)
                acc = (preds == digit).float().mean().item()
                per_class[digit] = acc
    
    return {
        'mnist_on_emnist': mnist_on_emnist_acc,
        'emnist_on_mnist': emnist_on_mnist_acc,
        'bidirectional_avg': avg_accuracy,
        'per_class_mnist_on_emnist': per_class,
        'mnist_test_size': len(mnist_test),
        'emnist_test_size': len(emnist_digits_test),
    }


# ============================================================================
# Step 2: USPS Transfer Test
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
# Step 1: Mechanism Stability Test (MNIST vs EMNIST-Digits)
# ============================================================================

def mechanism_stability_test(
    mnist_model: Model,
    mnist_eigenvalues: torch.Tensor,
    mnist_eigenvectors: torch.Tensor,
    emnist_digits_model: Model,
    emnist_digits_eigenvalues: torch.Tensor,
    emnist_digits_eigenvectors: torch.Tensor,
    device: str,
    k: int = 3,
) -> Dict:
    """
    Compare MNIST vs EMNIST-Digits using two complementary metrics.
    
    A. Functional Similarity: Cross-dataset classification accuracy
    B. Representational Similarity: Eigenvector subspace overlap
    
    **Hypothesis:** If mechanisms are truly universal and not writer-specific,
    both functional accuracy and eigenvector overlap should be high.
    
    Args:
        mnist_model: Trained MNIST model
        mnist_eigenvalues: MNIST model eigenvalues [10, n_components]
        mnist_eigenvectors: MNIST model eigenvectors [10, n_components, 784]
        emnist_digits_model: Trained EMNIST-Digits model
        emnist_digits_eigenvalues: EMNIST-Digits eigenvalues [10, n_components]
        emnist_digits_eigenvectors: EMNIST-Digits eigenvectors [10, n_components, 784]
        device: Device for computation
        k: Number of top eigenvectors to compare
    
    Returns:
        Dictionary with functional and representational similarity metrics
    """
    from src.analysis.subspace import compute_subspace_overlap
    
    print("\n" + "=" * 60)
    print("STEP 1: MECHANISM STABILITY TEST")
    print("=" * 60)
    print("Testing writer-independent mechanisms via:")
    print("  A. Functional Similarity (cross-dataset accuracy)")
    print("  B. Representational Similarity (eigenvector overlap)")
    print("=" * 60)
    
    # A. Functional Similarity
    functional_results = cross_dataset_accuracy_test(
        mnist_model, emnist_digits_model, device
    )
    
    # B. Representational Similarity
    print("\nB. Representational Similarity (Eigenvector Subspace Overlap):")
    print("-" * 60)
    
    # Sort eigenvectors by eigenvalue magnitude
    from src.analysis.subspace import sort_eigenvectors_by_magnitude
    mnist_vecs_sorted = sort_eigenvectors_by_magnitude(mnist_eigenvalues, mnist_eigenvectors)
    emnist_vecs_sorted = sort_eigenvectors_by_magnitude(emnist_digits_eigenvalues, emnist_digits_eigenvectors)
    
    print(f"Comparing top-{k} eigenvector subspaces for each digit:")
    print("-" * 60)
    print(f"{'Digit':<8} {'Mean Cos':<12} {'Grassmann':<12} {'Projection':<12}")
    print("-" * 60)
    
    per_class_results = {}
    overlaps_mean_cos = []
    overlaps_grassmann = []
    overlaps_projection = []
    
    for digit in range(10):
        vecs_mnist = mnist_vecs_sorted[digit, :k]  # [k, 784]
        vecs_emnist = emnist_vecs_sorted[digit, :k]  # [k, 784]
        
        # Compute overlaps using different methods
        mean_cos = compute_subspace_overlap(vecs_mnist, vecs_emnist, k=k, method='mean_cos')
        grassmann = compute_subspace_overlap(vecs_mnist, vecs_emnist, k=k, method='grassmann')
        projection = compute_subspace_overlap(vecs_mnist, vecs_emnist, k=k, method='projection')
        
        per_class_results[digit] = {
            'mean_cos': mean_cos,
            'grassmann': grassmann,
            'projection': projection,
        }
        
        overlaps_mean_cos.append(mean_cos)
        overlaps_grassmann.append(grassmann)
        overlaps_projection.append(projection)
        
        print(f"{digit:<8} {mean_cos:<12.4f} {grassmann:<12.4f} {projection:<12.4f}")
    
    # Compute aggregate statistics
    mean_overlap = np.mean(overlaps_mean_cos)
    std_overlap = np.std(overlaps_mean_cos)
    min_overlap = np.min(overlaps_mean_cos)
    max_overlap = np.max(overlaps_mean_cos)
    
    print("-" * 60)
    print(f"{'MEAN':<8} {mean_overlap:<12.4f} (±{std_overlap:.4f})")
    print(f"{'RANGE':<8} [{min_overlap:.4f}, {max_overlap:.4f}]")
    
    # Combined conclusion
    func_score = functional_results['bidirectional_avg']
    repr_score = mean_overlap
    
    print("\n" + "=" * 60)
    print("COMBINED CONCLUSION:")
    print("=" * 60)
    print(f"Functional similarity:       {func_score:.2%}")
    print(f"Representational similarity: {repr_score:.4f}")
    print()
    
    if func_score > 0.90 and repr_score > 0.8:
        conclusion = "✓✓ STRONG: Both functional and representational evidence"
        interpretation = "High cross-dataset accuracy AND eigenvector overlap prove universal mechanisms."
    elif func_score > 0.85 or repr_score > 0.75:
        conclusion = "✓ MODERATE: One metric shows strong evidence"
        interpretation = "Partial evidence for universal mechanisms; one metric is strong."
    else:
        conclusion = "✗ WEAK: Insufficient evidence for universality"
        interpretation = "Low scores suggest writer-specific artifacts."
    
    print(conclusion)
    print(interpretation)
    
    results = {
        'functional': functional_results,
        'representational': {
            'per_class_overlap': per_class_results,
            'aggregate': {
                'mean': mean_overlap,
                'std': std_overlap,
                'min': min_overlap,
                'max': max_overlap,
                'mean_grassmann': np.mean(overlaps_grassmann),
                'mean_projection': np.mean(overlaps_projection),
            }
        },
        'conclusion': conclusion,
        'interpretation': interpretation,
    }
    
    return results


# ============================================================================
# Step 3: Subspace Geometry Metric
# ============================================================================

def run_subspace_geometry_test(
    mnist_eigenvalues: torch.Tensor,
    mnist_eigenvectors: torch.Tensor,
    emnist_eigenvalues: torch.Tensor,
    emnist_eigenvectors: torch.Tensor,
    k: int = 3,
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
    from src.analysis.subspace import sort_eigenvectors_by_magnitude
    
    print("\n" + "=" * 60)
    print("STEP 3: SUBSPACE GEOMETRY METRIC")
    print("=" * 60)
    print("Comparing eigenvector subspaces between MNIST and EMNIST models...")
    print("(High overlap for similar shapes = universal mechanism learning)")
    
    results = {}
    
    # Sort eigenvectors by eigenvalue magnitude
    mnist_vecs_sorted = sort_eigenvectors_by_magnitude(mnist_eigenvalues, mnist_eigenvectors)
    emnist_vecs_sorted = sort_eigenvectors_by_magnitude(emnist_eigenvalues, emnist_eigenvectors)
    
    # Test specific pairs (digit, letter, mnist_idx, emnist_idx)
    test_pairs = [
        ('0', 'O', 0, 14),   # Zero vs O
        ('1', 'I', 1, 8),    # One vs I
        ('2', 'Z', 2, 25),   # Two vs Z
        ('5', 'S', 5, 18),   # Five vs S
        ('6', 'B', 6, 1),    # Six vs B
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
    Run the complete Extension 2 validation pipeline (4 steps).
    
    1. Train baseline MNIST model (no noise)
    2. Train regularized MNIST model (noise σ=0.15)
    3. Train EMNIST-Digits model (noise σ=0.15)
    4. Step 1: Mechanism Stability (functional + representational)
    5. Step 2: USPS Transfer Test
    6. Step 3: Semantic Confusion Test
    7. Train EMNIST-Letters model (noise σ=0.15)
    8. Step 4: Universal Geometry Test
    9. Save all results
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
    
    print("\n[2/9] Training regularized MNIST model (noise σ=0.15)...")
    model_regularized, vals_regularized, vecs_regularized = train_mnist_model(
        device=device, noise_std=0.15, epochs=epochs, seed=seed
    )
    
    # ---- Train EMNIST-Digits model ----
    print("\n[3/9] Training EMNIST-Digits model...")
    model_emnist_digits, vals_emnist_digits, vecs_emnist_digits = train_emnist_digits_model(
        device=device, noise_std=0.15, epochs=epochs, seed=seed
    )
    
    # ---- Step 1: Mechanism Stability Test ----
    print("\n[4/9] Running Mechanism Stability Test (MNIST vs EMNIST-Digits)...")
    mechanism_results = mechanism_stability_test(
        model_regularized, vals_regularized, vecs_regularized,
        model_emnist_digits, vals_emnist_digits, vecs_emnist_digits,
        device=device,
        k=3
    )
    
    # Save mechanism stability results
    mechanism_path = output_dir / "mechanism_stability_results.json"
    with open(mechanism_path, 'w') as f:
        json_results = json.dumps(mechanism_results, indent=2, default=str)
        f.write(json_results)
    print(f"Mechanism stability results saved to {mechanism_path}")
    
    # ---- Step 2: USPS Transfer Test ----
    print("\n[5/9] Running USPS Transfer Test...")
    usps_results = run_usps_transfer_comparison(
        model_baseline, model_regularized, device
    )
    
    # Save USPS results
    usps_path = output_dir / "usps_transfer_results.json"
    with open(usps_path, 'w') as f:
        json_results = json.dumps(usps_results, indent=2, default=str)
        f.write(json_results)
    print(f"USPS transfer results saved to {usps_path}")
    
    # ---- Step 3: Semantic Confusion Test ----
    print("\n[6/9] Running Semantic Confusion Test...")
    semantic_results = run_semantic_confusion_comparison(
        model_baseline, model_regularized, device
    )
    
    # Save semantic results
    semantic_path = output_dir / "semantic_confusion_results.json"
    with open(semantic_path, 'w') as f:
        json_results = json.dumps(semantic_results, indent=2, default=str)
        f.write(json_results)
    print(f"Semantic confusion results saved to {semantic_path}")
    
    # ---- Train EMNIST-Letters model ----
    print("\n[7/9] Training EMNIST-Letters model...")
    model_emnist, vals_emnist, vecs_emnist = train_emnist_model(
        device=device, noise_std=0.15, epochs=epochs, seed=seed
    )
    
    # ---- Step 4: Universal Geometry Test ----
    print("\n[8/9] Running Universal Geometry Test (Subspace Geometry)...")
    subspace_results = run_subspace_geometry_test(
        vals_regularized, vecs_regularized,
        vals_emnist, vecs_emnist,
        k=3
    )
    
    # Save subspace results
    subspace_path = output_dir / "subspace_geometry_results.json"
    with open(subspace_path, 'w') as f:
        json_results = json.dumps(subspace_results, indent=2, default=str)
        f.write(json_results)
    print(f"Subspace geometry results saved to {subspace_path}")
    
    # ---- Save checkpoints ----
    print("\n[9/9] Saving checkpoints...")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    for name, (model, vals, vecs) in [
        ('mnist_baseline', (model_baseline, vals_baseline, vecs_baseline)),
        ('mnist_regularized', (model_regularized, vals_regularized, vecs_regularized)),
        ('emnist_digits_regularized', (model_emnist_digits, vals_emnist_digits, vecs_emnist_digits)),
        ('emnist_letters_regularized', (model_emnist, vals_emnist, vecs_emnist)),
    ]:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'eigenvalues': vals.cpu(),
            'eigenvectors': vecs.cpu(),
            'seed': seed,
        }
        torch.save(checkpoint, checkpoint_dir / f"{name}_seed{seed}.pt")
    
    print(f"Checkpoints saved to {checkpoint_dir}")
    
    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("EXTENSION 2 COMPLETE - FINAL SUMMARY (4 STEPS)")
    print("=" * 60)
    
    print(f"\nStep 1 (Mechanism Stability):")
    print(f"  Cross-dataset accuracy: {mechanism_results['functional']['bidirectional_avg']:.2%}")
    print(f"  Eigenvector overlap: {mechanism_results['representational']['aggregate']['mean']:.4f}")
    print(f"  Conclusion: {mechanism_results['conclusion']}")
    
    print(f"\nStep 2 (USPS Transfer):")
    print(f"  Baseline accuracy: {usps_results['conclusion']['baseline_accuracy']:.2%}")
    print(f"  Regularized accuracy: {usps_results['conclusion']['regularized_accuracy']:.2%}")
    print(f"  Improvement: {usps_results['conclusion']['improvement']:+.2%}")
    print(f"  Regularization helps: {'YES' if usps_results['conclusion']['regularization_helps'] else 'NO'}")
    
    print(f"\nStep 3 (Semantic Confusion):")
    print(f"  Baseline accuracy: {semantic_results['conclusion']['baseline_accuracy']:.2%}")
    print(f"  Regularized accuracy: {semantic_results['conclusion']['regularized_accuracy']:.2%}")
    print(f"  Improvement: {semantic_results['conclusion']['improvement']:+.2%}")
    
    print(f"\nStep 4 (Universal Geometry):")
    print(f"  Expected pairs overlap: {subspace_results['summary']['expected_mean']:.4f}")
    print(f"  Random pairs overlap: {subspace_results['summary']['random_mean']:.4f}")
    print(f"  Ratio: {subspace_results['summary']['ratio']:.2f}x")
    print(f"  Conclusion: {subspace_results['summary']['conclusion']}")
    
    return {
        'mechanism': mechanism_results,
        'usps': usps_results,
        'semantic': semantic_results,
        'subspace': subspace_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Extension 2: Cross-Dataset Robustness")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full pipeline")
    parser.add_argument("--mechanism-test", action="store_true", help="Run only mechanism stability test (MNIST vs EMNIST-Digits)")
    parser.add_argument("--usps-test", action="store_true", help="Run only USPS transfer test")
    parser.add_argument("--semantic-test", action="store_true", help="Run only semantic confusion test")
    parser.add_argument("--subspace-test", action="store_true", help="Run only subspace geometry test")
    parser.add_argument("--checkpoint-baseline", type=str, help="Path to baseline checkpoint")
    parser.add_argument("--checkpoint-regularized", type=str, help="Path to regularized checkpoint")
    parser.add_argument("--checkpoint-emnist", type=str, help="Path to EMNIST-Letters checkpoint")
    parser.add_argument("--checkpoint-emnist-digits", type=str, help="Path to EMNIST-Digits checkpoint")
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
    elif args.mechanism_test:
        if not args.checkpoint_regularized or not args.checkpoint_emnist_digits:
            print("Error: --checkpoint-regularized and --checkpoint-emnist-digits required")
            print("Example: python src/evaluate_extension2.py --mechanism-test \\")
            print("           --checkpoint-regularized results/phase1/checkpoints/mnist_dense_full_seed42.pt \\")
            print("           --checkpoint-emnist-digits results/extension2/checkpoints/emnist_digits_regularized_seed42.pt")
            return
        
        # Load checkpoints
        print(f"Loading MNIST checkpoint: {args.checkpoint_regularized}")
        ckpt_mnist = torch.load(args.checkpoint_regularized, map_location=device)
        
        print(f"Loading EMNIST-Digits checkpoint: {args.checkpoint_emnist_digits}")
        ckpt_emnist_digits = torch.load(args.checkpoint_emnist_digits, map_location=device)
        
        # Reconstruct models
        mnist_model = Model(Config(d_hidden=256, d_output=10)).to(device)
        mnist_model.load_state_dict(ckpt_mnist['model_state_dict'])
        
        emnist_model = Model(Config(d_hidden=256, d_output=10)).to(device)
        emnist_model.load_state_dict(ckpt_emnist_digits['model_state_dict'])
        
        # Run test with both models and eigenvalues/eigenvectors
        results = mechanism_stability_test(
            mnist_model,
            ckpt_mnist['eigenvalues'],
            ckpt_mnist['eigenvectors'],
            emnist_model,
            ckpt_emnist_digits['eigenvalues'],
            ckpt_emnist_digits['eigenvectors'],
            device=device,
            k=3,
        )
        
        # Save results
        output_path = Path(args.output_dir) / "mechanism_stability_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
        
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
