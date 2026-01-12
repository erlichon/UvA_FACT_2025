"""
Diagnostic script to investigate the "Baseline Inversion" anomaly.

The "No Regularization" Effective Rank (38.5) is far lower than expected (~150),
and surprisingly lower than the "Noise" setting (75).

This script investigates three hypotheses:
1. Hidden Weight Decay - is AdamW using default wd=0.01?
2. Initialization Bias - is the Bilinear layer initialized with low rank?
3. MPS vs CPU Precision - does MPS cause spectral collapse?
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from image.model import Model, Config
from image.datasets import MNIST
from torch.optim import AdamW
from einops import einsum

from src.utils import load_config, set_seed
from src.analysis.spectral import effective_rank


def compute_bilinear_tensor_rank(model, device="cpu"):
    """
    Compute the effective rank of the bilinear interaction tensor B.

    B[cls, in1, in2] = sum_out w_u[cls, out] * w_l[out, in1] * w_r[out, in2]

    For a fresh model, this should have high rank (~d_hidden) if weights are random.
    """
    # Move to CPU for eigendecomposition
    w_u = model.w_u.to(device)  # [cls, out] = [10, 256]
    w_lr = model.w_lr[0].to(device)  # [2, out, hidden] = [2, 256, 256]
    w_e = model.w_e.to(device)  # [hidden, input] = [256, 784]

    l, r = w_lr.unbind(0)  # Each: [out, hidden] = [256, 256]

    # Compute third-order tensor: b[cls, in1, in2]
    b = einsum(w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")

    # Symmetrize
    b_sym = 0.5 * (b + b.mT)

    # Eigendecomposition
    vals, vecs = torch.linalg.eigh(b_sym)

    return vals, effective_rank(vals)


def check_weight_matrix_rank(model):
    """
    Check the rank of individual weight matrices.
    """
    results = {}

    # Embedding matrix
    w_e = model.w_e.cpu()  # [256, 784]
    u, s, v = torch.linalg.svd(w_e, full_matrices=False)
    results['w_e_singular_values'] = s[:20].tolist()  # Top 20 singular values
    results['w_e_effective_rank'] = (s.sum() ** 2 / (s ** 2).sum()).item()

    # Bilinear left/right weights
    w_lr = model.w_lr[0].cpu()  # [2, 256, 256]
    w_l = w_lr[0]  # [256, 256]
    w_r = w_lr[1]  # [256, 256]

    u_l, s_l, v_l = torch.linalg.svd(w_l, full_matrices=False)
    u_r, s_r, v_r = torch.linalg.svd(w_r, full_matrices=False)

    results['w_l_singular_values'] = s_l[:20].tolist()
    results['w_l_effective_rank'] = (s_l.sum() ** 2 / (s_l ** 2).sum()).item()
    results['w_r_singular_values'] = s_r[:20].tolist()
    results['w_r_effective_rank'] = (s_r.sum() ** 2 / (s_r ** 2).sum()).item()

    # Head matrix
    w_u = model.w_u.cpu()  # [10, 256]
    u_u, s_u, v_u = torch.linalg.svd(w_u, full_matrices=False)
    results['w_u_singular_values'] = s_u.tolist()
    results['w_u_effective_rank'] = (s_u.sum() ** 2 / (s_u ** 2).sum()).item()

    return results


def hypothesis_1_hidden_weight_decay():
    """
    Check if AdamW is using the default weight_decay=0.01 instead of 0.0.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Hidden Weight Decay")
    print("="*70)

    # Load config
    config = load_config(PROJECT_ROOT / "configs/mnist_dense_none.yaml")

    print(f"Config weight_decay value: {config['regularization']['weight_decay']}")

    # Create model and optimizer exactly as in train.py
    model_config = Config(
        epochs=100,
        d_hidden=config['model']['d_hidden'],
        wd=config['regularization']['weight_decay'],
        lr=config['training'].get('lr', 1e-3),
        seed=42,
    )

    print(f"Model Config.wd: {model_config.wd}")

    model = Model(model_config)
    optimizer = AdamW(model.parameters(), lr=model_config.lr, weight_decay=model_config.wd)

    # Check optimizer param groups
    wd_in_optimizer = optimizer.param_groups[0]['weight_decay']
    print(f"Optimizer param_groups[0]['weight_decay']: {wd_in_optimizer}")

    if wd_in_optimizer == 0.0:
        print("\n[PASS] Weight decay is correctly set to 0.0")
        return True
    else:
        print(f"\n[FAIL] Weight decay is {wd_in_optimizer}, not 0.0!")
        return False


def hypothesis_2_initialization_bias():
    """
    Check the effective rank of the Bilinear layer BEFORE training.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Initialization Bias")
    print("="*70)

    set_seed(42)

    # Create model
    model_config = Config(
        epochs=100,
        d_hidden=256,
        wd=0.0,
        lr=1e-3,
        seed=42,
    )
    model = Model(model_config)

    # Check weight matrix ranks
    print("\nIndividual weight matrix effective ranks:")
    weight_ranks = check_weight_matrix_rank(model)
    print(f"  w_e (embed):  {weight_ranks['w_e_effective_rank']:.2f}")
    print(f"  w_l (left):   {weight_ranks['w_l_effective_rank']:.2f}")
    print(f"  w_r (right):  {weight_ranks['w_r_effective_rank']:.2f}")
    print(f"  w_u (head):   {weight_ranks['w_u_effective_rank']:.2f}")

    # Compute bilinear tensor rank
    print("\nBilinear interaction tensor B[cls, in1, in2]:")
    eigenvalues, eff_ranks = compute_bilinear_tensor_rank(model)

    mean_rank = eff_ranks.mean().item()
    print(f"  Mean effective rank across classes: {mean_rank:.2f}")
    print(f"  Per-class effective ranks: {[f'{r:.1f}' for r in eff_ranks.tolist()]}")

    # Check eigenvalue distribution for class 0
    print(f"\n  Top 10 eigenvalues (class 0): {eigenvalues[0, -10:].tolist()}")
    print(f"  Bottom 10 eigenvalues (class 0): {eigenvalues[0, :10].tolist()}")

    # Theoretical expectation for random matrix
    # For a random symmetric matrix of size n, effective rank should be ~n
    print(f"\n  Expected for random matrix: ~{model_config.d_hidden}")

    if mean_rank < 100:
        print(f"\n[FAIL] Initialization rank ({mean_rank:.2f}) is already low!")
        print("  This suggests the initialization creates structure.")
        return False
    else:
        print(f"\n[PASS] Initialization rank ({mean_rank:.2f}) is high as expected.")
        return True


def hypothesis_3_mps_precision():
    """
    Compare training dynamics on MPS vs CPU.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: MPS vs CPU Precision")
    print("="*70)

    # Check if MPS is available
    has_mps = torch.backends.mps.is_available()
    print(f"MPS available: {has_mps}")

    if not has_mps:
        print("Skipping MPS comparison (not available)")
        return None

    # Load small subset of data
    print("\nLoading MNIST data...")

    results = {}

    for device_name in ['cpu', 'mps']:
        print(f"\n--- Training on {device_name.upper()} ---")
        set_seed(42)

        # Load data
        train_data = MNIST(train=True, device=device_name)
        test_data = MNIST(train=False, device=device_name)

        # Create model
        model_config = Config(
            epochs=5,  # Short training
            d_hidden=256,
            wd=0.0,
            lr=1e-3,
            seed=42,
        )
        model = Model(model_config).to(device_name)

        # Check dtype of weights
        print(f"  Weight dtype: {model.embed.weight.dtype}")

        # Train for a few epochs
        history = model.fit(train_data, test_data, transform=None)

        # Compute eigendecomposition
        model.to('cpu')
        vals, vecs = model.decompose()
        eff_rank = effective_rank(vals).mean().item()

        results[device_name] = {
            'effective_rank': eff_rank,
            'final_val_acc': history['val/acc'].iloc[-1],
            'weight_dtype': str(model.embed.weight.dtype),
        }

        print(f"  Final val accuracy: {results[device_name]['final_val_acc']:.4f}")
        print(f"  Effective rank: {eff_rank:.2f}")

    # Compare
    cpu_rank = results['cpu']['effective_rank']
    mps_rank = results['mps']['effective_rank']
    diff = abs(cpu_rank - mps_rank)

    print(f"\n--- Comparison ---")
    print(f"CPU effective rank: {cpu_rank:.2f}")
    print(f"MPS effective rank: {mps_rank:.2f}")
    print(f"Difference: {diff:.2f}")

    if diff > 10:
        print(f"\n[FAIL] Large difference ({diff:.2f}) between CPU and MPS!")
        return False
    else:
        print(f"\n[PASS] CPU and MPS results are similar.")
        return True


def check_paper_reproduction():
    """
    Additional check: verify we're using the exact paper settings.
    """
    print("\n" + "="*70)
    print("ADDITIONAL: Paper Configuration Check")
    print("="*70)

    # Paper settings (from Section 4)
    paper_settings = {
        'd_hidden': 256,
        'epochs': 100,  # Paper might use 20?
        'lr': 1e-3,
        'batch_size': 2048,
        'noise_std_reg': 0.4,
        'weight_decay_reg': 0.5,
        'data_normalization': '[0, 1]',  # /255
    }

    # Our settings
    config = load_config(PROJECT_ROOT / "configs/mnist_dense_none.yaml")
    our_settings = {
        'd_hidden': config['model']['d_hidden'],
        'epochs': config['training']['epochs'],
        'lr': config['training']['lr'],
        'batch_size': config['training'].get('batch_size', 2048),
    }

    print("Paper settings vs Our settings:")
    for key in ['d_hidden', 'epochs', 'lr', 'batch_size']:
        paper_val = paper_settings.get(key, 'N/A')
        our_val = our_settings.get(key, 'N/A')
        match = "OK" if paper_val == our_val else "DIFF"
        print(f"  {key}: paper={paper_val}, ours={our_val} [{match}]")

    # Check data normalization
    print("\nData normalization check:")
    train_data = MNIST(train=True, device='cpu')
    print(f"  Data range: [{train_data.x.min():.4f}, {train_data.x.max():.4f}]")
    print(f"  Data dtype: {train_data.x.dtype}")
    print(f"  Data shape: {train_data.x.shape}")


def training_dynamics_check():
    """
    Check how effective rank evolves during training.
    """
    print("\n" + "="*70)
    print("ADDITIONAL: Training Dynamics Check")
    print("="*70)

    set_seed(42)

    # Load data
    device = 'cpu'
    train_data = MNIST(train=True, device=device)
    test_data = MNIST(train=False, device=device)

    # Create model with NO regularization
    model_config = Config(
        epochs=20,  # Shorter run to see dynamics
        d_hidden=256,
        wd=0.0,
        lr=1e-3,
        seed=42,
    )
    model = Model(model_config).to(device)

    # Check initial rank
    vals_init, _ = model.decompose()
    rank_init = effective_rank(vals_init).mean().item()
    print(f"Initial effective rank: {rank_init:.2f}")

    # Custom training loop to track rank
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    optimizer = AdamW(model.parameters(), lr=model_config.lr, weight_decay=model_config.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=model_config.epochs)

    def collator(batch):
        x = torch.stack([item[0] for item in batch]).float()
        y = torch.stack([item[1] for item in batch])
        return x, y

    loader = DataLoader(train_data, batch_size=2048, shuffle=True, drop_last=True, collate_fn=collator)

    ranks_over_time = [rank_init]

    print("\nTraining with rank tracking (every 5 epochs)...")
    for epoch in tqdm(range(model_config.epochs)):
        model.train()
        for x, y in loader:
            y_hat = model(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                vals, _ = model.decompose()
                rank = effective_rank(vals).mean().item()
                ranks_over_time.append(rank)
                print(f"  Epoch {epoch+1}: rank = {rank:.2f}")

    print(f"\nRank evolution: {[f'{r:.1f}' for r in ranks_over_time]}")
    print(f"Initial -> Final: {rank_init:.2f} -> {ranks_over_time[-1]:.2f}")
    print(f"Change: {ranks_over_time[-1] - rank_init:.2f}")


def main():
    print("="*70)
    print("BASELINE ANOMALY DIAGNOSTIC")
    print("Investigating why 'No Reg' has lower rank than expected")
    print("="*70)

    # Run all hypothesis checks
    h1_result = hypothesis_1_hidden_weight_decay()
    h2_result = hypothesis_2_initialization_bias()
    h3_result = hypothesis_3_mps_precision()

    # Additional checks
    check_paper_reproduction()
    training_dynamics_check()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"H1 (Hidden WD):      {'PASS' if h1_result else 'FAIL'}")
    print(f"H2 (Init Bias):      {'PASS' if h2_result else 'FAIL'}")
    print(f"H3 (MPS Precision):  {'PASS' if h3_result else 'FAIL' if h3_result is not None else 'SKIPPED'}")


if __name__ == "__main__":
    main()
