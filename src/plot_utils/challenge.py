"""
Challenge task (Figure 6) visualization functions.

Paper-style visualization of the binary similarity classification task.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Tuple

from src.plot_utils.style import COLORS


def plot_challenge_figure(
    ckpt_path: Path,
    device: str = "cpu",
    title: str = "Challenge Task: Eigendecomposition (True − False direction)",
) -> Optional[plt.Figure]:
    """
    Paper-style Figure 6 renderer for the challenge task checkpoint.
    
    Shows: positive/negative eigenvalue decay, top eigenvectors, target image, and bias.
    
    Args:
        ckpt_path: Path to the challenge task checkpoint
        device: Device for model loading
        title: Figure title
        
    Returns:
        matplotlib Figure, or None if checkpoint doesn't exist
    """
    import sys
    from pathlib import Path
    
    # Add original code path for Bilinear/Linear imports
    project_root = Path(__file__).parent.parent.parent
    orig_path = project_root / "bilinear-decomposition-main"
    if str(orig_path) not in sys.path:
        sys.path.insert(0, str(orig_path))
    
    from shared.components import Bilinear, Linear
    
    if not ckpt_path.exists():
        return None
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    eigenvalues = ckpt["eigenvalues"].cpu()
    eigenvectors = ckpt["eigenvectors"].cpu()
    
    # Handle checkpoints that don't have target_image (older format)
    if "target_image" in ckpt:
        target_image = ckpt["target_image"].cpu()
    else:
        # Load target image from MNIST dataset (default: digit=1, index=0)
        from image.datasets import MNIST as OrigMNIST
        train_mnist = OrigMNIST(train=True, device="cpu")
        mask = train_mnist.y == 1  # digit=1
        target_image = train_mnist.x[mask][0].cpu()  # index=0

    d_hidden = int(ckpt.get("config", {}).get("d_hidden", 256))

    # Define ChallengeModel class
    class ChallengeModel(torch.nn.Module):
        def __init__(self, d_input: int = 784, d_hidden_: int = 256, d_output: int = 2, bias: bool = True):
            super().__init__()
            self.embed = Linear(d_input, d_hidden_, bias=False)
            self.bilinear = Bilinear(d_hidden_, d_hidden_, bias=bias)
            self.head = Linear(d_hidden_, d_output, bias=False)

        @property
        def w_e(self) -> torch.Tensor:
            return self.embed.weight.data

        @property
        def w_u(self) -> torch.Tensor:
            return self.head.weight.data

        @property
        def w_l(self) -> torch.Tensor:
            return self.bilinear.w_l

        @property
        def w_r(self) -> torch.Tensor:
            return self.bilinear.w_r

        @property
        def bilinear_bias(self) -> Optional[torch.Tensor]:
            return None if self.bilinear.bias is None else self.bilinear.bias.data

    model = ChallengeModel(d_hidden_=d_hidden, bias=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Bias visualization in input space for True−False direction
    w_diff = (model.w_u[1] - model.w_u[0]).detach().cpu()  # [d_hidden]
    bias = model.bilinear_bias
    if bias is None:
        print("Warning: Challenge model has no bilinear bias; using zeros for bias panel.")
        bias_inp = torch.zeros(784)
    else:
        b_l = bias[:d_hidden].detach().cpu()
        b_r = bias[d_hidden:].detach().cpu()
        w_l = model.w_l.detach().cpu()
        w_r = model.w_r.detach().cpu()
        w_e = model.w_e.detach().cpu()

        v = torch.einsum("o,o,oi->i", w_diff, b_l, w_r) + torch.einsum("o,o,oi->i", w_diff, b_r, w_l)
        bias_inp = (v @ w_e).detach().cpu()  # [784]

    # Indices by sign
    vals = eigenvalues.detach().cpu()
    vecs = eigenvectors.detach().cpu()
    pos_idx = torch.where(vals > 0)[0]
    neg_idx = torch.where(vals < 0)[0]
    pos_sorted = pos_idx[vals[pos_idx].argsort(descending=True)] if len(pos_idx) else torch.tensor([], dtype=torch.long)
    neg_sorted = neg_idx[vals[neg_idx].argsort()] if len(neg_idx) else torch.tensor([], dtype=torch.long)

    fig = plt.figure(figsize=(11, 6.0))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.2, 1.0, 1.0, 1.0], wspace=0.25, hspace=0.45)

    # Positive eigenvalue decay
    ax = fig.add_subplot(gs[0, 0])
    top_pos = vals[pos_sorted[:20]] if len(pos_sorted) else torch.tensor([])
    ax.plot(
        np.arange(1, len(top_pos) + 1),
        top_pos.numpy(),
        "-o",
        color=COLORS.get("full", "C0"),
        linewidth=2,
        markersize=3,
    )
    ax.set_title("Positive eigenvalues", fontsize=11)
    ax.set_xlabel("Index", fontsize=9)
    ax.set_ylabel("λ")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    for j in range(2):
        axv = fig.add_subplot(gs[0, 1 + j])
        if len(pos_sorted) > j:
            idx = int(pos_sorted[j].item())
            img = vecs[idx].reshape(28, 28).numpy()
            vmax = float(np.abs(img).max() or 1.0)
            axv.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axv.set_title(f"λ={vals[idx].item():.2f}", fontsize=10)
        axv.axis("off")

    # Negative eigenvalue decay
    ax = fig.add_subplot(gs[1, 0])
    top_neg = vals[neg_sorted[:20]] if len(neg_sorted) else torch.tensor([])
    ax.plot(
        np.arange(1, len(top_neg) + 1),
        top_neg.numpy(),
        "-o",
        color=COLORS.get("none", "C3"),
        linewidth=2,
        markersize=3,
    )
    ax.set_title("Negative eigenvalues", fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("λ")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    for j in range(2):
        axv = fig.add_subplot(gs[1, 1 + j])
        if len(neg_sorted) > j:
            idx = int(neg_sorted[j].item())
            img = vecs[idx].reshape(28, 28).numpy()
            vmax = float(np.abs(img).max() or 1.0)
            axv.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axv.set_title(f"λ={vals[idx].item():.2f}", fontsize=10)
        axv.axis("off")

    # Right column: Target + Bias
    ax_t = fig.add_subplot(gs[0, 3])
    ax_t.imshow(target_image.reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
    ax_t.set_title("Target", fontsize=11)
    ax_t.axis("off")

    ax_b = fig.add_subplot(gs[1, 3])
    bias_img = bias_inp.reshape(28, 28).numpy()
    vmax = float(np.abs(bias_img).max() or 1.0)
    ax_b.imshow(bias_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_b.set_title("Bias", fontsize=11)
    ax_b.axis("off")

    fig.suptitle(title, fontsize=12, y=1.02)
    return fig
