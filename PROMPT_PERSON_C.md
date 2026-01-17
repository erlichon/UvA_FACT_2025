# Claude Code Prompt: Person C (Robustness Testing Lead)

> **Instructions**: Copy this entire file content into a new Claude Code session. The prompt is self-contained with all necessary context.

---

## PROMPT START

You are implementing **Phase 2 Extension 1** of a FACT-AI course project as **Person C (Robustness Testing Lead)**.

### Project Summary
We are reproducing and extending "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., arXiv:2410.08417). Phase 1 (reproduction) is complete. You are now working on:
- **Extension 1**: Robustness testing (Rotated MNIST, EMNIST)

> **NOTE**: Extension 2 (CP Implementation) is handled by Person F separately.

### Your Role
- Implement data loaders for distribution shift experiments
- Evaluate Phase 1 regularized models on shifted data
- Analyze whether learned eigenvectors generalize to distribution shifts
- Generate robustness analysis figures

### Timeline
- **Days 8-9** (Jan 15-16): Extension 1 - Robustness testing
- **Day 10**: Complete analysis and figures

### Dependencies
- Phase 1 checkpoints from Person A: `results/vision/checkpoints/`
- Analysis utilities from Person B: `src/vision/spectral.py`, `src/vision/visualization.py`

---

## CRITICAL CONSTRAINTS

1. **USE PHASE 1 CHECKPOINTS**: Evaluate existing trained models, don't retrain

2. **5 SEEDS**: Use checkpoints from all seeds `[42, 43, 44, 45, 46]` for statistical analysis

3. **OUTPUT FORMAT**: Save results to paths Person E expects (see Expected Outputs)

4. **FIGURE QUALITY**: Publication-quality figures using matplotlib (PDF export)

---

## VERIFICATION & BUDGET REQUIREMENTS

> **IMPORTANT**: The team has a total budget of **25,000 SBUs** on Snellius.

### Local Verification (MANDATORY)

Before running any evaluation, verify your code works locally:

```bash
# Test data loaders
python -c "
from src.data.rotated_mnist import RotatedMNIST
ds = RotatedMNIST(angle=30, device='cpu')
print(f'Loaded {len(ds)} samples, shape: {ds.x.shape}')
"

python -c "
from src.data.emnist import EMNISTLetters
ds = EMNISTLetters(device='cpu')
print(f'Loaded {len(ds)} samples, classes: {ds.num_classes}')
"
```

**Verification Checklist**:
- [ ] RotatedMNIST loads correctly at all angles [0, 15, 30, 45, 60, 90]
- [ ] EMNISTLetters loads correctly with 26 classes
- [ ] Can load Phase 1 checkpoints
- [ ] Evaluation script runs without errors

### Budget Note

E1 (Robustness Testing) uses **minimal Snellius budget** since it's evaluation only (no training):
- Inference is fast (~10 SBUs total)
- Can run most tests locally on CPU

| Task | Est. SBUs |
|------|-----------|
| Robustness evaluation | 20 |
| Total E1 | 20 |

---

## EXTENSION 1: ROBUSTNESS TESTING

### 1.1 Concept

Test whether the interpretable eigenvectors learned with regularization generalize to:
1. **Rotated MNIST**: Same digits, but rotated (15, 30, 45, 60, 90 degrees)
2. **EMNIST Letters**: Different symbols (letters instead of digits)

**Key Questions**:
- Does accuracy degrade gracefully or catastrophically?
- Do the learned eigenvectors still capture meaningful features?
- Is there a correlation between effective rank and robustness?

### 1.2 Data Loaders

**`src/data/rotated_mnist.py`**:
```python
"""
Rotated MNIST Dataset

Creates MNIST test sets with fixed rotation angles for robustness evaluation.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from typing import List, Optional, Tuple


class RotatedMNIST(Dataset):
    """
    MNIST dataset with fixed rotation applied to all images.

    Args:
        root: Data directory
        angle: Rotation angle in degrees
        train: If True, use training set (default False for evaluation)
        download: Download if not present
        device: Device to load tensors to
    """

    def __init__(
        self,
        root: str = './data',
        angle: float = 0.0,
        train: bool = False,
        download: bool = True,
        device: str = 'cuda',
    ):
        self.angle = angle
        self.device = device

        # Load base MNIST
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )

        # Pre-rotate and move to device for efficiency
        self._preprocess()

    def _preprocess(self):
        """Pre-rotate all images and move to device."""
        images = []
        labels = []

        for img, label in self.mnist:
            # Rotate image
            if self.angle != 0:
                img = TF.rotate(img, self.angle)
            images.append(img)
            labels.append(label)

        # Stack and move to device
        self.x = torch.stack(images).squeeze(1).to(self.device)  # [N, 28, 28]
        self.x = self.x.view(self.x.shape[0], -1)  # Flatten to [N, 784]
        self.y = torch.tensor(labels).to(self.device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_rotated_datasets(
    angles: List[float] = [0, 15, 30, 45, 60, 90],
    device: str = 'cuda'
) -> dict:
    """
    Get dictionary of rotated MNIST test sets.

    Args:
        angles: List of rotation angles
        device: Device for tensors

    Returns:
        Dict mapping angle to RotatedMNIST dataset
    """
    return {
        angle: RotatedMNIST(angle=angle, train=False, device=device)
        for angle in angles
    }
```

**`src/data/emnist.py`**:
```python
"""
EMNIST Letters Dataset

Wrapper for EMNIST letters subset, compatible with our model interface.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class EMNISTLetters(Dataset):
    """
    EMNIST Letters dataset.

    The letters subset has 26 classes (A-Z, case-insensitive).
    Images are 28x28, same as MNIST.

    Note: EMNIST images are transposed compared to MNIST, so we fix that.

    Args:
        root: Data directory
        train: Training or test split
        download: Download if not present
        device: Device to load tensors to
    """

    def __init__(
        self,
        root: str = './data',
        train: bool = False,
        download: bool = True,
        device: str = 'cuda',
    ):
        self.device = device

        # Load EMNIST letters
        self.emnist = datasets.EMNIST(
            root=root,
            split='letters',
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )

        self._preprocess()

    def _preprocess(self):
        """Preprocess and move to device."""
        images = []
        labels = []

        for img, label in self.emnist:
            # EMNIST images are transposed - fix it
            img = img.transpose(1, 2)
            images.append(img)
            # EMNIST labels are 1-indexed, convert to 0-indexed
            labels.append(label - 1)

        self.x = torch.stack(images).squeeze(1).to(self.device)  # [N, 28, 28]
        self.x = self.x.view(self.x.shape[0], -1)  # Flatten to [N, 784]
        self.y = torch.tensor(labels).to(self.device)

        # Normalize to [0, 1] like MNIST
        self.x = self.x / 255.0 if self.x.max() > 1.0 else self.x

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @property
    def num_classes(self):
        return 26  # A-Z
```

### 1.3 Robustness Evaluation Script

**`src/evaluate_robustness.py`**:
```python
"""
Robustness Evaluation Script

Evaluate trained models on distribution-shifted data.
"""

import sys
from pathlib import Path
import argparse
import torch
import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent / "bilinear-decomposition-main"))

from src.data.rotated_mnist import get_rotated_datasets
from src.data.emnist import EMNISTLetters
from src.vision.spectral import extract_from_checkpoint, compute_all_metrics


def evaluate_accuracy(model, dataset):
    """Compute accuracy on a dataset."""
    model.eval()
    with torch.no_grad():
        # Batch evaluation
        logits = model(dataset.x)
        preds = logits.argmax(dim=-1)
        acc = (preds == dataset.y).float().mean().item()
    return acc


def evaluate_rotated_mnist(checkpoint_path: str, angles: list = [0, 15, 30, 45, 60, 90]):
    """
    Evaluate model on rotated MNIST.

    Returns DataFrame with accuracy per angle.
    """
    from image.model import Model, Config

    # Load checkpoint
    data = extract_from_checkpoint(checkpoint_path)
    config = Config(**{k: v for k, v in data['config'].items()
                       if k in ['d_hidden', 'epochs']})

    model = Model(config)
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    # Get rotated datasets
    datasets = get_rotated_datasets(angles)

    results = []
    for angle, dataset in datasets.items():
        acc = evaluate_accuracy(model, dataset)
        results.append({
            'angle': angle,
            'accuracy': acc,
        })

    return pd.DataFrame(results)


def evaluate_emnist(checkpoint_path: str):
    """
    Evaluate MNIST-trained model on EMNIST letters.

    Note: This is expected to fail since model was trained on 10 classes.
    We evaluate feature quality, not accuracy.
    """
    from image.model import Model, Config

    data = extract_from_checkpoint(checkpoint_path)

    # For EMNIST, we can't directly evaluate accuracy (different classes)
    # Instead, we analyze if eigenvectors capture letter-like features
    # This is a qualitative analysis done in the notebook

    return {
        'eigenvalues': data['eigenvalues'],
        'eigenvectors': data['eigenvectors'],
        'metrics': compute_all_metrics(data['eigenvalues']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    # IMPORTANT: Path must match Person E's expected path!
    parser.add_argument('--output', type=str, default='results/phase2/robustness/rotated_mnist_results.csv')
    args = parser.parse_args()

    # Ensure output directory exists
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = evaluate_rotated_mnist(args.checkpoint)
    results.to_csv(args.output, index=False)
    print(results)


if __name__ == '__main__':
    main()
```

### 1.4 Robustness Configs

**`configs/eval_rotated_mnist.yaml`**:
```yaml
name: eval_rotated_mnist
experiment_id: E1.1

# Use best regularized model from Phase 1
checkpoint: results/vision/checkpoints/mnist_dense_full_seed42.pt

# Rotation angles to test
angles: [0, 15, 30, 45, 60, 90]

# Output
output_dir: results/phase2/robustness
```

---

## NOTEBOOK STRUCTURE

**`notebooks/02a_robustness.ipynb`**:
```markdown
# Extension 1: Robustness Analysis

## 1. Setup
- Load Phase 1 checkpoints
- Initialize rotated MNIST datasets

## 2. Rotated MNIST Evaluation
- Evaluate best regularized model on rotations [0, 15, 30, 45, 60, 90]
- Plot accuracy vs rotation angle
- Compare regularized vs non-regularized

## 3. EMNIST Analysis
- Qualitative analysis of feature quality
- Do digit features transfer to letters?

## 4. Eigenspectrum Stability
- Compare eigenspectrum at different rotations
- Does low-rank structure persist?

## 5. Results Summary
- Generate figures for report
- Save to results/phase2/figures/ and Report/figures/
```

---

## EXPECTED OUTPUTS

### Files to Create
```
src/
├── data/
│   ├── __init__.py
│   ├── rotated_mnist.py
│   └── emnist.py
└── evaluate_robustness.py

configs/
└── eval_rotated_mnist.yaml

notebooks/
└── 02a_robustness.ipynb

results/phase2/
├── robustness/
│   ├── rotated_mnist_results.csv    # MUST match Person E's expected path
│   └── emnist_analysis.json         # MUST match Person E's expected path
└── figures/
    ├── robustness_rotated.pdf
    └── robustness_emnist.pdf
```

### Handoff to Person E
On Day 10, provide:
1. `results/phase2/robustness/rotated_mnist_results.csv`
2. `results/phase2/robustness/emnist_analysis.json`
3. Robustness figures in `Report/figures/`
4. Completed `notebooks/02a_robustness.ipynb`

> **NOTE**: Person F handles CP implementation separately and hands off to Person D.

---

## EXECUTION CHECKLIST

### Day 8-9: Data Loaders
- [ ] Create `src/data/rotated_mnist.py`
- [ ] Create `src/data/emnist.py`
- [ ] Test data loaders locally

### Day 9-10: Evaluation
- [ ] Evaluate Phase 1 regularized model on rotated MNIST
- [ ] Generate `robustness_rotated.pdf`
- [ ] Qualitative EMNIST analysis
- [ ] Complete `notebooks/02a_robustness.ipynb`

### Day 10: Finalize
- [ ] All robustness figures saved to `Report/figures/`
- [ ] Results CSV saved to expected path
- [ ] Notify Person E that robustness results are ready

---

**Begin by creating `src/data/rotated_mnist.py`, then test it with a Phase 1 checkpoint. Show me each file and ask for confirmation before proceeding.**

## PROMPT END
