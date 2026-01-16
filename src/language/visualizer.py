"""
SAE Feature Visualization for Language Models.

Provides tools for visualizing which tokens activate specific SAE features,
and understanding feature semantics through logit influence analysis.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import torch
from torch import Tensor
import numpy as np

# Add paths for original code
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _get_color_for_value(value: float, max_val: float, cmap_name: str = 'Blues') -> Tuple[int, int, int]:
    """Get RGB color for a normalized activation value."""
    if max_val <= 0:
        return (255, 255, 255)
    normalized = min(value / max_val, 1.0) * 0.8  # Cap at 0.8 for visibility
    cmap = plt.cm.get_cmap(cmap_name)
    rgba = cmap(normalized)
    return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


def _ansi_color_str(text: str, rgb: Tuple[int, int, int], value: float) -> str:
    """Create ANSI-colored string for terminal output."""
    r, g, b = rgb
    if value == 0:
        return text
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


class FeatureVisualizer:
    """
    Visualize SAE feature activations on text.

    Supports:
    - Terminal output with ANSI colors
    - HTML output for notebooks/web
    - Matplotlib heatmaps
    - Plotly interactive visualizations
    """

    def __init__(
        self,
        model,
        sae,
        tokenizer,
        device: str = "cpu"
    ):
        """
        Initialize visualizer.

        Args:
            model: Transformer model with .forward() method
            sae: SAE with .encode() method
            tokenizer: Tokenizer with encode/decode methods
            device: Device for computation
        """
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = device

        # Move SAE to device if needed
        if hasattr(sae, 'to'):
            self.sae = sae.to(device)

    def get_feature_activations(
        self,
        text: str,
        feature_idx: int,
    ) -> Tuple[List[str], Tensor]:
        """
        Get per-token activations for a specific feature.

        Args:
            text: Input text
            feature_idx: Which SAE feature to analyze

        Returns:
            Tuple of (tokens, activations)
        """
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(text, return_tensors='pt')
        else:
            input_ids = self.tokenizer(text, return_tensors='pt')['input_ids']

        input_ids = input_ids.to(self.device)

        # Get model activations at SAE hook point
        with torch.no_grad():
            # Run model and capture MLP output
            if hasattr(self.model, 'forward'):
                # For models with hook interface
                outputs = self.model(input_ids, output_hidden_states=True)
                # Get hidden states at the layer where SAE is applied
                if hasattr(self.sae, 'point'):
                    layer = self.sae.point.layer
                    hidden = outputs.hidden_states[layer]
                else:
                    # Default to last hidden state
                    hidden = outputs.hidden_states[-1]
            else:
                # Fallback: just get embeddings
                hidden = self.model(input_ids)

            # Encode through SAE
            sae_acts = self.sae.encode(hidden.squeeze(0))  # [seq_len, n_features]

            # Get activations for specific feature
            feature_acts = sae_acts[:, feature_idx]

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        return tokens, feature_acts.cpu()

    def color_tokens_terminal(
        self,
        text: str,
        feature_idx: int,
        cmap: str = 'Blues',
        context_window: int = 50,
    ) -> str:
        """
        Return ANSI-colored string for terminal display.

        Args:
            text: Input text
            feature_idx: Feature to visualize
            cmap: Colormap name ('Blues', 'magma', etc.)
            context_window: Number of tokens around max activation to show

        Returns:
            ANSI-colored string
        """
        tokens, activations = self.get_feature_activations(text, feature_idx)

        max_val = activations.max().item()
        max_idx = activations.argmax().item()

        # Determine display window
        start = max(0, max_idx - context_window // 2)
        end = min(len(tokens), max_idx + context_window // 2)

        colored_parts = []
        for i in range(start, end):
            token = tokens[i]
            act = activations[i].item()
            rgb = _get_color_for_value(act, max_val, cmap)
            colored_parts.append(_ansi_color_str(token, rgb, act))

        return f"{max_val:.2f}: " + "".join(colored_parts)

    def color_tokens_html(
        self,
        text: str,
        feature_idx: int,
        cmap: str = 'Blues',
    ) -> str:
        """
        Return HTML with colored tokens for notebook/web display.

        Args:
            text: Input text
            feature_idx: Feature to visualize
            cmap: Colormap name

        Returns:
            HTML string
        """
        tokens, activations = self.get_feature_activations(text, feature_idx)

        max_val = activations.max().item()

        html_parts = ['<div style="font-family: monospace; line-height: 1.8;">']
        for token, act in zip(tokens, activations):
            act_val = act.item()
            rgb = _get_color_for_value(act_val, max_val, cmap)
            bg_color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

            # Clean token for display
            display_token = token.replace('▁', ' ').replace('Ġ', ' ')
            if act_val > 0:
                html_parts.append(
                    f'<span style="background-color:{bg_color}; padding:2px; '
                    f'border-radius:2px;" title="act={act_val:.3f}">{display_token}</span>'
                )
            else:
                html_parts.append(f'<span>{display_token}</span>')

        html_parts.append('</div>')
        return "".join(html_parts)

    def plot_activation_heatmap(
        self,
        texts: List[str],
        feature_idx: int,
        max_tokens: int = 50,
        figsize: Tuple[float, float] = (12, 6),
        save_path: Optional[str] = None,
    ):
        """
        Plot heatmap of feature activations across multiple texts.

        Args:
            texts: List of input texts
            feature_idx: Feature to visualize
            max_tokens: Maximum tokens to show per text
            figsize: Figure size
            save_path: If provided, save figure

        Returns:
            matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available")
            return None

        all_tokens = []
        all_activations = []

        for text in texts:
            tokens, acts = self.get_feature_activations(text, feature_idx)
            # Truncate to max_tokens
            tokens = tokens[:max_tokens]
            acts = acts[:max_tokens]
            # Pad if needed
            while len(tokens) < max_tokens:
                tokens.append('')
                acts = torch.cat([acts, torch.zeros(1)])

            all_tokens.append(tokens)
            all_activations.append(acts.numpy())

        # Create heatmap
        act_matrix = np.stack(all_activations)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(act_matrix, aspect='auto', cmap='Blues')

        # Labels
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Text Index')
        ax.set_title(f'Feature {feature_idx} Activations')

        plt.colorbar(im, ax=ax, label='Activation')

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def show_logit_influence(
        self,
        feature_idx: int,
        top_k: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Show which output logits are most influenced by this feature.

        Uses the SAE decoder weights to find tokens most associated with the feature.

        Args:
            feature_idx: Feature to analyze
            top_k: Number of top positive and negative tokens to show
            save_path: If provided, save figure

        Returns:
            matplotlib Figure or dict of results
        """
        # Get decoder weights for this feature
        if hasattr(self.sae, 'w_dec'):
            # Original SAE format: w_dec.weight is [d_model, n_features]
            feature_vec = self.sae.w_dec.weight[:, feature_idx]
        elif hasattr(self.sae, 'decoder'):
            feature_vec = self.sae.decoder.weight[:, feature_idx]
        else:
            print("Cannot find SAE decoder weights")
            return None

        # Project to vocabulary via unembedding
        if hasattr(self.model, 'w_u'):
            # Original model format
            logit_influence = torch.einsum('d, vd -> v', feature_vec, self.model.w_u)
        elif hasattr(self.model, 'lm_head'):
            logit_influence = feature_vec @ self.model.lm_head.weight.T
        else:
            print("Cannot find model unembedding weights")
            return None

        # Get top positive and negative
        pos_vals, pos_idx = logit_influence.topk(top_k)
        neg_vals, neg_idx = logit_influence.topk(top_k, largest=False)

        pos_tokens = [self.tokenizer.decode([idx.item()]) for idx in pos_idx]
        neg_tokens = [self.tokenizer.decode([idx.item()]) for idx in neg_idx]

        results = {
            'positive': list(zip(pos_tokens, pos_vals.tolist())),
            'negative': list(zip(neg_tokens, neg_vals.tolist())),
        }

        if MATPLOTLIB_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            # Positive influence
            ax1.barh(range(top_k), pos_vals.cpu().numpy(), color='green', alpha=0.7)
            ax1.set_yticks(range(top_k))
            ax1.set_yticklabels(pos_tokens)
            ax1.set_xlabel('Logit Influence')
            ax1.set_title(f'Feature {feature_idx}: Positive Influence')
            ax1.invert_yaxis()

            # Negative influence
            ax2.barh(range(top_k), neg_vals.cpu().numpy(), color='red', alpha=0.7)
            ax2.set_yticks(range(top_k))
            ax2.set_yticklabels(neg_tokens)
            ax2.set_xlabel('Logit Influence')
            ax2.set_title(f'Feature {feature_idx}: Negative Influence')
            ax2.invert_yaxis()

            plt.tight_layout()

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight')
                print(f"Saved: {save_path}")

            return fig

        return results

    def visualize_feature_interactive(
        self,
        texts: List[str],
        feature_idx: int,
        save_path: Optional[str] = None,
    ):
        """
        Interactive Plotly visualization of feature activations.

        Args:
            texts: List of input texts
            feature_idx: Feature to visualize
            save_path: If provided, save as HTML

        Returns:
            Plotly Figure or None
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available")
            return None

        all_data = []
        for i, text in enumerate(texts):
            tokens, acts = self.get_feature_activations(text, feature_idx)
            for j, (token, act) in enumerate(zip(tokens, acts)):
                all_data.append({
                    'text_idx': i,
                    'token_idx': j,
                    'token': token.replace('▁', ' ').replace('Ġ', ' '),
                    'activation': act.item()
                })

        import pandas as pd
        df = pd.DataFrame(all_data)

        # Pivot for heatmap
        pivot = df.pivot(index='text_idx', columns='token_idx', values='activation')

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            colorscale='Blues',
            hovertemplate='Text %{y}<br>Token %{x}<br>Activation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Feature {feature_idx} Activations',
            xaxis_title='Token Position',
            yaxis_title='Text Index',
            template='plotly_white',
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            print(f"Saved: {save_path}")

        return fig
