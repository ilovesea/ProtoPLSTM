from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset_sisfall import SiSFallDataset

def plot_sample(sample_idx, sample, save_dir):
    fig, axes = plt.subplots(9, 1, figsize=(12, 12), dpi=300)
    x = range(3000)
    for c in range(9):
        ax = axes[c]
        ax.plot(x, sample[c])
        y_max = np.max(sample[c]) * 1.05
        y_min = np.min(sample[c]) * 0.95
        ax.tick_params(direction="in")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 3000)
        ax.set_xticks(range(0, 3000, 300))
        ax.set_xticklabels([])
        # ax.set_ylabel(channel_labels[c])
    fig.tight_layout()
    fig.savefig(
        save_dir / f"sample{sample_idx}.png", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    fig_dir = Path(__file__).resolve().parent / "figs"
    test_dataset = SiSFallDataset(train=False, expand_dims=True, sample="3000")
    for i, (sample, label) in enumerate(test_dataset):
        plot_sample(i, sample[0], fig_dir / "raw")