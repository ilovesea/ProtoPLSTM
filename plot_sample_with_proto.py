from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import torch

from protopnet import AProtoPNet
from dataset_sisfall import SiSFallDataset

samples = {
    99: [31, 66, 87],
    284: [107, 122, 144]
}


def find_high_activation_crop(activation_map, percentile=0.99):
    threshold = torch.quantile(activation_map, percentile)
    mask = torch.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    for c in range(9):
        start = 0
        end = 0
        for i in range(mask.shape[1]):
            if torch.amax(mask[c, i]) > 0.5:
                start = i
                break
        for i in reversed(range(mask.shape[1])):
            if torch.amax(mask[c, i]) > 0.5:
                end = i
                break
        if end != 0:
            return (start, end), c


def plot_sample_with_prototype_rect(sample_idx,
                                    sample,
                                    rects,
                                    proto_idx,
                                    label,
                                    save_dir):
    fig, axes = plt.subplots(9, 1, figsize=(12, 12), dpi=300)
    x = range(3000)
    for c, rect in enumerate(rects):
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
        if rect[1] > 0:
            patch = patches.Rectangle((rect[0], y_min - (y_max - y_min) * 0.05),
                                      rect[1] - rect[0],
                                      (y_max - y_min) * 1.1,
                                      linewidth=1,
                                      edgecolor='r',
                                      facecolor='none',
                                      clip_on=False)
            ax.add_patch(patch)
    fig.tight_layout()
    fig.savefig(
        save_dir / f"sample{sample_idx}_class{label}_proto{proto_idx}.png", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda:1")
    model: torch.nn.Module = AProtoPNet().to(device)
    model.eval()
    sd = torch.load(Path(__file__).resolve().parent /
                    "saved_models/best_model_a6057c79.pt", map_location="cpu")
    fig_dir = Path(__file__).resolve().parent / "figs"
    model.load_state_dict(sd)

    prototypes = pd.read_csv(Path(__file__).resolve(
    ).parent / "saved_models/prototypes.csv", sep=" ")
    train_dataset = SiSFallDataset(train=True, expand_dims=True, sample="3000")
    test_dataset = SiSFallDataset(train=False, expand_dims=True, sample="3000")

    for sample_idx, protos in samples.items():
        sample = 
        model.prototype_visualize(torch.FloatTensor(batch0[0]).to(device), torch.LongTensor(batch0[1]).to(device))

    for prototype in prototypes.itertuples():
        sample = train_dataset[prototype.train_sample]
        visual_pattern = model.prototype_visualize(torch.FloatTensor(
            sample[0]).unsqueeze(0).to(device), torch.LongTensor([sample[1]]).to(device))
        visual_pattern = visual_pattern[0]
        sim_idx = visual_pattern[1].index(prototype.Index)
        rects = find_high_activation_crop(visual_pattern[0][sim_idx])
        plot_sample_with_prototype_rect(
            prototype.train_sample, sample[0].squeeze(0), rects, prototype.Index, sample[1], fig_dir / "prototype")

    test_dataset = SiSFallDataset(train=False, expand_dims=True, sample="3000")
    batch0 = test_dataset[:]
    visual_patterns = model.prototype_visualize(torch.FloatTensor(
        batch0[0]).to(device), torch.LongTensor(batch0[1]).to(device))
    torch.save(visual_patterns, fig_dir / "visual_patterns.pt")

    visual_patterns = torch.load(fig_dir / "visual_patterns.pt")

    for i, (sample_visual_pattern, proto_idxs, label) in enumerate(visual_patterns):
        for visual_pattern, proto_idx in zip(sample_visual_pattern, proto_idxs):
            rects = find_high_activation_crop(visual_pattern)
            plot_sample_with_prototype_rect(
                i, batch0[0][i].squeeze(0), rects, proto_idx, label, fig_dir
                / "test")
