import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Output directory
# ============================================================

os.makedirs("figures", exist_ok=True)

# ============================================================
# Qwen2.5-VL-3B-Instruct results
# ============================================================

results = {
    "Cap-GRPO": {
        "BT-V": {"S": 34.58, "H": 52.54, "HM": 41.71},
        "MMS":  {"S": 42.51, "H": 60.16, "HM": 49.82},
        "VLG":  {"S": 76.51, "H": 54.56, "HM": 63.69},
        "VLS":  {"S": 51.32, "H": 80.01, "HM": 62.53},
        "SPA":  {"S": 51.70, "H": 47.92, "HM": 49.74},
    },

    "Safe-GRPO": {
        "BT-V": {"S": 78.64, "H": 61.86, "HM": 69.25},
        "MMS":  {"S": 70.25, "H": 66.44, "HM": 68.30},
        "VLG":  {"S": 89.99, "H": 61.04, "HM": 72.74},
        "VLS":  {"S": 67.25, "H": 81.53, "HM": 73.70},
        "SPA":  {"S": 94.34, "H": 62.45, "HM": 75.15},
    },

    "Multi-GRPO": {
        "BT-V": {"S": 49.15, "H": 52.20, "HM": 50.63},
        "MMS":  {"S": 53.14, "H": 61.16, "HM": 56.87},
        "VLG":  {"S": 83.31, "H": 57.06, "HM": 67.73},
        "VLS":  {"S": 60.15, "H": 80.99, "HM": 69.03},
        "SPA":  {"S": 72.83, "H": 54.91, "HM": 62.61},
    },

    "SC-GRPO": {
        "BT-V": {"S": 83.56, "H": 74.07, "HM": 78.53},
        "MMS":  {"S": 75.07, "H": 78.23, "HM": 76.62},
        "VLG":  {"S": 91.59, "H": 69.13, "HM": 78.79},
        "VLS":  {"S": 70.19, "H": 85.50, "HM": 77.09},
        "SPA":  {"S": 94.34, "H": 80.00, "HM": 86.58},
    },
}

# ============================================================
# Configuration
# ============================================================

methods = [
    "Cap-GRPO",
    "Safe-GRPO",
    "Multi-GRPO",
    "SC-GRPO",
]

benchmarks = [
    "BT-V",
    "MMS",
    "VLG",
    "VLS",
    "SPA",
]

metrics = [
    "S",
    "H",
    "HM",
]

metric_labels = [
    "Safety",
    "Helpfulness",
    "HM",
]

# Pastel colours, consistent with previous figures
colors = {
    "Cap-GRPO":   "#B8D8E8",
    "Safe-GRPO":  "#CDE8C5",
    "Multi-GRPO": "#F3E3A1",
    "SC-GRPO":    "#C9C3E6",
}

# ============================================================
# Compute averages across benchmarks
# ============================================================

averages = {}

for method in methods:
    averages[method] = {}

    for metric in metrics:
        values = [
            results[method][benchmark][metric]
            for benchmark in benchmarks
        ]

        averages[method][metric] = np.mean(values)

# Print values for sanity checking
print("\nAverage performance across safety benchmarks:\n")

for method in methods:
    print(
        f"{method:12s} | "
        f"Safety = {averages[method]['S']:.2f} | "
        f"Helpfulness = {averages[method]['H']:.2f} | "
        f"HM = {averages[method]['HM']:.2f}"
    )

# ============================================================
# Plot
# ============================================================

x = np.arange(len(metrics))

width = 0.19

# Four bars centred around each metric
offsets = np.array([
    -1.5,
    -0.5,
     0.5,
     1.5,
]) * width

fig, ax = plt.subplots(
    figsize=(7.4, 4.8)
)

all_bars = []

for method, offset in zip(methods, offsets):

    values = [
        averages[method][metric]
        for metric in metrics
    ]

    bars = ax.bar(
        x + offset,
        values,
        width=width,
        label=method,
        color=colors[method],
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    all_bars.append(bars)

# ============================================================
# Add numerical labels
# ============================================================

for bars in all_bars:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.8,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

# ============================================================
# Axis formatting
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels(
    metric_labels,
    fontsize=11,
)

ax.set_ylabel(
    "Average Score (%)",
    fontsize=11,
)

ax.set_ylim(
    0,
    100,
)

ax.tick_params(
    axis="y",
    labelsize=10,
)

# ============================================================
# Grid
# ============================================================

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.25,
    zorder=0,
)

ax.set_axisbelow(True)

# ============================================================
# Clean borders
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ============================================================
# Legend
# ============================================================

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper left",
    fontsize=9,
)

# ============================================================
# Layout + save
# ============================================================

plt.tight_layout()

plt.savefig(
    "figures/grpo_objective_comparison.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "figures/grpo_objective_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()