import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# ============================================================
# Configuration
# ============================================================

BENCHMARKS = [
    "MMMU-Pro",
    "MathVista",
    "MMStar",
    "AI2D",
]

METHODS = [
    "Direct",
    "Safe RLHF-V",
    "Cap-GRPO",
    "Safe-GRPO",
    "Multi-GRPO",
    "SC-GRPO",
]

# Pastel colours
METHOD_COLORS = {
    "Direct":      "#B8D8E8",
    "Safe RLHF-V": "#F4C2C2",
    "Cap-GRPO":    "#F6D6AD",
    "Safe-GRPO":   "#CDE8C5",
    "Multi-GRPO":  "#F3E3A1",
    "SC-GRPO":     "#A8D5BA",
}


# ============================================================
# Results
# ============================================================

capability_results = {

    "Qwen2.5-VL-3B-Instruct": {
        "Direct": {
            "MMMU-Pro": 98.21,
            "MathVista": 95.10,
            "MMStar": 93.79,
            "AI2D": 94.82,
        },

        # No public checkpoint
        "Safe RLHF-V": {
            "MMMU-Pro": np.nan,
            "MathVista": np.nan,
            "MMStar": np.nan,
            "AI2D": np.nan,
        },

        "Cap-GRPO": {
            "MMMU-Pro": 98.51,
            "MathVista": 99.20,
            "MMStar": 97.80,
            "AI2D": 98.90,
        },

        "Safe-GRPO": {
            "MMMU-Pro": 94.28,
            "MathVista": 94.90,
            "MMStar": 94.47,
            "AI2D": 94.27,
        },

        "Multi-GRPO": {
            "MMMU-Pro": 95.90,
            "MathVista": 95.00,
            "MMStar": 94.39,
            "AI2D": 94.56,
        },

        "SC-GRPO": {
            "MMMU-Pro": 93.41,
            "MathVista": 95.00,
            "MMStar": 94.26,
            "AI2D": 94.75,
        },
    },


    "Qwen2-VL-7B-Instruct": {
        "Direct": {
            "MMMU-Pro": 99.94,
            "MathVista": 93.10,
            "MMStar": 94.60,
            "AI2D": 95.34,
        },

        "Safe RLHF-V": {
            "MMMU-Pro": 99.36,
            "MathVista": 92.60,
            "MMStar": 94.93,
            "AI2D": 93.20,
        },

        "Cap-GRPO": {
            "MMMU-Pro": 99.88,
            "MathVista": 92.50,
            "MMStar": 93.26,
            "AI2D": 94.11,
        },

        "Safe-GRPO": {
            "MMMU-Pro": 99.71,
            "MathVista": 94.10,
            "MMStar": 92.80,
            "AI2D": 93.13,
        },

        "Multi-GRPO": {
            "MMMU-Pro": 99.77,
            "MathVista": 93.69,
            "MMStar": 92.40,
            "AI2D": 93.17,
        },

        "SC-GRPO": {
            "MMMU-Pro": 99.48,
            "MathVista": 93.30,
            "MMStar": 92.67,
            "AI2D": 93.39,
        },
    },


    "LLaVA-v1.6-Mistral-7B-HF": {
        "Direct": {
            "MMMU-Pro": 99.13,
            "MathVista": 87.79,
            "MMStar": 88.73,
            "AI2D": 93.72,
        },

        "Safe RLHF-V": {
            "MMMU-Pro": 99.83,
            "MathVista": 84.95,
            "MMStar": 91.00,
            "AI2D": 97.41,
        },

        "Cap-GRPO": {
            "MMMU-Pro": 99.36,
            "MathVista": 87.49,
            "MMStar": 88.73,
            "AI2D": 95.05,
        },

        "Safe-GRPO": {
            "MMMU-Pro": 92.31,
            "MathVista": 87.07,
            "MMStar": 82.40,
            "AI2D": 90.97,
        },

        "Multi-GRPO": {
            "MMMU-Pro": 99.02,
            "MathVista": 89.07,
            "MMStar": 88.73,
            "AI2D": 92.50,
        },

        "SC-GRPO": {
            "MMMU-Pro": 98.38,
            "MathVista": 87.56,
            "MMStar": 88.73,
            "AI2D": 94.82,
        },
    },
}


# ============================================================
# Utility
# ============================================================

def available_values(method_results):
    return np.array([
        method_results[b]
        for b in BENCHMARKS
        if not np.isnan(method_results[b])
    ])


def average_accuracy(method_results):
    vals = available_values(method_results)

    if len(vals) == 0:
        return np.nan

    return np.mean(vals)


# ============================================================
# 1. Radar plot
# ============================================================

def plot_capability_radar(
    model_name,
    model_results,
    filename,
):

    n = len(BENCHMARKS)

    angles = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False,
    )

    angles_closed = np.concatenate([
        angles,
        [angles[0]],
    ])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    # First benchmark at top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for method in METHODS:

        if method not in model_results:
            continue

        values = np.array([
            model_results[method][b]
            for b in BENCHMARKS
        ])

        # Do not fabricate missing results
        if np.isnan(values).any():
            continue

        values_closed = np.concatenate([
            values,
            [values[0]],
        ])

        # Stronger emphasis for SC-GRPO
        if method == "SC-GRPO":
            linewidth = 3.0
            alpha = 0.14
            zorder = 10
        else:
            linewidth = 1.8
            alpha = 0.04
            zorder = 3

        ax.plot(
            angles_closed,
            values_closed,
            linewidth=linewidth,
            label=method,
            color=METHOD_COLORS[method],
            zorder=zorder,
        )

        ax.fill(
            angles_closed,
            values_closed,
            alpha=alpha,
            color=METHOD_COLORS[method],
            zorder=zorder - 1,
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xticks(angles)

    ax.set_xticklabels(
        BENCHMARKS,
        fontsize=11,
    )

    # Capability values are very close together.
    # Zooming to 85-100 makes differences visible.
    ax.set_ylim(80, 100)

    ax.set_yticks([
        80,
        85,
        90,
        95,
        100,
    ])

    ax.set_yticklabels(
        ["80", "85", "90", "95", "100"],
        fontsize=9,
    )

    ax.set_title(
        model_name,
        fontsize=13,
        fontweight="bold",
        pad=25,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        filename.replace(".png", ".pdf"),
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 2. Capability tax / Delta Accuracy
# ============================================================

def plot_capability_delta(
    model_name,
    model_results,
    filename,
):

    # --------------------------------------------------------
    # IMPORTANT:
    # Compare each method against Direct using only benchmarks
    # available for BOTH methods.
    # --------------------------------------------------------

    direct = model_results["Direct"]

    values = []

    for method in METHODS:

        if method == "Direct":
            continue

        common_benchmarks = [
            b for b in BENCHMARKS
            if not np.isnan(direct[b])
            and not np.isnan(model_results[method][b])
        ]

        if not common_benchmarks:
            continue

        direct_avg = np.mean([
            direct[b]
            for b in common_benchmarks
        ])

        method_avg = np.mean([
            model_results[method][b]
            for b in common_benchmarks
        ])

        delta = method_avg - direct_avg

        values.append(
            (method, delta, len(common_benchmarks))
        )

    # Sort from largest capability loss to strongest retention
    values.sort(key=lambda x: x[1])

    methods = [x[0] for x in values]
    deltas = [x[1] for x in values]

    x = np.arange(len(methods))

    fig, ax = plt.subplots(
        figsize=(8, 4.8)
    )

    bars = ax.bar(
        x,
        deltas,
        width=0.65,
        color=[
            METHOD_COLORS[m]
            for m in methods
        ],
        edgecolor="black",
        linewidth=0.6,
    )

    # --------------------------------------------------------
    # Direct baseline
    # --------------------------------------------------------

    ax.axhline(
        0,
        linewidth=1.2,
        linestyle="--",
        color="black",
        label="Direct",
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_ylabel(
        r"$\Delta$ Average Accuracy (pp)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        methods,
        rotation=20,
        ha="right",
    )

    ax.set_title(
        model_name,
        fontsize=12,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Values
    # --------------------------------------------------------

    for bar, delta in zip(bars, deltas):

        if delta >= 0:
            y = delta + 0.1
            va = "bottom"
        else:
            y = delta - 0.1
            va = "top"

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            y,
            f"{delta:+.2f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    # Emphasise SC-GRPO
    for bar, method in zip(bars, methods):
        if method == "SC-GRPO":
            bar.set_linewidth(1.5)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        filename.replace(".png", ".pdf"),
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Generate figures
# ============================================================

for model_name, model_results in capability_results.items():

    short_name = (
        model_name
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "")
    )

    plot_capability_radar(
        model_name,
        model_results,
        f"figures/capability_radar_{short_name}.png",
    )

    plot_capability_delta(
        model_name,
        model_results,
        f"figures/capability_delta_{short_name}.png",
    )