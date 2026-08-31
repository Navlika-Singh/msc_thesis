import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# ============================================================
# Data
# Each entry: HM improvement, Safety improvement,
#             Helpfulness improvement
# ============================================================

results = {
    "Qwen2.5-VL-3B": {
        "BeaverTails-V": {
            "HM": 30.78,
            "Safety": 31.70,
            "Helpfulness": 29.83,
        },
        "MM-SafetyBench": {
            "HM": 22.57,
            "Safety": 18.95,
            "Helpfulness": 26.10,
        },
        "VLGuard": {
            "HM": 17.12,
            "Safety": 7.96,
            "Helpfulness": 20.29,
        },
        "VLSBench": {
            "HM": 12.11,
            "Safety": 11.87,
            "Helpfulness": 12.14,
        },
        "SPA-VL": {
            "HM": 33.96,
            "Safety": 15.28,
            "Helpfulness": 40.57,
        },
    },

    "Qwen2-VL-7B": {
        "BeaverTails-V": {
            "HM": 32.73,
            "Safety": 34.41,
            "Helpfulness": 29.83,
        },
        "MM-SafetyBench": {
            "HM": 21.82,
            "Safety": 18.85,
            "Helpfulness": 20.92,
        },
        "VLGuard": {
            "HM": 14.24,
            "Safety": 7.64,
            "Helpfulness": 15.47,
        },
        "VLSBench": {
            "HM": 25.91,
            "Safety": 28.70,
            "Helpfulness": 23.53,
        },
        "SPA-VL": {
            "HM": 28.59,
            "Safety": 13.77,
            "Helpfulness": 29.25,
        },
    },

    "LLaVA-v1.6-7B": {
        "BeaverTails-V": {
            "HM": 27.00,
            "Safety": 22.49,
            "Helpfulness": 31.35,
        },
        "MM-SafetyBench": {
            "HM": 17.11,
            "Safety": 12.23,
            "Helpfulness": 21.01,
        },
        "VLGuard": {
            "HM": 15.50,
            "Safety": 7.45,
            "Helpfulness": 18.49,
        },
        "VLSBench": {
            "HM": 10.46,
            "Safety": 11.60,
            "Helpfulness": 12.23,
        },
        "SPA-VL": {
            "HM": 25.76,
            "Safety": 19.72,
            "Helpfulness": 28.30,
        },
    },
}


# ============================================================
# Configuration
# ============================================================

MODELS = list(results.keys())

BENCHMARKS = [
    "BeaverTails-V",
    "MM-SafetyBench",
    "VLGuard",
    "VLSBench",
    "SPA-VL",
]

# Different markers make individual backbones distinguishable
MODEL_MARKERS = {
    "Qwen2.5-VL-3B": "o",
    "Qwen2-VL-7B": "s",
    "LLaVA-v1.6-7B": "^",
}

# Pastel colours for each metric
METRIC_COLORS = {
    "Safety": "#B8D8E8",
    "Helpfulness": "#CDE8C5",
    "HM": "#C9C3E6",
}


# ============================================================
# Compute benchmark-level averages
# ============================================================

def benchmark_average(benchmark, metric):
    values = [
        results[model][benchmark][metric]
        for model in MODELS
    ]
    return np.mean(values)


# Print averages for sanity checking
for metric in ["Safety", "Helpfulness", "HM"]:
    print(f"\n{metric}")
    for benchmark in BENCHMARKS:
        avg = benchmark_average(benchmark, metric)
        print(f"{benchmark:18s}: {avg:.2f}")


# ============================================================
# Plot
# ============================================================

def plot_benchmark_improvements(metric, filename):

    # --------------------------------------------------------
    # Calculate averages
    # --------------------------------------------------------

    averages = {
        benchmark: benchmark_average(benchmark, metric)
        for benchmark in BENCHMARKS
    }

    # Sort lowest -> highest
    ordered_benchmarks = sorted(
        BENCHMARKS,
        key=lambda b: averages[b]
    )

    avg_values = [
        averages[b]
        for b in ordered_benchmarks
    ]

    x = np.arange(len(ordered_benchmarks))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    # --------------------------------------------------------
    # Average bars
    # --------------------------------------------------------

    bars = ax.bar(
        x,
        avg_values,
        width=0.65,
        color=METRIC_COLORS[metric],
        edgecolor="black",
        linewidth=0.7,
        zorder=2,
    )

    # --------------------------------------------------------
    # Mark BeaverTails-V as ID
    # --------------------------------------------------------

    for bar, benchmark in zip(bars, ordered_benchmarks):

        if benchmark == "BeaverTails-V":
            bar.set_hatch("///")
            bar.set_linewidth(1.2)

    # --------------------------------------------------------
    # Overlay individual backbone values
    # --------------------------------------------------------

    offsets = np.linspace(-0.12, 0.12, len(MODELS))

    for offset, model in zip(offsets, MODELS):

        values = [
            results[model][benchmark][metric]
            for benchmark in ordered_benchmarks
        ]

        ax.scatter(
            x + offset,
            values,
            marker=MODEL_MARKERS[model],
            s=42,
            edgecolor="black",
            linewidth=0.6,
            label=model,
            zorder=4,
        )

    # --------------------------------------------------------
    # Average value labels
    # --------------------------------------------------------

    for bar, value in zip(bars, avg_values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # Benchmark labels
    # --------------------------------------------------------

    labels = []

    for benchmark in ordered_benchmarks:

        if benchmark == "BeaverTails-V":
            labels.append("BeaverTails-V\n(ID)")
        else:
            labels.append(benchmark)

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=10,
    )

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------

    ax.set_ylabel(
        f"Average $\\Delta$ {metric} (points)",
        fontsize=11,
    )

    ax.set_xlabel(
        "Safety Benchmark",
        fontsize=11,
    )

    ax.set_ylim(
        0,
        max(
            results[m][b][metric]
            for m in MODELS
            for b in BENCHMARKS
        ) + 6
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
        zorder=0,
    )

    ax.set_axisbelow(True)

    # Remove unnecessary borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        ncol=1,
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
# Generate all three figures
# ============================================================

plot_benchmark_improvements(
    "Safety",
    "figures/benchmark_avg_safety_improvement.png",
)

plot_benchmark_improvements(
    "Helpfulness",
    "figures/benchmark_avg_helpfulness_improvement.png",
)

plot_benchmark_improvements(
    "HM",
    "figures/benchmark_avg_hm_improvement.png",
)