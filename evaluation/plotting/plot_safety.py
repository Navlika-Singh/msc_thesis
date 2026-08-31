import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

USE_PLACEHOLDERS = False   # Set True ONLY to preview incomplete figures
SAVE_DIR = "figures"

# Methods to show.
# Keeping this relatively small prevents the radar plots becoming unreadable.
METHODS = [
    "Direct",
    "CoT",
    "Safe RLHF-V",
    "Cap-GRPO",
    "Safe-GRPO",
    "Multi-GRPO",
    "SC-GRPO",
]

BENCHMARKS = ["BT-V", "MMS", "VLG", "VLS", "SPA"]

# ============================================================
# Safety results
# Each entry is: benchmark -> (Safety, Helpfulness)
# np.nan = result not currently available
# ============================================================

results = {
    "Qwen2.5-VL-3B-Instruct": {
        "Direct": {
            "BT-V": (51.86, 44.24),
            "MMS":  (56.12, 52.13),
            "VLG":  (83.63, 48.84),
            "VLS":  (58.32, 73.36),
            "SPA":  (79.06, 39.43),
        },
        "CoT": {
            "BT-V": (57.29, 50.68),
            "MMS":  (60.33, 58.16),
            "VLG":  (87.93, 54.24),
            "VLS":  (61.22, 79.61),
            "SPA":  (83.58, 37.36),
        },
        "Safe RLHF-V": {
            "BT-V": (np.nan, np.nan),
            "MMS":  (np.nan, np.nan),
            "VLG":  (np.nan, np.nan),
            "VLS":  (np.nan, np.nan),
            "SPA":  (np.nan, np.nan),
        },
        "Cap-GRPO": {
            "BT-V": (34.58, 52.54),
            "MMS":  (42.51, 60.16),
            "VLG":  (76.51, 54.56),
            "VLS":  (51.32, 80.01),
            "SPA":  (51.70, 47.92),
        },
        "Safe-GRPO": {
            "BT-V": (78.64, 61.86),
            "MMS":  (70.25, 66.44),
            "VLG":  (89.99, 61.04),
            "VLS":  (67.25, 81.53),
            "SPA":  (94.34, 62.45),
        },
        "Multi-GRPO": {
            "BT-V": (49.15, 52.20),
            "MMS":  (53.14, 61.16),
            "VLG":  (83.31, 57.06),
            "VLS":  (60.15, 80.99),
            "SPA":  (72.83, 54.91),
        },
        "SC-GRPO": {
            "BT-V": (83.56, 74.07),
            "MMS":  (75.07, 78.23),
            "VLG":  (91.59, 69.13),
            "VLS":  (70.19, 85.50),
            "SPA":  (94.34, 80.00),
        },
    },

    "Qwen2-VL-7B-Instruct": {
        "Direct": {
            "BT-V": (62.88, 36.10),
            "MMS":  (70.36, 36.34),
            "VLG":  (86.71, 46.79),
            "VLS":  (63.32, 57.83),
            "SPA":  (84.53, 33.58),
        },
        "CoT": {
            "BT-V": (68.81, 37.29),
            "MMS":  (73.95, 35.62),
            "VLG":  (87.93, 43.26),
            "VLS":  (63.36, 64.39),
            "SPA":  (89.43, 30.75),
        },
        "Safe RLHF-V": {
            "BT-V": (75.08, 53.39),
            "MMS":  (78.44, 53.04),
            "VLG":  (89.35, 56.87),
            "VLS":  (71.89, 62.65),
            "SPA":  (91.51, 51.51),
        },
        "Cap-GRPO": {
            "BT-V": (42.54, 46.27),
            "MMS":  (52.26, 55.33),
            "VLG":  (78.95, 55.07),
            "VLS":  (59.25, 74.79),
            "SPA":  (66.79, 50.75),
        },
        "Safe-GRPO": {
            "BT-V": (92.71, 57.46),
            "MMS":  (85.70, 52.77),
            "VLG":  (92.88, 57.57),
            "VLS":  (87.43, 76.98),
            "SPA":  (97.92, 56.04),
        },
        "Multi-GRPO": {
            "BT-V": (69.45, 52.56),
            "MMS":  (75.58, 55.45),
            "VLG":  (87.74, 62.64),
            "VLS":  (71.58, 69.30),
            "SPA":  (87.74, 38.87),
        },
        "SC-GRPO": {
            "BT-V": (97.29, 65.93),
            "MMS":  (89.21, 57.26),
            "VLG":  (94.35, 62.26),
            "VLS":  (92.02, 81.36),
            "SPA":  (98.30, 62.83),
        },
    },

    "LLaVA-v1.6-Mistral-7B-HF": {
        "Direct": {
            "BT-V": (50.22, 44.75),
            "MMS":  (61.74, 51.13),
            "VLG":  (84.08, 51.09),
            "VLS":  (66.22, 81.21),
            "SPA":  (70.47, 46.42),
        },
        "CoT": {
            "BT-V": (69.15, 38.81),
            "MMS":  (70.05, 42.55),
            "VLG":  (90.37, 49.49),
            "VLS":  (71.58, 69.30),
            "SPA":  (87.74, 38.87),
        },
        "Safe RLHF-V": {
            "BT-V": (51.69, 52.37),
            "MMS":  (60.07, 53.78),
            "VLG":  (82.99, 51.54),
            "VLS":  (62.61, 77.64),
            "SPA":  (74.91, 53.77),
        },
        "Cap-GRPO": {
            "BT-V": (31.86, 46.95),
            "MMS":  (48.80, 53.55),
            "VLG":  (73.04, 51.86),
            "VLS":  (52.34, 78.22),
            "SPA":  (45.66, 40.94),
        },
        "Safe-GRPO": {
            "BT-V": (65.00, 56.40),
            "MMS":  (68.65, 55.72),
            "VLG":  (88.91, 53.15),
            "VLS":  (89.24, 80.95),
            "SPA":  (90.00, 62.26),
        },
        "Multi-GRPO": {
            "BT-V": (57.12, 57.29),
            "MMS":  (64.22, 58.31),
            "VLG":  (86.65, 55.46),
            "VLS":  (70.91, 85.23),
            "SPA":  (78.68, 53.21),
        },
        "SC-GRPO": {
            "BT-V": (72.71, 76.10),
            "MMS":  (73.97, 72.14),
            "VLG":  (91.53, 69.58),
            "VLS":  (77.82, 89.87),
            "SPA":  (90.19, 74.72),
        },
    },
}


# ============================================================
# Optional placeholder filling
# ============================================================

def fill_missing(model_results):
    """
    ONLY for visual prototyping.

    Missing values are estimated from the mean of the available
    results for that method. Do NOT use placeholder-filled plots
    in the final thesis.
    """
    output = {}

    for method, vals in model_results.items():
        available_s = [
            s for s, h in vals.values()
            if not np.isnan(s)
        ]
        available_h = [
            h for s, h in vals.values()
            if not np.isnan(h)
        ]

        mean_s = np.mean(available_s) if available_s else 70
        mean_h = np.mean(available_h) if available_h else 60

        output[method] = {}

        for benchmark, (s, h) in vals.items():
            if np.isnan(s):
                s = mean_s
            if np.isnan(h):
                h = mean_h

            output[method][benchmark] = (s, h)

    return output


# ============================================================
# Radar plotting
# ============================================================

def plot_radar(model_name, model_results, filename):
    if USE_PLACEHOLDERS:
        model_results = fill_missing(model_results)

    # Alternate S/H around the radar
    labels = []
    for benchmark in BENCHMARKS:
        labels.extend([
            f"{benchmark}\nSafety",
            f"{benchmark}\nHelpfulness",
        ])

    n_axes = len(labels)

    angles = np.linspace(
        0,
        2 * np.pi,
        n_axes,
        endpoint=False
    )

    # close polygon
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)

    # Put first dimension at the top and go clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for method in METHODS:
        if method not in model_results:
            continue

        values = []

        for benchmark in BENCHMARKS:
            s, h = model_results[method][benchmark]
            values.extend([s, h])

        values = np.array(values, dtype=float)

        # A radar polygon cannot sensibly cross missing dimensions.
        # Skip incomplete methods unless placeholders are enabled.
        if np.isnan(values).any():
            continue

        values_closed = np.concatenate([values, [values[0]]])

        # Make SC-GRPO visually stronger
        if method == "SC-GRPO":
            linewidth = 3.0
            alpha = 0.10
            zorder = 10
        else:
            linewidth = 1.6
            alpha = 0.025
            zorder = 3

        line, = ax.plot(
            angles_closed,
            values_closed,
            linewidth=linewidth,
            label=method,
            zorder=zorder,
        )

        ax.fill(
            angles_closed,
            values_closed,
            alpha=alpha,
            zorder=zorder - 1,
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)

    # All reported metrics are percentages
    ax.set_ylim(0, 100)

    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(
        ["20", "40", "60", "80", "100"],
        fontsize=8,
    )

    ax.set_title(
        model_name,
        fontsize=14,
        pad=25,
        fontweight="bold",
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.savefig(filename.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# Generate plots
# ============================================================

import os
os.makedirs(SAVE_DIR, exist_ok=True)

for model_name, model_results in results.items():
    short_name = (
        model_name
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "")
    )

    plot_radar(
        model_name,
        model_results,
        f"{SAVE_DIR}/radar_{short_name}.png",
    )

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# ============================================================
# Methods
# ============================================================

METHODS = [
    "Direct",
    "CoT",
    "Safe RLHF-V",
    "Cap-GRPO",
    "Safe-GRPO",
    "Multi-GRPO",
    "SC-GRPO",
]


# ============================================================
# Metric computation
# ============================================================

def harmonic_mean(s, h):
    if np.isnan(s) or np.isnan(h) or (s + h) == 0:
        return np.nan
    return 2 * s * h / (s + h)


def compute_average_metrics(model_results):
    """
    Average Safety, Helpfulness, and benchmark-level HM
    across all available safety benchmarks.
    """
    averages = {}

    for method in METHODS:
        if method not in model_results:
            continue

        safety = []
        helpfulness = []
        hm = []

        for benchmark in BENCHMARKS:
            s, h = model_results[method][benchmark]

            if np.isnan(s) or np.isnan(h):
                continue

            safety.append(s)
            helpfulness.append(h)
            hm.append(harmonic_mean(s, h))

        if safety:
            averages[method] = {
                "Safety": np.mean(safety),
                "Helpfulness": np.mean(helpfulness),
                "HM": np.mean(hm),
            }

    return averages


# ============================================================
# Pastel method colors
# ============================================================

METHOD_COLORS = {
    "Direct":       "#B8D8E8",  # pastel blue
    "CoT":          "#C9C3E6",  # pastel purple
    "Safe RLHF-V":  "#F4C2C2",  # pastel red
    "Cap-GRPO":     "#F6D6AD",  # pastel orange
    "Safe-GRPO":    "#CDE8C5",  # pastel green
    "Multi-GRPO":   "#F3E3A1",  # pastel yellow
    "SC-GRPO":      "#A8D5BA",  # stronger pastel green
}


# ============================================================
# Plot one metric
# ============================================================

def plot_metric(
    model_name,
    averages,
    metric,
    filename,
):

    values = [
        (method, scores[metric])
        for method, scores in averages.items()
        if not np.isnan(scores[metric])
    ]

    # Sort ascending -> best method appears furthest right
    values.sort(key=lambda x: x[1])

    methods = [x[0] for x in values]
    scores = [x[1] for x in values]

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars = ax.bar(
        x,
        scores,
        width=0.68,
        color=[METHOD_COLORS[m] for m in methods],
        edgecolor="black",
        linewidth=0.6,
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_ylabel(f"Average {metric} (%)")
    ax.set_ylim(0, 100)

    ax.set_xticks(x)
    ax.set_xticklabels(
        methods,
        rotation=20,
        ha="right",
    )

    ax.set_title(
        f"{model_name} — {metric}",
        fontsize=12,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Values above bars
    # --------------------------------------------------------

    for bar, value in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Slightly emphasize SC-GRPO
    # --------------------------------------------------------

    # for bar, method in zip(bars, methods):
    #     if method == "SC-GRPO":
    #         bar.set_linewidth(1.5)

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
# Generate all plots
# ============================================================

for model_name, model_results in results.items():

    averages = compute_average_metrics(model_results)

    short_name = (
        model_name
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "")
    )

    for metric in ["Safety", "Helpfulness", "HM"]:

        plot_metric(
            model_name=model_name,
            averages=averages,
            metric=metric,
            filename=(
                f"figures/{short_name}_"
                f"{metric.lower()}_average.png"
            ),
        )