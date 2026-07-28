#%%
"""
Create visual verification figures for the correlated-Beta implementation.

Outputs:
1. Target(assumed) Gaussian Pearson correlation heatmap.
2. Empirical Gaussian Pearson correlation heatmaps for:
   - CS1 -> CS2
   - CS2 -> CS3
   - CS3 -> CS4
3. A Beta probability histogram for every element and every transition.

All figures are saved inside:
    verification_correlated_beta
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    BETA_PROBABILITY_VARIANCE,
    DO_NOTHING_TRANSITIONS,
    ELEMENT_CORRELATION_MATRIX,
    ELEMENT_NUMBERS,
    gamma,
    include_step_count,
    max_steps,
    reset_prob,
)

# ============================================================
# User settings
# ============================================================
N_SAMPLES = 50_000
SEED = 20260728
OUTPUT_DIR = Path("verification_correlated_beta")
HISTOGRAM_BINS = 80

TRANSITION_NAMES = (
    "CS1_to_CS2",
    "CS2_to_CS3",
    "CS3_to_CS4",
)

TRANSITION_TITLES = (
    "CS1 to CS2",
    "CS2 to CS3",
    "CS3 to CS4",
)


class RecordingGenerator:
    """
    Wrap NumPy's random generator and record multivariate-normal draws.

    Every method other than multivariate_normal is delegated to the original
    NumPy generator, so the environment's random behavior is unchanged.
    """

    def __init__(self, generator):
        self._generator = generator
        self.multivariate_normal_draws = []

    def multivariate_normal(self, *args, **kwargs):
        draw = self._generator.multivariate_normal(*args, **kwargs)
        self.multivariate_normal_draws.append(np.asarray(draw, dtype=float).copy())
        return draw

    def __getattr__(self, name):
        return getattr(self._generator, name)


def save_correlation_heatmap(matrix, title, output_path, element_labels):
    """
    Save an annotated Pearson correlation heatmap.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, aspect="equal")

    ax.set_xticks(np.arange(len(element_labels)))
    ax.set_yticks(np.arange(len(element_labels)))
    ax.set_xticklabels(element_labels, rotation=45,ha="right")
    ax.set_yticklabels(element_labels)

    ax.set_xlabel("Element")
    ax.set_ylabel("Element")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    colorbar.set_label("Pearson correlation")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_beta_histogram(samples, target_mean, target_variance, transition_title, transition_name, element_number, output_dir):
    """
    Save one Beta transition-probability histogram.
    """
    samples = np.asarray(samples, dtype=float)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples, ddof=1))

    maximum_sample = float(np.max(samples))
    x_upper = max(maximum_sample * 1.05, target_mean * 4.0,0.05)
    x_upper = min(x_upper, 1.0)

    weights = np.full(samples.shape, 100.0 / samples.size, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(samples, bins=HISTOGRAM_BINS, range=(0.0, x_upper), weights=weights)

    ax.axvline(target_mean, linestyle="--", linewidth=2.0)

    statistics_text = (
        f"Target mean = {target_mean:.5f}\n"
        f"Target variance = {target_variance:.6f}\n"
        f"Sample mean = {sample_mean:.5f}\n"
        f"Sample variance = {sample_variance:.6f}"
    )

    ax.text(
        0.98,
        0.98,
        statistics_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
        },
    )

    ax.set_xlim(0.0, x_upper)
    ax.set_xlabel("Sampled transition probability")
    ax.set_ylabel("Percent of samples")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

    ax.set_title("Beta probability distribution: " f"{transition_title}, element {element_number}")

    fig.tight_layout()

    output_path = (output_dir/ f"{transition_name}_element_{element_number}_beta_distribution.png"
    )

    fig.savefig(output_path, dpi=300,bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    element_labels = [str(int(element)) for element in ELEMENT_NUMBERS]

    number_of_elements = len(ELEMENT_NUMBERS)
    number_of_transitions = len(TRANSITION_NAMES)

    env = BridgeBHIEnv(
        max_steps=max_steps,
        discount=gamma,
        include_step_count=include_step_count,
        reset_prob=reset_prob,
        transition_mode="correlated_beta",
        beta_transition_variance=BETA_PROBABILITY_VARIANCE,
        element_correlation_matrix=ELEMENT_CORRELATION_MATRIX,
        seed=SEED,
    )
    env.reset(seed=SEED)

    recording_rng = RecordingGenerator(env.np_random)
    env.np_random = recording_rng

    # Shape:
    # samples x transitions x elements
    latent_samples = np.empty(
        (
            N_SAMPLES,
            number_of_transitions,
            number_of_elements,
        ),
        dtype=float,
    )

    beta_samples = np.empty_like(latent_samples)

    for sample_index in range(N_SAMPLES):
        draw_start = len(recording_rng.multivariate_normal_draws)

        sampled_transition_matrices = (env._sample_correlated_beta_transition_matrices())

        new_draws = (recording_rng.multivariate_normal_draws[draw_start:])


        latent_samples[sample_index] = np.asarray(
            new_draws,
            dtype=float)

        for transition_index in range(number_of_transitions):
            for element_index, element_number_raw in enumerate(
                ELEMENT_NUMBERS):
                element_number = int(element_number_raw)

                beta_samples[sample_index, transition_index, element_index] = sampled_transition_matrices[element_number][transition_index,transition_index + 1]

        if (sample_index + 1) % 5_000 == 0:
            print(
                f"Generated {sample_index + 1:,} "
                f"of {N_SAMPLES:,} samples."
            )

    # ========================================================
    # Pearson correlation heatmaps
    # ========================================================
    save_correlation_heatmap(
        matrix=ELEMENT_CORRELATION_MATRIX,
        title="Target(assumed) Gaussian Pearson correlation matrix",
        output_path=(
            OUTPUT_DIR
            / "target_gaussian_pearson_correlation.png"
        ),
        element_labels=element_labels,
    )

    for transition_index, transition_name in enumerate(
        TRANSITION_NAMES):
        empirical_correlation = np.corrcoef(
            latent_samples[:, transition_index, :],
            rowvar=False,
        )

        save_correlation_heatmap(
            matrix=empirical_correlation,
            title=("Empirical Gaussian Pearson correlation matrix\n" f"{TRANSITION_TITLES[transition_index]}"),
            output_path=(OUTPUT_DIR/ ("empirical_gaussian_pearson_" f"{transition_name}.png")),
            element_labels=element_labels
        )

    # ========================================================
    # Beta marginal histograms
    # ========================================================
    for transition_index, transition_name in enumerate(TRANSITION_NAMES):
        for element_index, element_number_raw in enumerate(
            ELEMENT_NUMBERS):
            element_number = int(element_number_raw)

            target_mean = float(
                DO_NOTHING_TRANSITIONS[element_number][
                    transition_index,
                    transition_index + 1,
                ]
            )

            save_beta_histogram(
                samples=beta_samples[
                    :,
                    transition_index,
                    element_index,
                ],
                target_mean=target_mean,
                target_variance=float(
                    BETA_PROBABILITY_VARIANCE
                ),
                transition_title=TRANSITION_TITLES[
                    transition_index
                ],
                transition_name=transition_name,
                element_number=element_number,
                output_dir=OUTPUT_DIR,
            )

    env.close()

    print()
    print(f"All figures were saved in: "f"{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
# %%
