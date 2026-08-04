# %%
"""
End-to-end visual verification of the correlated-Beta implementation.

Verification methods
--------------------
1. Recovered latent Pearson correlation from final Beta outputs:
       Q -> Beta CDF -> U -> Normal inverse CDF -> recovered Z
       Pearson(recovered Z) is compared with the target latent matrix.

2. Empirical Spearman correlation of final Beta outputs:
       Spearman(Q) is compared with the theoretical Gaussian-copula
       Spearman matrix.

3. Empirical Pearson correlation of final Beta outputs:
       Pearson(Q) shows how much the ordinary linear correlation changes
       after the nonlinear Beta inverse-CDF transformation.

4. Beta marginal histograms:
       For every element and deterioration transition, compare the target
       mean and variance with the sample mean and variance.

All figures are saved inside:
    verification_correlated_beta
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import beta as beta_distribution
from scipy.stats import norm, spearmanr

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    BETA_PROBABILITY_VARIANCES,
    DO_NOTHING_TRANSITIONS,
    ELEMENT_CORRELATION_MATRIX,
    ELEMENT_NUMBERS,
    NCS,
    gamma,
    include_step_count,
    max_steps,
    reset_prob,
)


# ============================================================
# Verification settings
# ============================================================
N_SAMPLES = 50_000
SEED = 20260728
OUTPUT_DIR = Path("verification_correlated_beta")
HISTOGRAM_BINS = 80

# Prevent norm.ppf(0) and norm.ppf(1).
CDF_EPSILON = 1.0e-12


def save_correlation_heatmap(
    matrix,
    title,
    output_path,
    element_labels,
    colorbar_label,
):
    """
    Save one annotated correlation heatmap.
    """
    matrix = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0,aspect="equal")

    ax.set_xticks(np.arange(len(element_labels)))
    ax.set_yticks(np.arange(len(element_labels)))
    ax.set_xticklabels(element_labels, rotation=45, ha="right")
    ax.set_yticklabels(element_labels)

    ax.set_xlabel("Element")
    ax.set_ylabel("Element")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = "NaN" if not np.isfinite(value) else f"{value:.3f}"

            ax.text(column, row,label, ha="center", va="center", fontsize=8)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_beta_histogram(
    samples,
    target_mean,
    target_variance,
    alpha,
    beta_parameter,
    transition_title,
    transition_name,
    element_number,
    output_dir,):
    """
    Save one histogram of final Beta transition-probability samples.
    """
    samples = np.asarray(samples, dtype=float)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples, ddof=1))

    minimum_sample = float(np.min(samples))
    maximum_sample = float(np.max(samples))

    x_lower = minimum_sample
    x_upper = maximum_sample

    weights = np.full(
        samples.shape,
        100.0 / samples.size,
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        samples,
        bins=HISTOGRAM_BINS,
        range=(x_lower, x_upper),
        weights=weights,
    )

    ax.axvline(
        target_mean,
        linestyle="--",
        linewidth=2.0,
        label="Target mean",
    )

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

    ax.set_xlim(x_lower, x_upper)
    ax.set_xlabel("Sampled transition probability")
    ax.set_ylabel("Percent of samples")
    ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=100.0, decimals=0)
    )

    ax.set_title(
        "Final Beta probability distribution: "
        f"{transition_title}, element {element_number}"
    )

    ax.legend()
    fig.tight_layout()

    output_path = (
        output_dir
        / (
            f"06_{transition_name}_element_{element_number}"
            "_beta_distribution.png"
        )
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "transition": transition_title,
        "element_number": element_number,
        "target_mean": target_mean,
        "target_variance": target_variance,
        "alpha": float(alpha),
        "beta": float(beta_parameter),
        "sample_mean": sample_mean,
        "sample_variance": sample_variance,
        "x_axis_min_sampled_probability": x_lower,
        "x_axis_max_sampled_probability": x_upper,
    }





def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    element_labels = [
        str(int(element))
        for element in ELEMENT_NUMBERS
    ]

    transition_names = tuple(
        f"CS{current_cs + 1}_to_CS{current_cs + 2}"
        for current_cs in range(NCS - 1)
    )

    transition_titles = tuple(
        f"CS{current_cs + 1} to CS{current_cs + 2}"
        for current_cs in range(NCS - 1)
    )

    number_of_elements = len(ELEMENT_NUMBERS)
    number_of_transitions = NCS - 1

    # The environment creates:
    # - the correlation validation and Cholesky factor,
    # - the Beta(alpha, beta) parameters,
    # - the correlated-Beta transition matrices.
    env = BridgeBHIEnv(
        max_steps=max_steps,
        discount=gamma,
        include_step_count=include_step_count,
        reset_prob=reset_prob,
        transition_mode="correlated_beta",
        element_correlation_matrix=ELEMENT_CORRELATION_MATRIX,
        seed=SEED,
    )
    env.reset(seed=SEED)

    # Shape:
    # samples x transitions x elements
    beta_samples = np.empty(
        (
            N_SAMPLES,
            number_of_transitions,
            number_of_elements,
        ),
        dtype=float,
    )

    # ========================================================
    # Collect only the final outputs returned by the existing
    # correlated-Beta function in rl_env.py.
    # ========================================================
    for sample_index in range(N_SAMPLES):
        sampled_transition_matrices = (
            env._sample_correlated_beta_transition_matrices()
        )

        for current_cs in range(number_of_transitions):
            for element_index, element_number_raw in enumerate(ELEMENT_NUMBERS):
                element_number = int(element_number_raw)

                beta_samples[
                    sample_index,
                    current_cs,
                    element_index,
                ] = sampled_transition_matrices[element_number][
                    current_cs,
                    current_cs + 1,
                ]

        if (sample_index + 1) % 5_000 == 0:
            print(
                f"Generated {sample_index + 1:,} "
                f"of {N_SAMPLES:,} final Beta samples."
            )

    # ========================================================
    # Target latent Gaussian Pearson matrix
    # ========================================================
    save_correlation_heatmap(
        matrix=ELEMENT_CORRELATION_MATRIX,
        title="Target latent Gaussian Pearson correlation matrix",
        output_path=(
            OUTPUT_DIR
            / "01_target_latent_pearson_correlation.png"
        ),
        element_labels=element_labels,
        colorbar_label="Pearson correlation",
    )

    # ========================================================
    # Method 1:
    # Recover latent Gaussian values from final Beta outputs.
    #
    # The exact alpha and beta values are taken from:
    #     env.beta_transition_parameters
    #
    # They were built by BridgeBHIEnv._build_beta_transition_parameters().
    # This verification file does not recalculate them.
    # ========================================================
    recovered_gaussian_samples = np.empty_like(
        beta_samples,
        dtype=float,
    )

    for current_cs in range(number_of_transitions):
        for element_index, element_number_raw in enumerate(ELEMENT_NUMBERS):
            element_number = int(element_number_raw)

            alpha, beta_parameter = (env.beta_transition_parameters[element_number][current_cs])

            q_values = beta_samples[
                :,
                current_cs,
                element_index,
            ]

            # Reverse the production Beta inverse-CDF step:
            # Q -> U = F_Beta(Q)
            recovered_uniform_values = beta_distribution.cdf(
                q_values,
                alpha,
                beta_parameter,
            )

            recovered_uniform_values = np.clip(
                recovered_uniform_values,
                CDF_EPSILON,
                1.0 - CDF_EPSILON,
            )

            # Reverse the production normal-CDF step:
            # U -> Z = Phi^(-1)(U)
            recovered_gaussian_samples[
                :,
                current_cs,
                element_index,
            ] = norm.ppf(recovered_uniform_values)

    for current_cs, transition_name in enumerate(transition_names):
        recovered_latent_pearson = np.corrcoef(
            recovered_gaussian_samples[
                :,
                current_cs,
                :,
            ],
            rowvar=False,
        )

        save_correlation_heatmap(
            matrix=recovered_latent_pearson,
            title=(
                "Recovered latent Pearson correlation "
                "from final Beta outputs\n"
                f"{transition_titles[current_cs]}"
            ),
            output_path=(
                OUTPUT_DIR
                / (
                    "02_recovered_latent_pearson_"
                    f"{transition_name}.png"
                )
            ),
            element_labels=element_labels,
            colorbar_label="Pearson correlation",
        )

    # ========================================================
    # Method 2:
    # Empirical Spearman correlation of final Beta outputs.
    #
    # SciPy's existing spearmanr() function is used directly.
    # ========================================================
    theoretical_spearman_matrix = (
        6.0
        / np.pi
        * np.arcsin(
            np.asarray(
                ELEMENT_CORRELATION_MATRIX,
                dtype=float,
            )
            / 2.0
        )
    )
    np.fill_diagonal(theoretical_spearman_matrix, 1.0)

    save_correlation_heatmap(
        matrix=theoretical_spearman_matrix,
        title=(
            "Theoretical Gaussian-copula "
            "Spearman correlation matrix"
        ),
        output_path=(
            OUTPUT_DIR
            / "03_theoretical_spearman_correlation.png"
        ),
        element_labels=element_labels,
        colorbar_label="Spearman correlation",
    )

    for current_cs, transition_name in enumerate(transition_names):
        empirical_spearman = spearmanr(
            beta_samples[
                :,
                current_cs,
                :,
            ],
            axis=0,
        ).statistic

        save_correlation_heatmap(
            matrix=empirical_spearman,
            title=(
                "Empirical Spearman correlation "
                "of final Beta outputs\n"
                f"{transition_titles[current_cs]}"
            ),
            output_path=(
                OUTPUT_DIR
                / (
                    "04_empirical_spearman_final_beta_"
                    f"{transition_name}.png"
                )
            ),
            element_labels=element_labels,
            colorbar_label="Spearman correlation",
        )

    # ========================================================
    # Method 3:
    # Direct empirical Pearson correlation of final Beta outputs.
    #
    # This is not expected to equal the target latent Pearson matrix,
    # because the element-specific inverse Beta CDFs are nonlinear.
    # It is plotted to show how much the ordinary linear correlation
    # changes in the final transition probabilities.
    # ========================================================
    for current_cs, transition_name in enumerate(transition_names):
        empirical_final_beta_pearson = np.corrcoef(
            beta_samples[
                :,
                current_cs,
                :,
            ],
            rowvar=False,
        )

        save_correlation_heatmap(
            matrix=empirical_final_beta_pearson,
            title=(
                "Empirical Pearson correlation "
                "of final Beta outputs\n"
                f"{transition_titles[current_cs]}"
            ),
            output_path=(
                OUTPUT_DIR
                / (
                    "05_empirical_pearson_final_beta_"
                    f"{transition_name}.png"
                )
            ),
            element_labels=element_labels,
            colorbar_label="Pearson correlation",
        )


    # ========================================================
    # Method 4:
    # Beta histograms with target and sample mean/variance.
    # ========================================================
    histogram_axis_limits = []

    for current_cs, transition_name in enumerate(transition_names):
        for element_index, element_number_raw in enumerate(
            ELEMENT_NUMBERS
        ):
            element_number = int(element_number_raw)



            target_mean = float(
                DO_NOTHING_TRANSITIONS[element_number][
                    current_cs,
                    current_cs + 1,
                ])

            alpha, beta_parameter = (
                env.beta_transition_parameters[element_number][current_cs])




            histogram_axis_limits.append(
                save_beta_histogram(
                    samples=beta_samples[
                        :,
                        current_cs,
                        element_index,
                    ],
                    target_mean=target_mean,
                    target_variance=float(
                        BETA_PROBABILITY_VARIANCES[element_number]
                    ),
                    alpha=alpha,
                    beta_parameter=beta_parameter,
                    transition_title=transition_titles[current_cs],
                    transition_name=transition_name,
                    element_number=element_number,
                    output_dir=OUTPUT_DIR,
                )
            )










    histogram_limits_path = (
        OUTPUT_DIR
        / "07_histogram_sampled_probability_min_max.csv"
    )

    pd.DataFrame(histogram_axis_limits).to_csv(
        histogram_limits_path,
        index=False,
    )




    env.close()

    print()
    print(
        "Verification completed. "
        f"All figures were saved in: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()

# %%
