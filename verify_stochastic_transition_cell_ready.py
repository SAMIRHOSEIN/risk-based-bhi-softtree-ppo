# %%
"""
Verification of deterministic and stochastic condition-state transitions.

Purpose
-------
This script verifies the stochastic transition implementation in the existing
BridgeBHIEnv class. It does NOT duplicate the transition equations from
rl_env.py. Instead, it imports BridgeBHIEnv and calls the existing env.reset()
and env.step() methods.

Experiment
----------
1. Start every bridge from the pristine state by using reset_prob=None.
2. Apply Action 0 (Do nothing) at every year.
3. Run one deterministic bridge trajectory.
4. Run stochastic ensembles with n=4 and n=1000 bridge realizations.
5. For every bridge element and CS1-CS4, plot:
      - deterministic trajectory: blue solid line,
      - individual stochastic bridges: orange transparent lines,
      - stochastic ensemble mean: red dashed line.
6. Save only the figures. 

Important interpretation
------------------------
The number n is the number of independent bridge realizations. It is different
from ELEMENT_QUANTITIES, which controls how many units of each bridge element
are sampled inside one stochastic bridge realization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    ELEMENT_NAMES,
    ELEMENT_NUMBERS,
    ELEMENT_QUANTITIES,
    NCS,
    gamma,
    max_steps,
)

# %% Configuration
DO_NOTHING_ACTION = 0
SAMPLE_SIZES = (4, 1000)
BASE_SEED = 1034
YEARS = max_steps

DETERMINISTIC_COLOR = "blue"
STOCHASTIC_SAMPLE_COLOR = "orange"
STOCHASTIC_MEAN_COLOR = "red"


SCRIPT_DIR = Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / "verification_stochastic"


# %% Environment construction

def make_env(*, years: int, transition_mode: str, seed: int) -> BridgeBHIEnv:
    """Create the existing project environment in the requested transition mode."""
    return BridgeBHIEnv(
        max_steps=years,
        discount=gamma,
        element_numbers=ELEMENT_NUMBERS,
        include_step_count=False,
        reset_prob=None,
        reward_normalizer=1,
        transition_mode=transition_mode,
        render_mode=None,
        seed=seed,
    )


# %% Verification checks

def validate_stochastic_counts(
    info: dict,
    *,
    sample_id: int,
    year: int,
) -> None:
    """Check the hidden integer CS counts returned by the existing environment."""
    counts = info["cs_counts"]
    state = info["cs"]

    if counts is None:
        raise AssertionError(
            f"Sample {sample_id}, year {year}: "
            "stochastic mode returned cs_counts=None."
        )

    if not np.issubdtype(counts.dtype, np.integer):
        raise AssertionError(
            f"Sample {sample_id}, year {year}: cs_counts is not integer-valued."
        )

    if np.any(counts < 0):
        raise AssertionError(
            f"Sample {sample_id}, year {year}: "
            "negative condition-state counts found."
        )

    expected_quantities = np.array(
        [ELEMENT_QUANTITIES[int(element_no)] for element_no in ELEMENT_NUMBERS],
        dtype=np.int64,
    )

    observed_quantities = counts.sum(axis=1)

    if not np.array_equal(observed_quantities, expected_quantities):
        raise AssertionError(
            f"Sample {sample_id}, year {year}: stochastic counts do not preserve "
            "element quantities."
        )

    reconstructed_state = counts / expected_quantities[:, None]

    if not np.allclose(state, reconstructed_state, atol=1e-7, rtol=0.0):
        raise AssertionError(
            f"Sample {sample_id}, year {year}: state != counts / quantity."
        )


# %% Deterministic trajectory

def run_deterministic_trajectory(years: int) -> np.ndarray:
    """Run one deterministic bridge trajectory using existing environment methods."""
    env = make_env(
        years=years,
        transition_mode="deterministic",
        seed=BASE_SEED,
    )

    _, info = env.reset()

    if info["cs_counts"] is not None:
        raise AssertionError(
            "Deterministic mode should return cs_counts=None."
        )
    
    trajectory = np.empty(
        (years + 1, len(ELEMENT_NUMBERS), NCS),
        dtype=np.float32,
    )
    trajectory[0] = info["cs"].copy()

    for year in range(1, years + 1):
        _, _, _, _, info = env.step(DO_NOTHING_ACTION)
        trajectory[year] = info["cs"].copy()


    env.close()

    return trajectory

# %% Stochastic bridge ensemble

def run_stochastic_ensemble(
    *,
    n_samples: int,
    years: int,
    base_seed: int,
) -> np.ndarray:
    """
    Run n_samples independent stochastic bridge realizations.

    Each sample represents one complete bridge containing all bridge elements.
    The stochastic transitions themselves are performed only by BridgeBHIEnv.
    """

    env = make_env(
        years=years,
        transition_mode="stochastic",
        seed=base_seed,
    )

    ensemble = np.empty(
        (
            n_samples,
            years + 1,
            len(ELEMENT_NUMBERS),
            NCS,
        ),
        dtype=np.float32,
    )

    progress_every = max(1, n_samples // 10)

    for sample_idx in range(n_samples):
        sample_id = sample_idx + 1
        sample_seed = base_seed + sample_idx

        _, info = env.reset(seed=sample_seed)
        validate_stochastic_counts(
            info,
            sample_id=sample_id,
            year=0,
        )
        ensemble[sample_idx, 0] = info["cs"].copy()

        for year in range(1, years + 1):
            _, _, _, _, info = env.step(DO_NOTHING_ACTION)

            validate_stochastic_counts(
                info,
                sample_id=sample_id,
                year=year,
            )

            ensemble[sample_idx, year] = info["cs"].copy()

        if (
            sample_id == 1
            or sample_id % progress_every == 0
            or sample_id == n_samples
        ):
            print(
                f"  stochastic n={n_samples}: "
                f"completed {sample_id}/{n_samples} bridges"
            )

    env.close()


    return ensemble


# %% Plotting

def plot_element_comparisons(
    *,
    deterministic: np.ndarray,
    stochastic_results: Dict[int, np.ndarray],
    years: int,
    output_dir: Path,
) -> None:
    """Save one 2 x 4 CS comparison figure for every bridge element."""
    sample_sizes = list(stochastic_results.keys())

    if len(sample_sizes) != 2:
        raise ValueError(
            "The comparison figure is designed for exactly two sample sizes."
        )

    time = np.arange(years + 1)

    for element_idx, element_no_raw in enumerate(ELEMENT_NUMBERS):
        element_no = int(element_no_raw)
        element_name = ELEMENT_NAMES[element_no]
        element_quantity = ELEMENT_QUANTITIES[element_no]

        fig, axes = plt.subplots(
            nrows=2,
            ncols=NCS,
            figsize=(18, 8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        for row_idx, n_samples in enumerate(sample_sizes):
            ensemble = stochastic_results[n_samples]

            # Mean across independent bridge realizations.
            ensemble_mean = ensemble.mean(axis=0)

            sample_alpha = 0.65 if n_samples <= 10 else 0.025
            sample_linewidth = 0.9 if n_samples <= 10 else 0.45

            for cs_idx in range(NCS):
                ax = axes[row_idx, cs_idx]

                # Shape after transpose: (years + 1, n_samples).
                # This draws one line for every stochastic bridge realization.
                stochastic_lines = ensemble[:, :, element_idx, cs_idx].T

                ax.plot(
                    time,
                    stochastic_lines,
                    color=STOCHASTIC_SAMPLE_COLOR,
                    alpha=sample_alpha,
                    linewidth=sample_linewidth,
                    zorder=1,
                )

                ax.plot(
                    time,
                    deterministic[:, element_idx, cs_idx],
                    color=DETERMINISTIC_COLOR,
                    linewidth=2.4,
                    zorder=3,
                )

                ax.plot(
                    time,
                    ensemble_mean[:, element_idx, cs_idx],
                    color=STOCHASTIC_MEAN_COLOR,
                    linestyle="--",
                    linewidth=2.2,
                    zorder=4,
                )

                ax.set_title(f"CS{cs_idx + 1}")
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, alpha=0.25)

                if cs_idx == 0:
                    ax.set_ylabel(f"n={n_samples}\nProbability")

                if row_idx == 1:
                    ax.set_xlabel("Year")

                if row_idx == 0 and cs_idx == 0:
                    legend_handles = [
                        Line2D(
                            [0],
                            [0],
                            color=DETERMINISTIC_COLOR,
                            linewidth=2.4,
                            label="Deterministic",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color=STOCHASTIC_SAMPLE_COLOR,
                            linewidth=1.0,
                            alpha=0.7,
                            label="Individual stochastic bridges",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color=STOCHASTIC_MEAN_COLOR,
                            linestyle="--",
                            linewidth=2.2,
                            label="Stochastic ensemble mean",
                        ),
                    ]

                    ax.legend(
                        handles=legend_handles,
                        loc="best",
                    )

        fig.suptitle(
            "Deterministic vs stochastic condition-state trajectories\n"
            f"Element {element_no}: {element_name} | "
            f"Quantity = {element_quantity} | "
            "Action A0: Do nothing",
            fontsize=15,
        )

        output_path = output_dir / (
            f"element_{element_no}_deterministic_vs_stochastic.png"
        )

        fig.savefig(
            output_path,
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(fig)
        print(f"Saved: {output_path}")


# %% Main experiment

def main() -> None:

    if len(SAMPLE_SIZES) != 2 or len(set(SAMPLE_SIZES)) != 2:
        raise ValueError(
            "SAMPLE_SIZES must contain exactly two different positive values."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("================ Stochastic transition verification ================")
    print(f"Years                    = {YEARS}")
    print(f"Action every year        = A{DO_NOTHING_ACTION} (Do nothing)")
    print("Initial state            = pristine for every element")
    print(f"Stochastic sample sizes  = {SAMPLE_SIZES}")
    print("Each stochastic sample   = one complete bridge realization")
    print(f"Output directory         = {OUTPUT_DIR}")

    print("\nRunning deterministic trajectory...")
    deterministic = run_deterministic_trajectory(YEARS)

    stochastic_results: Dict[int, np.ndarray] = {}

    for experiment_idx, n_samples in enumerate(SAMPLE_SIZES):
        print(f"\nRunning stochastic ensemble with n={n_samples} bridges...")

        stochastic_results[n_samples] = run_stochastic_ensemble(
            n_samples=n_samples,
            years=YEARS,
            base_seed=BASE_SEED + experiment_idx * 1_000_000,
        )

    print("\nCreating per-element CS1-CS4 figures...")

    plot_element_comparisons(
        deterministic=deterministic,
        stochastic_results=stochastic_results,
        years=YEARS,
        output_dir=OUTPUT_DIR,
    )

    print("\nVerification completed successfully.")


# %% Run
if __name__ == "__main__":
    main()
