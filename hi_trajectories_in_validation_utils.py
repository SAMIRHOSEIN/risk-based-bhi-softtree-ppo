# %%
# I provided this file to encapsulate the validation plotting utilities and run for 
# validation in NN and custome policy and oblique tree
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from bridge_gym.example_bridge_bhi.settings import GROUP_ORDER


def plot_validation_hi_trajectories(
    *,
    eval_log,
    hi_calculator: Callable[[torch.Tensor], torch.Tensor],
    save_prefix: str | None = None,
    title_prefix: str = "Validation",
    show_individual_episodes: bool = False,
):
    """Compute and plot the four group HIs and aggregate BHI.

    ``hi_calculator`` must accept a two-dimensional observation tensor and
    return the five health indices in this order:
    deck, superstructure, bearings, substructure, aggregate BHI.
    """

    observations = np.stack(eval_log["observations"], axis=0)

    num_episodes, num_steps, observation_size = observations.shape
    flat_observations = torch.as_tensor(
        observations.reshape(-1, observation_size),
        dtype=torch.float32,
    )

    # Keep the observations on the same device and dtype as the calculator.
    calculator_owner = getattr(hi_calculator, "__self__", None)
    if isinstance(calculator_owner, torch.nn.Module):
        reference_tensor = next(iter(calculator_owner.parameters()),None)
        
        if reference_tensor is None:
            reference_tensor = next(
                iter(calculator_owner.buffers()),
                None,
            )
        if reference_tensor is not None:
            flat_observations = flat_observations.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )

    with torch.no_grad():
        flat_hi = hi_calculator(flat_observations)

 
    hi_names = [*GROUP_ORDER, "BHI_aggregate"]


    all_hi = (
        flat_hi.detach()
        .cpu()
        .numpy()
        .reshape(num_episodes, num_steps, len(hi_names))
    )

    hi_trajectories = {hi_name: all_hi[:, :, hi_idx] for hi_idx, hi_name in enumerate(hi_names)}

    print("\n================ Validation HI trajectories ================")
    print(f"Validation episodes = {num_episodes}")
    print(f"Observed decision steps per episode = {num_steps}")

    time_steps = np.arange(num_steps)

    for hi_name, trajectories in hi_trajectories.items():
        mean_trajectory = trajectories.mean(axis=0)
        lower_trajectory = np.percentile(trajectories, 5, axis=0)
        upper_trajectory = np.percentile(trajectories, 95, axis=0)

        print(
            f"{hi_name:<20} | "
            f"initial mean={mean_trajectory[0]:.4f} | "
            f"final observed mean={mean_trajectory[-1]:.4f}"
        )

        fig, ax = plt.subplots(
            figsize=(9, 5),
            tight_layout=True,
        )

        if show_individual_episodes:
            for episode_trajectory in trajectories:
                ax.plot(
                    time_steps,
                    episode_trajectory,
                    linewidth=0.6,
                    alpha=0.08,
                )

        ax.fill_between(
            time_steps,
            lower_trajectory,
            upper_trajectory,
            alpha=0.25,
            label="5th–95th percentile",
        )
        ax.plot(
            time_steps,
            mean_trajectory,
            linewidth=2.0,
            label="Mean",
        )

        ax.set_xlabel("Decision step")
        ax.set_ylabel("Health Index")
        ax.set_title(f"{title_prefix}: {hi_name}")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_prefix is not None:
            figure_path = Path(f"{save_prefix}_{hi_name}.png")
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                figure_path,
                dpi=300,
                bbox_inches="tight",
            )
            print(f"[*] Saved: {figure_path}")

        plt.show()
        plt.close(fig)

    return hi_trajectories
