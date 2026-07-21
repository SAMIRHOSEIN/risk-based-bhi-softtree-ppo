# %%
"""
Validate the fixed expert-based maintenance policy in a separate file.

This script does not train or load an actor and does not save any CSV file.
It reuses:
    * PerNodeGHISelector._compute_all_hi() for group health indices.
    * PPOTrainer._setup_actor() for the TorchRL actor wrapper.
    * PPOTrainer.evaluate() for validation rollouts.
    * mean_and_ci() for mean, 95% confidence interval, and standard deviation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchrl.envs import GymWrapper

from bridge_bhi_validation_nn import mean_and_ci
from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    ELEMENT_NUMBERS,
    ELEMENT_TO_GROUP_IDX,
    ELEMENT_WEIGHTS,
    GROUP_TO_IDX,
    HEALTH_COEFFICIENTS,
    NA,
    NCS,
    gamma,
    include_step_count,
    max_steps,
    reset_prob,
)
from softtree.bhi_softtree import PerNodeGHISelector
from softtree_ppo.training import PPOTrainer


POLICY_THRESHOLD = 0.30


class ExpertThresholdPolicy(nn.Module):
    """Deterministic engineering policy returned as categorical-action logits."""

    def __init__(self, threshold: float = POLICY_THRESHOLD):
        super().__init__()
        self.threshold = float(threshold)

        fixed_element_weights = [
            ELEMENT_WEIGHTS[int(element_no)] for element_no in ELEMENT_NUMBERS
        ]

        # Reuse the existing project implementation for all group-HI calculations.
        self.hi_calculator = PerNodeGHISelector(
            num_elements=len(ELEMENT_NUMBERS),
            ncs=NCS,
            num_nodes=1,
            health_coefficients=HEALTH_COEFFICIENTS,
            element_to_group_idx=ELEMENT_TO_GROUP_IDX,
            initial_element_weights=fixed_element_weights,
            include_step_count=include_step_count,
            learnable_element_weights=False,
        )
        self.hi_calculator.requires_grad_(False)

    def _select_action(
        self,
        deck_hi: float,
        superstructure_hi: float,
        bearings_hi: float,
        substructure_hi: float,
    ):
        """Return the expert-policy action ID for one bridge state."""
        if substructure_hi < self.threshold:
            return 5  # Full bridge replacement
        elif superstructure_hi < self.threshold:
            return 4  # Replace deck + superstructure + bearings
        elif deck_hi < self.threshold and bearings_hi < self.threshold:
            return 3  # Replace deck + bearings
        elif deck_hi < self.threshold:
            return 2  # Replace deck
        elif bearings_hi < self.threshold:
            return 1  # Replace bearings
        else:
            return 0  # Do nothing

    @torch.no_grad()
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Apply the 0.70 threshold policy and return deterministic logits."""
        original_batch_shape = observation.shape[:-1]

        # Existing function returns:
        # [deck HI, superstructure HI, bearings HI, substructure HI, aggregate BHI]
        all_hi = self.hi_calculator._compute_all_hi(observation)

        deck_hi = all_hi[:, GROUP_TO_IDX["deck"]]
        superstructure_hi = all_hi[:, GROUP_TO_IDX["superstructure"]]
        bearings_hi = all_hi[:, GROUP_TO_IDX["bearings"]]
        substructure_hi = all_hi[:, GROUP_TO_IDX["substructure"]]

        action_ids = [
            self._select_action(
                deck_hi=deck_hi[i].item(),
                superstructure_hi=superstructure_hi[i].item(),
                bearings_hi=bearings_hi[i].item(),
                substructure_hi=substructure_hi[i].item(),
            )
            for i in range(all_hi.shape[0])
        ]

        action = torch.tensor(
            action_ids,
            dtype=torch.long,
            device=observation.device,
        ).reshape(original_batch_shape)

        logits = torch.full(
            (*action.shape, NA),
            fill_value=-1.0e9,
            dtype=observation.dtype,
            device=observation.device,
        )
        logits.scatter_(-1, action.unsqueeze(-1), 0.0)
        return logits


# %%
if __name__ == "__main__":
    # Same main validation inputs as bridge_bhi_validation_stBHI.py.
    env_seed = 508
    num_episodes = 1000
    reward_normalizer = 1

    gym_env = BridgeBHIEnv(
        max_steps=max_steps,
        discount=gamma,
        include_step_count=include_step_count,
        reset_prob=reset_prob,
        reward_normalizer=reward_normalizer,
        render_mode="ansi",
        seed=env_seed,
    )
    env = GymWrapper(gym_env, categorical_action_encoding=True)

    policy_core = ExpertThresholdPolicy(threshold=POLICY_THRESHOLD)

    actor = PPOTrainer._setup_actor(env.action_spec, policy_core)
    eval_log = PPOTrainer.evaluate(
        actor,
        env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=True,
    )

    reward_stats = mean_and_ci(eval_log["eval_reward"])

    print(
        f"Custom-policy validation over {reward_stats['n']} complete episodes "
        f"(threshold = {POLICY_THRESHOLD:.2f}):\n"
        f"Mean unnormalized discounted episode return = "
        f"{reward_stats['mean']:.4f}\n"
        f"95% confidence interval for the mean = "
        f"[{reward_stats['ci_low']:.4f}, {reward_stats['ci_high']:.4f}]\n"
        f"Standard deviation of episode returns = "
        f"{reward_stats['sd']:.4f}"
    )