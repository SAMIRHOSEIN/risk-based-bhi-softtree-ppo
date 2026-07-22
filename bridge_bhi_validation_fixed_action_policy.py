# %%
"""Validate the fixed baseline policy.
The policy always selects fixed action. No actor is trained or loaded.
For each episode, cumulative return is the sum of the discounted step rewards
already returned by BridgeBHIEnv.step().
"""

from __future__ import annotations
import numpy as np

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    ACTION_NAMES,
    STATE_TRANSITION_MODE,
    gamma,
    include_step_count,
    max_steps,
    reset_prob,
)

def mean_and_ci(values):
    """Return sample mean, standard deviation, and normal 95% CI."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    ci = 1.96 * se

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - ci,
        "ci_high": mean + ci,
    }


# %%
# %%
if __name__ == "__main__":
    env_seed = 508
    num_episodes = 1000
    reward_normalizer = 1

    # 0 = Do nothing
    # 1 = Replace bearings
    # 2 = Replace deck
    # 3 = Replace deck + bearings
    # 4 = Replace deck + superstructure + bearings
    # 5 = Full bridge replacement
    fixed_action = 0



    gym_env = BridgeBHIEnv(
        max_steps=max_steps,
        discount=gamma,
        include_step_count=include_step_count,
        reset_prob=reset_prob,
        reward_normalizer=reward_normalizer,
        render_mode=None,
        seed=env_seed,
    )

    episode_returns = []
    episode_lengths = []

    for _ in range(num_episodes):
        gym_env.reset()

        cumulative_return = 0.0
        episode_steps = 0

        while True:
            _, reward, terminated, _, _ = gym_env.step(fixed_action)

            # BridgeBHIEnv already applies gamma**time to each step reward.
            # Therefore, do not discount the reward again.
            cumulative_return += float(reward)
            episode_steps += 1

            if terminated:
                break

        episode_returns.append(cumulative_return)
        episode_lengths.append(episode_steps)

    reward_stats = mean_and_ci(episode_returns)
    unique_episode_lengths = sorted(set(episode_lengths))

    print(
        f"Fixed-action validation over "
        f"{reward_stats['n']} complete episodes:\n"
        f"Transition mode = {STATE_TRANSITION_MODE}\n"
        f"Action at every step = {fixed_action} "
        f"({ACTION_NAMES[fixed_action]})\n"
        f"Configured time steps per episode = {max_steps}\n"
        f"Mean unnormalized discounted cumulative episode return = "
        f"{reward_stats['mean']:.4f}\n"
        f"95% confidence interval for the mean = "
        f"[{reward_stats['ci_low']:.4f}, "
        f"{reward_stats['ci_high']:.4f}]\n"
        f"Standard deviation of episode returns = "
        f"{reward_stats['sd']:.4f}"
    )

    gym_env.close()
