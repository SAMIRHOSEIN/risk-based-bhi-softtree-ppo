#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torchrl.envs import GymWrapper

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    NCS,
    ELEMENT_NUMBERS,
    ELEMENT_WEIGHTS,
    HEALTH_COEFFICIENTS,
    max_steps,
    gamma,
    include_step_count,
    reset_prob,
    RUN_MODE_TAG,  # "<STATE_TRANSITION_MODE>_<learnSF|fixedSF>" tag embedded in every saved filename
)
from softtree_ppo.training import PPOTrainer

from bridge_bhi_training_nn import actor_neurons, actor_layers

def mean_and_ci(values):
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


def compute_bhi_from_observation_fixed_weights(obs):
    obs = np.asarray(obs, dtype=float)

    if include_step_count:
        obs = obs[:-1]

    cs_probs = obs.reshape(len(ELEMENT_NUMBERS), NCS)
    health_coefficients = np.asarray(HEALTH_COEFFICIENTS, dtype=float)
    element_health = cs_probs @ health_coefficients

    weights = np.asarray(
        [ELEMENT_WEIGHTS[int(element_no)] for element_no in ELEMENT_NUMBERS],
        dtype=float,
    )
    weights = weights / weights.sum()

    return float(np.sum(weights * element_health))



# %%

if __name__ == '__main__':
    env_seed = 508
    num_episodes = 1000                # David's assumption 1000
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
    
    # Same RUN_MODE_TAG as the training script, so validation always loads the
    # actor that matches the current STATE_TRANSITION_MODE and
    # LEARNABLE_SIGNIFICANCE_FACTOR settings.
    actor_path = f"./actors/nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr_{RUN_MODE_TAG}.pt"

    actor = PPOTrainer.load_actor(
        actor_path,
        env.action_spec,
    )

    eval_log = PPOTrainer.evaluate(
        actor,
        env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=True,
    )

    # plot testing results
    init_states = np.array(eval_log["init_state"])
    eval_rewards = np.array(eval_log["eval_reward"])

    init_bhi = np.array([
        compute_bhi_from_observation_fixed_weights(obs)
        for obs in init_states
    ])

    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style("ticks")
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        sns.scatterplot(
            x=init_bhi,
            y=eval_rewards,
            ax=ax,
        )

        ax.set_xlabel("Initial Bridge Health Index with fixed BHI weights")
        ax.set_ylabel("Unnormalized episode reward")
        ax.set_title("NN Actor Validation")



    pd.DataFrame({
        "init_bhi_fixed_weights": init_bhi,
        "eval_reward_unnormalized": eval_rewards,
    }).to_csv(
        f"./results/val_nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr_{RUN_MODE_TAG}.csv",
        index=False,
    )



    reward_stats = mean_and_ci(eval_log["eval_reward"])

    print(f"Validation (episode return for {reward_stats['n']} episodes): "
        f"mean={reward_stats['mean']:.4f}, "
        f"95% CI=[{reward_stats['ci_low']:.4f}, {reward_stats['ci_high']:.4f}], "
        f"SD={reward_stats['sd']:.4f}")     
# %%