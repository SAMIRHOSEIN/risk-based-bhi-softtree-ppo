#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from torchrl.envs import GymWrapper

from softtree_ppo.training import SofttreePPOTrainer

from bridge_bhi_training_stBHI import actor_tree_depth, tree_beta, reg_coef

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    NCS,
    ELEMENT_NUMBERS,
    HEALTH_COEFFICIENTS,
    max_steps,
    gamma,
    include_step_count,
    reset_prob,
)

# For plotting
def compute_bhi_from_observation(actor, obs):
    """Compute the Bridge Health Index (BHI) from the observation using 
    the learned element weights from the actor's inner nodes."""
    core = actor.module[0].module

    # actor gives us raw element weights(learned), we need to apply softplus to get the positive weights.
    weights = torch.nn.functional.softplus(
        core.inner_nodes.raw_element_weights
    ).detach().cpu().numpy()

    # normalize
    weights = weights / weights.sum()

    if include_step_count:
        obs = obs[:-1]

    cs_probs = obs.reshape(len(ELEMENT_NUMBERS), NCS)

    element_health = cs_probs @ HEALTH_COEFFICIENTS

    bhi = np.sum(weights * element_health)

    return bhi

# %%
if __name__ == '__main__':
    actor_path = f"./actors/stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{reg_coef:.0e}_{max_steps:d}yr.pt"
    save_path = f"./results/val_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{reg_coef:.0e}_{max_steps:d}yr.csv"

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

    gamma_sum = sum(gamma**t for t in range(max_steps))

    expected_do_nothing_return_raw = gym_env.C0 * gamma_sum
    expected_do_nothing_return_normalized = gamma_sum

    print("To verify the reward structure:")
    print("C0 =", gym_env.C0)
    print("reward_normalizer =", gym_env.reward_normalizer)
    print("discounted gamma sum =", gamma_sum)
    print(f"Expected do-nothing return unnormalized = {expected_do_nothing_return_raw}")
    print(f"Expected do-nothing return normalized = {expected_do_nothing_return_normalized}\n")

    env = GymWrapper(gym_env, categorical_action_encoding=True)
    
    actor = SofttreePPOTrainer.load_actor(
        actor_path,
        env.action_spec,
    )

    loaded_core = actor.module[0].module
    print("Loaded actor type:", type(loaded_core).__name__)
    print("Loaded inner-node type:", type(loaded_core.inner_nodes).__name__)


    eval_log = SofttreePPOTrainer.evaluate(
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
        compute_bhi_from_observation(actor, obs)
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

        ax.set_xlabel("Initial Bridge Health Index with new element weights")
        ax.set_ylabel("Unnormalized episode reward(original weights)")


    # save results
    val_res = {
        "init_bhi": init_bhi,
        "eval_reward _unnormalized": eval_rewards,
    }
    pd.DataFrame(val_res).to_csv(
        save_path,
        index=False
    )