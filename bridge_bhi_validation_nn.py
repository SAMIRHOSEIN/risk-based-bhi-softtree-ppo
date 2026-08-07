#%%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torchrl.envs import GymWrapper

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    NCS,
    ELEMENT_NUMBERS,
    ELEMENT_WEIGHTS,
    ELEMENT_TO_GROUP_IDX,
    HEALTH_COEFFICIENTS,
    max_steps,
    gamma,
    include_step_count,
    reset_prob,
    STATE_TRANSITION_MODE,
    NA,
    ACTION_NAMES,
)

# The NN actor does not contain BHI element weights. 
# Therefore, we need a fixed PerNodeGHISelector solely to calculate the validation HIs using ELEMENT_WEIGHTS.
from softtree.bhi_softtree import PerNodeGHISelector
from action_validation_utils import (
    save_action_summary,
    save_first_ten_episode_hi_and_probabilities,
)



from softtree_ppo.training import PPOTrainer

from hi_trajectories_in_validation_utils import plot_validation_hi_trajectories
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
        transition_mode=STATE_TRANSITION_MODE,
        render_mode="ansi",
        seed=env_seed,
    )
    env = GymWrapper(gym_env, categorical_action_encoding=True)
    

    actor_path = (f"./actors/nn_" f"{actor_neurons:d}x{actor_layers:d}_" f"{max_steps:d}yr_{STATE_TRANSITION_MODE}.pt")
    actor = PPOTrainer.load_actor(
        actor_path,
        env.action_spec,
    )






    # A plain NN actor has no element-significance-factor parameters.
    # Fixed engineering weights are used only to calculate validation HIs.
    fixed_hi_calculator = PerNodeGHISelector(
        num_elements=len(ELEMENT_NUMBERS),
        ncs=NCS,
        num_nodes=1,
        health_coefficients=HEALTH_COEFFICIENTS,
        element_to_group_idx=ELEMENT_TO_GROUP_IDX,
        initial_element_weights=[
            ELEMENT_WEIGHTS[int(element_no)]
            for element_no in ELEMENT_NUMBERS
        ],
        include_step_count=include_step_count,
        learnable_element_weights=False,
    )

    fixed_hi_calculator.requires_grad_(False)
    fixed_hi_calculator.eval()







    eval_log = PPOTrainer.evaluate(
        actor,
        env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=True,
    )







    # Save for validation
    action_file_stem = (f"nn_{actor_neurons:d}x{actor_layers:d}_"f"{max_steps:d}yr_{STATE_TRANSITION_MODE}")
    save_action_summary(eval_log, action_file_stem)
    save_first_ten_episode_hi_and_probabilities(actor, eval_log,action_file_stem, fixed_hi_calculator._compute_all_hi)












    os.makedirs(".results/result_hi_directories/nn", exist_ok=True)
    hi_trajectories = plot_validation_hi_trajectories(
        eval_log=eval_log,
        hi_calculator=fixed_hi_calculator._compute_all_hi,
        save_prefix=(
            f".results/result_hi_directories/hi_trajectory_nn_"
            f"{actor_neurons:d}x{actor_layers:d}_"
            f"{max_steps:d}yr_{STATE_TRANSITION_MODE}"
        ),
        title_prefix="NN actor validation",
        show_individual_episodes=True,
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
        f"./results/val_nn_"
        f"{actor_neurons:d}x{actor_layers:d}_"
        f"{max_steps:d}yr_{STATE_TRANSITION_MODE}.csv",
        index=False,
    )



    reward_stats = mean_and_ci(eval_log["eval_reward"])  

    print(
        f"NN validation over {reward_stats['n']} complete episodes(validation):\n"
        f"Mean unnormalized discounted episode return = "
        f"{reward_stats['mean']:.4f}\n"
        f"95% confidence interval for the mean = "
        f"[{reward_stats['ci_low']:.4f}, {reward_stats['ci_high']:.4f}]\n"
        f"Standard deviation of episode returns = "
        f"{reward_stats['sd']:.4f}"
    )



# %%