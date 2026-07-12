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

from bridge_bhi_validation_nn import compute_bhi_from_observation_fixed_weights

from bridge_bhi_validation_nn import mean_and_ci


# ---------------------------------------------------------------------
# Learnable-significance-factor mode helpers
# ---------------------------------------------------------------------
# When the actor was trained with LEARNABLE_SIGNIFICANCE_FACTOR = False the
# element weights are NOT learned: they are held fixed at ELEMENT_WEIGHTS for
# the whole run. The trainer saves that flag inside the actor, so any loaded
# actor can tell us which mode it is in. Validation output must reflect this so
# it never reports "learned" weights when nothing was actually learned.
def element_weights_are_learnable(actor):
    """True if the actor's element significance factors were trained;
    False if they were held fixed at ELEMENT_WEIGHTS."""
    inner = actor.module[0].module.inner_nodes
    return bool(getattr(inner, "learnable_element_weights", True))


def weight_mode_label(actor):
    """'learned' or 'fixed' — for honest labels in prints, plots, and CSVs."""
    return "learned" if element_weights_are_learnable(actor) else "fixed"


def get_actor_element_weights(actor):
    """Positive element weights the actor actually uses (softplus of raw).
    In fixed mode these are exactly ELEMENT_WEIGHTS; in learnable mode they are
    the trained values."""
    core = actor.module[0].module
    return torch.nn.functional.softplus(
        core.inner_nodes.raw_element_weights
    ).detach().cpu().numpy()


# For plotting
def compute_bhi_from_observation_learned_weights(actor, obs):
    """
    Compute BHI from one observation using the element weights stored in the
    SoftTreeBHI actor.

    NOTE on naming: this uses the actor's ACTUAL element weights. When the actor
    was trained with LEARNABLE_SIGNIFICANCE_FACTOR = True these are the learned
    weights; when False they are the FIXED ELEMENT_WEIGHTS (nothing was learned).
    Use weight_mode_label(actor) to label the result correctly.
    """
    core = actor.module[0].module

    # Positive element weights the actor uses. In learnable mode these are the
    # trained values; in fixed mode softplus() reproduces ELEMENT_WEIGHTS.
    learned_weights = torch.nn.functional.softplus(
        core.inner_nodes.raw_element_weights
    ).detach().cpu().numpy()

    normalized_learned_weights = learned_weights / learned_weights.sum()

    obs = np.asarray(obs, dtype=float)

    if include_step_count:
        obs = obs[:-1]

    cs_probs = obs.reshape(len(ELEMENT_NUMBERS), NCS)

    health_coefficients = np.asarray(HEALTH_COEFFICIENTS, dtype=float)

    element_health = cs_probs @ health_coefficients

    bhi = np.sum(normalized_learned_weights * element_health)

    return float(bhi)




def report_node_hi_selection(actor):
    """
    Print and return which health index each internal node selected.

    The selected HI per node = argmax of that node's selection logits.
    The confidence is the softmax probability of that HI at
    the actor's current tau, which after training is near one-hot.
    """
    from softtree.bhi_softtree import GROUP_NAMES

    inner = actor.module[0].module.inner_nodes

    # argmax-selected HI name per node (matches extraction)
    selected_names = inner.get_selected_hi_names()           # list[str], len = num_nodes
    # soft confidences at current tau
    probs = inner.get_selection_probs().detach().cpu().numpy()   # (num_nodes, 6)
    selected_idx = inner.selection_logits.argmax(dim=1).cpu().numpy()

    print("\n" + "=" * 64)
    print("PER-NODE HEALTH INDEX SELECTION")
    print(f"(current tau = {inner.tau:.4f}; argmax is tau-independent)")
    print("=" * 64)
    print(f"{'Node':>4}  {'Selected HI':<40}  {'Confidence':>10}")
    rows = []
    for n, name in enumerate(selected_names):
        conf = float(probs[n, selected_idx[n]])
        print(f"{n:>4}  {name:<40}  {conf:>10.4f}")
        rows.append({"node": n, "selected_hi": name, "confidence": conf})

    return rows


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
    print(f"Expected do-nothing return unnormalized = {expected_do_nothing_return_raw}\n")

    env = GymWrapper(gym_env, categorical_action_encoding=True)
    
    actor = SofttreePPOTrainer.load_actor(
        actor_path,
        env.action_spec,
    )

    loaded_core = actor.module[0].module
    print("Loaded actor type:", type(loaded_core).__name__)
    print("Loaded inner-node type:", type(loaded_core.inner_nodes).__name__)

    # Report whether the element significance factors were learned or fixed.
    mode = weight_mode_label(actor)
    if element_weights_are_learnable(actor):
        print("\n[Element significance factors] LEARNABLE "
              "(LEARNABLE_SIGNIFICANCE_FACTOR = True): weights were trained by PPO.")
    else:
        print("\n[Element significance factors] FIXED "
              "(LEARNABLE_SIGNIFICANCE_FACTOR = False): weights are held at "
              "ELEMENT_WEIGHTS, so there are NO learned weights.")
    actor_weights = get_actor_element_weights(actor)
    for element_no, w in zip(ELEMENT_NUMBERS, actor_weights):
        print(f"  Element {int(element_no):>4} | {mode} weight = {w:>8.4f}")


    # Report which HI each internal node selected (interpretability).
    node_hi_rows = report_node_hi_selection(actor)
    pd.DataFrame(node_hi_rows).to_csv(
        f"./results/node_hi_selection_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{reg_coef:.0e}_{max_steps:d}yr.csv",
        index=False,
    )


    
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


    # Actor-weight BHI: uses the actor's own element weights (learned OR fixed).
    init_bhi_learned = np.array([
        compute_bhi_from_observation_learned_weights(actor, obs)
        for obs in init_states
    ])

    init_bhi_fixed = np.array([
        compute_bhi_from_observation_fixed_weights(obs)
        for obs in init_states
    ])


    # Plot 1: actor-weight BHI vs reward (labelled learned/fixed by mode)
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style("ticks")
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        sns.scatterplot(
            x=init_bhi_learned,
            y=eval_rewards,
            ax=ax,
        )

        ax.set_xlabel(f"Initial BHI using {mode} soft-tree weights")
        ax.set_ylabel("Unnormalized episode reward using fixed reward weights")
        ax.set_title(f"Soft-Tree Actor: {mode.capitalize()}-Weight BHI vs Reward")


    # Plot 2: fixed-weight BHI vs reward
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style("ticks")
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        sns.scatterplot(
            x=init_bhi_fixed,
            y=eval_rewards,
            ax=ax,
        )

        ax.set_xlabel("Initial BHI using fixed environment weights")
        ax.set_ylabel("Unnormalized episode reward using fixed reward weights")
        ax.set_title("Soft-Tree Actor: Fixed-Weight BHI(To compare with NN) vs Reward")


    # save results
    val_res = {
        "init_bhi_fixed_weights": init_bhi_fixed,
        "eval_reward_unnormalized": eval_rewards,
    }
    # Only emit a "learned weights" column when weights were actually learned.
    # In fixed mode the actor-weight BHI equals init_bhi_fixed_weights, so a
    # separate "learned" column would be both redundant and misleading.
    if element_weights_are_learnable(actor):
        val_res["init_bhi_learned_weights"] = init_bhi_learned
    pd.DataFrame(val_res).to_csv(
        save_path,
        index=False
    )




    reward_stats = mean_and_ci(eval_log["eval_reward"])

    print(f"Validation (episode return for {reward_stats['n']} episodes): "
        f"mean={reward_stats['mean']:.4f}, "
        f"95% CI=[{reward_stats['ci_low']:.4f}, {reward_stats['ci_high']:.4f}], "
        f"SD={reward_stats['sd']:.4f}")   
