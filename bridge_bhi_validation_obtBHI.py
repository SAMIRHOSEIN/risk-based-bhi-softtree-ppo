#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from torchrl.envs import GymWrapper

from softtree_ppo.training import SofttreePPOTrainer

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    ELEMENT_NUMBERS,
    ELEMENT_WEIGHTS,
    max_steps,
    gamma,
    include_step_count,
    reset_prob,
    ACTION_NAMES,
    ELEMENT_NAMES,
)
from bridge_bhi_training_stBHI import actor_tree_depth, tree_beta, reg_coef

from bridge_bhi_validation_stBHI import compute_bhi_from_observation_learned_weights


def summarize_element_weights(actor):
    """
    Print BHI-soft-tree parameters for each internal node.
    Print leaf-node classes/actions from the extracted oblique tree.

    Each node uses: logit_n = beta * (BHI + b_n)

    Hard split: BHI + b_n > 0 equivalent to: BHI > -b_n
    """
    core = actor.module[0].module

    learned_weights = torch.nn.functional.softplus(
        core.inner_nodes.raw_element_weights
    ).detach().cpu().numpy()

    normalized_weights = learned_weights / learned_weights.sum()

    print("\n================ BHI-soft-tree learned element weights ================")
    print(ELEMENT_NAMES)
    for element_no, w_raw, w_norm in zip(ELEMENT_NUMBERS, learned_weights, normalized_weights):
        original_w = ELEMENT_WEIGHTS[int(element_no)]
        print(
            f"Element {int(element_no):>4} | "
            f"original W={original_w:>8.4f} | "
            f"learned W={w_raw:>8.4f} | "
            f"normalized learned W={w_norm:>8.4f} | "
            f"ratio learned/original={w_raw/original_w:>8.4f}"
        )




def print_sorted_element_weights(actor):
    """
    Print element ranking based on:
    1. Learned weights
    2. Original engineering weights
    3. Rank change after learning
    """
    core = actor.module[0].module

    learned_weights = torch.nn.functional.softplus(
        core.inner_nodes.raw_element_weights
    ).detach().cpu().numpy()

    normalized_learned_weights = (
        learned_weights / learned_weights.sum()
    )

    # Build dataframe
    rows = []

    for idx, element_no in enumerate(ELEMENT_NUMBERS):
        element_no = int(element_no)

        rows.append({
            "element_no": element_no,
            "element_name": ELEMENT_NAMES[element_no],
            "original_weight": ELEMENT_WEIGHTS[element_no],
            "learned_weight": learned_weights[idx],
            "normalized_learned_weight": normalized_learned_weights[idx],
            "ratio_learned_original":
                learned_weights[idx] / ELEMENT_WEIGHTS[element_no],
        })

    df = pd.DataFrame(rows)

    # ===================================================
    # Ranking based on learned weights
    # ===================================================
    df_learned = df.sort_values(
        by="learned_weight",
        ascending=False
    ).reset_index(drop=True)

    print("\n================ Ranking based on LEARNED weights ================")

    for rank, row in df_learned.iterrows():
        print(
            f"Rank {rank+1:>2} | "
            f"EN={int(row['element_no']):>3} | "
            f"{row['element_name']:<30} | "
            f"learned_W={row['learned_weight']:>8.4f} | "
            f"normalized={row['normalized_learned_weight']:>8.4f}"
        )

    # ===================================================
    # Ranking based on original weights
    # ===================================================
    df_original = df.sort_values(
        by="original_weight",
        ascending=False
    ).reset_index(drop=True)

    print("\n================ Ranking based on ORIGINAL weights ================")

    for rank, row in df_original.iterrows():
        print(
            f"Rank {rank+1:>2} | "
            f"EN={int(row['element_no']):>3} | "
            f"{row['element_name']:<30} | "
            f"original_W={row['original_weight']:>8.4f}"
        )

    # ===================================================
    # Rank change
    # ===================================================
    learned_rank = {
        en: rank + 1
        for rank, en in enumerate(df_learned["element_no"])
    }

    original_rank = {
        en: rank + 1
        for rank, en in enumerate(df_original["element_no"])
    }

    df["original_rank"] = df["element_no"].map(original_rank)
    df["learned_rank"] = df["element_no"].map(learned_rank)
    df["rank_change"] = (
        df["original_rank"] - df["learned_rank"]
    )

    print("\n================ Rank change after learning ================")

    df_rank = df.sort_values(
        by="rank_change",
        ascending=False
    )

    for _, row in df_rank.iterrows():
        print(
            f"EN={int(row['element_no']):>3} | "
            f"{row['element_name']:<30} | "
            f"original_rank={int(row['original_rank'])} | "
            f"learned_rank={int(row['learned_rank'])} | "
            f"change={int(row['rank_change']):+d}"
        )




def summarize_full_oblique_tree_before_pruning(STC_actor):
    """
    Print all internal nodes and leaf nodes of the full extracted oblique tree
    before zero-weight, infeasible-path, or identical-leaf pruning.

    For ParameterizedObliqueTree:
        score = w^T x + b
        if score < 0  -> right branch
        if score >= 0 -> left branch
    """
    core = STC_actor.module[0].module

    max_depth = core.depth

    biases = core.inner_nodes.bias.detach().cpu().numpy()
    leaf_logits = core.leaf_nodes.leaf_scores.detach().cpu().numpy()
    leaf_actions = np.argmax(leaf_logits, axis=1)

    learned_weights = torch.nn.functional.softplus(
        core.inner_nodes.raw_element_weights
    ).detach().cpu().numpy()

    health_coefficients = core.inner_nodes.health_coefficients.detach().cpu().numpy()

    weight_sum = learned_weights.sum()

    bhi_feature_weights = []
    for w_i in learned_weights:
        for k_s in health_coefficients:
            bhi_feature_weights.append((w_i * k_s) / weight_sum)

    bhi_feature_weights = np.asarray(bhi_feature_weights, dtype=float)

    if core.inner_nodes.include_step_count:
        bhi_feature_weights = np.append(bhi_feature_weights, 0.0)

    print("\n================ Full oblique tree BEFORE pruning ================")
    print("Branch rule in ParameterizedObliqueTree:")
    print("score = BHI + b_n")
    print("if score >= 0  -> left")
    print("if score <  0  -> right\n")

    def node_index_from_path(path_bits):
        idx = 0
        for bit in path_bits:
            idx = 2 * idx + 1 + bit
        return idx

    def leaf_index_from_path(path_bits):
        idx = 0
        for bit in path_bits:
            idx = 2 * idx + bit
        return idx

    def recurse(depth, path_bits, path_name):
        # Internal node
        if depth < max_depth:
            node_id = node_index_from_path(path_bits)
            bias = biases[node_id]
            threshold = -bias


            print(
                f"Node {node_id:>3} | "
                f"depth={depth} | "
                f"path={path_name} | "
                f"bias={bias:>10.5f} | "
                f"BHI > {threshold:>10.5f} | "
            )

            recurse(depth + 1, path_bits + [0], path_name + "_Left")
            recurse(depth + 1, path_bits + [1], path_name + "_Right")

        # Leaf node
        else:
            leaf_id = leaf_index_from_path(path_bits)
            action = int(leaf_actions[leaf_id])


            print(
                f"Leaf {leaf_id:>3} | "
                f"depth={depth} | "
                f"path={path_name} | "
                f"action={action}"
            )

    recurse(depth=0, path_bits=[], path_name="root")

    print(f"\nFull internal nodes before pruning = {2**max_depth - 1}")
    print(f"Full leaf nodes before pruning     = {2**max_depth}")




def summarize_oblique_tree_after_pruning(OBT_actor):
    """
    Print all active internal nodes and leaf nodes in the extracted oblique tree.
    """
    tree = OBT_actor.module.tree

    internal_counter = 0
    leaf_counter = 0

    print("\n================ Oblique-tree structure AFTER pruning ================")

    def traverse(node, path="root", depth=0):
        nonlocal internal_counter, leaf_counter

        if node is None:
            return

        # Leaf node
        if node.is_leaf:
            action = int(node.value)

            print(
                f"Leaf | "
                f"original_path={getattr(node, 'id', None)} | "
                f"new_depth={depth} | "
                f"new_path={path} | "
                f"action={ACTION_NAMES[action]}(key={action})"
            )

            leaf_counter += 1
            return

        # Internal node
        bias = float(node.bias)
        threshold = -bias

        print(
            "Node |"
            f"original_path={getattr(node, 'id', None)} | "
            f"new_depth={depth} | "
            f"new_path={path} | "
            f"rule: BHI > {threshold:.5f}"
        )
        traverse(node.left, path + "_Left", depth + 1)
        traverse(node.right, path + "_Right", depth + 1)

    traverse(tree.root)

    print(f"\nTotal active internal nodes = {internal_counter}")
    print(f"Total active leaf nodes     = {leaf_counter}")


# %%

if __name__ == '__main__':
    env_seed = 1034
    obs_episodes = 10
    pruning_threshold = -np.inf #1e-3
    num_episodes = 1000

    lp_threshold = 1e-6

    reward_normalizer = 1

    actor_path = f"./actors/stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{reg_coef:.0e}_{max_steps:d}yr.pt"
    save_path = f"./results/val_obtBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{reg_coef:.0e}_{max_steps:d}yr_{pruning_threshold:.0e}prune.csv"


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

    # get oblique tree actor
    STC_actor = SofttreePPOTrainer.load_actor(
        actor_path,
        env.action_spec,
    )

    loaded_core = STC_actor.module[0].module
    print("Loaded actor type:", type(loaded_core).__name__)
    print("Loaded inner-node type:", type(loaded_core.inner_nodes).__name__)

    # Print the learned element weights in the BHI-soft-tree actor
    summarize_element_weights(STC_actor)
    # Print the learned element weights in the BHI-soft-tree actor sorted by learned weights and original weights, and print the rank change.
    print_sorted_element_weights(STC_actor)
    # Print the full extracted oblique tree before pruning
    summarize_full_oblique_tree_before_pruning(STC_actor)


    OBT_actor, prune_mask = SofttreePPOTrainer.convert_to_obtree_actor(
        STC_actor, pruning_threshold=pruning_threshold,
        lp_threshold=lp_threshold,
        A_ub=[],
        b_ub=[],
        bounds=(0, 1),
    )

    # Print the extracted oblique tree after pruning
    summarize_oblique_tree_after_pruning(OBT_actor)


    # evaluate oblique tree actor
    eval_log = SofttreePPOTrainer.evaluate(
        OBT_actor,
        env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=True,
    )

    # plot testing results
    init_states = np.array(eval_log["init_state"])
    eval_rewards = np.array(eval_log["eval_reward"])

    # In the follwoing lines, I used STC_actor instead of OBT_actor because in this line, 
    # I just want to compute BHI, so it doesn't matter which actor I use. Actually, the eval_rewards
    # is important, and it is computed by OBT_actor in "eval_log = SofttreePPOTrainer.evaluate" line.
    # 
    init_bhi = np.array([
        compute_bhi_from_observation_learned_weights(STC_actor, obs)
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

        ax.set_xlabel("Initial Bridge Health Index(new element weights)")
        ax.set_ylabel("Unnormalized episode reward(original weights)")


    # save results
    candidate_nodes = np.sum(prune_mask).item()
    internal_nodes = OBT_actor.module.tree.internal_num
    leaf_nodes = OBT_actor.module.tree.leaf_num
    pruned_internal = 2**OBT_actor.module.tree.max_depth - 1 - internal_nodes
    pruned_leaf = 2**OBT_actor.module.tree.max_depth - leaf_nodes
    val_res = {
        'init_bhi': init_bhi,
        'eval_reward': eval_rewards,
        'internal_nodes': internal_nodes,
        'leaf_nodes': leaf_nodes,
        'candidate_nodes': candidate_nodes,
        'pruned_internal': pruned_internal,
        'pruned_leaf': pruned_leaf
    }


    pd.DataFrame(val_res).to_csv(
        save_path,
        index=False
    )