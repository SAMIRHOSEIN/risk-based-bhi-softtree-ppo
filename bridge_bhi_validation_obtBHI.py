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

from bridge_bhi_validation_nn import compute_bhi_from_observation_fixed_weights

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
    BEFORE zero-weight, infeasible-path, or identical-leaf pruning.
 
    PER-NODE GHI VERSION (Solution 1):
        Each internal node routes on its OWN selected health index, not a single
        shared BHI. The selected HI per node = argmax of that node's selection
        logits (the same rule convert_to_obtree_actor uses for extraction).
 
    Branch rule in ParameterizedObliqueTree:
        score = HI_selected(n) + b_n
        if score >= 0  -> left
        if score <  0  -> right
        equivalently:  HI_selected(n) > -b_n
    """
    from softtree.bhi_softtree import GROUP_NAMES
 
    core = STC_actor.module[0].module
    max_depth = core.depth
 
    biases = core.inner_nodes.bias.detach().cpu().numpy()
    leaf_logits = core.leaf_nodes.leaf_scores.detach().cpu().numpy()
    leaf_actions = np.argmax(leaf_logits, axis=1)
 
    # Per-node selected HI (argmax of selection logits). This is what makes the
    # printout match the actual extracted decision rule. argmax is tau-independent.
    selected_idx = core.inner_nodes.selection_logits.argmax(dim=1).cpu().numpy()  # (num_nodes,)
    selected_names = [GROUP_NAMES[k] for k in selected_idx]
 
    # Also show each node's selection confidence (softmax prob at current tau).
    selection_probs = core.inner_nodes.get_selection_probs().detach().cpu().numpy()  # (num_nodes, 6)
 
    print("\n================ Full oblique tree BEFORE pruning ================")
    print("Branch rule in ParameterizedObliqueTree:")
    print("score = HI_selected(n) + b_n")
    print("if score >= 0  -> left")
    print("if score <  0  -> right")
    print(" If HI_selected(n) > -b_n, go left; else go right.")
    print("(each node uses its OWN selected health index)\n")
 
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
 
            hi_name = selected_names[node_id]
            hi_conf = float(selection_probs[node_id, selected_idx[node_id]])
 
            print(
                f"Node {node_id:>3}| "
                f"depth={depth}| "
                f"path={path_name}| "
                f"selected_HI={hi_name}| "
                f"conf={hi_conf:>6.4f}| "
                f"bias={bias:>10.5f}| "
                f"rule: HI {hi_name}>{threshold:>10.5f}| "
            )
 
            recurse(depth + 1, path_bits + [0], path_name + "_L")
            recurse(depth + 1, path_bits + [1], path_name + "_R")
 
        # Leaf node
        else:
            leaf_id = leaf_index_from_path(path_bits)
            action = int(leaf_actions[leaf_id])
 
            print(
                f"Leaf {leaf_id:>3}| "
                f"depth={depth}| "
                f"path={path_name}| "
                f"action={action}"
            )
 
    recurse(depth=0, path_bits=[], path_name="root")
 
    print(f"\nFull internal nodes before pruning = {2**max_depth - 1}")
    print(f"Full leaf nodes before pruning     = {2**max_depth}")
 




def summarize_oblique_tree_after_pruning(OBT_actor, STC_actor):
    """
    Print active internal nodes and leaf nodes in the pruned oblique tree.
    """
    from softtree.bhi_softtree import GROUP_NAMES

    tree = OBT_actor.module.tree
    core = STC_actor.module[0].module


    selected_idx = (
        core.inner_nodes.selection_logits
        .argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
    )

    selected_names = [
        GROUP_NAMES[int(k)]
        for k in selected_idx
    ]

    selection_probs = (
        core.inner_nodes.get_selection_probs()
        .detach()
        .cpu()
        .numpy()
    )

    def path_to_bits(path):
        """
        Convert path string to bits.

        Examples:
            root              -> []
            root_Left         -> [0]
            root_Right_Left   -> [1, 0]
            root_L_R          -> [0, 1]
        """
        if path is None or path == "root":
            return []

        parts = path.split("_")[1:]
        bits = []

        for part in parts:
            if part in ["L"]: # Left
                bits.append(0)
            elif part in ["R"]: # Right
                bits.append(1)
            else:
                return []

        return bits

    def path_to_depth(path):
        return len(path_to_bits(path))

    def internal_node_no_from_path(path):
        """
        Full-tree internal node number.

        Examples:
            root       -> 0
            root_L     -> 1
            root_R     -> 2
        """
        bits = path_to_bits(path)

        idx = 0
        for bit in bits:
            idx = 2 * idx + 1 + bit

        return idx

    def leaf_no_from_path(path):
        """
        Left-to-right leaf number from path bits.
        """
        bits = path_to_bits(path)

        leaf_no = 0
        for bit in bits:
            leaf_no = 2 * leaf_no + bit

        return leaf_no

    def collect_entries(node, new_path="root", new_depth=0, entries=None):
        """
        Collect all active nodes and leaves from the pruned tree.
        """
        if entries is None:
            entries = []

        if node is None:
            return entries

        original_path = getattr(node, "id", None)
        original_depth = path_to_depth(original_path)

        if node.is_leaf:
            entries.append({
                "kind": "Leaf",
                "node": node,
                "original_path": original_path,
                "new_path": new_path,
                "original_depth": original_depth,
                "new_depth": new_depth,
                "new_bits": path_to_bits(new_path),
            })
            return entries

        entries.append({
            "kind": "Node",
            "node": node,
            "original_path": original_path,
            "new_path": new_path,
            "original_depth": original_depth,
            "new_depth": new_depth,
            "new_bits": path_to_bits(new_path),
        })

        collect_entries(
            node.left,
            new_path=new_path + "_Left",
            new_depth=new_depth + 1,
            entries=entries,
        )

        collect_entries(
            node.right,
            new_path=new_path + "_Right",
            new_depth=new_depth + 1,
            entries=entries,
        )

        return entries

    entries = collect_entries(tree.root)

    internal_entries = [
        e for e in entries
        if e["kind"] == "Node"
    ]

    leaf_entries = [
        e for e in entries
        if e["kind"] == "Leaf"
    ]

    # New internal node numbers:
    # breadth-first order, left-to-right within each depth.
    internal_entries_sorted = sorted(
        internal_entries,
        key=lambda e: (e["new_depth"], e["new_bits"])
    )

    for new_no, entry in enumerate(internal_entries_sorted):
        entry["new_node_no"] = new_no

    # New leaf numbers:
    # terminal leaves from left to right.
    leaf_entries_sorted = sorted(
        leaf_entries,
        key=lambda e: e["new_bits"]
    )

    for new_no, entry in enumerate(leaf_entries_sorted):
        entry["new_leaf_no"] = new_no

    active_internal_nodes = len(internal_entries)
    active_leaf_nodes = len(leaf_entries)


    print("\n================ Oblique-tree structure AFTER pruning ================")
    print("Active nodes and leaves:")

    # Print in readable tree order: root, then left branch, then right branch.
    for entry in entries:
        node = entry["node"]
        original_path = entry["original_path"]
        new_path = entry["new_path"]


        if entry["kind"] == "Leaf":
            action = int(node.value)

            original_leaf_no = leaf_no_from_path(original_path)
            new_leaf_no = entry["new_leaf_no"]

            print(
                f"Leaf | "
                f"original_leaf_no={original_leaf_no} | "
                f"new_leaf_no={new_leaf_no} | "
                f"new_path={new_path} | "
                f"action={ACTION_NAMES[action]}(key={action})"
            )

        else:
            bias = float(node.bias)
            threshold = -bias

            original_node_no = internal_node_no_from_path(original_path)
            new_node_no = entry["new_node_no"]

            if original_node_no < len(selected_names):
                hi_name = selected_names[original_node_no]
                hi_conf = float(
                    selection_probs[original_node_no, selected_idx[original_node_no]]
                )
                rule_text = f"{hi_name} > {threshold:.5f}"
                conf_text = f"{hi_conf:.4f}"
            else:
                rule_text = f"selected_HI_unknown > {threshold:.5f}"
                conf_text = "NA"

            print(
                f"Node | "
                f"original_node_no={original_node_no} | "
                f"new_node_no={new_node_no} | "
                f"new_path={new_path} | "
                f"rule: {rule_text} | "
                f"conf={conf_text}"

            )

    print("\nPruned oblique tree:")
    print(f"  active internal nodes   = {active_internal_nodes}")
    print(f"  active leaf nodes       = {active_leaf_nodes}")
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
    # summarize_oblique_tree_after_pruning(OBT_actor)
    # I need to pass STC_actor to summarize_oblique_tree_after_pruning because I need to use STC_actor to get the selected HI name for each internal node in the pruned oblique tree.
    summarize_oblique_tree_after_pruning(OBT_actor, STC_actor)


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

    # In "init_bhi_learned", I used STC_actor instead of OBT_actor because in this line, 
    # I just want to compute BHI, so it doesn't matter which actor I use. Actually, the eval_rewards
    # is important, and it is computed by OBT_actor in "eval_log = SofttreePPOTrainer.evaluate" line.
    init_bhi_learned = np.array([
        compute_bhi_from_observation_learned_weights(STC_actor, obs)
        for obs in init_states
    ])

    init_bhi_fixed = np.array([
        compute_bhi_from_observation_fixed_weights(obs)
        for obs in init_states
    ])

    # Plot 1: learned-weight BHI vs reward
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style("ticks")
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        sns.scatterplot(
            x=init_bhi_learned,
            y=eval_rewards,
            ax=ax,
        )

        ax.set_xlabel("Initial BHI using learned soft-tree weights")
        ax.set_ylabel("Unnormalized episode reward using fixed reward weights")
        ax.set_title("Oblique Tree Actor: Learned-Weight BHI vs Reward")


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
        ax.set_title("Oblique Tree Actor: Fixed-Weight BHI(To compare with NN) vs Reward")


    # save results
    candidate_nodes = np.sum(prune_mask).item()
    internal_nodes = OBT_actor.module.tree.internal_num
    leaf_nodes = OBT_actor.module.tree.leaf_num
    pruned_internal = 2**OBT_actor.module.tree.max_depth - 1 - internal_nodes
    pruned_leaf = 2**OBT_actor.module.tree.max_depth - leaf_nodes
    val_res = {
        "init_bhi_learned_weights": init_bhi_learned,
        "init_bhi_fixed_weights": init_bhi_fixed,
        "eval_reward_unnormalized": eval_rewards,
        "internal_nodes": internal_nodes,
        "leaf_nodes": leaf_nodes,
        "candidate_nodes": candidate_nodes,
        "pruned_internal": pruned_internal,
        "pruned_leaf": pruned_leaf,
    }


    pd.DataFrame(val_res).to_csv(
        save_path,
        index=False
    )