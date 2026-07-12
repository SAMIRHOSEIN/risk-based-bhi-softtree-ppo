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












from bridge_gym.example_bridge_bhi.settings import (
    NCS,
    ELEMENT_NUMBERS,
    ELEMENT_WEIGHTS,
    HEALTH_COEFFICIENTS,
    GROUP_ORDER,
    ELEMENT_TO_GROUP_IDX,
    max_steps,
    gamma,
    include_step_count,
    reset_prob,
    ACTION_NAMES,
    ELEMENT_NAMES,
    RUN_MODE_TAG,  # "<STATE_TRANSITION_MODE>_<learnSF|fixedSF>" tag embedded in every saved filename
)



















from bridge_bhi_training_stBHI import actor_tree_depth, tree_beta, reg_coef

from bridge_bhi_validation_nn import compute_bhi_from_observation_fixed_weights

from bridge_bhi_validation_stBHI import compute_bhi_from_observation_learned_weights
from bridge_bhi_validation_stBHI import (
    element_weights_are_learnable,
    weight_mode_label,
    get_actor_element_weights,
)

from bridge_bhi_validation_nn import mean_and_ci

def summarize_element_weights(actor):
    """
    Print BHI-soft-tree parameters for each internal node.
    Print leaf-node classes/actions from the extracted oblique tree.

    Each node uses: logit_n = beta * (BHI + b_n)

    Hard split: BHI + b_n > 0 equivalent to: BHI > -b_n
    """
    core = actor.module[0].module

    learned_weights = get_actor_element_weights(actor)
    normalized_weights = learned_weights / learned_weights.sum()

    # If the actor was trained with LEARNABLE_SIGNIFICANCE_FACTOR = False, the
    # weights are NOT learned: they are fixed at ELEMENT_WEIGHTS. Report them as
    # such instead of pretending anything was learned.
    if not element_weights_are_learnable(actor):
        print("\n================ BHI-soft-tree FIXED element significance factors "
              "(LEARNABLE_SIGNIFICANCE_FACTOR = False) ================")
        print("These weights were held fixed at ELEMENT_WEIGHTS; nothing was learned.")
        print(ELEMENT_NAMES)
        for element_no, w_raw, w_norm in zip(ELEMENT_NUMBERS, learned_weights, normalized_weights):
            print(
                f"Element {int(element_no):>4} | "
                f"fixed W={w_raw:>8.4f} | "
                f"normalized fixed W={w_norm:>8.4f}"
            )
        return

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

    learned_weights = get_actor_element_weights(actor)

    normalized_learned_weights = (
        learned_weights / learned_weights.sum()
    )

    # In fixed mode there is no "learning": ranking-vs-learned and rank-change
    # are all trivial (weights == ELEMENT_WEIGHTS). Print a single fixed ranking
    # instead of the learned/original/rank-change comparison.
    if not element_weights_are_learnable(actor):
        fixed_rows = sorted(
            (
                {
                    "element_no": int(en),
                    "element_name": ELEMENT_NAMES[int(en)],
                    "fixed_weight": float(w),
                    "normalized_fixed_weight": float(wn),
                }
                for en, w, wn in zip(
                    ELEMENT_NUMBERS, learned_weights, normalized_learned_weights
                )
            ),
            key=lambda r: r["fixed_weight"],
            reverse=True,
        )
        print("\n================ Ranking based on FIXED significance factors "
              "(LEARNABLE_SIGNIFICANCE_FACTOR = False) ================")
        print("Weights were held fixed at ELEMENT_WEIGHTS; there is nothing learned to rank.")
        for rank, row in enumerate(fixed_rows):
            print(
                f"Rank {rank+1:>2} | "
                f"EN={row['element_no']:>3} | "
                f"{row['element_name']:<30} | "
                f"fixed_W={row['fixed_weight']:>8.4f} | "
                f"normalized={row['normalized_fixed_weight']:>8.4f}"
            )
        return

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
            new_path=new_path + "_L",
            new_depth=new_depth + 1,
            entries=entries,
        )

        collect_entries(
            node.right,
            new_path=new_path + "_R",
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
    print("Deck = 12, 331, 306; \nSuperstructure = 109; \nBearings = 310;\n Substructure = 205, 215, 234")
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



































































def summarize_leaf_visits_from_eval(
    OBT_actor,
    eval_log,
    save_path=None,
):
    """
    Count visits to each active leaf using observations already generated
    during policy evaluation.
    """
    tree = OBT_actor.module.tree



    # ----------------------------------------------------------
    # Collect active leaves from left to right
    # ----------------------------------------------------------
    def collect_leaves(node, leaves):

        if node is None:
            return

        if node.is_leaf:
            leaves.append(node)
            return

        collect_leaves(node.left, leaves)
        collect_leaves(node.right, leaves)
    # ----------------------------------------------------------
    # Get original leaf number from original tree path
    # ----------------------------------------------------------
    def get_original_leaf_no(node):

        parts = node.id.split("_")[1:]

        bit_map = {"L": "0",
                   "R": "1",}

        bits = "".join(bit_map[part] for part in parts)

        return int(bits, 2) if bits else 0
    # ----------------------------------------------------------
    # Route one observation through the hard oblique tree
    # ----------------------------------------------------------
    def route_to_leaf(obs):

        node = tree.root

        while not node.is_leaf:

            obs = np.asarray(obs).reshape(-1)

            score = float(
                np.dot(obs, node.weights) + node.bias
            )

            if score < 0:
                node = node.right
            else:
                node = node.left

        return node
    # ----------------------------------------------------------
    # Collect all active leaves
    # ----------------------------------------------------------
    leaves = []
    collect_leaves(tree.root,leaves)
    # ----------------------------------------------------------
    # Initialize statistics
    # ----------------------------------------------------------
    leaf_stats = {
        id(leaf): {
            "visits": 0,
            "reward_sum": 0.0,
        }
        for leaf in leaves
    }
    # ----------------------------------------------------------
    # Count visits using evaluation trajectories
    # ----------------------------------------------------------
    for episode_obs, episode_rewards in zip(
        eval_log["observations"],
        eval_log["step_rewards"],
    ):

        for obs, reward in zip(
            episode_obs,
            episode_rewards,
        ):

            leaf = route_to_leaf(obs)

            stats = leaf_stats[id(leaf)]

            stats["visits"] += 1
            stats["reward_sum"] += float(reward)
    # ----------------------------------------------------------
    # Total visits
    # ----------------------------------------------------------
    total_visits = sum(stats["visits"] for stats in leaf_stats.values())

    # ----------------------------------------------------------
    # Create result table
    # ----------------------------------------------------------
    output_rows = []

    for new_leaf_no, leaf in enumerate(leaves):

        stats = leaf_stats[id(leaf)]
        visits = stats["visits"]
        visit_pct = (100.0 * visits / total_visits if total_visits > 0 else 0.0)
        mean_reward = (stats["reward_sum"] / visits if visits > 0 else np.nan)
        action = int(leaf.value)

        output_rows.append({
            "new_leaf_no": new_leaf_no,
            "original_leaf_no": get_original_leaf_no(leaf),
            "action_key": action,
            "action_name": ACTION_NAMES[action],
            "visits": visits,
            "visit_pct": visit_pct,
            "mean_reward_when_visited": mean_reward,
        })
    # ----------------------------------------------------------
    # Save and print results
    # ----------------------------------------------------------
    df = pd.DataFrame(output_rows)
    print("\n================ Oblique-tree leaf visit counts ================")
    print(f"Total leaf visits = {total_visits}")
    print(df.to_string(index=False))

    if save_path is not None:
        df.to_csv(save_path,index=False,)
        print(f"\n[*] Leaf visit counts saved to {save_path}")

    return df


















def plot_validation_hi_trajectories(
    STC_actor,
    eval_log,
    save_prefix=None,
):
    """
    Compute and plot validation trajectories for:

        1. Deck HI
        2. Superstructure HI
        3. Bearings HI
        4. Substructure HI
        5. Aggregate BHI

    For each health index, one separate figure is created.

    The solid line is the mean trajectory across validation episodes.
    The shaded band shows the 5th to 95th percentile range.

    Health-index calculations use the learned positive element weights
    from the trained BHI soft-tree actor.
    """

    # ==========================================================
    # 1. Get learned positive element weights
    # ==========================================================
    core = STC_actor.module[0].module

    learned_weights = torch.nn.functional.softplus(
        core.inner_nodes.raw_element_weights
    ).detach().cpu().numpy()


    # ==========================================================
    # 2. Collect validation observations
    #
    # Shape:
    #   (num_episodes, num_steps, observation_size)
    # ==========================================================
    observations = np.stack(
        eval_log["observations"],
        axis=0,
    )

    num_episodes, num_steps, _ = observations.shape


    # ==========================================================
    # 3. Remove step-count feature if it is present
    # ==========================================================
    cs_observations = observations[
        ...,
        :len(ELEMENT_NUMBERS) * NCS
    ]


    # ==========================================================
    # 4. Reshape condition states
    #
    # Shape:
    #   (episodes, time, elements, condition_states)
    # ==========================================================
    cs_probabilities = cs_observations.reshape(
        num_episodes,
        num_steps,
        len(ELEMENT_NUMBERS),
        NCS,
    )


    # ==========================================================
    # 5. Calculate element health
    #
    # H_e = CS_e dot HEALTH_COEFFICIENTS
    #
    # Shape:
    #   (episodes, time, elements)
    # ==========================================================
    health_coefficients = np.asarray(
        HEALTH_COEFFICIENTS,
        dtype=float,
    )

    element_health = (
        cs_probabilities @ health_coefficients
    )


    # ==========================================================
    # 6. Calculate the four group health indices
    #
    # GHI_k =
    # sum(w_e * H_e) / sum(w_e)
    # for elements belonging to group k
    # ==========================================================
    group_idx = np.asarray(
        ELEMENT_TO_GROUP_IDX,
        dtype=int,
    )

    hi_trajectories = {}

    for k, group_name in enumerate(GROUP_ORDER):

        mask = group_idx == k

        group_weights = learned_weights[mask]

        group_element_health = element_health[:, :, mask]

        group_hi = (
            group_element_health * group_weights[None, None, :]
        ).sum(axis=2) / group_weights.sum()

        hi_trajectories[group_name] = group_hi


    # ==========================================================
    # 7. Calculate aggregate learned-weight BHI
    # ==========================================================
    normalized_weights = (
        learned_weights / learned_weights.sum()
    )

    aggregate_bhi = np.sum(
        element_health
        * normalized_weights[None, None, :],
        axis=2,
    )

    hi_trajectories["BHI_aggregate"] = aggregate_bhi


    # ==========================================================
    # 8. Display useful information
    # ==========================================================
    print(
        "\n================ Validation HI trajectories ================"
    )

    print(f"Validation episodes = {num_episodes}")
    print(f"Time steps per episode = {num_steps}")

    for hi_name, trajectories in hi_trajectories.items():

        print(
            f"{hi_name:<20} | "
            f"initial mean={trajectories[:, 0].mean():.4f} | "
            f"final mean={trajectories[:, -1].mean():.4f}"
        )


    # ==========================================================
    # 9. Create five separate trajectory figures
    # ==========================================================
    time_steps = np.arange(num_steps)

    for hi_name, trajectories in hi_trajectories.items():

        fig, ax = plt.subplots(
            figsize=(9, 5),
            tight_layout=True,
        )


        # Plot every validation episode as one trajectory
        for episode_idx in range(num_episodes):

            ax.plot(
                time_steps,
                trajectories[episode_idx, :],
                linewidth=0.6,
                alpha=0.08,
            )


        ax.set_xlabel("Time step")
        ax.set_ylabel("Health Index")
        ax.set_title(
            f"Validation trajectories: {hi_name}"
        )

        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)


        if save_prefix is not None:

            figure_path = (
                f"{save_prefix}_{hi_name}.png"
            )

            fig.savefig(
                figure_path,
                dpi=300,
                bbox_inches="tight",
            )

            print(
                f"[*] Saved: {figure_path}"
            )


        plt.show()
        plt.close(fig)
        # ======================================================
        # Save figure
        # ======================================================
        if save_prefix is not None:

            figure_path = (
                f"{save_prefix}_{hi_name}.png"
            )

            fig.savefig(
                figure_path,
                dpi=300,
                bbox_inches="tight",
            )

            print(
                f"[*] Saved: {figure_path}"
            )


        plt.show()
        plt.close(fig)


    return hi_trajectories



# %%

if __name__ == '__main__':
    env_seed = 1034
    obs_episodes = 10
    pruning_threshold = -np.inf #1e-3
    num_episodes = 1000

    lp_threshold = 1e-6

    reward_normalizer = 1

    # Same RUN_MODE_TAG as the training script, so validation always loads the
    # actor that matches the current STATE_TRANSITION_MODE and
    # LEARNABLE_SIGNIFICANCE_FACTOR settings.
    actor_path = f"./actors/stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{reg_coef:.0e}_{max_steps:d}yr_{RUN_MODE_TAG}.pt"
    save_path = f"./results/val_obtBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{reg_coef:.0e}_{max_steps:d}yr_{RUN_MODE_TAG}_{pruning_threshold:.0e}prune.csv"


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

    # Report whether the element significance factors were learned or fixed.
    mode = weight_mode_label(STC_actor)
    if element_weights_are_learnable(STC_actor):
        print("\n[Element significance factors] LEARNABLE "
              "(LEARNABLE_SIGNIFICANCE_FACTOR = True): weights were trained by PPO.")
    else:
        print("\n[Element significance factors] FIXED "
              "(LEARNABLE_SIGNIFICANCE_FACTOR = False): weights are held at "
              "ELEMENT_WEIGHTS, so there are NO learned weights.")

    # Print the element weights in the BHI-soft-tree actor (learned or fixed)
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




    leaf_visit_path = (
        f"./results/leaf_visits_obtBHI_d{actor_tree_depth:d}"
        f"b{tree_beta:.0f}le{reg_coef:.0e}_{max_steps:d}yr_"
        f"{RUN_MODE_TAG}_"
        f"{pruning_threshold:.0e}prune.csv"
    )




    # # evaluate oblique tree actor
    # eval_log = SofttreePPOTrainer.evaluate(
    #     OBT_actor,
    #     env,
    #     num_episodes=num_episodes,
    #     max_steps=max_steps,
    #     deterministic=True,
    # )



    # leaf_visit_df = summarize_leaf_visits_from_eval(
    #     OBT_actor=OBT_actor,
    #     eval_log=eval_log,
    #     save_path=leaf_visit_path,
    # )

    # evaluate oblique tree actor
    eval_log = SofttreePPOTrainer.evaluate(
        OBT_actor,
        env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=True,
    )


    # Plot validation trajectories for:
    # Deck HI
    # Superstructure HI
    # Bearings HI
    # Substructure HI
    # Aggregate BHI
    hi_trajectories = plot_validation_hi_trajectories(
        STC_actor=STC_actor,
        eval_log=eval_log,
        save_prefix=(
            f"./results/hi_trajectory_obtBHI_"
            f"d{actor_tree_depth:d}"
            f"b{tree_beta:.0f}"
            f"le{reg_coef:.0e}_"
            f"{max_steps:d}yr_"
            f"{RUN_MODE_TAG}"
        ),
    )


    leaf_visit_df = summarize_leaf_visits_from_eval(
        OBT_actor=OBT_actor,
        eval_log=eval_log,
        save_path=leaf_visit_path,
    )

































    # plot testing results
    init_states = np.array(eval_log["init_state"])
    eval_rewards = np.array(eval_log["eval_reward"])





    reward_stats = mean_and_ci(eval_log["eval_reward"])

    print(f"Validation (episode return for {reward_stats['n']} episodes): "
        f"mean={reward_stats['mean']:.4f}, "
        f"95% CI=[{reward_stats['ci_low']:.4f}, {reward_stats['ci_high']:.4f}], "
        f"SD={reward_stats['sd']:.4f}")   








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
        ax.set_title(f"Oblique Tree Actor: {mode.capitalize()}-Weight BHI vs Reward")


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
        "init_bhi_fixed_weights": init_bhi_fixed,
        "eval_reward_unnormalized": eval_rewards,
        "internal_nodes": internal_nodes,
        "leaf_nodes": leaf_nodes,
        "candidate_nodes": candidate_nodes,
        "pruned_internal": pruned_internal,
        "pruned_leaf": pruned_leaf,
    }
    # Only emit a "learned weights" column when weights were actually learned.
    # In fixed mode the actor-weight BHI equals init_bhi_fixed_weights, so a
    # separate "learned" column would be both redundant and misleading.
    if element_weights_are_learnable(STC_actor):
        val_res["init_bhi_learned_weights"] = init_bhi_learned


    pd.DataFrame(val_res).to_csv(
        save_path,
        index=False
    )

