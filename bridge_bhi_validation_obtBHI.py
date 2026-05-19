#%%
# import numpy as np
# import pandas as pd
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# import seaborn as sns

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torchrl.envs import GymWrapper
# from torchrl.modules import ProbabilisticActor
# from torch.distributions import Categorical as CategoricalDist
# from tensordict.nn import TensorDictModule

# from bridge_gym.example_nbe107.rl_env import SingleElement
# from bridge_gym.example_nbe107.settings import CS_PFS
# from softtree_ppo.training import SofttreePPOTrainer
# from softtree.softtree_classification import SoftTreeClassifier

# from nbe107_training_nn import max_steps, gamma
# from nbe107_training_nn import include_step_count
# from nbe107_training_nn import alpha_vector

# # %%

# if __name__ == '__main__':
#     env_seed = 1034
#     obs_episodes = 10
#     pruning_threshold = 5e-4
#     num_episodes = 1000
#     cost_kwargs = {"normalizer": 1}

#     actor_path = "./actors/stBHI_d9b1le1e-04_200yr.pt"
#     save_path = f"./results/val_obtBHI_d9b1le1e-04_200yr_{pruning_threshold:.0e}prune.csv"

#     gym_env = SingleElement(
#         max_steps=max_steps, discount=gamma,
#         include_step_count=include_step_count,
#         reset_prob=None,
#         dirichlet_alpha=alpha_vector,
#         render_mode="ansi",
#         seed=env_seed,
#         cost_kwargs=cost_kwargs,
#     )
#     env = GymWrapper(gym_env, categorical_action_encoding=True)

#     # load softtree_BHI
#     load_dict = torch.load(actor_path)
#     # create a placeholder STC model
#     STC_model = SoftTreeClassifier(
#         input_dim=load_dict['actor_core_hyperparams']['input_dim'],
#         output_dim=load_dict['actor_core_hyperparams']['output_dim'],
#         depth=load_dict['actor_core_hyperparams']['depth'],
#         beta=load_dict['actor_core_hyperparams']['beta'],
#     )

#     # transfer the weights to a conventional soft tree model
#     with torch.no_grad(): # Essential when modifying parameters directly
#         # Calculate and expand the positive weights to shape (out_features, in_features)
#         trained_pos_weight = F.softplus(
#             load_dict['actor_core_state']['inner_nodes.weight']
#         )
#         expanded_weights = trained_pos_weight.unsqueeze(0).expand(
#             STC_model.internal_node_num_, -1
#         )
    
#         # Create a dictionary matching nn.Linear's expected keys
#         load_dict['actor_core_state']['inner_nodes.weight'] = expanded_weights
#         STC_model.load_state_dict(load_dict['actor_core_state'])

#     # recreate actor
#     actor_module = TensorDictModule(STC_model, in_keys=['observation'], out_keys=['logits'])
#     STC_actor = ProbabilisticActor(
#         module=actor_module,
#         spec=env.action_spec,
#         distribution_class=CategoricalDist,
#         in_keys=['logits'],  # Key in the input tensor containing the observation
#         out_keys=['action'],  # Key where the sampled action will be written
#         return_log_prob=True,
#     ).to(env.device)
#     print(f"[*] Actor created successfully from {actor_path}")
    
#     # use actor
#     obs = env.rollout(
#         max_steps=gym_env.max_steps*obs_episodes,
#         policy=STC_actor,
#         break_when_any_done=False,
#         auto_reset=True,
#         auto_cast_to_device=True,
#     )
#     OBT_actor, prune_mask = SofttreePPOTrainer.convert_to_obtree_actor(
#         STC_actor,
#         observations_t=obs["observation"],
#         pruning_threshold=pruning_threshold,
#     )

#     # evaluate oblique tree actor
#     eval_log = SofttreePPOTrainer.evaluate(
#         OBT_actor,
#         env,
#         num_episodes=num_episodes,
#         max_steps=max_steps,
#         deterministic=True,
#     )

#     # plot testing results
#     if include_step_count:
#         init_states = np.array(eval_log["init_state"])[:, :-1]
#     else:
#         init_states = np.array(eval_log["init_state"])
#     init_pf = init_states @ CS_PFS
#     init_beta = -stats.norm.ppf(init_pf)
#     eval_costs = -np.array(eval_log["eval_reward"])
#     with sns.plotting_context("notebook", font_scale=1.0):
#         sns.set_style('ticks')
#         fig, ax = plt.subplots(1, 1, tight_layout=True)
#         sns.scatterplot(x=init_beta, y=eval_costs, ax=ax)
#         # ax.set_ylim(0, 1e6)

#     # save results
#     candidate_nodes = np.sum(prune_mask != None).item()
#     internal_nodes = OBT_actor.module.tree.internal_num
#     leaf_nodes = OBT_actor.module.tree.leaf_num
#     pruned_nodes = 2**OBT_actor.module.tree.max_depth - 1 - (internal_nodes + leaf_nodes)
#     val_res = {
#         'init_beta': init_beta,
#         'eval_costs': eval_costs,
#         'internal_nodes': internal_nodes,
#         'leaf_nodes': leaf_nodes,
#         'candidate_nodes': candidate_nodes,
#         'pruned_nodes': pruned_nodes
#     }
#     pd.DataFrame(val_res).to_csv(
#         save_path,
#         index=False
#     )
# %%




#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from torchrl.envs import GymWrapper
from torchrl.modules import ProbabilisticActor
from torch.distributions import Categorical as CategoricalDist
from tensordict.nn import TensorDictModule

from softtree_ppo.training import SofttreePPOTrainer

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from bridge_gym.example_bridge_bhi.settings import (
    NCS,
    ELEMENT_NUMBERS,
    ELEMENT_WEIGHTS,
    HEALTH_COEFFICIENTS,
)




from bridge_bhi_training_stBHI import SoftTreeBHI

from bridge_gym.example_bridge_bhi.settings import (
    NCS,
    ELEMENT_NUMBERS,
    ELEMENT_WEIGHTS,
    HEALTH_COEFFICIENTS,
    max_steps,
    gamma,
    include_step_count,
    reset_prob,
)


def compute_node_logits(model, observations):
    model.eval()
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32)

    with torch.no_grad():
        raw_logits = model.inner_nodes(obs_tensor)
        beta_logits = model.beta * raw_logits

    return raw_logits.cpu().numpy(), beta_logits.cpu().numpy()


def plot_node_logits_vs_bhi(model, observations, init_bhi, max_nodes_to_plot=15):
    raw_logits, beta_logits = compute_node_logits(model, observations)

    nodes_to_plot = min(beta_logits.shape[1], max_nodes_to_plot)

    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style("ticks")
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        for node_id in range(nodes_to_plot):
            ax.scatter(
                init_bhi,
                beta_logits[:, node_id],
                s=18,
                label=f"Node {node_id}",
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Initial Bridge Health Index")
        ax.set_ylabel(r"Node logit: $\beta(BHI+b_n)$")
        ax.legend(ncol=2, fontsize=8)

    return raw_logits, beta_logits








def compute_bhi_from_observation(obs):
    """
    Compute bridge health index from a flattened observation.

    obs shape:
        (num_elements * NCS,)
    or:
        (num_elements * NCS + 1,) if include_step_count=True
    """

    if include_step_count:
        obs = obs[:-1]

    state = obs.reshape(len(ELEMENT_NUMBERS), NCS)

    health_coefficients = np.asarray(HEALTH_COEFFICIENTS, dtype=float)

    element_health = state @ health_coefficients

    weights = np.array(
        [ELEMENT_WEIGHTS[int(element_no)] for element_no in ELEMENT_NUMBERS],
        dtype=float,
    )

    bhi = np.sum(weights * element_health) / np.sum(weights)

    return float(bhi)












# %%

if __name__ == '__main__':
    env_seed = 1034
    obs_episodes = 10
    pruning_threshold = 5e-4
    num_episodes = 1000

    reward_normalizer = 1

    actor_path = f"./actors/stBHI_d10b1le1e-04_{max_steps:d}yr.pt"
    save_path = f"./results/val_obtBHI_d10b1le1e-04_{max_steps:d}yr_{pruning_threshold:.0e}prune.csv"


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

    # load softtree_BHI
    load_dict = torch.load(actor_path, map_location="cpu")
    # create a placeholder STC model


    initial_element_weights = [
        ELEMENT_WEIGHTS[int(element_no)] for element_no in ELEMENT_NUMBERS
    ]

    STC_model = SoftTreeBHI(
        input_dim=load_dict["actor_core_hyperparams"]["input_dim"],
        output_dim=load_dict["actor_core_hyperparams"]["output_dim"],
        depth=load_dict["actor_core_hyperparams"]["depth"],
        beta=load_dict["actor_core_hyperparams"]["beta"],
        num_elements=len(ELEMENT_NUMBERS),
        ncs=NCS,
        health_coefficients=HEALTH_COEFFICIENTS,
        initial_element_weights=initial_element_weights,
        include_step_count=include_step_count,
        apply_batchNorm=False,
    )

    STC_model.load_state_dict(load_dict["actor_core_state"])
    STC_model.eval()



    # recreate actor
    actor_module = TensorDictModule(STC_model, in_keys=['observation'], out_keys=['logits'])
    STC_actor = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        distribution_class=CategoricalDist,
        in_keys=['logits'],  # Key in the input tensor containing the observation
        out_keys=['action'],  # Key where the sampled action will be written
        return_log_prob=True,
    ).to(env.device)
    print(f"[*] Actor created successfully from {actor_path}")
    
    # use actor
    obs = env.rollout(
        max_steps=gym_env.max_steps*obs_episodes,
        policy=STC_actor,
        break_when_any_done=False,
        auto_reset=True,
        auto_cast_to_device=True,
    )






    print("inner_nodes.weight shape:", STC_model.inner_nodes.weight.shape)
    print("raw_element_weights shape:", STC_model.inner_nodes.raw_element_weights.shape)
    print("bias shape:", STC_model.inner_nodes.bias.shape)
    print("observation shape:", obs["observation"].shape)




    OBT_actor, prune_mask = SofttreePPOTrainer.convert_to_obtree_actor(
        STC_actor,
        observations_t=obs["observation"],
        pruning_threshold=pruning_threshold,
    )

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

    init_bhi = np.array([
        compute_bhi_from_observation(obs)
        for obs in init_states
    ])

    raw_logits, beta_logits = plot_node_logits_vs_bhi(
        STC_model,
        init_states,
        init_bhi,
        max_nodes_to_plot=15,
    )

    # save results
    candidate_nodes = np.sum(prune_mask != None).item()
    internal_nodes = OBT_actor.module.tree.internal_num
    leaf_nodes = OBT_actor.module.tree.leaf_num
    pruned_nodes = 2**OBT_actor.module.tree.max_depth - 1 - (internal_nodes + leaf_nodes)
    val_res = {
        "init_bhi": init_bhi,
        "eval_reward": eval_rewards,
        "internal_nodes": internal_nodes,
        "leaf_nodes": leaf_nodes,
        "candidate_nodes": candidate_nodes,
        "pruned_nodes": pruned_nodes,
    }


    num_logits_to_save = min(15, beta_logits.shape[1])
    for node_id in range(num_logits_to_save):
        val_res[f"node_{node_id}_raw_logit"] = raw_logits[:, node_id]
        val_res[f"node_{node_id}_beta_logit"] = beta_logits[:, node_id]
    pd.DataFrame(val_res).to_csv(
        save_path,
        index=False
    )