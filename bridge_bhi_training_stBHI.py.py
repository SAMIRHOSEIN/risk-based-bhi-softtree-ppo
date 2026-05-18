#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.envs import GymWrapper

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from softtree_ppo.training import SofttreePPOTrainer
from softtree_ppo.rl_util import CriticNet
from softtree.softtree_classification import SoftTreeClassifier

import os


from bridge_gym.example_bridge_bhi.settings import (
    NCS,
    ELEMENT_NUMBERS,
    ELEMENT_WEIGHTS,
    HEALTH_COEFFICIENTS,
)



class SharedBHILinear(nn.Module):
    def __init__(
        self,
        num_elements,
        ncs,
        out_features,
        health_coefficients,
        initial_element_weights,
        include_step_count=False,
    ):
        super().__init__()

        self.num_elements = num_elements
        self.ncs = ncs
        self.out_features = out_features
        self.include_step_count = include_step_count

        health_coefficients = torch.as_tensor(health_coefficients, dtype=torch.float32)
        self.register_buffer("health_coefficients", health_coefficients)

        initial_weights = torch.as_tensor(initial_element_weights, dtype=torch.float32)

        # Reason: this initializes the learnable element weights from our engineering-based weights VF×SSF. 
        # Later, we can compare the learned weights with the original ELEMENT_WEIGHTS from Valenzuela or Rashidi.
        # starts close to the engineering-based element weights.
        initial_raw = torch.log(torch.expm1(initial_weights).clamp_min(1e-6))
        self.raw_element_weights = nn.Parameter(initial_raw)

        # Alias for compatibility with SofttreePPOTrainer regularization code.
        # The trainer expects inner_nodes.weight to exist.
        self.weight = self.raw_element_weights

        # One bias/threshold per internal node.
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, x):
        # If time is appended to the observation, ignore it for BHI.
        # BHI must be computed only from element condition states.
        x_cs = x[..., : self.num_elements * self.ncs]

        state = x_cs.reshape(-1, self.num_elements, self.ncs)

        # H_i = CS_i dot K
        element_health = torch.matmul(state, self.health_coefficients)

        # Positive learnable element weights.
        element_weights = F.softplus(self.raw_element_weights)

        # BHI = sum_i W_i H_i / sum_i W_i
        bhi = (element_health * element_weights).sum(dim=1) / element_weights.sum()

        # Same BHI for all internal nodes, different bias per node.
        return bhi.unsqueeze(-1) + self.bias
    



class SoftTreeBHI(SoftTreeClassifier):
    def __init__(
        self,
        input_dim,
        output_dim,
        depth,
        beta,
        num_elements,
        ncs,
        health_coefficients,
        initial_element_weights,
        include_step_count=False,
        apply_batchNorm=False,
        **kwargs,
    ):
        super().__init__(input_dim, output_dim, depth, beta, apply_batchNorm, **kwargs)

        self.inner_nodes = SharedBHILinear(
            num_elements=num_elements,
            ncs=ncs,
            out_features=self.internal_node_num_,
            health_coefficients=health_coefficients,
            initial_element_weights=initial_element_weights,
            include_step_count=include_step_count,
        )

# %%

if __name__ == '__main__':


    # env parameters(in BHI-softtree version, we don't import env parameters from nbe107_training_nn.py since we don't have nbe107_training_nn.py in our directory)
    max_steps, gamma = 200, 1/1.03
    include_step_count = False


    reset_prob = None # this means all elements are reset with [1, 0, 0, 0] probability distribution. 
    # reset_prob = np.array([
    #     [1, 0, 0, 0],  # EL12
    #     [1, 0, 0, 0],  # EL109
    #     [1, 0, 0, 0],  # EL205
    #     [1, 0, 0, 0],  # EL215
    #     [1, 0, 0, 0],  # EL234
    #     [1, 0, 0, 0],  # EL306
    #     [1, 0, 0, 0],  # EL310
    #     [1, 0, 0, 0],  # EL331
    #     [1, 0, 0, 0],  # EL510
    # ], dtype=np.float32)

    reward_normalizer = None # when reward_normalizer is None, we will use C0 as the normalizer.

    env_seed = 305

    # actor and critic net parameters
    torch_seed = 503
    actor_tree_depth, tree_beta = 9, 1.0
    critic_neurons, critic_layers = 32, 2

    # training configuration
    train_config = {
        "total_frames": 2_000_000,
        "frames_per_batch": 20_000,

        "clip_epsilon": 0.1,
        "entropy_eps": 0.001,
        "critic_coef": 1e-5,
        "GAE_gamma": 1.0,
        "GAE_lmbda": 0.95,
        "average_GAE": True,

        "learning_rate": 1e-3,
        "scheduler_type": None,
        "lr_min": 1e-3,

        "actor_l2_coef": 1e-4,
        "beta_anneal": 100**(1/100),
        "beta_update_freq": 1,

        "epochs_per_batch": 100,
        "frames_per_minibatch": 200,
        "max_grad_norm": None, 
        "eval_freq": 10,
        "eval_episodes": 100,
        "eval_deterministic": True,
    }

    # create environment
    gym_env = BridgeBHIEnv(
        max_steps=max_steps,
        discount=gamma,
        include_step_count=include_step_count,
        reset_prob=reset_prob,
        reward_normalizer= reward_normalizer,
        render_mode="ansi",
        seed=env_seed,
    )
    env = GymWrapper(gym_env, categorical_action_encoding=True)

    # create actor and critic nets
    torch.manual_seed(torch_seed)


    initial_element_weights = [
        ELEMENT_WEIGHTS[int(element_no)] for element_no in ELEMENT_NUMBERS
    ]

    actor_tree = SoftTreeBHI(
        input_dim=gym_env.state_size + int(gym_env.include_step_count),
        output_dim=gym_env.action_size,
        depth=actor_tree_depth,
        beta=tree_beta,
        num_elements=len(ELEMENT_NUMBERS),
        ncs=NCS,
        health_coefficients=HEALTH_COEFFICIENTS,
        initial_element_weights=initial_element_weights,
        include_step_count=include_step_count,
        apply_batchNorm=False,
    )

















    critic_net = CriticNet(
        input_dim=gym_env.state_size + int(gym_env.include_step_count),
        critic_cells=critic_neurons,
        critic_layers=critic_layers,
    )

    # create trainer
    trainer = SofttreePPOTrainer(
        env=env,
        actor_tree=actor_tree,
        critic_net=critic_net,
        config=train_config,
    )

    # train
    train_log, eval_log = trainer.train()

    # plot learning curves
    unscaled_rewards = np.array(train_log["reward"])
    unscaled_eval_rewards = np.array(eval_log["eval_reward"])

    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.plot(train_log["batch"], unscaled_rewards, label="training")
        ax.plot(eval_log["batch"], unscaled_eval_rewards, label="evaluation")







    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./actors", exist_ok=True)
    os.makedirs("./results", exist_ok=True)





    # save checkpoint (debug) and actor
    trainer.save_checkpoint(f"./checkpoints/checkpoint_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l2_coef']:.0e}_{max_steps:d}yr.pt")
    # trainer.load_checkpoint("./checkpoints/checkpoint_softtree.pt")
    trainer.save_actor(f"./actors/stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l2_coef']:.0e}_{max_steps:d}yr.pt")

    # save log
    pd.DataFrame(train_log).to_csv(
        f"./results/train_log_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l2_coef']:.0e}_{max_steps:d}yr.csv",
        index=False
    )
    pd.DataFrame(eval_log).to_csv(
        f"./results/eval_log_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l2_coef']:.0e}_{max_steps:d}yr.csv",
        index=False
    )

#%%
# import numpy as np
# import pandas as pd
# import math

# import matplotlib.pyplot as plt
# import seaborn as sns

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchrl.envs import GymWrapper

# from bridge_gym.example_bridge_bhi.rl_env import SingleElement
# from softtree_ppo.training import SofttreePPOTrainer
# from softtree_ppo.rl_util import CriticNet
# from softtree.softtree_classification import SoftTreeClassifier

# from nbe107_training_nn import max_steps, gamma
# from nbe107_training_nn import include_step_count
# from nbe107_training_nn import alpha_vector
# from nbe107_training_nn import cost_kwargs


# class SharedPositiveLinear(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         # 1. Shared weights: A single 1D vector of size (in_features,)
#         self.weight = nn.Parameter(torch.empty(in_features))
        
#         # 2. Unique biases: A 1D vector of size (out_features,)
#         self.bias = nn.Parameter(torch.empty(out_features))
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         # Initialize weights similarly to standard nn.Linear
#         # We unsqueeze temporarily so kaiming_uniform_ can calculate fan-in properly
#         nn.init.kaiming_uniform_(self.weight.unsqueeze(0), a=math.sqrt(5))
        
#         # Initialize biases
#         bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
#         nn.init.uniform_(self.bias, -bound, bound)

#     def forward(self, x):
#         # Apply Softplus to ensure all weights are strictly > 0.
#         # (You could also use torch.exp(self.weight) or torch.abs(self.weight))
#         pos_weight = F.softplus(self.weight)
        
#         # Calculate the shared output.
#         # x is shape: (batch_size, in_features)
#         # pos_weight.unsqueeze(0) is shape: (1, in_features)
#         # Resulting shared_out is shape: (batch_size, 1)
#         shared_out = F.linear(x, pos_weight.unsqueeze(0))
        
#         # Add the unique biases using broadcasting.
#         # (batch_size, 1) + (out_features,) -> (batch_size, out_features)
#         return shared_out + self.bias


# class SoftTreeBHI(SoftTreeClassifier):
#     def __init__(self, input_dim, output_dim, depth, beta, apply_batchNorm=False, **kwargs):
#         super().__init__(input_dim, output_dim, depth, beta, apply_batchNorm, **kwargs)
#         self.inner_nodes = SharedPositiveLinear(input_dim, self.internal_node_num_)


# # %%

# if __name__ == '__main__':
#     env_seed = 305

#     # actor and critic net parameters
#     torch_seed = 503
#     actor_tree_depth, tree_beta = 9, 1.0
#     critic_neurons, critic_layers = 32, 2

#     # training configuration
#     train_config = {
#         "total_frames": 2_000_000,
#         "frames_per_batch": 20_000,

#         "clip_epsilon": 0.1,
#         "entropy_eps": 0.001,
#         "critic_coef": 1e-5,
#         "GAE_gamma": 1.0,
#         "GAE_lmbda": 0.95,
#         "average_GAE": True,

#         "learning_rate": 1e-3,
#         "scheduler_type": None,
#         "lr_min": 1e-3,

#         "actor_l2_coef": 1e-4,
#         "beta_anneal": 100**(1/100),
#         "beta_update_freq": 1,

#         "epochs_per_batch": 100,
#         "frames_per_minibatch": 200,
#         "max_grad_norm": None, 
#         "eval_freq": 10,
#         "eval_episodes": 100,
#         "eval_deterministic": True,
#     }

#     # create environment
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

#     # create actor and critic nets
#     torch.manual_seed(torch_seed)
#     actor_tree = SoftTreeBHI(
#         input_dim=gym_env.state_size + int(gym_env.include_step_count),
#         output_dim=gym_env.action_size,
#         depth=actor_tree_depth,
#         beta=tree_beta,
#         apply_batchNorm=False,
#     )
#     critic_net = CriticNet(
#         input_dim=gym_env.state_size + int(gym_env.include_step_count),
#         critic_cells=critic_neurons,
#         critic_layers=critic_layers,
#     )

#     # create trainer
#     trainer = SofttreePPOTrainer(
#         env=env,
#         actor_tree=actor_tree,
#         critic_net=critic_net,
#         config=train_config,
#     )

#     # train
#     train_log, eval_log = trainer.train()

#     # plot learning curves
#     unscaled_rewards = np.array(train_log["reward"])*cost_kwargs["normalizer"]
#     unscaled_eval_rewards = np.array(eval_log["eval_reward"])*cost_kwargs["normalizer"]
#     with sns.plotting_context("notebook", font_scale=1.0):
#         sns.set_style('ticks')
#         fig, ax = plt.subplots(1, 1, tight_layout=True)
#         ax.plot(train_log["batch"], unscaled_rewards, label="training")
#         ax.plot(eval_log["batch"], unscaled_eval_rewards, label="evaluation")

#     # save checkpoint (debug) and actor
#     trainer.save_checkpoint(f"./checkpoints/checkpoint_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l2_coef']:.0e}_{max_steps:d}yr.pt")
#     # trainer.load_checkpoint("./checkpoints/checkpoint_softtree.pt")
#     trainer.save_actor(f"./actors/stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l2_coef']:.0e}_{max_steps:d}yr.pt")

#     # save log
#     pd.DataFrame(train_log).to_csv(
#         f"./results/train_log_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l2_coef']:.0e}_{max_steps:d}yr.csv",
#         index=False
#     )
#     pd.DataFrame(eval_log).to_csv(
#         f"./results/eval_log_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l2_coef']:.0e}_{max_steps:d}yr.csv",
#         index=False
#     )
# # %%


