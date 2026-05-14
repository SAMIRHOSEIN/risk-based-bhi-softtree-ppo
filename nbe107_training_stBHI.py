#%%
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.envs import GymWrapper

from bridge_gym.example_bridge_bhi.rl_env import SingleElement
from softtree_ppo.training import SofttreePPOTrainer
from softtree_ppo.rl_util import CriticNet
from softtree.softtree_classification import SoftTreeClassifier

from nbe107_training_nn import max_steps, gamma
from nbe107_training_nn import include_step_count
from nbe107_training_nn import alpha_vector
from nbe107_training_nn import cost_kwargs


class SharedPositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Shared weights: A single 1D vector of size (in_features,)
        self.weight = nn.Parameter(torch.empty(in_features))
        
        # 2. Unique biases: A 1D vector of size (out_features,)
        self.bias = nn.Parameter(torch.empty(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similarly to standard nn.Linear
        # We unsqueeze temporarily so kaiming_uniform_ can calculate fan-in properly
        nn.init.kaiming_uniform_(self.weight.unsqueeze(0), a=math.sqrt(5))
        
        # Initialize biases
        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Apply Softplus to ensure all weights are strictly > 0.
        # (You could also use torch.exp(self.weight) or torch.abs(self.weight))
        pos_weight = F.softplus(self.weight)
        
        # Calculate the shared output.
        # x is shape: (batch_size, in_features)
        # pos_weight.unsqueeze(0) is shape: (1, in_features)
        # Resulting shared_out is shape: (batch_size, 1)
        shared_out = F.linear(x, pos_weight.unsqueeze(0))
        
        # Add the unique biases using broadcasting.
        # (batch_size, 1) + (out_features,) -> (batch_size, out_features)
        return shared_out + self.bias


class SoftTreeBHI(SoftTreeClassifier):
    def __init__(self, input_dim, output_dim, depth, beta, apply_batchNorm=False, **kwargs):
        super().__init__(input_dim, output_dim, depth, beta, apply_batchNorm, **kwargs)
        self.inner_nodes = SharedPositiveLinear(input_dim, self.internal_node_num_)


# %%

if __name__ == '__main__':
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
    gym_env = SingleElement(
        max_steps=max_steps, discount=gamma,
        include_step_count=include_step_count,
        reset_prob=None,
        dirichlet_alpha=alpha_vector,
        render_mode="ansi",
        seed=env_seed,
        cost_kwargs=cost_kwargs,
    )
    env = GymWrapper(gym_env, categorical_action_encoding=True)

    # create actor and critic nets
    torch.manual_seed(torch_seed)
    actor_tree = SoftTreeBHI(
        input_dim=gym_env.state_size + int(gym_env.include_step_count),
        output_dim=gym_env.action_size,
        depth=actor_tree_depth,
        beta=tree_beta,
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
    unscaled_rewards = np.array(train_log["reward"])*cost_kwargs["normalizer"]
    unscaled_eval_rewards = np.array(eval_log["eval_reward"])*cost_kwargs["normalizer"]
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.plot(train_log["batch"], unscaled_rewards, label="training")
        ax.plot(eval_log["batch"], unscaled_eval_rewards, label="evaluation")

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
# %%