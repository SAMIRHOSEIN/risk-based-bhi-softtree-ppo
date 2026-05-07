#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torchrl
from torchrl.envs import GymWrapper

from bridge_gym.example_nbe107.rl_env import SingleElement
from softtree_ppo.training import SofttreePPOTrainer
from softtree_ppo.rl_util import CriticNet
from softtree.softtree_classification import SoftTreeClassifier

from nbe107_training_nn import max_steps, gamma
from nbe107_training_nn import include_step_count
from nbe107_training_nn import alpha_vector
from nbe107_training_nn import cost_kwargs

# %%
# I need them outside the main block for loading the trained actor in validation
actor_tree_depth, tree_beta = 9, 1.0  # David's assumption 10, 1.0


if __name__ == '__main__':
    env_seed = 305

    # actor and critic net parameters
    torch_seed = 503
    critic_neurons, critic_layers = 32, 2

    # training configuration
    train_config = {
        "total_frames": 200_000,                # Amir's assumption 163840, David's assumption 100_000
        "frames_per_batch": 2_000,              # Amir's assumption 160, David's assumption 1000

        "clip_epsilon": 0.2,                  # Amir's assumption 1e-3, David's assumption 0.1,
        "entropy_eps": 0.01,                   # the same as Amir's version
        "critic_coef": 1.0,                    # the same as Amir's version
        "GAE_gamma": 1.0,                      # the same as Amir's version
        "GAE_lmbda": 0.95,                     # the same as Amir's version
        "average_GAE": True,                   # the same as Amir's version

        "learning_rate": 1e-2,                  # Amir's assumption 1e-3, David's assumption 1e-2,
        "scheduler_type": "cosine",
        "lr_min": 1e-4,                         # Amir's assumption 1e-5, David's assumption 1e-4,

        "actor_l1_coef": 1e-4,                   # David's assumption 1e-4, 
        "beta_anneal": 100**0.01,                 # David's assumption 1.5848931925,
        "beta_update_freq": 1,

        "epochs_per_batch": 50,                 # the same as Amir's version
        "frames_per_minibatch": 100,            # Amir's assumption 160, David's assumption 100
        "max_grad_norm": 1.0,                   # the same as Amir's version
        "eval_freq": 1,                         # the same as Amir's version
        "eval_episodes": 10,                    # Amir's assumption 1
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
    actor_tree = SoftTreeClassifier(
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
        ax.set_xlabel("# of training episodes") #ax.set_xlabel("Training Batch")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Learning Curve")
        ax.legend()



    # # save learning curve data
    # learning_curve_df = pd.DataFrame({
    #     "train_batch": train_log["batch"],
    #     "train_reward": unscaled_rewards,
    #     "eval_batch": eval_log["batch"],
    #     "eval_reward": unscaled_eval_rewards
    # })

    # learning_curve_df.to_csv(
    #     f"./learning_curve_nn.csv",
    #     index=False
    # )

    # cause the train and evaluation batches are not necessarily the same, we save them separately
    train_curve_df = pd.DataFrame({
        "train_batch": train_log["batch"],
        "train_reward": unscaled_rewards,
    })

    eval_curve_df = pd.DataFrame({
        "eval_batch": eval_log["batch"],
        "eval_reward": unscaled_eval_rewards,
    })

    train_curve_df.to_csv("./learning_curve_nn_train.csv", index=False)
    eval_curve_df.to_csv("./learning_curve_nn_eval.csv", index=False)


    # save checkpoint (debug) and actor
    trainer.save_checkpoint(f"./checkpoints/checkpoint_softtree_d{actor_tree_depth:d}b{tree_beta:.1f}_{max_steps:d}yr.pt")
    # trainer.load_checkpoint("./checkpoints/checkpoint_softtree.pt")
    trainer.save_actor(f"./actors/softtree_d{actor_tree_depth:d}b{tree_beta:.1f}_{max_steps:d}yr.pt")

# %%