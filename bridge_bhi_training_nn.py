# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 

import torch
from torchrl.envs import GymWrapper

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from softtree_ppo.training import PPOTrainer
from softtree_ppo.rl_util import ActorNetLogit, CriticNet

from bridge_gym.example_bridge_bhi.settings import (
    max_steps,
    gamma,
    include_step_count,
    reset_prob,
)

# %%

# Neural-network actor parameters
actor_neurons, actor_layers = 32, 2
critic_neurons, critic_layers = 32, 2

# Use the same reward normalization as the soft-tree training script.
# If reward_normalizer is None, BridgeBHIEnv uses C0 as the normalizer.
reward_normalizer = None

if __name__ == '__main__':
    env_seed = 42

    # actor and critic net parameters
    torch_seed = 46 


    # training configuration
    train_config = {
        "total_frames": 2_000_000,
        "frames_per_batch": 20_000,

        "clip_epsilon": 0.1,
        "entropy_eps": 0.05,
        "critic_coef": 0.5,
        "GAE_gamma": 1.0,
        "GAE_lmbda": 0.95,
        "average_GAE": True,
        "reward_decay": None,

        "learning_rate": 1e-3,
        "scheduler_type": None,
        "lr_min": 1e-3,

        "actor_l1_coef": 0.0,
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
        reward_normalizer=reward_normalizer,
        render_mode="ansi",
        seed=env_seed,
    )
    env = GymWrapper(gym_env, categorical_action_encoding=True)

    # create actor and critic nets
    torch.manual_seed(torch_seed)
    actor_net = ActorNetLogit(
        input_dim=gym_env.state_size + int(gym_env.include_step_count),
        output_dim=gym_env.action_size,
        actor_cells=actor_neurons,
        actor_layers=actor_layers,
    )
    critic_net = CriticNet(
        input_dim=gym_env.state_size + int(gym_env.include_step_count),
        critic_cells=critic_neurons,
        critic_layers=critic_layers,
    )

    # create trainer
    trainer = PPOTrainer(
        env=env,
        actor_net=actor_net,
        critic_net=critic_net,
        config=train_config,
    )

    # train
    train_log, eval_log = trainer.train()

    # plot learning curves
    rewards = np.array(train_log["reward"])
    eval_rewards = np.array(eval_log["eval_reward"])
    
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.plot(train_log["batch"], rewards, label="training")
        ax.plot(eval_log["batch"], eval_rewards, label="evaluation")
        ax.set_xlabel("# of training episodes") #ax.set_xlabel("Training Batch")
        ax.set_ylabel("Normalized episode reward")
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

    # create output folders before saving anything
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./actors", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # cause the train and evaluation batches are not necessarily the same, we save them separately
    train_curve_df = pd.DataFrame({"train_batch": train_log["batch"],"train_reward_normalized": rewards,})

    eval_curve_df = pd.DataFrame({"eval_batch": eval_log["batch"], "eval_reward_normalized": eval_rewards,})

    train_curve_df.to_csv(f"./results/train_log_nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr.csv",index=False,)
    eval_curve_df.to_csv(f"./results/eval_log_nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr.csv",index=False,)



    # save checkpoint (debug) and actor
    trainer.save_checkpoint(f"./checkpoints/checkpoint_nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr.pt")
    # trainer.load_checkpoint("./checkpoints/checkpoint_test.pt")
    actor_save_path = f"./actors/nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr.pt"
    print(f"Saving NN actor to: {actor_save_path}")
    trainer.save_actor(actor_save_path)

# %%
