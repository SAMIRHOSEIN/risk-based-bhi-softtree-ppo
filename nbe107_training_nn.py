# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchrl.envs import GymWrapper

from bridge_gym.example_nbe107.rl_env import SingleElement
from softtree_ppo.training import PPOTrainer
from softtree_ppo.rl_util import ActorNetLogit, CriticNet

# %%

# env parameters
max_steps, gamma = 200, 1/1.03
include_step_count = False
alpha_vector = np.array([0.14964171, 0.11136174, 0.05003725, 0.03926025])
cost_kwargs = {"normalizer": 1e4}

if __name__ == '__main__':
    env_seed = 42

    # actor and critic net parameters
    torch_seed = 46
    actor_neurons, actor_layers = 64, 1
    critic_neurons, critic_layers = 32, 2

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

        "learning_rate": 1e-3,
        "scheduler_type": None,
        "lr_min": 1e-3,

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
    unscaled_rewards = np.array(train_log["reward"])*cost_kwargs["normalizer"]
    unscaled_eval_rewards = np.array(eval_log["eval_reward"])*cost_kwargs["normalizer"]
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.plot(train_log["batch"], unscaled_rewards, label="training")
        ax.plot(eval_log["batch"], unscaled_eval_rewards, label="evaluation")

    # save checkpoint (debug) and actor
    trainer.save_checkpoint(f"./checkpoints/checkpoint_nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr.pt")
    # trainer.load_checkpoint("./checkpoints/checkpoint_test.pt")
    trainer.save_actor(f"./actors/nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr.pt")

    # save log
    pd.DataFrame(train_log).to_csv(
        f"./results/train_log_nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr.csv",
        index=False
    )
    pd.DataFrame(eval_log).to_csv(
        f"./results/eval_log_nn_{actor_neurons:d}x{actor_layers:d}_{max_steps:d}yr.csv",
        index=False
    )

# %%
