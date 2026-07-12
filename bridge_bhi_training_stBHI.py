#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchrl.envs import GymWrapper

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv
from softtree_ppo.training import SofttreePPOTrainer
from softtree_ppo.rl_util import CriticNet

import os

from softtree.bhi_softtree import SoftTreeBHI

from bridge_gym.example_bridge_bhi.settings import (
    NCS,
    ELEMENT_NUMBERS,
    ELEMENT_WEIGHTS,
    LEARNABLE_SIGNIFICANCE_FACTOR,  # True -> learn element weights; False -> keep them fixed at ELEMENT_WEIGHTS
    RUN_MODE_TAG,           # "<STATE_TRANSITION_MODE>_<learnSF|fixedSF>" tag embedded in every saved filename
    ELEMENT_TO_GROUP_IDX,   # per-element group index for the GHI actor
    HEALTH_COEFFICIENTS,
    max_steps,
    gamma,
    include_step_count,
    reset_prob,
    STATE_TRANSITION_MODE,
)

# Inputs for all files
actor_tree_depth, tree_beta = 8, 1.0 #6, 1.0 #8, 1.0 #3, 1.0
reg_coef = 0.0 #1e-1 # we don't need regularizaion becaue we have tau and it is already a regularization for the selection of the elements.

# %%
if __name__ == '__main__':

    env_seed = 305

    # actor and critic net parameters
    torch_seed = 503
    critic_neurons, critic_layers = 128, 2

    reward_normalizer = None # when reward_normalizer is None, we will use C0 as the normalizer.

    # # training configuration
    # train_config = {
    #     "total_frames": 2_000_000, 
    #     "frames_per_batch": 20_000,

    #     "clip_epsilon": 0.1,
    #     "entropy_eps": 0.05,
    #     "critic_coef": 0.5,
    #     "GAE_gamma": 1.0,
    #     "GAE_lmbda": 0.95,
    #     "average_GAE": True,
    #     "reward_decay": None,

    #     "learning_rate": 1e-3,
    #     "scheduler_type": None,
    #     "lr_min": 1e-3,

    #     "actor_l1_coef": reg_coef, # "actor_l2_coef": 1e-4, 
    #     "beta_anneal": 100**(1/100),
    #     "beta_update_freq": 1, 


    #     # tau (selection-temperature) annealing for per-node HI selection.
    #     # tau_anneal > beta_anneal so selection commits before routing hardens.
    #     # With ~100 batches: 100**(1/60) reaches tau_min ~0.01 around batch 60.
    #     "tau_anneal": 100**(1/60),
    #     "tau_update_freq": 1,
    #     # ----- tau annealing (per-node HI selection temperature) --
    #     # WHY TAU_MIN (don't let tau go to zero or decrease like beta):
    #     #   1. SOFTMAX NUMERICAL SAFETY: When tau → 0, the softmax denominator
    #     #      exp(logits/tau) overflows to infinity. tau_min=0.01 keeps
    #     #      softmax numerically stable during backprop.
    #     #   2. GRADIENT FLOW: Very small tau makes softmax nearly one-hot, so
    #     #      gradients through non-selected HIs vanish → weights stop learning.
    #     #      tau_min=0.01 keeps soft enough that all HIs receive meaningful
    #     #      gradient signals, preventing selection collapse.
    #     #   3. SINGLE-ELEMENT WEIGHT LEARNING : Element weights
    #     #      for single-element groups (superstructure, bearings, wearing surface)
    #     #      ONLY receive gradients through the aggregate BHI. If tau hardens
    #     #      too much, those gradients vanish and those element weights freeze.
    #     #      tau_min ensures the aggregate-BHI pathway stays active throughout.
    #     "tau_min": 0.01,



    #     "epochs_per_batch": 100,
    #     "frames_per_minibatch": 200,
    #     "max_grad_norm": None, 
    #     "eval_freq": 10,
    #     "eval_episodes": 100,
    #     "eval_deterministic": True,
    # }




    train_config = {
        "total_frames": 5_000_000,
        "frames_per_batch": 50_000,

        "clip_epsilon": 0.1,
        "entropy_eps": 0.001,
        "critic_coef": 1.0,
        "GAE_gamma": 1.0,
        "GAE_lmbda": 0.95,
        "average_GAE": True,
        "reward_decay": None,

        "learning_rate": 1e-3,
        "scheduler_type": "cosine",
        "lr_min": 1e-5,

        "actor_l1_coef": reg_coef, # "actor_l2_coef": 1e-4, 
        "beta_anneal": 100**(1/100),
        "beta_update_freq": 1, 


        # tau (selection-temperature) annealing for per-node HI selection.
        # tau_anneal > beta_anneal so selection commits before routing hardens.
        # With ~100 batches: 100**(1/60) reaches tau_min ~0.01 around batch 60.
        "tau_anneal": 100**(1/60),
        "tau_update_freq": 1,
        # ----- tau annealing (per-node HI selection temperature) --
        # WHY TAU_MIN (don't let tau go to zero or decrease like beta):
        #   1. SOFTMAX NUMERICAL SAFETY: When tau → 0, the softmax denominator
        #      exp(logits/tau) overflows to infinity. tau_min=0.01 keeps
        #      softmax numerically stable during backprop.
        #   2. GRADIENT FLOW: Very small tau makes softmax nearly one-hot, so
        #      gradients through non-selected HIs vanish → weights stop learning.
        #      tau_min=0.01 keeps soft enough that all HIs receive meaningful
        #      gradient signals, preventing selection collapse.
        #   3. SINGLE-ELEMENT WEIGHT LEARNING : Element weights
        #      for single-element groups (superstructure, bearings, wearing surface)
        #      ONLY receive gradients through the aggregate BHI. If tau hardens
        #      too much, those gradients vanish and those element weights freeze.
        #      tau_min ensures the aggregate-BHI pathway stays active throughout.
        "tau_min": 0.01,

        "epochs_per_batch": 10,
        "frames_per_minibatch": 2500,
        "max_grad_norm": 0.5,

        "eval_freq": 2,
        "eval_episodes": 20,
        "eval_deterministic": True,
    }





    # create environment
    gym_env = BridgeBHIEnv(
        max_steps=max_steps,
        discount=gamma,
        include_step_count=include_step_count,
        reset_prob=reset_prob,
        reward_normalizer=reward_normalizer,
        transition_mode=STATE_TRANSITION_MODE,
        render_mode="ansi",
        seed=env_seed,
    )






    print(f"\nC0 = {gym_env.C0:.2f}")
    print(f"Discounted sum(Caluclate this for verification) = {sum(gamma**t for t in range(max_steps)):.2f}")


    env = GymWrapper(gym_env, categorical_action_encoding=True)

    # create actor and critic nets
    torch.manual_seed(torch_seed)


    initial_element_weights = [
        ELEMENT_WEIGHTS[int(element_no)] for element_no in ELEMENT_NUMBERS
    ]

    # Make the significance-factor mode explicit in the training log.
    if LEARNABLE_SIGNIFICANCE_FACTOR:
        print("\n[Element significance factors] LEARNABLE "
              "(LEARNABLE_SIGNIFICANCE_FACTOR = True): warm-started from "
              "ELEMENT_WEIGHTS and updated by PPO.")
    else:
        print("\n[Element significance factors] FIXED "
              "(LEARNABLE_SIGNIFICANCE_FACTOR = False): held at ELEMENT_WEIGHTS; "
              "no weights are learned during training.")



    actor_tree = SoftTreeBHI(
            input_dim=gym_env.state_size + int(gym_env.include_step_count),
            output_dim=gym_env.action_size,
            depth=actor_tree_depth,
            beta=tree_beta,
            num_elements=len(ELEMENT_NUMBERS),
            ncs=NCS,
            health_coefficients=HEALTH_COEFFICIENTS,
            initial_element_weights=initial_element_weights,
            element_to_group_idx=ELEMENT_TO_GROUP_IDX,
            include_step_count=include_step_count,
            tau_init=1.0,                                # start soft/uniform, I mean conisider all elements equally at the beginning of training. Then, gradually anneal tau to make the selection more deterministic.
            learnable_element_weights=LEARNABLE_SIGNIFICANCE_FACTOR,  # on -> learn significance factors; off -> fix them at ELEMENT_WEIGHTS
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
    normalized_rewards = np.array(train_log["reward"])
    normalized_eval_rewards = np.array(eval_log["eval_reward"])



    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.plot(train_log["batch"], normalized_rewards, label="training")
        ax.plot(eval_log["batch"], normalized_eval_rewards, label="evaluation")
        ax.set_xlabel("PPO batch(Policy update iteration)")
        ax.set_ylabel("Normalized reward")
        ax.set_title("Learning Curve(st)")
        ax.legend()


    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./actors", exist_ok=True)
    os.makedirs("./results", exist_ok=True)



    # save checkpoint (debug) and actor
    # RUN_MODE_TAG embeds STATE_TRANSITION_MODE and the significance-factor mode
    # (learnSF/fixedSF) in every saved filename.
    trainer.save_checkpoint(f"./checkpoints/checkpoint_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l1_coef']:.0e}_{max_steps:d}yr_{RUN_MODE_TAG}.pt")
    # trainer.load_checkpoint("./checkpoints/checkpoint_softtree.pt")
    trainer.save_actor(f"./actors/stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l1_coef']:.0e}_{max_steps:d}yr_{RUN_MODE_TAG}.pt")

    # save log
    pd.DataFrame(train_log).to_csv(
        f"./results/train_log_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l1_coef']:.0e}_{max_steps:d}yr_{RUN_MODE_TAG}.csv",
        index=False
    )
    pd.DataFrame(eval_log).to_csv(
        f"./results/eval_log_stBHI_d{actor_tree_depth:d}b{tree_beta:.0f}le{train_config['actor_l1_coef']:.0e}_{max_steps:d}yr_{RUN_MODE_TAG}.csv",
        index=False
    )