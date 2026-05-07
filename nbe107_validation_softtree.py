#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from torchrl.envs import GymWrapper

from bridge_gym.example_nbe107.rl_env import SingleElement
from bridge_gym.example_nbe107.settings import CS_PFS
from softtree_ppo.training import SofttreePPOTrainer

from nbe107_training_nn import max_steps, gamma
from nbe107_training_nn import include_step_count
from nbe107_training_nn import alpha_vector
from nbe107_training_softtree import actor_tree_depth, tree_beta

from eval_stats import mean_and_ci

# %%

if __name__ == '__main__':
    env_seed = 1034
    num_episodes = 1000 # David's assumption 1000
    cost_kwargs = {"normalizer": 1}

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
    
    actor = SofttreePPOTrainer.load_actor(
        f"./actors/softtree_d{actor_tree_depth:d}b{tree_beta:.1f}_{max_steps:d}yr.pt",
        env.action_spec,
    )

    eval_log = SofttreePPOTrainer.evaluate(
        actor,
        env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=True,
    )

    # plot testing results
    init_states_raw = np.array(eval_log["init_state"])
    init_states = init_states_raw[:, :-1] if include_step_count else init_states_raw
    init_pf = init_states @ CS_PFS
    init_beta = -stats.norm.ppf(init_pf)
    eval_costs = -np.array(eval_log["eval_reward"])
    norm_eval_costs = eval_costs / max_steps
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        if gamma == 1:
            sns.scatterplot(x=init_beta, y=norm_eval_costs, ax=ax)
            ax.set_xlabel("β (Reliability Index)")
            ax.set_ylabel("LCC / max_steps")
        else:
            sns.scatterplot(x=init_beta, y=eval_costs, ax=ax)
            ax.set_xlabel("β (Reliability Index)")
            ax.set_ylabel("LCC")
        # ax.set_ylim(0, 1e6)


    reward_stats = mean_and_ci(eval_log["eval_reward"])

    print(f"Validation (episode return for {reward_stats['n']} episodes): "
        f"mean={reward_stats['mean']:.4f}, "
        f"95% CI=[{reward_stats['ci_low']:.4f}, {reward_stats['ci_high']:.4f}], "
        f"SD={reward_stats['sd']:.4f}")
# %%
