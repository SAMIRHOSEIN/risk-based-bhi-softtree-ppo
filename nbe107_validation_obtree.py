#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from torchrl.envs import GymWrapper

from bridge_gym.example_nbe107.rl_env import SingleElement
from bridge_gym.example_nbe107.settings import CS_PFS
from softtree_ppo.training import SofttreePPOTrainer

from nbe107_training_nn import max_steps, gamma
from nbe107_training_nn import include_step_count
from nbe107_training_nn import alpha_vector

# %%

if __name__ == '__main__':
    env_seed = 1034
    obs_episodes = 10
    pruning_threshold = 1e-5
    num_episodes = 1000
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
    
    # get oblique tree actor
    STC_actor = SofttreePPOTrainer.load_actor(
        "./actors/softtree_d10b1.0_20yr.pt",
        env.action_spec,
    )
    obs = env.rollout(
        max_steps=gym_env.max_steps*obs_episodes,
        policy=STC_actor,
        break_when_any_done=False,
        auto_reset=True,
        auto_cast_to_device=True,
    )
    OBT_actor = SofttreePPOTrainer.convert_to_obtree_actor(
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
    init_states = np.array(eval_log["init_state"])[:, :-1]
    init_pf = init_states @ CS_PFS
    init_beta = -stats.norm.ppf(init_pf)
    eval_costs = -np.array(eval_log["eval_reward"])
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        sns.scatterplot(x=init_beta, y=eval_costs, ax=ax)
        # ax.set_ylim(0, 1e6)
# %%
