#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bridge_gym.example_bridge_bhi.rl_env import BridgeBHIEnv


from bridge_gym.example_bridge_bhi.settings import (
    ACTION_NAMES,
    gamma,
    include_step_count,
    reset_prob,
)

max_years = 200


#%%
def simulate_policy_for_bhi(policy_name, max_years=100, bhi_threshold=50):
    env = BridgeBHIEnv(
        max_steps=max_years,
        discount=gamma,                     # not important here because we only verify BHI
        include_step_count=include_step_count,
        reset_prob=reset_prob               # starts from [1, 0, 0, 0] for all elements
    )

    obs, info = env.reset()

    years = []
    bhi_values = []
    actions = []

    for year in range(max_years + 1):
        current_bhi = env._compute_bhi(env._state)

        years.append(year)
        bhi_values.append(current_bhi)

        if year == max_years:
            actions.append("End")
            break

        if policy_name == "do_nothing":
            action = 0

        elif policy_name == f"replace_when_bhi_below_{bhi_threshold * 100:.0f}_percent":
            if current_bhi < bhi_threshold:
                action = 7   # Full bridge replacement
            else:
                action = 0   # Do nothing

        else:
            raise ValueError("Unknown policy name.")

        actions.append(ACTION_NAMES[action])

        obs, reward, terminated, truncated, info = env.step(action)

    df = pd.DataFrame({
        "year": years,
        "BHI": bhi_values,
        "BHI_percent": np.array(bhi_values) * 100,
        "action": actions
    })

    return df


if __name__ == "__main__":

    bhi_threshold = 0.7
    max_years = 100

    os.makedirs("verification_figures", exist_ok=True)

    df_do_nothing = simulate_policy_for_bhi("do_nothing", max_years=max_years, bhi_threshold=bhi_threshold)

    df_replacement = simulate_policy_for_bhi(
        f"replace_when_bhi_below_{bhi_threshold * 100:.0f}_percent",
        max_years=max_years,
        bhi_threshold=bhi_threshold
    )


    plt.figure(figsize=(7, 4))

    plt.plot(
        df_do_nothing["year"],
        df_do_nothing["BHI_percent"],
        marker="o",
        label="Do nothing"
    )

    plt.plot(
        df_replacement["year"],
        df_replacement["BHI_percent"],
        marker="s",
        label=f"Replacement if BHI < {bhi_threshold * 100:.0f}%"
    )

    plt.axhline(
        y=bhi_threshold * 100,
        linestyle="--",
        label=f"Replacement threshold = {bhi_threshold * 100:.0f}%"
    )

    plt.xlabel("Year")
    plt.ylabel("Bridge Health Index (%)")
    plt.title(f"Verification of BHI Evolution Over {max_years} Years")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%

def simulate_policy_for_year(policy_name, max_years=max_years, every_year=10):
    env = BridgeBHIEnv(
        max_steps=max_years,
        discount=gamma,                     # not important here because we only verify BHI
        include_step_count=include_step_count,
        reset_prob=reset_prob               # starts from [1, 0, 0, 0] for all elements
    )

    obs, info = env.reset()

    years = []
    bhi_values = []
    actions = []


    for year in range(max_years + 1):
        current_bhi = env._compute_bhi(env._state)

        years.append(year)
        bhi_values.append(current_bhi)

        if year == max_years:
            actions.append("End")
            break

        if policy_name == "do_nothing":
            action = 0

        elif policy_name == f"replace_every_{year_threshold}_years":
            if year % year_threshold == 0:
                action = 7   # Full bridge replacement
            else:
                action = 0   # Do nothing

        else:
            raise ValueError("Unknown policy name.")

        actions.append(ACTION_NAMES[action])

        obs, reward, terminated, truncated, info = env.step(action)

    df = pd.DataFrame({
        "year": years,
        "BHI": bhi_values,
        "BHI_percent": np.array(bhi_values) * 100,
        "action": actions
    })

    return df


if __name__ == "__main__":

    year_threshold = 10

    os.makedirs("verification_figures", exist_ok=True)

    df_do_nothing = simulate_policy_for_year("do_nothing", max_years=max_years, every_year=year_threshold)

    df_replacement = simulate_policy_for_year(
        f"replace_every_{year_threshold}_years",
        max_years=max_years,
        every_year=year_threshold
    )

    plt.figure(figsize=(7, 4))

    plt.plot(
        df_do_nothing["year"],
        df_do_nothing["BHI_percent"],
        marker="o",
        label="Do nothing"
    )

    plt.plot(
        df_replacement["year"],
        df_replacement["BHI_percent"],
        marker="s",
        label=f"Replacement every {year_threshold} years"
    )

    plt.axhline(
        y=bhi_threshold * 100,
        linestyle="--",
        label=f"Replacement every {year_threshold} years"
    )

    plt.xlabel("Year")
    plt.ylabel("Bridge Health Index (%)")
    plt.title(f"Verification of BHI Evolution Over {max_years} Years")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
