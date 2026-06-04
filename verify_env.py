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
    ELEMENT_NUMBERS,
    ELEMENT_NAMES,
)


# Inputs for all files
max_years = 100
bhi_threshold_for_BHI_based_policy = 0.6
every_year_for_time_based_policy =10 


#%%
# BHI policy simulation and plotting
def simulate_bhi_policy_all_elements(policy_name, max_year=100, bhi_threshold=0.6):
    env = BridgeBHIEnv(
        max_steps=max_year,
        discount=gamma,
        include_step_count=include_step_count,
        reset_prob=reset_prob
    )

    obs, info = env.reset()

    records = []

    for year in range(max_year + 1):

        current_bhi = env._compute_bhi(env._state)

        if policy_name == "do_nothing":
            action = 0

        elif policy_name == "bhi_based_replacement":
            if current_bhi < bhi_threshold:
                action = 7
            else:
                action = 0

        else:
            raise ValueError("Unknown policy name.")

        action_name = ACTION_NAMES[action]

        for i, element_no in enumerate(ELEMENT_NUMBERS):
            records.append({
                "year": year,
                "element_no": int(element_no),
                "element_name": ELEMENT_NAMES[int(element_no)],
                "CS1": env._state[i, 0],
                "CS2": env._state[i, 1],
                "CS3": env._state[i, 2],
                "CS4": env._state[i, 3],
                "BHI_percent": current_bhi * 100,
                "action": action_name
            })

        if year == max_year:
            break

        obs, reward, terminated, truncated, info = env.step(action)

    df = pd.DataFrame(records)
    return df


def plot_bhi_policy_only(df_do_nothing, df_replacement, bhi_threshold, output_file):
    df_bhi_dn = (
        df_do_nothing[["year", "BHI_percent"]].sort_values("year")
    )

    df_bhi_replac = (
        df_replacement[["year", "BHI_percent"]].sort_values("year")
    )

    plt.figure(figsize=(7, 4))

    plt.plot(
        df_bhi_dn["year"],
        df_bhi_dn["BHI_percent"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Do nothing"
    )

    plt.plot(
        df_bhi_replac["year"],
        df_bhi_replac["BHI_percent"],
        marker="s",
        markersize=3,
        linewidth=1.5,
        label=f"Replacement if BHI < {bhi_threshold * 100:.0f}%"
    )

    plt.axhline(
        y=bhi_threshold * 100,
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold = {bhi_threshold * 100:.0f}%"
    )

    plt.xlabel("Time (years)")
    plt.ylabel("Bridge Health Index (%)")
    plt.title("BHI-Based Replacement Policy")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()



def plot_condition_states_for_each_element(df, policy_label, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for element_no in df["element_no"].unique():

        df_e = df[df["element_no"] == element_no].copy()
        element_name = df_e["element_name"].iloc[0]

        plt.figure(figsize=(7, 4))

        plt.bar(df_e["year"], df_e["CS1"], label="CS1")
        plt.bar(df_e["year"], df_e["CS2"], bottom=df_e["CS1"], label="CS2")
        plt.bar(df_e["year"], df_e["CS3"], bottom=df_e["CS1"] + df_e["CS2"], label="CS3")
        plt.bar(df_e["year"], df_e["CS4"], bottom=df_e["CS1"] + df_e["CS2"] + df_e["CS3"], label="CS4")

        plt.xlabel("Time (years)")
        plt.ylabel("Condition-state distribution")
        plt.title(f"{policy_label}\nElement {element_no}: {element_name}")
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filename = (
            f"{output_folder}/element_{element_no}_"
            f"{policy_label.replace(' ', '_').replace('-', '_')}.png"
        )

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved element-level CS figures in: {output_folder}")



if __name__ == "__main__":

    bhi_threshold = bhi_threshold_for_BHI_based_policy

    os.makedirs("verification_figures", exist_ok=True)

    df_do_nothing = simulate_bhi_policy_all_elements(
        policy_name="do_nothing",
        max_year=max_years,
        bhi_threshold=bhi_threshold
    )

    df_replacement = simulate_bhi_policy_all_elements(
        policy_name="bhi_based_replacement",
        max_year=max_years,
        bhi_threshold=bhi_threshold
    )


    #  BHI
    plot_bhi_policy_only(
        df_do_nothing=df_do_nothing,
        df_replacement=  df_replacement,
        bhi_threshold=bhi_threshold,
        output_file="verification_figures/bhi_policy_only.png"
    )

    # One CS figure per element
    plot_condition_states_for_each_element(
        df= df_replacement,
        policy_label="BHI-based replacement policy",
        output_folder="verification_figures/bhi_policy_condition_states"
    )

#%%

def simulate_time_policy_all_elements(policy_name, max_year=max_years, every_year=10):
    env = BridgeBHIEnv(
        max_steps=max_year,
        discount=gamma,
        include_step_count=include_step_count,
        reset_prob=reset_prob
    )

    obs, info = env.reset()

    records = []

    for year in range(max_year + 1):

        current_bhi = env._compute_bhi(env._state)

        if policy_name == "do_nothing":
            action = 0

        elif policy_name == f"time_based_for_every_{every_year}_years":
            if year > 0 and year % every_year == 0:
                action = 7
            else:
                action = 0

        else:
            raise ValueError("Unknown policy name.")

        action_name = ACTION_NAMES[action]

        for i, element_no in enumerate(ELEMENT_NUMBERS):
            records.append({
                "year": year,
                "element_no": int(element_no),
                "element_name": ELEMENT_NAMES[int(element_no)],
                "CS1": env._state[i, 0],
                "CS2": env._state[i, 1],
                "CS3": env._state[i, 2],
                "CS4": env._state[i, 3],
                "BHI_percent": current_bhi * 100,
                "action": action_name
            })

        if year == max_year:
            break

        obs, reward, terminated, truncated, info = env.step(action)

    return pd.DataFrame(records)




def plot_time_policy_only(df_do_nothing, df_policy, every_year, output_file):
    df_bhi_dn = (
        df_do_nothing[["year", "BHI_percent"]].sort_values("year")
    )

    df_bhi_policy = (
        df_policy[["year", "BHI_percent"]].sort_values("year")
    )

    plt.figure(figsize=(7, 4))

    plt.plot(
        df_bhi_dn["year"],
        df_bhi_dn["BHI_percent"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Do nothing"
    )

    plt.plot(
        df_bhi_policy["year"],
        df_bhi_policy["BHI_percent"],
        marker="s",
        markersize=3,
        linewidth=1.5,
        label=f"Replacement every {every_year} years"
    )

    for yr in range(every_year, int(df_bhi_policy["year"].max()) + 1, every_year):
        plt.axvline(
            x=yr,
            linestyle="--",
            linewidth=0.8,
            alpha=0.4
        )

    plt.xlabel("Time (years)")
    plt.ylabel("Bridge Health Index (%)")
    plt.title(f"Time-Based Replacement Policy")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    every_year = every_year_for_time_based_policy

    os.makedirs("verification_figures", exist_ok=True)

    df_do_nothing_time = simulate_time_policy_all_elements(
        policy_name="do_nothing",
        max_year=max_years,
        every_year=every_year
    )

    df_time_policy = simulate_time_policy_all_elements(
        policy_name=f"time_based_for_every_{every_year}_years",
        max_year=max_years,
        every_year=every_year
    )


    plot_time_policy_only(
        df_do_nothing=df_do_nothing_time,
        df_policy=df_time_policy,
        every_year=every_year,
        output_file="verification_figures/time_policy_bhi_only.png"
    )

    plot_condition_states_for_each_element(
        df=df_time_policy,
        policy_label=f"Time-based replacement every {every_year} years",
        output_folder="verification_figures/time_policy_condition_states"
    )
#%%








