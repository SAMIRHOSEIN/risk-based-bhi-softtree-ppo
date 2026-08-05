from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bridge_gym.example_bridge_bhi.settings import ACTION_NAMES, GROUP_ORDER


OUTPUT_DIR = Path("./results/action_validation")


def save_first_ten_episode_hi_and_probabilities(
    actor,
    eval_log,
    file_stem,
    hi_calculator,
):
    """Save HIs, action probabilities, and chosen actions for up to 10 episodes."""

    number_to_save = min(10, len(eval_log["actions"]))
    all_episode_data = []

    parameter = next(actor.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32

    actor_core = actor.module[0].module
    actor.eval()

    for episode_id in range(number_to_save):
        observations = eval_log["observations"][episode_id]
        chosen_actions = eval_log["actions"][episode_id].astype(int)

        observation_tensor = torch.as_tensor(
            observations,
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            hi_values = hi_calculator(observation_tensor).cpu().numpy()
            logits = actor_core(observation_tensor)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        data = {
            "episode": episode_id + 1,
            "step": np.arange(1, len(chosen_actions) + 1),
        }

        hi_names = [*GROUP_ORDER, "BHI_aggregate"]
        for hi_id, hi_name in enumerate(hi_names):
            data[f"HI_{hi_name}"] = hi_values[:, hi_id]

        for action_id in sorted(ACTION_NAMES):
            data[f"prob_A{action_id}"] = probabilities[:, action_id]

        data["chosen_action_id"] = chosen_actions
        data["chosen_action_name"] = [
            ACTION_NAMES[action_id]
            for action_id in chosen_actions
        ]

        all_episode_data.append(pd.DataFrame(data))

    result_df = pd.concat(all_episode_data, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"hi_action_probabilities_{file_stem}_first10episodes.csv"
    )

    result_df.to_csv(output_path, index=False)

    print(f"\nSaved HI and action probabilities to: {output_path}")

    return result_df