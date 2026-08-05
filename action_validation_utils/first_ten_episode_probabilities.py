from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bridge_gym.example_bridge_bhi.settings import ACTION_NAMES


OUTPUT_DIR = Path("./results/action_validation")


def save_first_ten_episode_probabilities(actor, eval_log, file_stem):

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

        observation_tensor = torch.as_tensor(observations, dtype=dtype, device=device)

        with torch.no_grad():
            logits = actor_core(observation_tensor)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        data = {
            "episode": episode_id + 1,
            "step": np.arange(1, len(chosen_actions) + 1),
        }

        for action_id in sorted(ACTION_NAMES):
            data[f"prob_A{action_id}"] = probabilities[:, action_id]

        data["chosen_action_id"] = chosen_actions
        data["chosen_action_name"] = [
            ACTION_NAMES[action_id]
            for action_id in chosen_actions]

        all_episode_data.append(pd.DataFrame(data))

    probability_df = pd.concat(all_episode_data, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"action_probabilities_{file_stem}_first10episodes.csv"
    )

    probability_df.to_csv(output_path, index=False)

    print(f"\nSaved action probabilities to: {output_path}")

    return probability_df