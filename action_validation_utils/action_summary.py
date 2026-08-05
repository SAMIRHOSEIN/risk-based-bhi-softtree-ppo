from pathlib import Path

import numpy as np
import pandas as pd

from bridge_gym.example_bridge_bhi.settings import ACTION_NAMES


OUTPUT_DIR = Path("./results/action_validation")


def save_action_summary(eval_log, file_stem):
    """Print and save action counts and percentages for all validation steps."""
    actions = np.concatenate(eval_log["actions"]).astype(int)
    total_actions = len(actions)

    rows = []
    for action_id in sorted(ACTION_NAMES):
        count = int(np.sum(actions == action_id))
        percentage = 100.0 * count / total_actions

        rows.append({
            "action_id": action_id,
            "action_name": ACTION_NAMES[action_id],
            "chosen_count": count,
            "chosen_percentage": percentage,
        })

    summary_df = pd.DataFrame(rows)

    print("\nAction-selection summary:")
    print(summary_df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"action_summary_{file_stem}.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"\nSaved action summary to: {output_path}")
    return summary_df
