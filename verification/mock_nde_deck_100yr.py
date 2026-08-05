#%% 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BRIDGE_ID = "01577A016 04612"
ELEMENT_NO = 12
YEARS = 100
SEED = 1234
TRANSITION_MATRIX = np.array(
    [[0.98866331, 0.01133669, 0.0, 0.0],
     [0.0, 0.97793375, 0.02206625, 0.0],
     [0.0, 0.0, 0.89663155, 0.10336845],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=float)
BETA_VARIANCE = 1.0e-3
ELEMENT_QUANTITY = 8462
state = np.array([1., 0., 0., 0.])

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "results" / "mock_nde_records"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
records = []
means = TRANSITION_MATRIX[np.arange(3), np.arange(1, 4)] # Pij
kappa = means * (1 - means) / BETA_VARIANCE - 1
alpha, beta = means * kappa, (1 - means) * kappa

# for year in range(1, YEARS + 1):
#     q12, q23, q34 = rng.beta(alpha, beta)
#     transition = np.array([[1-q12, q12, 0, 0], [0, 1-q23, q23, 0],
#                            [0, 0, 1-q34, q34], [0, 0, 0, 1.]])
#     state = transition.T @ state
#     raw = state * ELEMENT_QUANTITY
#     counts = np.floor(raw).astype(int)

#     missing = ELEMENT_QUANTITY - counts.sum()    
#     largest_fraction = np.argmax(raw - counts)
#     counts[largest_fraction] += missing # fix the rounding to largest one

#     records.append([year, ELEMENT_QUANTITY, *counts])

for year in range(1, YEARS + 1):
    # Convert the current probability state to integer quantities.
    # Q4 receives the current-count rounding remainder.
    q1 = int(np.floor(state[0] * ELEMENT_QUANTITY))
    q2 = int(np.floor(state[1] * ELEMENT_QUANTITY))
    q3 = int(np.floor(state[2] * ELEMENT_QUANTITY))
    q4 = ELEMENT_QUANTITY - q1 - q2 - q3

    # Sample the annual Beta transition probabilities.
    p12, p23, p34 = rng.beta(alpha, beta)

    transition = np.array(
        [
            [1 - p12, p12, 0, 0],
            [0, 1 - p23, p23, 0],
            [0, 0, 1 - p34, p34],
            [0, 0, 0, 1.0],
        ]
    )

    # Continuous state before integer rounding.
    new_state = transition.T @ state

    # Convert annual probability changes to integer quantity changes.
    dq1 = int(min((new_state[0] - state[0]) * ELEMENT_QUANTITY,0.0))

    dq4 = int(
        max((new_state[3] - state[3]) * ELEMENT_QUANTITY,0.0))

    dq2 = int((new_state[1] - state[1]) * ELEMENT_QUANTITY)

    # Q3 absorbs the annual-change rounding imbalance.
    dq3 = -dq1 - dq2 - dq4

    counts = np.array(
        [
            q1 + dq1,
            q2 + dq2,
            q3 + dq3,
            q4 + dq4,
        ],
        dtype=int,
    )

    assert counts.sum() == ELEMENT_QUANTITY
    assert (counts >= 0).all()

    records.append([year, ELEMENT_QUANTITY, *counts])

    state = counts / ELEMENT_QUANTITY



columns = [ "year","total_quantity", "Q1", "Q2", "Q3", "Q4"]
df = pd.DataFrame(records, columns=columns)
csv_path = OUT / f"mock_nde_bridge_{BRIDGE_ID}_element{ELEMENT_NO}_{YEARS}yr.csv"
plot_path = OUT / f"mock_nde_bridge_{BRIDGE_ID}_element{ELEMENT_NO}_{YEARS}yr.png"
df.to_csv(csv_path, index=False)
plt.figure(figsize=(10, 6))
plt.stackplot(df["year"], *(df[f"Q{i}"] for i in range(1, 5)), labels=[f"Q{i}" for i in range(1, 5)])
plt.xlabel("Year"); plt.ylabel("Element quantity")
plt.title(f"Mock NDE Record: RC Deck, Element {ELEMENT_NO}")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5)); plt.tight_layout()
plt.savefig(plot_path, dpi=300); plt.close()
print(f"Generated mock NDE record for {BRIDGE_ID}, Element {ELEMENT_NO}, {YEARS} years.")
assert (df[["Q1", "Q2", "Q3", "Q4"]].sum(axis=1) == ELEMENT_QUANTITY).all()
