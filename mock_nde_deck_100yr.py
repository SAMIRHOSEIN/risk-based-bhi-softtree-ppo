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

OUT = Path("mock_nde_records")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)
records = []
means = TRANSITION_MATRIX[np.arange(3), np.arange(1, 4)] # Pij
kappa = means * (1 - means) / BETA_VARIANCE - 1
alpha, beta = means * kappa, (1 - means) * kappa

for year in range(1, YEARS + 1):
    q12, q23, q34 = rng.beta(alpha, beta)
    transition = np.array([[1-q12, q12, 0, 0], [0, 1-q23, q23, 0],
                           [0, 0, 1-q34, q34], [0, 0, 0, 1.]])
    state = transition.T @ state
    raw = state * ELEMENT_QUANTITY
    counts = np.floor(raw).astype(int)

    missing = ELEMENT_QUANTITY - counts.sum()    
    largest_fraction = np.argmax(raw - counts)
    counts[largest_fraction] += missing # fix the rounding to largest one

    records.append([year, ELEMENT_QUANTITY, *counts])
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
