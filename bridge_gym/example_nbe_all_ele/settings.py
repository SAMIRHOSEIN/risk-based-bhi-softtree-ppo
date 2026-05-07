"""
"""
import numpy as np
import scipy.stats as stats


__all__ = [
    "NCS", "NA",
    "CS_PFS", "FAILURE_COST",
    "ACTION_MODEL", "UNIT_COSTS",
]


# RL parameters
NCS, NA = 4, 2

# Failure probabilities
CS_PFS = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])  
cost_base = 10
FAILURE_COST = cost_base**5

# Action 0 — Do nothing
action0 = np.array([
    [0.9381, 0.0619, 0, 0],
    [0, 0.9356, 0.0644, 0],
    [0, 0, 0.8888, 0.1112],
    [0, 0, 0, 1]
])
unit_price0 = np.zeros(NCS)

# Action 4 — Replacement
action4 = np.array([
    [1.0, 0, 0, 0],
    [1.0, 0, 0, 0],
    [1.0, 0, 0, 0],
    [1.0, 0, 0, 0]
])
unit_price4 = np.array([2*cost_base**3]*NCS)  # drop CS5

# Pack into final arrays
ACTION_MODEL = np.array([action0, action4])
UNIT_COSTS = np.array([unit_price0, unit_price4])


