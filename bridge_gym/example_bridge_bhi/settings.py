import numpy as np
import scipy.stats as stats

from TP_and_Preprocessing.results.transition_matrices import TRANSITION_MATRICES


__all__ = [
    "NCS",
    "NA",
    "ELEMENT_NUMBERS",
    "ELEMENT_GROUPS",
    "ELEMENT_TO_GROUP",
    "ELEMENT_NAMES",
    "ELEMENT_WEIGHTS",
    "ELEMENT_QUANTITIES",
    "ACTION_NAMES",
    "ACTION_REPLACEMENT_MASK",
    "DO_NOTHING_TRANSITIONS",
    "REPLACEMENT_TRANSITION",
    "HEALTH_COEFFICIENTS",
    "CS_PFS",
    "FAILURE_COST",
]


# ---------------------------------------------------------------------
# Basic condition-state and action settings
# ---------------------------------------------------------------------
# Number of condition states
NCS = 4

# Number of bridge-level actions: A0 to A7
NA = 8

# Fixed California-style health coefficients for CS1 to CS4.
HEALTH_COEFFICIENTS = np.array([1.00, 0.66, 0.33, 0.00], dtype=float)

# ---------------------------------------------------------------------
# Element set used in the bridge-level BHI environment
# ---------------------------------------------------------------------

# Element numbers currently available in the bridge-level case.
ELEMENT_NUMBERS = np.array([12, 109, 205, 215, 234, 306, 310, 331, 510], dtype=int)


# Element names used for reporting, debugging, and interpretation.
ELEMENT_NAMES = {
    12: "RC Deck",
    109: "Girder Beam (PSC)",
    205: "Column (RC)",
    215: "Abutment (RC)",
    234: "Pier Cap (RC)",
    306: "Other Joint",
    310: "Elastomeric Bearings",
    331: "RC Bridge Railing",
    510: "Wearing Surface",
}

# Non-economic element weights computed as:
# element weight = VF(Table 10) * SSF(Table 3)
ELEMENT_WEIGHTS = {
    12: 6.0,
    109: 4.0,
    205: 8.0,
    215: 4.0,
    234: 8.0,
    306: 1.5,
    310: 4.5,
    331: 2.0,
    510: 1.5,
}

# Total quantity of each element in the selected bridge case. Bridge ID: 01577A016 04612
# C0 = sum_i W_i * Q_i
# C(a) = sum_j W_j * Q_j and j is the set of elements that are fully replaced under action a
ELEMENT_QUANTITIES = {
    12: 8462,
    109: 1198,
    205: 2,
    215: 78,
    234: 78,
    306: 76,
    310: 30,
    331: 541,
    510: 8462,
}

# These groups are used to map bridge-level maintenance actions to elements.
ELEMENT_GROUPS = [
    "deck",
    "wearing_surface_or_protective_coating",
    "superstructure",
    "bearings",
    "substructure",
]

# Mapping from element number to the engineering group because bridge-level actions are defined in terms of groups but the transition matrices are defined at the element level.
ELEMENT_TO_GROUP = {
    12: "deck",
    109: "superstructure",
    205: "substructure",
    215: "substructure",
    234: "substructure",
    306: "deck",
    310: "bearings",
    331: "deck",
    510: "wearing_surface_or_protective_coating",
}

# ---------------------------------------------------------------------
# Transition matrices
# ---------------------------------------------------------------------

# Do-nothing transition matrix for each element.
DO_NOTHING_TRANSITIONS = {
    element_no: TRANSITION_MATRICES[element_no]
    for element_no in ELEMENT_NUMBERS
}

# Replacement transition matrix.
REPLACEMENT_TRANSITION = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)

# ---------------------------------------------------------------------
# Bridge-level action definitions
# ---------------------------------------------------------------------
# # D: deck
# WS_PC: wearing surface or protective coating
# Sup: superstructure
# B: bearings
# Sub: substructure
ACTION_NAMES = {
    0: "Do nothing",
    1: "Replace WS_PC",
    2: "Replace B",
    3: "Replace D + WS_PC",
    4: "Replace WS_PC + B",
    5: "Replace D + WS_PC + B",
    6: "Replace D + WS_PC + Sup + B",
    7: "Full bridge replacement",
}

# ACTION_REPLACEMENT_MASK defines which engineering groups are fully replaced
# under each bridge-level maintenance action.
# If a group appears in the selected action mask, then ALL elements belonging to that group are assigned the REPLACEMENT_TRANSITION matrix.
ACTION_REPLACEMENT_MASK = {
    0: set(),
    1: {"wearing_surface_or_protective_coating"},
    2: {"bearings"},
    3: {"deck", "wearing_surface_or_protective_coating"},
    4: {"wearing_surface_or_protective_coating", "bearings"},
    5: {"deck", "wearing_surface_or_protective_coating", "bearings"},
    6: {
        "deck",
        "wearing_surface_or_protective_coating",
        "superstructure",
        "bearings",
    },
    7: {
        "deck",
        "wearing_surface_or_protective_coating",
        "superstructure",
        "bearings",
        "substructure",
    },
}

# ---------------------------------------------------------------------
# Optional reliability / failure-risk settings
# ---------------------------------------------------------------------

# These values can remain for now because your reward function is not finalized.
# Later, if reward is based only on BHI and intervention cost, these may be removed.
CS_PFS = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])

cost_base = 10
FAILURE_COST = cost_base**5




#original code
# import numpy as np
# import scipy.stats as stats


# __all__ = [
#     "NCS", "NA",
#     "CS_PFS", "FAILURE_COST",
#     "ACTION_MODEL", "UNIT_COSTS",
# ]


# # RL parameters
# NCS, NA = 4, 2

# # Failure probabilities
# CS_PFS = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])  
# cost_base = 10
# FAILURE_COST = cost_base**5

# # Action 0 — Do nothing
# action0 = np.array([
#     [0.9381, 0.0619, 0, 0],
#     [0, 0.9356, 0.0644, 0],
#     [0, 0, 0.8888, 0.1112],
#     [0, 0, 0, 1]
# ])
# unit_price0 = np.zeros(NCS)

# # Action 4 — Replacement
# action4 = np.array([
#     [1.0, 0, 0, 0],
#     [1.0, 0, 0, 0],
#     [1.0, 0, 0, 0],
#     [1.0, 0, 0, 0]
# ])
# unit_price4 = np.array([2*cost_base**3]*NCS)  # drop CS5

# # Pack into final arrays
# ACTION_MODEL = np.array([action0, action4])
# UNIT_COSTS = np.array([unit_price0, unit_price4])
