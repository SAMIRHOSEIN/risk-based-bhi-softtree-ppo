import numpy as np

from TP_and_Preprocessing.results.transition_matrices import TRANSITION_MATRICES

__all__ = [
    "NCS",
    "NA",
    "ELEMENT_NUMBERS",
    "GROUP_ORDER",
    "GROUP_TO_IDX",
    "ELEMENT_TO_GROUP_IDX",
    "ELEMENT_TO_GROUP",
    "ELEMENT_NAMES",
    "ELEMENT_WEIGHTS",
    "ELEMENT_QUANTITIES",
    "STATE_TRANSITION_MODE",
    "ELEMENT_UNIT_COSTS",
    "ACTION_NAMES",
    "ACTION_REPLACEMENT_MASK",
    "DO_NOTHING_TRANSITIONS",
    "REPLACEMENT_TRANSITION",
    "HEALTH_COEFFICIENTS",
    "CS_PFS",
    "FAILURE_COST",
]

# ----------------------------------------------------------
# RL parameters inputs
# ----------------------------------------------------------
# Input parameters for the BHI environment AND RL training AND validation. That's why I put them here. 
# env parameters(in BHI-softtree version, we don't import env parameters from nbe107_training_nn.py because we don't have nbe107_training_nn.py in our directory)
max_steps, gamma = 200, 1/1.03 #200, 1/1.03
include_step_count = False


# ---------------------------------------------------------------------
# Initial-state distribution
# ---------------------------------------------------------------------
# None means that all elements start in pristine condition:
# [1.0, 0.0, 0.0, 0.0]
reset_prob = None

# For a non-pristine initial state without wearing-surface element
# reset_prob = np.array([
#     [1.0, 0.0, 0.0, 0.0],  # EL12
#     [1.0, 0.0, 0.0, 0.0],  # EL109
#     [1.0, 0.0, 0.0, 0.0],  # EL205
#     [1.0, 0.0, 0.0, 0.0],  # EL215
#     [1.0, 0.0, 0.0, 0.0],  # EL234
#     [1.0, 0.0, 0.0, 0.0],  # EL306
#     [0.0, 0.0, 0.2, 0.8],  # EL310
#     [1.0, 0.0, 0.0, 0.0],  # EL331
# ], dtype=np.float32)






# ---------------------------------------------------------------------
# State transition mode
# ---------------------------------------------------------------------
# "deterministic" uses the expected Markov-chain transition.
# "stochastic" samples condition-state counts.
# STATE_TRANSITION_MODE = "deterministic"
STATE_TRANSITION_MODE = "stochastic"







# ---------------------------------------------------------------------
# Basic condition-state and action settings
# ---------------------------------------------------------------------
# Number of condition states and Number of bridge-level actions: A0 to A7
# NCS, NA = 4, 8 # with all elements including the wearing-surface actions
NCS, NA = 4, 6 # removing the wearing-surface actions (wearging sirface and protective coating doesn't afffect structural safety so I removed it)


# Fixed California-style health coefficients for CS1 to CS4.
HEALTH_COEFFICIENTS = np.array([1.00, 0.66, 0.33, 0.00], dtype=float)

# ---------------------------------------------------------------------
# Element set used in the bridge-level BHI environment
# ---------------------------------------------------------------------
# Element numbers currently available in the bridge-level case.
# ELEMENT_NUMBERS = np.array([12, 109, 205, 215, 234, 306, 310, 331, 510], dtype=int)
ELEMENT_NUMBERS = np.array([12, 109, 205, 215, 234, 306, 310, 331], dtype=int) # removing the wearing-surface element 510 because it doesn't afffect structural safety 


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
}

# # Non-economic element weights computed as:
# # element weight = VF(Table 10) * SSF(Table 3)
# ELEMENT_WEIGHTS = {
#     12: 6.0,
#     109: 4.0,
#     205: 8.0,
#     215: 4.0,
#     234: 8.0,
#     306: 1.5,
#     310: 4.5,
#     331: 2.0,
#     510: 1.5,
# } 


# Element importance weights based only on structural significance.
# Material vulnerability factor is excluded because deterioration is already
# represented through element-level transition probabilities.
ELEMENT_WEIGHTS = {
    12: 3.0,
    109: 4.0,
    205: 4.0,
    215: 2.0,
    234: 4.0,
    306: 1.0,
    310: 3.0,
    331: 1.0,
}



# Unit replacement cost for each element.
# Used only for C0 and C(a), not for BHI.
ELEMENT_UNIT_COSTS = {
    12: 30.0,
    109: 250.0,
    205: 15000.0,
    215: 800.0,
    234: 800.0,
    306: 100.0,
    310: 1500.0,
    331: 200.0,
}



# Total quantity of each element for Bridge ID: 01577A016 04612
# First version(not implemented): 
                # To calcualte the C0 and C(a) values for the reward function.
                # C0 = sum_i W_i * Q_i and i is the set of all elements in the bridge
                # C(a) = sum_j W_j * Q_j and j is the set of elements that are fully replaced under action a
# Second version: 
                # C0 and C(a) are calculated from quantity multiplied by unit replacement cost.
                # C0 = sum_i Q_i * UC_i
                # C(a) = sum_j Q_j * UC_j
ELEMENT_QUANTITIES = {
    12: 8462,
    109: 1198,
    205: 2,
    215: 78,
    234: 78,
    306: 76,
    310: 30,
    331: 541,
}






# Mapping from element number to the engineering group because bridge-level actions are defined in terms of groups but the transition matrices are defined at the element level.
# Groups: deck, superstructure, bearings, substructure, wearing_surface_or_protective_coating
ELEMENT_TO_GROUP = {
    12: "deck",
    109: "superstructure",
    205: "substructure",
    215: "substructure",
    234: "substructure",
    306: "deck",
    310: "bearings",
    331: "deck",
}





# ---------------------------------------------------------------------
# Group ordering and per-element group index (needed by SoftTreeGHI)
# ---------------------------------------------------------------------
# The per-node health-index selector must know which engineering group each
# element belongs to, expressed as an INTEGER INDEX, not a string. 
#
# Index k -> candidate group-level health index GHI_k:
#   0 -> deck
#   1 -> superstructure
#   2 -> bearings
#   3 -> substructure
#   4 -> wearing_surface_or_protective_coating
#
# A sixth candidate (k = 5 -> aggregate BHI over ALL elements) is added inside
# the actor itself, so it is intentionally NOT listed here.
GROUP_ORDER = [
    "deck",
    "superstructure",
    "bearings",
    "substructure",
]

# Reverse lookup: group name -> canonical integer index (0..4).
GROUP_TO_IDX = {group_name: idx for idx, group_name in enumerate(GROUP_ORDER)}

# Per-element group index, ALIGNED to the order of ELEMENT_NUMBERS.
# ELEMENT_TO_GROUP_IDX[i] is the group index of the i-th element in
# ELEMENT_NUMBERS. The actor uses this to build the five group health indices.
#   ELEMENT_NUMBERS = [12, 109, 205, 215, 234, 306, 310, 331]
#   ELEMENT_TO_GROUP_IDX -> [ 0,   3,   3,   3,   3,   0,   2,   0]
ELEMENT_TO_GROUP_IDX = [
    GROUP_TO_IDX[ELEMENT_TO_GROUP[int(element_no)]]
    for element_no in ELEMENT_NUMBERS
]





















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
# # D: deck / WS_PC: wearing surface or protective coating / Sup: superstructure / B: bearings / Sub: substructure
ACTION_NAMES = {
    0: "Do nothing",
    1: "Replace B",
    2: "Replace D",
    3: "Replace D + B",
    4: "Replace D + Sup + B",
    5: "Full bridge replacement",
}





# ACTION_REPLACEMENT_MASK defines which engineering groups are fully replaced
# under each bridge-level maintenance action.
# If a group appears in the selected action mask, then ALL elements belonging to that group are assigned the REPLACEMENT_TRANSITION matrix.
ACTION_REPLACEMENT_MASK = {
    0: set(),
    1: {"bearings"},
    2: {"deck"},
    3: {"deck", "bearings"},
    4: {
        "deck",
        "superstructure",
        "bearings",
    },
    5: {
        "deck",
        "superstructure",
        "bearings",
        "substructure",
    },
}