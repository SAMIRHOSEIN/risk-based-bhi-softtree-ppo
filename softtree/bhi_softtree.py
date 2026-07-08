#%%
# WHY THIS FILE IS SEPARATE:
#   bridge_bhi_training_stBHI.py imports SofttreePPOTrainer from
#   softtree_ppo.training. If training.py also imported the actor class from the
#   training script, we would create a circular import. So the actor class lives
#   here on its own.
#
# WHAT CHANGED (Solution 1 = per-node Group Health Index selection):
#   The new PerNodeGHISelector lets EACH internal node choose, via a
#   temperature-controlled softmax over learnable logits, WHICH health index to
#   route on. The Five candidates are:
#       k=0 deck GHI, k=1 superstructure GHI, k=2 bearings GHI,
#       k=3 substructure GHI, , k=4 aggregate BHI.
#
#   SOLUTION 1 (this file): softmax in BOTH forward and backward passes.
#     forward node feature:  phi_n(s) = sum_k p_n(k) * HI_k(s)     <-- soft mixture
#     selection probs:       p_n(k)   = softmax(logits_n / tau)_k
#   There is NO hardmax and NO straight-through here. The actor changes smoothly,
#   which keeps the PPO probability ratio stable. As tau anneals toward 0 during
#   training, p_n becomes nearly one-hot, so the trained soft actor is already
#   close to a hard one-HI-per-node tree at extraction time.
# =============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
 
from softtree.softtree_classification import SoftTreeClassifier

# Canonical candidate order. Indices 0..4 are the five engineering groups,
# in the SAME order as GROUP_ORDER in settings.py. Index 5 is the aggregate
# BHI over all elements, which we always keep in the candidate set so that the
# weights of single-element groups still receive a gradient (see report doc SAM-20260626_very important, point number: #4).
GROUP_NAMES = [
    "deck",
    "superstructure",
    "bearings",
    "substructure",
    "BHI_aggregate",
]
NUM_CANDIDATE_HI = len(GROUP_NAMES)


class PerNodeGHISelector(nn.Module):
    """
    Per-node health-index selector (Solution 1: soft mixture in forward & backward).
    For every internal node n and input bridge state s:
      1. Compute five candidate health indices HI_k(s), k = 0..4.
      2. Convert that node's learnable logits into selection probabilities using a temperature-controlled softmax:
      3. The node feature is the SOFT MIXTURE of candidate HIs:
             phi_n(s) = sum_k p_n(k) * HI_k(s)
      4. The routing logit fed into the soft tree is:
             node_logit_n(s) = phi_n(s) + bias_n
         (the surrounding SoftTreeClassifier multiplies by beta and applies
          the sigmoid; that part is unchanged.)
    """

 
    def __init__(
        self,
        num_elements,
        ncs,
        num_nodes,
        health_coefficients, # [1.00, 0.66, 0.33, 0.00]
        element_to_group_idx, # Comes from settings.ELEMENT_TO_GROUP_IDX -> [ 0,   3,   3,   3,   3,   0,   2,   0]
        initial_element_weights, # significance factors used to warm-start the learnable element weights
        include_step_count=False, # Excluded from all health-index computations
        tau_init=1.0,
    ):
        super().__init__()
 
        self.num_elements = num_elements
        self.ncs = ncs
        self.num_nodes = num_nodes
        self.include_step_count = include_step_count
        self.num_hi = NUM_CANDIDATE_HI
 
        # ----- fixed (non-learnable) buffers -------------------------------
        health_coefficients = torch.as_tensor(health_coefficients, dtype=torch.float32)
        self.register_buffer("health_coefficients", health_coefficients)  # (NCS,)
 
        element_to_group_idx = torch.as_tensor(element_to_group_idx, dtype=torch.long)
        self.register_buffer("element_to_group_idx", element_to_group_idx)  # (E,)
 
        # ----- learnable element weights (shared by all GHIs + aggregate BHI)
        # Warm-started from engineering significance factors via inverse-softplus
        # so that softplus(raw) ~= initial_weights (same trick as original code from david).
        initial_weights = torch.as_tensor(initial_element_weights, dtype=torch.float32)
        initial_raw = torch.log(torch.expm1(initial_weights).clamp_min(1e-6))
        self.raw_element_weights = nn.Parameter(initial_raw)              # (E,)
 
        # Alias kept so the trainer's regularization code that reads
        # inner_nodes.weight does not crash. (See note in training.py change.)
        self.weight = self.raw_element_weights
 
        # ----- per-node selection logits -----
        # Shape (num_nodes, 5). Initialized to zeros => uniform selection at the
        # start, i.e. no prior preference for any HI at any node.
        self.selection_logits = nn.Parameter(torch.zeros(num_nodes, self.num_hi))
 
        # ----- per-node bias / threshold  -------------
        self.bias = nn.Parameter(torch.empty(num_nodes))
        nn.init.uniform_(self.bias, -0.5, 0.5) # like david's original code
 
        # ----- selection temperature (annealed externally, not a Parameter) -
        self.tau = float(tau_init)
 
    # -----------------------------------------------------------------------
    # Compute the five candidate health indices for a batch of observations.
    # Returns hi_stack of shape (N, 5) in GROUP_NAMES order.
    # -----------------------------------------------------------------------
    def _compute_all_hi(self, x):
        # In the following lines, we first extract the last self.num_elements * self.ncs features from x and 
        # then rehape them into a 3D tensor to extract N because we need only condition stats to calcualte the HIs.
        x_cs = x[..., : self.num_elements * self.ncs]
        state = x_cs.reshape(-1, self.num_elements, self.ncs)            # (N, E, NCS)
        N = state.shape[0] #  Num of observations in the batch
 
        # Element health H_i = CS_i . K
        H = torch.matmul(state, self.health_coefficients)               # (N, E)
 
        # Strictly positive element weights.
        w = F.softplus(self.raw_element_weights)                        # (E,)
 
        # Four group-level health indices.
        # GHI_k = sum_{i in group k} w_i H_i / sum_{i in group k} w_i
        # NOTE: for a single-element group this reduces to H_i and the weight
        # cancels — which is exactly why the aggregate BHI below is essential
        # for learning single-element weights (see report SAM-20260626_very important, point number: #4).
        hi_groups = torch.zeros(N, NUM_CANDIDATE_HI-1, device=x.device, dtype=x.dtype)   # (N, 4)
        for k in range(NUM_CANDIDATE_HI-1):
            mask = (self.element_to_group_idx == k)                     # (E,)
            if mask.any():
                w_k = w[mask]                                           # (E_k,)
                H_k = H[:, mask]                                        # (N, E_k)
                hi_groups[:, k] = (H_k * w_k).sum(dim=1) / w_k.sum().clamp_min(1e-8)
 
        # Aggregate BHI over ALL elements (k = 4). 
        bhi_agg = (H * w).sum(dim=1) / w.sum().clamp_min(1e-8)          # (N,)
 
        # Stack into (N, 5).
        hi_stack = torch.cat([hi_groups, bhi_agg.unsqueeze(1)], dim=1)  # (N, 5)
        return hi_stack
 
    # -----------------------------------------------------------------------
    # SOLUTION 1 forward: soft mixture in BOTH forward and backward.
    # -----------------------------------------------------------------------
    def forward(self, x):
        hi_stack = self._compute_all_hi(x)                              # (N, 5)
 
        # Selection probabilities per node: p_n(k) = softmax(logits_n / tau).
        # Lower tau => sharper (closer to one-hot). Same probs are used in the
        # forward value AND in the backward gradient — there is no no argmax here. 
        # THIS LINE is the essence of Solution 1.
        p = F.softmax(self.selection_logits / max(self.tau, 1e-3), dim=1)  # (nodes, 5)
 
        # Node feature = soft mixture of candidate HIs.
        # phi[n, node] = sum_k p[node, k] * hi_stack[n, k]
        phi = hi_stack @ p.T                                           # (N, nodes)
 
        # Add per-node threshold. The SoftTreeClassifier applies beta + sigmoid.
        return phi + self.bias                                         # (N, nodes)
 
    # -----------------------------------------------------------------------
    # Inspection helpers (used by validation / figures).
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def get_selection_probs(self):
        """(num_nodes, 5) current soft selection probabilities."""
        return F.softmax(self.selection_logits / max(self.tau, 1e-3), dim=1)
 
    @torch.no_grad()
    def get_selected_hi_names(self):
        """Most-likely HI name for each node (for the extracted hard tree)."""
        idx = self.selection_logits.argmax(dim=1).tolist()
        return [GROUP_NAMES[k] for k in idx]
 
 




class SoftTreeBHI(SoftTreeClassifier):
    """
    Per-node Group-Health-Index soft-tree actor (Solution 1).
    """
 
    def __init__(
        self,
        input_dim,
        output_dim,
        depth,
        beta,
        num_elements,
        ncs,
        health_coefficients,
        initial_element_weights,
        element_to_group_idx,          
        include_step_count=False,
        apply_batchNorm=False,
        tau_init=1.0,                  
        **kwargs,
    ):
        super().__init__(
            input_dim,
            output_dim,
            depth,
            beta,
            apply_batchNorm,
            **kwargs,
        )
 
        # Bookkeeping kept for save/load round-tripping.
        self.bhi_num_elements = num_elements
        self.bhi_ncs = ncs
        self.bhi_include_step_count = include_step_count
        self.bhi_initial_element_weights = list(initial_element_weights)
        self.bhi_element_to_group_idx = list(element_to_group_idx)
 
        self.inner_nodes = PerNodeGHISelector(
            num_elements=num_elements,
            ncs=ncs,
            num_nodes=self.internal_node_num_,
            health_coefficients=health_coefficients,
            element_to_group_idx=element_to_group_idx,
            initial_element_weights=initial_element_weights,
            include_step_count=include_step_count,
            tau_init=tau_init,
        )
 
    # Convenience pass-throughs so the trainer can read/write tau on the actor.
    @property
    def tau(self):
        return self.inner_nodes.tau
 
    @tau.setter
    def tau(self, value):
        self.inner_nodes.tau = float(value)
 