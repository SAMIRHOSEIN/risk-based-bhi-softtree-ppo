#%%
# Reason: this file makes SoftTreeBHI reusable by training, validation, and SofttreePPOTrainer.load_actor().

# Why seperate file?
# bridge_bhi_training_stBHI.py imports " from softtree_ppo.training import SofttreePPOTrainer"
# So if training.py also imports SoftTreeBHI from bridge_bhi_training_stBHI.py, we create circular import nonsense.
# So I decided to move SoftTreeBHI in seperate file. 

import torch
import torch.nn as nn
import torch.nn.functional as F

from softtree.softtree_classification import SoftTreeClassifier


class SharedBHILinear(nn.Module):
    def __init__(
        self,
        num_elements,
        ncs,
        out_features,
        health_coefficients,
        initial_element_weights,
        include_step_count=False,
    ):
        super().__init__()

        self.num_elements = num_elements
        self.ncs = ncs
        self.out_features = out_features
        self.include_step_count = include_step_count

        health_coefficients = torch.as_tensor(health_coefficients, dtype=torch.float32)
        self.register_buffer("health_coefficients", health_coefficients)

        initial_weights = torch.as_tensor(initial_element_weights, dtype=torch.float32)

        # Reason: this initializes the learnable element weights from SSF-only engineering weights not engineering-based weights VF×SSF. 
        # Later, we can compare the learned weights with the original ELEMENT_WEIGHTS from Valenzuela or Rashidi.
        # Starts close to the engineering-based element weights, otherwise, it might not make learnable elemenet weights sense.
        initial_raw = torch.log(torch.expm1(initial_weights).clamp_min(1e-6))
        self.raw_element_weights = nn.Parameter(initial_raw)

        # Alias for compatibility with SofttreePPOTrainer regularization code.
        # The trainer expects inner_nodes.weight to exist.
        self.weight = self.raw_element_weights

        # One bias/threshold per internal node.
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, x):
        # If time is appended to the observation, ignore it for BHI.
        # BHI must be computed only from element condition states.
        x_cs = x[..., : self.num_elements * self.ncs]

        state = x_cs.reshape(-1, self.num_elements, self.ncs)

        # H_i = CS_i dot K
        element_health = torch.matmul(state, self.health_coefficients)

        # Positive learnable element weights.
        element_weights = F.softplus(self.raw_element_weights)

        # BHI = sum_i W_i H_i / sum_i W_i
        bhi = (element_health * element_weights).sum(dim=1) / element_weights.sum()

        # Same BHI for all internal nodes, different bias per node.
        return bhi.unsqueeze(-1) + self.bias
    



class SoftTreeBHI(SoftTreeClassifier):
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
        include_step_count=False,
        apply_batchNorm=False,
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

        self.bhi_num_elements = num_elements
        self.bhi_ncs = ncs
        self.bhi_include_step_count = include_step_count
        self.bhi_initial_element_weights = list(initial_element_weights)

        self.inner_nodes = SharedBHILinear(
            num_elements=num_elements,
            ncs=ncs,
            out_features=self.internal_node_num_,
            health_coefficients=health_coefficients,
            initial_element_weights=initial_element_weights,
            include_step_count=include_step_count,
        )