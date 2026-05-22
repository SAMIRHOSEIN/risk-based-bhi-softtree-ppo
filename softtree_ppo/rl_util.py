import torch
import torch.nn as nn

from .settings import CONST_ACTION_DEFAULT


class CriticNet(nn.Module):
    """Critic neural net giving state value
    """
    def __init__(
        self, input_dim,
        critic_cells, critic_layers,
        device=torch.device("cpu")
    ):
        # no need for input_dim due to LazyLinear
        super().__init__()
        layers = [nn.Linear(input_dim, critic_cells, device=device), nn.ELU()]
        layers = layers + [nn.Linear(critic_cells, critic_cells, device=device), nn.ELU()] * critic_layers
        layers.append(nn.Linear(critic_cells, 1, device=device))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ActorNetLogit(nn.Module):
    """Actor neural net giving action logits (input of softmax)
    """
    def __init__(
        self, input_dim, output_dim,
        actor_cells, actor_layers,
        device=torch.device("cpu")
    ):
        # Change LazyLinear to Linear
        super().__init__()
        layers = [nn.Linear(input_dim, actor_cells, device=device), nn.ELU()]
        layers = layers + [nn.Linear(actor_cells, actor_cells, device=device), nn.ELU()] * actor_layers
        layers.append(nn.Linear(actor_cells, output_dim, device=device))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConstantModule(nn.Module):
    """Always choose one action (a dummy actor for comparison)
    """
    def __init__(
        self, constant_value: int = CONST_ACTION_DEFAULT,
        device=torch.device("cpu")
    ):
        super().__init__()
        self.constant = torch.as_tensor(constant_value, device=device)

    def forward(self, x):
        # ignore input x and always return the constant
        return self.constant


class RunningRewardNormalizer:
    def __init__(
        self, decay=0.999, eps=1e-8,
        device=torch.device("cpu")
    ):
        self.decay = decay
        self.eps = eps
        self.device = device
        
        # Initialize running statistics
        self.running_mean = torch.zeros(1, device=device)
        self.running_var = torch.ones(1, device=device)
        self.initialized = False

    @torch.no_grad()
    def __call__(self, tensordict_batch):
        """Updates stats and normalizes the rewards in the TensorDict in-place."""
        
        # 1. Extract rewards (TorchRL stores them in ("next", "reward"))
        rewards = tensordict_batch.get(("next", "reward"))
        
        # Flatten to easily compute batch statistics
        flat_rewards = rewards.view(-1)
        
        batch_mean = flat_rewards.mean()
        # Handle edge case where batch size is 1
        if flat_rewards.numel() > 1:
            batch_var = flat_rewards.var(unbiased=False) 
        else:
            batch_var = torch.zeros_like(batch_mean)

        # 2. Update running statistics using EMA
        if not self.initialized:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
            self.initialized = True
        else:
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * batch_mean
            self.running_var = self.decay * self.running_var + (1 - self.decay) * batch_var

        # 3. Normalize the rewards
        std = torch.sqrt(self.running_var + self.eps)
        normalized_rewards = (rewards - self.running_mean) / std
        
        # 4. Write the normalized rewards back into the TensorDict
        tensordict_batch.set(("next", "reward"), normalized_rewards)
        
        return tensordict_batch
    
    def state_dict(self):
        return {
            "running_mean": self.running_mean.cpu(),
            "running_var": self.running_var.cpu(),
            "initialized": self.initialized
        }
        
    def load_state_dict(self, state_dict):
        self.running_mean.copy_(state_dict["running_mean"].to(self.device))
        self.running_var.copy_(state_dict["running_var"].to(self.device))
        self.initialized = state_dict["initialized"]