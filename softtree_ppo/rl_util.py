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