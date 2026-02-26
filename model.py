"""Shared PINN model architecture — maps (x, t) to u(x, t)."""

import torch
import torch.nn as nn


class PhysicsInformedNN(nn.Module):
    """Feed-forward (x,t) -> u with Tanh activations.

    Tanh instead of ReLU because we need nonzero second derivatives
    for the PDE residual.
    """

    def __init__(
        self,
        hidden_size: int = 20,
        num_layers: int = 4,
        nu: float | None = None,
    ):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

        # Kept for backward compat with old Burgers-only checkpoint loading.
        # New code should use equations.py for physics parameters.
        self.nu = nu if nu is not None else 0.01 / torch.pi

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    # Legacy helper — kept so old tests / old checkpoints still work.
    def physics_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the Burgers equation residual (legacy interface).

        New code should use equation.physics_loss(model, x, t) instead.
        """
        from equations import BurgersEquation

        eq = BurgersEquation(nu=float(self.nu))
        return eq.physics_loss(self, x, t)
