"""Policy network (discrete actor) — AS-IS from the paper code (``02-07``).

A 2-hidden-layer ReLU MLP that maps a fixed-dimension state vector to action
logits (not softmax) for numerical stability under PG/PPO. Used by RL and MBOM
agents for both phases:

* propose: ``action_dim = len(action_library)`` (PADDED uses ``K_max``);
* respond: ``action_dim = 2`` (``0 = accept``, ``1 = reject``).

In the new architecture the input front-end (set-encoder / padded encoder) is a
**sub-module placed in front of this network** (see ``03-02``); the core MLP is
unchanged.
"""

import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Discrete policy ``pi(a|s)`` producing logits over ``action_dim`` actions.

    Example:
        >>> import torch
        >>> net = PolicyNetwork(state_dim=8, action_dim=4, hidden_dim=16)
        >>> out = net(torch.zeros(2, 8))
        >>> tuple(out.shape)
        (2, 4)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(
                f"n_layers must be >= 1, got {n_layers}"
            )
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Return raw action logits for the batched state ``x`` of shape ``(B, state_dim)``.

        Example:
            >>> import torch
            >>> PolicyNetwork(3, 2, 8)(torch.zeros(1, 3)).shape[-1]
            2
        """
        return self.net(x)
