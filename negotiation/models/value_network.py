"""Value network ``V(s)`` — AS-IS from the paper code (``02-07``).

A 2-hidden-layer ReLU MLP mapping a fixed-dimension state vector to a scalar
state-value, used by PPO (and optionally PG). The optional critic enrichment
(``03-06``: ``critic_sees_candidate_pool`` / ``critic_privileged_info``) is
realized by *widening* the ``state_dim`` passed in here (the critic's front-end
appends pooled candidate / privileged features before this MLP); the MLP body
itself is unchanged and stays ``V(s)`` (never ``Q(s, a)``). See ``03-06``.
"""

import torch.nn as nn


class ValueNetwork(nn.Module):
    """State-value function ``V(s)`` producing a scalar per state.

    Example:
        >>> import torch
        >>> net = ValueNetwork(state_dim=8, hidden_dim=16)
        >>> out = net(torch.zeros(5, 8))
        >>> tuple(out.shape)
        (5,)
    """

    def __init__(
        self,
        state_dim: int,
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
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Return ``V(s)`` of shape ``(B,)`` for batched state ``x`` of shape ``(B, state_dim)``.

        Example:
            >>> import torch
            >>> ValueNetwork(3, 8)(torch.zeros(4, 3)).shape[0]
            4
        """
        return self.net(x).squeeze(-1)
