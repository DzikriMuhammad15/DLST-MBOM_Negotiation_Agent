"""DDPG networks for DLST-ANESIA — AS-IS from the paper code (``02-07``).

Three separate actor-critic architectures (paper Sec. 5.2):

1. Threshold utility (u-bar_t): actor = single-output regression ANN, output
   ``u-bar_t`` in ``(0, 1)`` via sigmoid.
2. Bidding strategy: actor = multi-output regression ANN producing ``(delta, c, p)``.
3. Acceptance strategy: actor = multi-output regression ANN producing ``(delta, c, p)``.

The critic is always ``Q(s, a)``: it receives state + action and returns a
scalar. Because ``Q`` already receives the action, the DLST/DDPG critic is
**not** enriched with the candidate pool (see ``03-06``). The continuous-action
head here is unchanged across input modes; only the front-end that produces the
state vector differs (``03-01``).
"""

import torch
import torch.nn as nn


class DDPGActorThreshold(nn.Module):
    """Actor for the utility threshold (single-output regression ANN).

    Output ``u-bar_t`` in ``(0, 1)`` via sigmoid.

    Example:
        >>> import torch
        >>> a = DDPGActorThreshold(state_dim=6, hidden_dim=8)
        >>> out = a(torch.zeros(3, 6))
        >>> tuple(out.shape)
        (3, 1)
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
        """Return ``u-bar_t`` in ``(0, 1)`` of shape ``(B, 1)``.

        Example:
            >>> import torch
            >>> float(DDPGActorThreshold(2, 4)(torch.zeros(1, 2)).item()) == 0.5
            True
        """
        return torch.sigmoid(self.net(x))


class DDPGActorStrategy(nn.Module):
    """Actor for bidding / acceptance strategy (multi-output regression ANN).

    Produces a flat vector of size ``action_dim`` split via
    ``split_sizes = (n_delta, n_choice, n_param)``: the ``delta`` and ``c`` parts
    are passed through sigmoid (in ``(0, 1)``), the ``p`` part through tanh (in
    ``(-1, 1)``, re-scaled inside the strategy template). This is the head that
    stays AS-IS in the new project.

    Example:
        >>> import torch
        >>> a = DDPGActorStrategy(state_dim=6, action_dim=3,
        ...                       split_sizes=(1, 1, 1), hidden_dim=8)
        >>> out = a(torch.zeros(2, 6))
        >>> tuple(out.shape)
        (2, 3)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        split_sizes,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(
                f"n_layers must be >= 1, got {n_layers}"
            )
        self.action_dim = action_dim
        self.split_sizes = tuple(split_sizes)  # (n_delta, n_choice, n_param)
        assert sum(self.split_sizes) == action_dim, (
            f"split_sizes {self.split_sizes} must sum to action_dim={action_dim}"
        )
        backbone_layers = []
        backbone_layers.append(nn.Linear(state_dim, hidden_dim))
        backbone_layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            backbone_layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone_layers)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """Return the bounded strategy vector ``[delta..., c..., p...]`` of shape ``(B, action_dim)``.

        Example:
            >>> import torch
            >>> DDPGActorStrategy(2, 2, (1, 1, 0), 4)(torch.zeros(1, 2)).shape[-1]
            2
        """
        h = self.backbone(x)
        raw = self.head(h)

        n_delta, n_choice, n_param = self.split_sizes
        parts = []
        idx = 0
        if n_delta > 0:
            parts.append(torch.sigmoid(raw[:, idx:idx + n_delta]))
            idx += n_delta
        if n_choice > 0:
            parts.append(torch.sigmoid(raw[:, idx:idx + n_choice]))
            idx += n_choice
        if n_param > 0:
            parts.append(torch.tanh(raw[:, idx:idx + n_param]))
            idx += n_param

        return torch.cat(parts, dim=-1) if parts else raw


class DDPGCritic(nn.Module):
    """Critic ``Q(s, a)``: receives state + action, returns a scalar Q-value.

    Example:
        >>> import torch
        >>> c = DDPGCritic(state_dim=6, action_dim=3, hidden_dim=8)
        >>> out = c(torch.zeros(4, 6), torch.zeros(4, 3))
        >>> tuple(out.shape)
        (4,)
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
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        """Return ``Q(s, a)`` of shape ``(B,)`` for batched ``state`` and ``action``.

        Example:
            >>> import torch
            >>> DDPGCritic(2, 1, 4)(torch.zeros(3, 2), torch.zeros(3, 1)).shape[0]
            3
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)
