"""Actor DDPG varian DLSTMBOMDirected — L5 hygiene B1 + B2 (spec §7).

Sama dengan ``DDPGActorStrategy`` / ``DDPGActorThreshold`` AS-IS, TAPI:
* **B2 LayerNorm** disisipkan setelah tiap ``Linear`` (sebelum ``ReLU``) di backbone →
  memaksa pre-activation mean-0/std-1 → `z` tetap di zona responsif → cegah drift saturasi.
* **B1 init kecil** pada lapisan-akhir (``head`` / ``net[-1]``): `uniform(±init_scale)` →
  `z≈0` di awal → `tanh'/sigmoid'` maksimal (titik-awal responsif).

``forward`` di-inherit dari kelas AS-IS (memakai ``self.backbone``+``self.head`` /
``self.net``) — hanya struktur backbone & init lapisan-akhir yang diganti. Dipakai HANYA
oleh DLSTMBOMDirected saat ``directed.hygiene`` ON; ``ddpg_networks.py`` tetap AS-IS.
"""

import torch.nn as nn

from negotiation.models.ddpg_networks import DDPGActorStrategy, DDPGActorThreshold


def _ln_backbone(state_dim: int, hidden_dim: int, n_layers: int):
    """Layer backbone dengan LayerNorm: (Linear → LayerNorm → ReLU) × n_layers."""
    layers = [nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    return layers


def _small_init(layer: nn.Linear, scale: float) -> None:
    """B1: init kecil `uniform(±scale)` pada bobot & bias lapisan-akhir."""
    nn.init.uniform_(layer.weight, -scale, scale)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -scale, scale)


class DirectedActorStrategy(DDPGActorStrategy):
    """Actor bidding/acceptance + LayerNorm (B2) + init kecil head (B1).

    Example:
        >>> import torch
        >>> a = DirectedActorStrategy(6, 3, (1, 1, 1), hidden_dim=8)
        >>> tuple(a(torch.zeros(2, 6)).shape)
        (2, 3)
        >>> any(isinstance(m, torch.nn.LayerNorm) for m in a.backbone)
        True
    """

    def __init__(self, state_dim, action_dim, split_sizes, hidden_dim: int = 128,
                 n_layers: int = 2, init_scale: float = 3e-3) -> None:
        super().__init__(state_dim, action_dim, split_sizes, hidden_dim, n_layers)
        self.backbone = nn.Sequential(*_ln_backbone(state_dim, hidden_dim, n_layers))  # B2
        _small_init(self.head, init_scale)                                             # B1


class DirectedActorThreshold(DDPGActorThreshold):
    """Actor threshold + LayerNorm (B2) + init kecil net[-1] (B1).

    Example:
        >>> import torch
        >>> a = DirectedActorThreshold(6, hidden_dim=8)
        >>> tuple(a(torch.zeros(2, 6)).shape)
        (2, 1)
        >>> any(isinstance(m, torch.nn.LayerNorm) for m in a.net)
        True
    """

    def __init__(self, state_dim, hidden_dim: int = 128, n_layers: int = 2,
                 init_scale: float = 3e-3) -> None:
        super().__init__(state_dim, hidden_dim, n_layers)
        layers = _ln_backbone(state_dim, hidden_dim, n_layers)      # B2
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        _small_init(self.net[-1], init_scale)                       # B1
