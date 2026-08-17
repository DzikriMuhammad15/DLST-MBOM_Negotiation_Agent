"""DirectedDDPGTrainer — DDPG trainer + L3 potential-based reward shaping (spec §6).

Subclass ``DDPGTrainer`` yang menambahkan istilah shaping **di atas** reward AS-IS
(Eq.9 dari ``RewardShaper`` TAK disentuh):

    r'_t = r_t + β·(γ·(1 − done_t)·Φ(s_{t+1}) − Φ(s_t)),   Φ(terminal)=0

Φ dihitung di AGEN saat episode (butuh output IOP respond) lalu di-cache ke
``step.meta["phi"]`` (lihat ``DLSTMBOMDirected`` §2.2). Di sini trainer hanya membaca
``phi`` dari step — jadi ``reward_shaper.py`` tetap IOP-agnostic & AS-IS.

Policy-invariance (Ng, Harada & Russell 1999): karena ``F = γΦ' − Φ`` dengan
``Φ(terminal)=0``, optimum kebijakan tak berubah; hanya distribusi kredit antar-langkah.

Config-gated: ``shaping_enabled=False`` (default) → identik ``DDPGTrainer``.
Dipakai HANYA untuk head **propose** DLSTMBOMDirected (tempat Eq.9 buta-aksi).
"""

from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from negotiation.trainers.ddpg_trainer import DDPGTrainer


class DirectedDDPGTrainer(DDPGTrainer):
    """DDPG + L3 PBRS shaping (config-gated). ``shaping_enabled=False`` → identik base.

    Example:
        >>> t = DirectedDDPGTrainer(gamma=0.99, tau=0.005, device="cpu", epoch=1,
        ...                         batch_size=8, reward_shaper=None,
        ...                         shaping_enabled=True, beta=0.3, gamma_shaping=0.95)
        >>> [round(x, 4) for x in t._shaping_terms([0, 0, 1], [0.3, 0.5, 0.0])]
        [0.0525, -0.15, 0.0]
    """

    def __init__(self, *args: Any, shaping_enabled: bool = False, beta: float = 0.30,
                 gamma_shaping: float = 0.95, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._shaping_enabled = bool(shaping_enabled)
        self._beta = float(beta)
        self._gamma_shaping = float(gamma_shaping)

    def _shaping_terms(self, dones: Any, phi: List[float]) -> List[float]:
        """F_t = β·(γ·(1−done_t)·Φ_{t+1} − Φ_t). ``Φ(terminal)=0`` via done_t."""
        n = len(phi)
        out: List[float] = []
        for t in range(n):
            terminal = float(dones[t]) >= 0.5
            phi_next = 0.0 if terminal else (phi[t + 1] if t + 1 < n else 0.0)
            out.append(self._beta * (self._gamma_shaping * phi_next - phi[t]))
        return out

    @staticmethod
    def _episode_phi(episode: Any, n: int) -> Optional[List[float]]:
        """Ambil Φ per langkah dari ``step.meta['phi']`` (None kalau tak lengkap/absen)."""
        steps = getattr(episode, "steps", None) or []
        if len(steps) != n:
            return None
        phi: List[float] = []
        any_phi = False
        for s in steps:
            meta = getattr(s, "meta", None)
            v = meta.get("phi") if isinstance(meta, dict) else None
            if v is not None:
                any_phi = True
            phi.append(float(v) if v is not None else 0.0)
        return phi if any_phi else None

    def episode_to_batch(self, episode: Any) -> Optional[Dict[str, Any]]:
        """Base pack (reward Eq.9 AS-IS), lalu tambahkan shaping L3 dari ``step.meta['phi']``."""
        pack = super().episode_to_batch(episode)
        if pack is None or not self._shaping_enabled or torch is None:
            return pack
        rewards = pack["rewards"]
        n = int(rewards.shape[0])
        phi = self._episode_phi(episode, n)
        if phi is None:
            return pack  # tak ada Φ (mis. episode shadow/base) → tanpa shaping
        dones = pack["dones"].detach().cpu().tolist()
        F = self._shaping_terms(dones, phi)
        pack["rewards"] = rewards + torch.tensor(
            F, dtype=rewards.dtype, device=rewards.device)
        return pack
