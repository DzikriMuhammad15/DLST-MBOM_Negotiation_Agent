"""Pengumpul nilai loss pelatihan per `update()` per (side, phase).

Modul ini menyediakan konstanta nama side/phase (LLR-A5) dan kelas
:class:`LossHistory` (LLR-N6.2) yang mengumpulkan nilai loss yang dihasilkan
setiap pemanggilan `trainer.update()`, dikelompokkan menurut pihak negosiasi
(`side`) dan fase mekanisme (`phase`).

Loss dicatat **per update()** (KR-8), bukan per episode, dan dipisahkan **per
pihak dan per fase** (KR-9) sehingga visualisasi (figur terpisah) dapat
dilakukan kemudian. Modul ini hanya bergantung pada pustaka standar dan
`typing`; ia TIDAK meng-import torch/numpy/matplotlib atau lapisan
`negotiation` (INV-1/INV-2).
"""

from math import isfinite
from typing import Dict, Iterable, List, Tuple

# --- Konstanta nama fase loss (LLR-A5 / LLR-N6.1) -----------------------------
# Fase mekanisme negosiasi (propose & respond terpisah penuh; threshold = DLST).
PHASE_PROPOSE: str = "propose"
PHASE_RESPOND: str = "respond"
PHASE_THRESHOLD: str = "utility_threshold"

# Pihak negosiasi.
SIDE_FLEET: str = "fleet"
SIDE_SPKLU: str = "spklu"


class LossHistory:
    """Kumpulan deret nilai loss per kombinasi (side, phase).

    Setiap kombinasi `(side, phase)` memetakan ke daftar nilai loss terurut
    menurut urutan event `update()`. Nilai non-finite (None/NaN/inf) diabaikan
    agar pelatihan tidak terganggu oleh nilai loss yang rusak.

    Kombinasi yang umum dilacak: (fleet, propose), (fleet, respond),
    (spklu, propose), (spklu, respond), serta (fleet, utility_threshold) dan
    (spklu, utility_threshold) untuk agent DLST.

    Contoh:
        >>> history = LossHistory()
        >>> history.record(SIDE_FLEET, PHASE_PROPOSE, 0.5)
        >>> history.record(SIDE_FLEET, PHASE_PROPOSE, 0.25)
        >>> history.get(SIDE_FLEET, PHASE_PROPOSE)
        [0.5, 0.25]
        >>> history.get(SIDE_SPKLU, PHASE_RESPOND)
        []
    """

    def __init__(self) -> None:
        """Inisialisasi penyimpanan deret loss kosong.

        Contoh:
            >>> LossHistory().is_empty()
            True
        """
        self.series: Dict[Tuple[str, str], List[float]] = {}

    def record(self, side: str, phase: str, loss: float) -> None:
        """Tambahkan satu nilai loss untuk kombinasi (side, phase).

        Membuat deret baru bila kombinasi belum ada. Nilai None atau non-finite
        (NaN/inf) diabaikan tanpa menimbulkan error.

        Contoh:
            >>> history = LossHistory()
            >>> history.record(SIDE_SPKLU, PHASE_RESPOND, 1.0)
            >>> history.record(SIDE_SPKLU, PHASE_RESPOND, float("nan"))
            >>> history.get(SIDE_SPKLU, PHASE_RESPOND)
            [1.0]
        """
        if loss is None:
            return
        try:
            value = float(loss)
        except (TypeError, ValueError):
            return
        if not isfinite(value):
            return
        self.series.setdefault((side, phase), []).append(value)

    def record_many(self, items: Iterable[Tuple[str, str, float]]) -> None:
        """Catat banyak nilai loss sekaligus dari iterable (side, phase, loss).

        Contoh:
            >>> history = LossHistory()
            >>> history.record_many([
            ...     (SIDE_FLEET, PHASE_PROPOSE, 0.1),
            ...     (SIDE_FLEET, PHASE_PROPOSE, 0.2),
            ... ])
            >>> history.get(SIDE_FLEET, PHASE_PROPOSE)
            [0.1, 0.2]
        """
        for side, phase, loss in items:
            self.record(side, phase, loss)

    def get(self, side: str, phase: str) -> List[float]:
        """Kembalikan deret loss untuk (side, phase); list kosong bila kosong.

        Mengembalikan referensi list internal agar pemanggil dapat membacanya
        langsung; deret yang belum pernah diisi menghasilkan list kosong (KR-8).

        Contoh:
            >>> history = LossHistory()
            >>> history.get(SIDE_FLEET, PHASE_THRESHOLD)
            []
            >>> history.record(SIDE_FLEET, PHASE_THRESHOLD, 0.3)
            >>> history.get(SIDE_FLEET, PHASE_THRESHOLD)
            [0.3]
        """
        return self.series.get((side, phase), [])

    def keys(self) -> List[Tuple[str, str]]:
        """Kembalikan daftar kombinasi (side, phase) yang sudah pernah dicatat.

        Contoh:
            >>> history = LossHistory()
            >>> history.record(SIDE_FLEET, PHASE_PROPOSE, 0.5)
            >>> history.keys()
            [('fleet', 'propose')]
        """
        return list(self.series.keys())

    def to_dict(self) -> Dict[str, List[float]]:
        """Serialisasi deret menjadi dict berkunci "side:phase" untuk JSON/debug.

        Contoh:
            >>> history = LossHistory()
            >>> history.record(SIDE_SPKLU, PHASE_PROPOSE, 0.5)
            >>> history.to_dict()
            {'spklu:propose': [0.5]}
        """
        return {
            f"{side}:{phase}": list(values)
            for (side, phase), values in self.series.items()
        }

    def is_empty(self) -> bool:
        """Kembalikan True bila belum ada satu pun nilai loss tercatat.

        Contoh:
            >>> history = LossHistory()
            >>> history.is_empty()
            True
            >>> history.record(SIDE_FLEET, PHASE_PROPOSE, 0.5)
            >>> history.is_empty()
            False
        """
        return not any(self.series.values())
