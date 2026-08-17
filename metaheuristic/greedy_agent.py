# -*- coding: utf-8 -*-
"""Agen negosiasi GREEDY dua-sisi (baseline metaheuristic) — desain v2 §5.

Aturan (JUJUR & wajar — guard kejujuran context.md §14.5; kolaps harus
EMERGENT dari kekakuan aspirasi, BUKAN disabotase):

- **Propose**: SELALU menawarkan bundle ber-utility-SENDIRI tertinggi di
  action_library (fleet → utility fleet maksimal; SPKLU → utility SPKLU
  maksimal, mis. harga tinggi tanpa geser). Tidak ada konsesi sepanjang
  negosiasi (tawaran konstan).
- **Respond**: ACCEPT hanya bila utility(offer) ≥ ambang aspirasi =
  utility bundle terbaik KE-2 milik sendiri (aspirasi tinggi, konstan
  terhadap waktu). Bila library hanya 1 bundle → ambang = terbaik.

Konsekuensi yang diharapkan (hipotesis pre-register V2): daerah penerimaan
kedua pihak jarang beririsan (bundle terbaik fleet ≈ harga rendah; bundle
terbaik SPKLU ≈ harga tinggi) → deal_rate rendah secara EMERGENT. Baseline
konsesi-terjadwal (mis. Boulware — lihat RuleBasedAgent) adalah baseline
lebih kuat yang tidak diuji di sini — diakui sebagai limitasi.

Implementasi meniru pola :class:`negotiation.agents.rule_agent.RuleBasedAgent`
(non-learning, payload episode tetap direkam, instrumentasi inference otomatis
dari ``BaseNegotiator.__init_subclass__``). Tidak menyentuh kode lama.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

try:
    from negmas import ResponseType
except Exception:  # pragma: no cover
    ResponseType = None  # type: ignore

from negotiation.agents.base_agent import RESPOND_ACCEPT, RESPOND_REJECT, BaseNegotiator
from negotiation.data_type.episode import Episode

AGENT_GREEDY = "GreedyMH"


class GreedyNegotiator(BaseNegotiator):
    """Greedy dua-sisi: propose terbaik-sendiri; accept hanya ≥ terbaik ke-2.

    ``accept_rank`` = peringkat utility sendiri yang menjadi ambang accept
    (1-indexed; default 2 = bundle terbaik ke-2, sesuai desain v2 §5).
    """

    def __init__(
        self,
        name: str,
        ufun: Any,
        domain_spec: Any,
        trainers: Optional[Dict[str, Any]] = None,
        accept_rank: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec,
                         trainers=trainers, **kwargs)
        self.accept_rank = max(1, int(accept_rank))
        self.propose_action_dim = len(self.action_library)
        self.respond_action_dim = 2

    # ------------------------------------------------------------------
    # Inti greedy (dihitung per panggilan — library berganti tiap negosiasi)
    # ------------------------------------------------------------------
    def _utilities_desc(self) -> List[Tuple[float, int]]:
        """[(utility_sendiri, idx)] terurut menurun atas action_library."""
        pairs = [(self._offer_utility(self.make_offer(i)), i)
                 for i in range(len(self.action_library))]
        pairs.sort(key=lambda p: (-p[0], p[1]))
        return pairs

    def _best_index(self) -> int:
        """Indeks bundle ber-utility-sendiri tertinggi."""
        return self._utilities_desc()[0][1]

    def _aspiration_utility(self) -> float:
        """Ambang accept = utility terbaik ke-``accept_rank`` (konstan)."""
        pairs = self._utilities_desc()
        return pairs[min(self.accept_rank - 1, len(pairs) - 1)][0]

    # ------------------------------------------------------------------
    # Stubs interface (non-learning) — pola RuleBasedAgent
    # ------------------------------------------------------------------
    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """No-op: greedy tidak belajar (simetri interface)."""

    def build_state(self, state: Any, offer: Any = None) -> Any:
        return None

    def sample_action_propose(self, obs: Any) -> Tuple[int, float, Optional[float], Any]:
        return self._best_index(), 0.0, None, None

    def sample_action_respond(self, obs: Any) -> Tuple[int, float, Optional[float], Any]:
        u_raw = obs if isinstance(obs, (int, float)) else 0.0
        decision = RESPOND_ACCEPT if u_raw >= self._aspiration_utility() else RESPOND_REJECT
        return decision, 0.0, None, None

    # ------------------------------------------------------------------
    # Protokol NegMAS
    # ------------------------------------------------------------------
    def respond(self, state: Any, source: Any = None) -> Any:
        """ACCEPT hanya bila utility(offer) ≥ utility terbaik ke-2 sendiri."""
        self.flush_pending_propose(state)
        self.flush_pending_respond(state)

        offer = self._state_offer(state)
        if offer is None:
            return ResponseType.REJECT_OFFER

        current_u = self._offer_utility(offer)
        accept = current_u >= self._aspiration_utility()
        action_id = RESPOND_ACCEPT if accept else RESPOND_REJECT

        self.current_state_data_respond = {
            "state": self.state_to_dict(state),
            "action": action_id,
            "utility": current_u,
            "value": None,
            "log_prob": None,
            "action_mask": None,
            "candidate_features": None,
        }
        return ResponseType.ACCEPT_OFFER if accept else ResponseType.REJECT_OFFER

    def propose(self, state: Any, dest: Any = None) -> Any:
        """Selalu tawarkan bundle terbaik-sendiri (tanpa konsesi)."""
        action_id = self._best_index()
        offer = self.make_offer(action_id)
        self.current_state_data_propose = {
            "state": self.state_to_dict(state),
            "action": action_id,
            "utility": self._offer_utility(offer),
            "value": None,
            "log_prob": None,
            "action_mask": None,
            "candidate_features": None,
        }
        return offer

    # ------------------------------------------------------------------
    # Payload episode (pola RuleBasedAgent)
    # ------------------------------------------------------------------
    def _build_payload(self, memory: List[Any], agreement: Any, phase: str) -> Episode:
        episode = Episode(
            agent_name=self.name,
            phase=phase,
            terminal_utility=self._offer_utility(agreement) if agreement is not None else -1.0,
            agreement=agreement,
            steps=deepcopy(memory),
        )
        return episode.finalize()

    def build_episode_payload_propose(self, agreement: Any) -> Episode:
        return self._build_payload(self.memory_propose, agreement, "propose")

    def build_episode_payload_respond(self, agreement: Any) -> Episode:
        return self._build_payload(self.memory_respond, agreement, "respond")
