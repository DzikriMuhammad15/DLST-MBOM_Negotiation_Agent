"""DLSTMBOMNewMechanism — opponent-perspective IOP hybrid + N·N fan-out (AS-IS).

Same as :class:`DLSTMBOMAgent` except the IOP pipeline is built from the
**opponent's** synthesized state (estimated opponent utility via
:class:`FrequencyOpponentModel`) rather than the user's NegMAS state — mirroring
:class:`MBOMNewMechanism`. The DLST head, action inference, and IOP / rollout /
mixer formulas are inherited unchanged (Invariant #1 preserved).

DEFINING FORMULA — N·N fan-out (paper "MBOM New Mechanism"): the combined
augmented state is **not** built from the single current IOP basis. Instead, for
every offer ``o`` the user could propose next, a hypothetical opponent state is
synthesized (``build_negmas_state_from_offer``) and its basis
(``build_opponent_agent_state_basis``) is stacked into a batch ``X`` of shape
``(N, state_dim)``. Then:

* IOP **respond** over ``X`` -> ``(N, 2)`` -> flatten ``(2N,)``;
* IOP **propose** over ``X`` -> ``(N, N)`` -> flatten ``(N*N,)``;

and the combined state consumed by the DDPG actor is
``[dlst_state_vec | iop_respond(2N) | iop_propose(N*N)]``
(see :meth:`_build_augmented_state_combined`). ``_enriched_dim`` is widened so the
three DLST heads (threshold / respond / propose) and the factory size the DDPG
``state_dim`` for the fan-out block, NOT the base single-state block. The DLST head
itself stays AS-IS (flat MLP); only its front-end input grows.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, List

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from negotiation.agents.dlst.opponent_model import FrequencyOpponentModel
from negotiation.agents.dlst_mbom_agent import DLSTMBOMAgent
from negotiation.agents.mbom.bayesian_mixer import BayesianMixer


class DLSTMBOMNewMechanism(DLSTMBOMAgent):
    """DLST+MBOM hybrid with an opponent-perspective IOP state + N·N fan-out (AS-IS).

    Example:
        >>> DLSTMBOMNewMechanism.__name__
        'DLSTMBOMNewMechanism'
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mbom_opponent_model = FrequencyOpponentModel(
            issue_names=[self._domain_attr("issue_name", "bundle_index") or "bundle_index"],
            value_lists=self._domain_attr("value_lists", {}) or {},
        )
        self.best_outcome_opponent = self._domain_attr("best_outcome_opponent", None)

        # Fan-out dimensions: N = k_max (the fixed propose action space; the real
        # action_library is injected per-negotiation via C3 and domain_spec is None
        # at construction). The combined state appends iop_respond (2N) then
        # iop_propose (N*N) to the DLST tactical vector. Sizing N from k_max keeps
        # the IOP propose net output and the DDPG enriched state-dim consistent.
        n = self.k_max
        self.fanout_n = n
        self.fanout_respond_dim = n * self.iop_respond_action_dim  # 2N
        self.fanout_propose_dim = n * self.iop_propose_action_dim   # N*N
        self.fanout_extra_dim = self.fanout_respond_dim + self.fanout_propose_dim

    # ------------------------------------------------------------------
    # Opponent-state synthesis
    # ------------------------------------------------------------------

    def build_negmas_state_from_offer(self, offer: Any, relative_time: float = 0.0) -> SimpleNamespace:
        """Synthesize a NegMAS-like state from an offer (``current_offer`` = offer)."""
        return SimpleNamespace(current_offer=offer, relative_time=float(relative_time))

    def build_opponent_agent_state_basis(self, state: Any, offer: Any = None) -> np.ndarray:
        """Flat dim-3 basis from the **opponent's** estimated utility (AS-IS)."""
        if isinstance(state, np.ndarray) and state.shape[0] == self.state_dim:
            return state.astype(np.float32)
        relative_time = self._relative_time(state)
        if offer is None:
            raw_offer = self._state_offer(state)
            offer = raw_offer if raw_offer is not None else self.best_outcome_opponent
        current_norm = float(self.mbom_opponent_model.estimate_utility(offer))
        has_offer = 1.0 if offer is not None else 0.0
        return np.array([current_norm, relative_time, has_offer], dtype=np.float32)

    def _state_basis_for_iop(self, state: Any, offer: Any = None) -> np.ndarray:
        """Override: IOP pipeline uses the opponent-perspective state basis."""
        return self.build_opponent_agent_state_basis(state, offer)

    def _record_opponent_state_after_propose(self, state: Any, offer: Any) -> SimpleNamespace:
        """Override: synthesize the opponent state from the just-proposed offer."""
        relative_time = self._relative_time(state)
        return self.build_negmas_state_from_offer(offer, relative_time)

    def _on_real_offer_received(self, offer: Any) -> None:
        """Update the MBOM frequency model on each real opponent offer."""
        if offer is not None:
            self.mbom_opponent_model.update(offer)

    # ------------------------------------------------------------------
    # Fan-out combined state (N·N) — DEFINING FORMULA (AS-IS)
    # ------------------------------------------------------------------

    def _enriched_dim(self, dlst_dim: int) -> int:
        """Enriched DDPG input dim = ``dlst_dim + 2N + N*N`` (fan-out block)."""
        return dlst_dim + self.fanout_extra_dim

    def _mix_probs_batch(self, all_probs_per_model: List[np.ndarray], mixer: BayesianMixer) -> np.ndarray:
        """Batched :meth:`BayesianMixer.get_mixed_probs` over ``N`` hypothetical states.

        ``all_probs_per_model``: list of length ``M``, each ``(N, action_dim)``.
        Returns ``(N, action_dim)`` — exact replication of :meth:`get_mixed_probs`
        (weighted sum, per-row normalize, uniform-row fallback). Identical to
        :meth:`MBOMNewMechanism._mix_probs_batch`.
        """
        alpha = mixer.get_alpha().astype(np.float64)                        # (M,)
        stacked = np.stack(all_probs_per_model, axis=0).astype(np.float64)  # (M, N, action_dim)
        mixed = np.tensordot(alpha, stacked, axes=(0, 0))                   # (N, action_dim)

        totals = mixed.sum(axis=-1, keepdims=True)                         # (N, 1)
        safe_totals = np.where(totals < 1e-10, 1.0, totals)
        result = mixed / safe_totals

        action_dim = mixed.shape[-1]
        uniform_row = np.ones(action_dim, dtype=np.float64) / action_dim
        fallback_mask = (totals < 1e-10).flatten()
        if fallback_mask.any():
            result[fallback_mask] = uniform_row

        return result.astype(np.float32)

    def _fanout_probs(self, X: "torch.Tensor", trainers_key: str, action_dim: int, mixer: BayesianMixer) -> np.ndarray:
        """IOP probs over the batch ``X`` of ``N`` hypothetical states -> ``(N, action_dim)`` (AS-IS)."""
        n = X.shape[0]
        iop_trainers = self.trainers.get(trainers_key)
        uniform = np.ones((n, action_dim), dtype=np.float32) / action_dim
        if iop_trainers is None or torch is None:
            return uniform

        if self.rollout_level is None:
            all_probs: List[np.ndarray] = []
            for m in range(self.M):
                entry = iop_trainers[m].models.get("policy")
                if entry is None:
                    all_probs.append(uniform.copy())
                    continue
                net = entry["model"]
                net.eval()
                with torch.no_grad():
                    probs = torch.softmax(net(X), dim=-1)
                all_probs.append(probs.cpu().numpy())
            return self._mix_probs_batch(all_probs, mixer)

        level = self.rollout_level
        entry = iop_trainers[level].models.get("policy")
        if entry is None:
            return uniform
        net = entry["model"]
        net.eval()
        with torch.no_grad():
            probs = torch.softmax(net(X), dim=-1)
        return probs.cpu().numpy()

    def _pad_fanout_rows(self, batch: np.ndarray) -> np.ndarray:
        """Zero-pad the candidate-row dimension from ``real_N`` up to ``k_max``.

        The fan-out is computed over the RUNTIME ``action_library`` (size
        ``real_N``), but the DDPG actors / factory size the enriched state from
        ``k_max`` (== ``max_action_library``); ``real_N`` is frequently smaller
        (the per-negotiation feasible library is injected via C3). The real
        candidate rows keep their exact computed IOP probabilities; the
        non-existent (padded) rows are zeroed so the flattened block is always
        ``k_max``-sized regardless of ``real_N`` — the same padded-and-masked shape
        adaptation the discrete propose head uses, NOT a paper-formula change.
        ``batch``: ``(real_N, action_dim)`` -> ``(k_max, action_dim)``.
        """
        real_n, action_dim = batch.shape
        if real_n == self.fanout_n:
            return batch
        padded = np.zeros((self.fanout_n, action_dim), dtype=batch.dtype)
        rows = min(real_n, self.fanout_n)
        padded[:rows] = batch[:rows]
        return padded

    def _build_augmented_state_combined(self, dlst_state_vec: np.ndarray, iop_state_basis: np.ndarray) -> np.ndarray:
        """Combined state ``[dlst_state_vec | iop_respond(2N) | iop_propose(N*N)]`` (fan-out).

        For every offer the user could propose next, synthesize the opponent's
        hypothetical state and basis, stack into ``X`` of shape
        ``(real_N, state_dim)``, then run IOP respond -> ``(real_N, 2)`` and IOP
        propose -> ``(real_N, N)`` (AS-IS fan-out math, ``N = k_max``). Each batch
        is then row-padded to ``k_max`` and flattened, so the block is always
        ``iop_respond(2N) | iop_propose(N*N)`` with ``N = k_max`` regardless of the
        runtime library size (shape adaptation only — real-candidate values
        unchanged; padded rows zero/masked). ``relative_time`` anchoring all
        hypothetical states is taken from ``iop_state_basis[1]`` (AS-IS).
        """
        relative_time = float(iop_state_basis[1]) if len(iop_state_basis) > 1 else 0.0
        hypothetical_basis = [
            self.build_opponent_agent_state_basis(self.build_negmas_state_from_offer(offer, relative_time))
            for offer in self.action_library
        ]
        X = torch.tensor(np.stack(hypothetical_basis, axis=0), dtype=torch.float32, device=self.device)  # (real_N, state_dim)

        resp_batch = self._fanout_probs(X, "iop_trainers_respond", self.iop_respond_action_dim, self.bayesian_mixer_respond)
        prop_batch = self._fanout_probs(X, "iop_trainers_propose", self.iop_propose_action_dim, self.bayesian_mixer_propose)
        iop_respond_probs = self._pad_fanout_rows(resp_batch).reshape(-1)  # (2N,)  N = k_max
        iop_propose_probs = self._pad_fanout_rows(prop_batch).reshape(-1)  # (N*N,) N = k_max
        return np.concatenate(
            [dlst_state_vec.astype(np.float32), iop_respond_probs, iop_propose_probs], axis=0
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Lifecycle / reset / clone
    # ------------------------------------------------------------------

    def respond(self, state: Any, source: Any = None) -> Any:
        """Update the MBOM opponent model, then run the hybrid respond flow."""
        offer = self._state_offer(state)
        if self.rollout_level is None:
            self._on_real_offer_received(offer)
        return super().respond(state, source)

    def reset_episode(self) -> None:
        """Clear memories + reset the MBOM opponent model."""
        super().reset_episode()
        self.mbom_opponent_model.reset()

    def clone_for_rollout(self) -> "DLSTMBOMNewMechanism":
        """Clone for rollout, deep-copying the MBOM opponent model state (AS-IS)."""
        clone = super().clone_for_rollout()
        clone.mbom_opponent_model = deepcopy(self.mbom_opponent_model)
        clone.best_outcome_opponent = self.best_outcome_opponent
        return clone
