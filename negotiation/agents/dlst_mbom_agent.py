"""DLSTMBOMAgent — hybrid DLST + MBOM opponent modeling (``02-06``), AS-IS.

Combines **DLST-style action inference** (DDPG actor + strategy templates) with
**MBOM-style opponent modeling** (IOP + recursive imagination + Bayesian mixer).
The action is decided 100% by the DLST head; the MBOM pipeline is used **only to
model the opponent and enrich the state** that the DDPG actors consume.

Inherits the full DLST machine (:class:`DLSTNegotiator`): three DDPG actor-critics
(threshold / propose / respond) with their own trainers / memories / payloads
(Invariant #1, head AS-IS / G13), the bidding & acceptance templates, and the
shared frequency opponent model. Adds an IOP network per reasoning level plus the
recursive-imagination rollout engines and Bayesian mixers.

State enrichment (new architecture): the DLST tactical state (fixed-dim flat
vector) is **concatenated** with the IOP propose / respond probability blocks
(``_build_augmented_state_combined``); the DDPG head stays a flat MLP, so only the
front-end input grows. The factory sizes each DDPG actor's ``state_dim`` as
``dlst_state_dim + iop_propose_dim + iop_respond_dim``. IOP / rollout / mixer
formulas are unchanged.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from negotiation.agents.dlst_agent import DLSTNegotiator
from negotiation.agents.mbom.bayesian_mixer import BayesianMixer
from negotiation.agents.mbom.rollout_engine import (
    NegotiationRolloutEnginePropose,
    NegotiationRolloutEngineRespond,
    refresh_rollout_engines,
)
from negotiation.data_type.step import Step


class DLSTMBOMAgent(DLSTNegotiator):
    """DLST agent enriched with MBOM opponent modeling (AS-IS).

    Example:
        >>> DLSTMBOMAgent.__name__
        'DLSTMBOMAgent'
    """

    def __init__(
        self,
        name: str,
        ufun: Any,
        domain_spec: Any,
        bidding_spec,
        acceptance_spec,
        u_fixed: float = 0.6,
        mode: str = "PADDED",
        state_dim: int = 3,
        M: int = 3,
        rollout_k: int = 2,
        gamma: float = 0.99,
        bayesian_decay: float = 0.9,
        bayesian_horizon: int = 10,
        bayesian_temperature: float = 1.0,
        n_rollout_samples: int = 5,
        max_steps: int = 10,
        k_max: Optional[int] = None,
        trainers: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, ufun=ufun, domain_spec=domain_spec,
            bidding_spec=bidding_spec, acceptance_spec=acceptance_spec,
            u_fixed=u_fixed, mode=mode, trainers=trainers, **kwargs,
        )
        self.state_dim = state_dim
        self.k_max = int(k_max) if k_max is not None else len(self.action_library)
        # IOP propose head width = k_max (the fixed propose action space). The real
        # action_library is injected per-negotiation (C3); domain_spec is None at
        # construction, so sizing from len(action_library) would size the IOP net
        # (and the enriched DDPG state) to 1 and crash at runtime. k_max ==
        # max_action_library (bridge-enforced).
        self.iop_propose_action_dim = max(1, self.k_max)
        self.iop_respond_action_dim = 2

        self.M = M
        self.rollout_k = rollout_k
        self.gamma = gamma
        self.n_rollout_samples = n_rollout_samples
        self.max_steps = max_steps

        issue_names = [self._domain_attr("issue_name", "bundle_index") or "bundle_index"]
        self.bayesian_mixer_propose = BayesianMixer(M, bayesian_decay, bayesian_horizon, bayesian_temperature)
        self.bayesian_mixer_respond = BayesianMixer(M, bayesian_decay, bayesian_horizon, bayesian_temperature)
        self.rollout_engine_propose = NegotiationRolloutEnginePropose(
            domain_spec=self.domain_spec, action_library=self.action_library,
            issue_names=issue_names, gamma=gamma, rollout_k=rollout_k, max_steps=max_steps,
        )
        self.rollout_engine_respond = NegotiationRolloutEngineRespond(
            domain_spec=self.domain_spec, action_library=self.action_library,
            issue_names=issue_names, gamma=gamma, rollout_k=rollout_k, max_steps=max_steps,
        )
        self.rollout_level: Optional[int] = None
        self.opponent_state: Any = None

    # ------------------------------------------------------------------
    # Trainer attach (DLST 3 + IOP lists)
    # ------------------------------------------------------------------

    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """Attach the three DLST trainers + ``iop_trainers_propose`` / ``_respond``."""
        for key, trainer in dict_of_trainer.items():
            self.trainers[key] = trainer
        iop_propose = dict_of_trainer.get("iop_trainers_propose")
        if iop_propose is not None:
            self.M = len(iop_propose)

    # ------------------------------------------------------------------
    # World-context injection (C3) — FIX E-1: engines ikut di-refresh
    # ------------------------------------------------------------------

    def configure_negotiation(self, raw_obs, cand_feats, cand_mask, action_library, world_ufun, utility_min, utility_max) -> None:
        """Inject world context (C3; DLST rebuild bidding) + refresh engine (FIX E-1)."""
        super().configure_negotiation(
            raw_obs, cand_feats, cand_mask, action_library,
            world_ufun, utility_min, utility_max,
        )
        refresh_rollout_engines(self)

    # ------------------------------------------------------------------
    # Imagined initial offer (FIX E-2) — alur propose DLST lengkap
    # ------------------------------------------------------------------

    def rollout_initial_offer(self, state_basis: Any, relative_time: float = 0.0) -> tuple:
        """Tawaran awal user untuk negosiasi imajiner lewat alur propose DLST.

        FIX E-2: aktor DDPG DLST mengembalikan vektor parameter template kontinu
        (bukan indeks bundle), sehingga unpacking 4-tuple lama di rollout engine
        selalu ``ValueError``. Di sini parameter di-sample lalu diterjemahkan
        oleh strategy template menjadi indeks bundle — persis alur
        :meth:`DLSTNegotiator.propose`, tanpa perekaman memory.
        """
        rt = float(relative_time)
        state = {"current_offer": None, "relative_time": rt}
        u_bar = self._predict_threshold(state, None)
        obs = self.build_state_propose(state, None)
        action_vec = self.sample_action_propose(obs)
        ctx_bid = {
            "u_bar": u_bar,
            "opponent_last_offer": self.last_received_offer,
            "opponent_last_idx": self._offer_to_action_idx(self.last_received_offer)
            if self.last_received_offer is not None else None,
        }
        idx = self.bidding_strategy.generate(rt, action_vec, ctx_bid)
        return self.make_offer(int(idx))

    def _warn_iop_failure(self, where: str, error: Exception) -> None:
        """Log kegagalan pipeline IOP sekali per (lokasi, jenis error) — FIX E-3."""
        key = (type(self).__name__, where, type(error).__name__)
        seen = getattr(DLSTMBOMAgent, "_iop_failures_warned", None)
        if seen is None:
            seen = set()
            DLSTMBOMAgent._iop_failures_warned = seen
        if key not in seen:
            seen.add(key)
            print(f"[WARN] {key[0]}: IOP pipeline gagal di {where}: "
                  f"{type(error).__name__}: {error} (negosiasi tetap berjalan)")

    # ------------------------------------------------------------------
    # MBOM state basis (flat dim-3, for IOP) + hooks
    # ------------------------------------------------------------------

    def build_state_basis(self, state: Any, offer: Any = None) -> np.ndarray:
        """Flat dim-3 basis ``[current_u_norm, relative_time, has_offer]`` (for IOP)."""
        if isinstance(state, np.ndarray) and state.shape[0] == self.state_dim:
            return state.astype(np.float32)
        relative_time = self._relative_time(state)
        if offer is None:
            offer = self._state_offer(state)
        current_norm = self._normalize_utility(self._offer_utility(offer))
        has_offer = 1.0 if offer is not None else 0.0
        return np.array([current_norm, relative_time, has_offer], dtype=np.float32)

    def _state_basis_for_iop(self, state: Any, offer: Any = None) -> np.ndarray:
        """Hook: state basis feeding the IOP pipeline (default user perspective)."""
        return self.build_state_basis(state, offer)

    def _record_opponent_state_after_propose(self, state: Any, offer: Any) -> Any:
        """Hook: what to store as ``opponent_state`` after our propose (AS-IS)."""
        return state

    def _opponent_utility_estimate(self, offer: Any) -> float:
        """Opponent utility for rollout reward — default ZERO-SUM ``-U_user`` (AS-IS).

        FIX: rollout memanggil ini dengan elemen ``action_library`` mentah
        (Bundle); pakai :meth:`_bundle_utility` agar tidak salah rute.
        """
        return -self._bundle_utility(offer)

    def _iop_offer_to_propose_idx(self, offer: Any) -> int:
        """Map an offer to an IOP propose index (AS-IS)."""
        idx = self._offer_to_action_idx(offer)
        return idx if 0 <= idx < len(self.action_library) else 0

    # ------------------------------------------------------------------
    # IOP probability blocks + combined augmented state
    # ------------------------------------------------------------------

    def _iop_probs(self, state_basis: np.ndarray, trainers_key: str, action_dim: int, mixer: BayesianMixer) -> np.ndarray:
        """Return the (mixed / single-level) IOP probability vector (AS-IS)."""
        iop_trainers = self.trainers.get(trainers_key)
        uniform = np.ones(action_dim, dtype=np.float32) / action_dim
        if iop_trainers is None or torch is None:
            return uniform
        x = torch.tensor(state_basis, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.rollout_level is None:
            all_probs: List[np.ndarray] = []
            for m in range(self.M):
                entry = iop_trainers[m].models.get("policy")
                if entry is None:
                    all_probs.append(uniform)
                    continue
                net = entry["model"]
                net.eval()
                with torch.no_grad():
                    probs = torch.softmax(net(x), dim=-1)
                all_probs.append(probs.cpu().numpy().flatten())
            return mixer.get_mixed_probs(all_probs)
        n = self.rollout_level
        entry = iop_trainers[n].models.get("policy")
        if entry is None:
            return uniform
        net = entry["model"]
        net.eval()
        with torch.no_grad():
            probs = torch.softmax(net(x), dim=-1)
        return probs.cpu().numpy().flatten()

    def _build_augmented_state_combined(self, dlst_state_vec: np.ndarray, state_basis: np.ndarray) -> np.ndarray:
        """Concatenate the DLST tactical state with the IOP propose / respond blocks.

        ``[dlst_state_vec | iop_respond_probs (2) | iop_propose_probs (K)]`` — the
        DDPG head stays a flat MLP; only the front-end input grows (``02-06``).
        """
        iop_respond = self._iop_probs(state_basis, "iop_trainers_respond", self.iop_respond_action_dim, self.bayesian_mixer_respond)
        iop_propose = self._iop_probs(state_basis, "iop_trainers_propose", self.iop_propose_action_dim, self.bayesian_mixer_propose)
        return np.concatenate([dlst_state_vec.astype(np.float32), iop_respond.astype(np.float32), iop_propose.astype(np.float32)], axis=0)

    # ------------------------------------------------------------------
    # State builders override — DLST tactical state + IOP enrichment
    # ------------------------------------------------------------------
    #
    # Each DLST head's state is enriched with the IOP blocks at the *builder*
    # level so that every DLST call site (``_predict_threshold`` / ``respond`` /
    # ``propose`` / the DDPG trainer's ``build_state(for_=...)``) consistently
    # sees the enriched vector. ``build_state(for_=...)`` (inherited from DLST)
    # dispatches to these, so the DDPG batch is rebuilt with the same dims.

    def _enriched_dim(self, dlst_dim: int) -> int:
        """Enriched DDPG input dim = ``dlst_dim + iop_respond + iop_propose``."""
        return dlst_dim + self.iop_respond_action_dim + self.iop_propose_action_dim

    def _maybe_passthrough(self, state: Any, dlst_dim: int) -> Optional[np.ndarray]:
        """Pass an already-enriched vector straight through (trainer rebuild)."""
        if isinstance(state, (np.ndarray, list)):
            arr = np.asarray(state, dtype=np.float32)
            if arr.shape == (self._enriched_dim(dlst_dim),):
                return arr
        return None

    def build_state_utility_threshold(self, state: Any, offer: Any = None) -> np.ndarray:
        """Enriched threshold state (DLST dim 6 + IOP blocks)."""
        passthrough = self._maybe_passthrough(state, self.STATE_DIM_THRESHOLD)
        if passthrough is not None:
            return passthrough
        dlst_vec = super().build_state_utility_threshold(state, offer)
        return self._build_augmented_state_combined(dlst_vec, self._state_basis_for_iop(state, offer))

    def build_state_respond(self, state: Any, offer: Any = None) -> np.ndarray:
        """Enriched acceptance state (DLST dim 10 + IOP blocks)."""
        passthrough = self._maybe_passthrough(state, self.STATE_DIM_RESPOND)
        if passthrough is not None:
            return passthrough
        dlst_vec = super().build_state_respond(state, offer)
        return self._build_augmented_state_combined(dlst_vec, self._state_basis_for_iop(state, offer))

    def build_state_propose(self, state: Any, offer: Any = None) -> np.ndarray:
        """Enriched bidding state (DLST dim 10 + IOP blocks)."""
        passthrough = self._maybe_passthrough(state, self.STATE_DIM_PROPOSE)
        if passthrough is not None:
            return passthrough
        dlst_vec = super().build_state_propose(state, offer)
        return self._build_augmented_state_combined(dlst_vec, self._state_basis_for_iop(state, offer))

    # ------------------------------------------------------------------
    # Recursive imagination + mixer (shared with MBOM logic)
    # ------------------------------------------------------------------

    def _run_recursive_imagination(self, engine, trainers_key: str, state_basis: np.ndarray, relative_time: float) -> None:
        """Recursive imagination for one phase's IOP pipeline (AS-IS formulas).

        FIX E-3: dibungkus ``try/finally`` agar ``rollout_level`` SELALU kembali
        ``None`` (mencegah "mode zombie" saat exception di tengah imajinasi).
        """
        iop_trainers = self.trainers.get(trainers_key)
        if iop_trainers is None:
            return
        try:
            self._run_recursive_imagination_inner(engine, trainers_key, state_basis, relative_time, iop_trainers)
        finally:
            self.rollout_level = None

    def _run_recursive_imagination_inner(self, engine, trainers_key: str, state_basis: np.ndarray, relative_time: float, iop_trainers) -> None:
        """Isi loop imajinasi rekursif (dipisah agar ``finally`` di atas rapi)."""
        for m in range(1, self.M):
            self.rollout_level = m - 1
            iop_trainers[m].copy_iop_policy_weight(
                iop_trainers[m - 1].models.get("policy").get("model")
            )

            def opponent_value_fn(sv: np.ndarray) -> float:
                return 0.0  # DLST critic is Q(s, a); no flat V(s) bootstrap (AS-IS-safe)

            def opponent_utility_fn(offer) -> float:
                return self._opponent_utility_estimate(offer)

            best_response_idx = engine.compute_best_response(
                initial_user_state_basis=state_basis,
                opponent_util_fn=opponent_utility_fn,
                opponent_value_fn=opponent_value_fn,
                user_agent=deepcopy(self),
                relative_time=relative_time,
                n_uniform_sample=1,
            )
            iop_trainers[m].add_step(Step(state=state_basis, action=best_response_idx))
            iop_trainers[m].update()
            iop_trainers[m].reset_step()
        self.rollout_level = None

    def _update_bayesian_mixer(self, trainers_key: str, action_dim: int, mixer: BayesianMixer, state_basis: np.ndarray, real_action_idx: int) -> None:
        """Update one phase's Bayesian mixer from the real opponent action (AS-IS)."""
        iop_trainers = self.trainers.get(trainers_key)
        if iop_trainers is None or torch is None:
            return
        iop_probs = []
        x = torch.tensor(state_basis, dtype=torch.float32, device=self.device).unsqueeze(0)
        for m in range(self.M):
            entry = iop_trainers[m].models.get("policy")
            if entry is None:
                iop_probs.append(1.0 / action_dim)
                continue
            net = entry["model"]
            net.eval()
            with torch.no_grad():
                probs = torch.softmax(net(x), dim=-1)
            iop_probs.append(float(probs[0, real_action_idx].item()))
        mixer.update(np.array(iop_probs, dtype=np.float64))

    # ------------------------------------------------------------------
    # Lifecycle (extends DLST: runs the IOP pipeline at the right moments)
    # ------------------------------------------------------------------

    def on_negotiation_start(self, state: Any) -> None:
        """Reset DLST memories + MBOM opponent state at episode start."""
        if self.rollout_level is None:
            super().on_negotiation_start(state)
            self.opponent_state = None

    def respond(self, state: Any, source: Any = None) -> Any:
        """Run the IOP pipeline on the received offer, then DLST acceptance flow."""
        offer = self._state_offer(state)
        if self.rollout_level is None and self.opponent_state is not None and offer is not None:
            opponent_propose_idx = self._iop_offer_to_propose_idx(offer)
            state_vec = self._state_basis_for_iop(self.opponent_state)
            rt = self._relative_time(self.opponent_state)
            iop_propose = self.trainers.get("iop_trainers_propose")
            if iop_propose is not None:
                try:  # FIX E-3: kegagalan imajinasi tidak mematikan respond().
                    iop_propose[0].add_step(Step(state=state_vec, action=opponent_propose_idx))
                    iop_propose[0].update()
                    iop_propose[0].reset_step()
                    self._run_recursive_imagination(self.rollout_engine_propose, "iop_trainers_propose", state_vec, rt)
                    self._update_bayesian_mixer("iop_trainers_propose", self.iop_propose_action_dim, self.bayesian_mixer_propose, state_vec, opponent_propose_idx)
                except Exception as error:
                    self._warn_iop_failure("respond/iop_propose", error)
            iop_respond = self.trainers.get("iop_trainers_respond")
            if iop_respond is not None:
                try:  # FIX E-3.
                    iop_respond[0].add_step(Step(state=state_vec, action=1))
                    iop_respond[0].update()
                    iop_respond[0].reset_step()
                    self._run_recursive_imagination(self.rollout_engine_respond, "iop_trainers_respond", state_vec, rt)
                    self._update_bayesian_mixer("iop_trainers_respond", self.iop_respond_action_dim, self.bayesian_mixer_respond, state_vec, 1)
                except Exception as error:
                    self._warn_iop_failure("respond/iop_respond", error)
            self.opponent_state = None
        return super().respond(state, source)

    def propose(self, state: Any, dest: Any = None) -> Any:
        """DLST bidding flow; record the opponent state for the IOP pipeline."""
        offer = super().propose(state, dest)
        if self.rollout_level is None:
            self.opponent_state = self._record_opponent_state_after_propose(state, offer)
        return offer

    def on_negotiation_end(self, state: Any) -> None:
        """Final IOP respond pipeline (accept), then DLST's three-payload store."""
        if self.rollout_level is not None:
            return
        if self.opponent_state is not None:
            try:  # FIX E-3: payload episode DLST HARUS tetap tersimpan di bawah.
                iop_respond = self.trainers.get("iop_trainers_respond")
                state_vec = self._state_basis_for_iop(self.opponent_state)
                rt = self._relative_time(self.opponent_state)
                if iop_respond is not None:
                    iop_respond[0].add_step(Step(state=state_vec, action=0))
                    iop_respond[0].update()
                    iop_respond[0].reset_step()
                    self._run_recursive_imagination(self.rollout_engine_respond, "iop_trainers_respond", state_vec, rt)
                    self._update_bayesian_mixer("iop_trainers_respond", self.iop_respond_action_dim, self.bayesian_mixer_respond, state_vec, 0)
            except Exception as error:
                self._warn_iop_failure("on_negotiation_end/iop_respond", error)
            finally:
                self.opponent_state = None
        super().on_negotiation_end(state)

    # ------------------------------------------------------------------
    # Reset / clone
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Clear DLST memories + MBOM opponent state."""
        super().reset_episode()
        self.opponent_state = None

    def clone_for_rollout(self) -> "DLSTMBOMAgent":
        """Return a clean clone sharing trainers/context for imagined rollouts."""
        clone = self.__class__(
            name=self.name + "_rollout_clone",
            ufun=self.ufun,
            domain_spec=self.domain_spec,
            bidding_spec=self.bidding_spec,
            acceptance_spec=self.acceptance_spec,
            u_fixed=self.acceptance_strategy.u_fixed,
            mode=self.mode,
            state_dim=self.state_dim,
            M=self.M,
            rollout_k=self.rollout_k,
            gamma=self.gamma,
            n_rollout_samples=self.n_rollout_samples,
            max_steps=self.max_steps,
        )
        clone.trainers = self.trainers
        clone.rollout_level = self.rollout_level
        # FIX: konstruktor clone membangun BiddingStrategy dengan library KOSONG
        # (domain_spec None); configure_negotiation membangun ulang strategi
        # bidding dengan library nyata + me-refresh rollout engine clone —
        # tanpa ini clone tidak bisa menawar di negosiasi imajiner.
        clone.configure_negotiation(
            self._raw_obs, self._cand_feats, self._cand_mask,
            self.action_library, self._world_ufun,
            self.utility_min, self.utility_max,
        )
        clone.opponent_state = None
        return clone
