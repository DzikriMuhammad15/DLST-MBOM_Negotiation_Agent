from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from negmas import ResponseType
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    ResponseType = None  # type: ignore

from negotiation.agents.base_agent import RESPOND_ACCEPT, BaseNegotiator
from negotiation.agents.mbom.bayesian_mixer import BayesianMixer
from negotiation.agents.mbom.rollout_engine import (
    NegotiationRolloutEnginePropose,
    NegotiationRolloutEngineRespond,
    refresh_rollout_engines,
)
from negotiation.agents.obs_utils import (
    action_mask_from_obs,
    candidate_features_from_obs,
    obs_to_batch,
)
from negotiation.data_type.episode import Episode
from negotiation.data_type.step import Step
from negotiation.trace_logger import get_logger as _get_trace_logger

MODE_PADDED: str = "PADDED"
MODE_ENCODER: str = "ENCODER"


class MBOMAgent(BaseNegotiator):
    def __init__(
        self,
        name: str,
        ufun: Any,
        domain_spec: Any,
        state_dim: int = 3,
        M: int = 3,
        rollout_k: int = 2,
        gamma: float = 0.99,
        iop_batch_size: int = 32,
        iop_update_steps: int = 3,
        bayesian_decay: float = 0.9,
        bayesian_horizon: int = 10,
        bayesian_temperature: float = 1.0,
        n_rollout_samples: int = 5,
        max_steps: int = 10,
        mode: str = MODE_PADDED,
        k_max: Optional[int] = None,
        trainers: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec, trainers=trainers, **kwargs)
        if mode not in (MODE_PADDED, MODE_ENCODER):
            raise ValueError(f"mode must be {MODE_PADDED!r} or {MODE_ENCODER!r}, not {mode!r}")
        self.mode = mode
        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        self.state_dim = state_dim
        self.k_max = int(k_max) if k_max is not None else len(self.action_library)
        self.propose_action_dim = self.k_max
        self.respond_action_dim = 2
        # IOP propose head width = k_max (the fixed propose action space). The real
        # action_library is injected per-negotiation (C3) and may be smaller, but
        # the IOP network output dim must equal k_max so any library index (and the
        # cross-entropy target) stays in bounds. k_max == max_action_library.
        self.iop_propose_action_dim = max(1, self.k_max)
        self.iop_respond_action_dim = 2

        # MBOM hyperparameters.
        self.M = M
        self.rollout_k = rollout_k
        self.gamma = gamma
        self.iop_batch_size = iop_batch_size
        self.iop_update_steps = iop_update_steps
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

        # rollout_level: None = normal acting; int n = imagined level-n only.
        self.rollout_level: Optional[int] = None
        self.opponent_state: Any = None

    # ------------------------------------------------------------------
    # Trainer attach
    # ------------------------------------------------------------------

    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """Attach propose / respond + ``iop_trainers_propose`` / ``_respond`` (lists)."""
        for key, trainer in dict_of_trainer.items():
            self.trainers[key] = trainer
        iop_propose = dict_of_trainer.get("iop_trainers_propose")
        if iop_propose is not None:
            self.M = len(iop_propose)

    # ------------------------------------------------------------------
    # World-context injection (C3) — FIX E-1: engines ikut di-refresh
    # ------------------------------------------------------------------

    def configure_negotiation(self, raw_obs, cand_feats, cand_mask, action_library, world_ufun, utility_min, utility_max) -> None:
        """Inject world context (C3) lalu refresh kedua rollout engine (FIX E-1)."""
        super().configure_negotiation(
            raw_obs, cand_feats, cand_mask, action_library,
            world_ufun, utility_min, utility_max,
        )
        refresh_rollout_engines(self)

    # ------------------------------------------------------------------
    # Imagined initial offer (FIX E-2) — dipakai rollout engine
    # ------------------------------------------------------------------

    def rollout_initial_offer(self, state_basis: np.ndarray, relative_time: float = 0.0) -> tuple:
        """Sample tawaran awal user untuk negosiasi imajiner (kepala diskret).

        Mengembalikan outcome NegMAS ``(idx,)`` dari kepala propose diskret —
        pengganti unpacking 4-tuple lama di rollout engine yang tidak polimorfik
        (lihat FIX E-2 di ``rollout_engine.py``).
        """
        obs = self.build_state(np.asarray(state_basis, dtype=np.float32))
        action_idx, _, _, _ = self.sample_action_propose(obs)
        if self.action_library:
            action_idx = min(int(action_idx), len(self.action_library) - 1)
        return self.make_offer(int(action_idx))

    # ------------------------------------------------------------------
    # State basis (flat dim-3, for IOP pipeline) + opponent-modeling hooks
    # ------------------------------------------------------------------

    def build_state_basis(self, state: Any, offer: Any = None) -> np.ndarray:
        """Flat dim-3 state basis ``[current_u_norm, relative_time, has_offer]`` (AS-IS)."""
        if isinstance(state, np.ndarray):
            return state.astype(np.float32)
        relative_time = self._relative_time(state)
        if offer is None:
            raw_offer = self._state_offer(state)
            offer = raw_offer
        current_u = self._offer_utility(offer)
        current_norm = self._normalize_utility(current_u)
        has_offer = 1.0 if offer is not None else 0.0
        return np.array([current_norm, relative_time, has_offer], dtype=np.float32)

    def _state_basis_for_iop(self, state: Any, offer: Any = None) -> np.ndarray:
        """Hook: state basis feeding the IOP pipeline (overridden by NewMechanism)."""
        return self.build_state_basis(state, offer)

    def _record_opponent_state_after_propose(self, state: Any, offer: Any) -> Any:
        """Hook: what to store as ``opponent_state`` after our propose (AS-IS)."""
        return state

    def _on_real_offer_received(self, offer: Any) -> None:
        """Hook: called on a real (non-imagined) opponent offer (default no-op)."""

    def _opponent_utility_estimate(self, offer: Any) -> float:
        """Opponent utility for rollout reward — default ZERO-SUM ``-U_user`` (AS-IS).

        FIX: rollout engine memanggil ini dengan elemen ``action_library`` mentah
        (objek Bundle); ``_offer_utility`` dirancang untuk outcome ``(idx,)`` dan
        salah rute untuk Bundle — pakai :meth:`_bundle_utility`.
        """
        return -self._bundle_utility(offer)

    def _iop_offer_to_propose_idx(self, offer: Any) -> int:
        """Map an offer to an IOP propose index (closest utility fallback, AS-IS)."""
        idx = self._offer_to_action_idx(offer)
        if 0 <= idx < len(self.action_library):
            return idx
        return 0

    # ------------------------------------------------------------------
    # IOP probability blocks (mixed over M levels / single level in rollout)
    # ------------------------------------------------------------------

    def _iop_probs(self, state_basis: np.ndarray, trainers_key: str, action_dim: int, mixer: BayesianMixer) -> np.ndarray:
        """Return the (mixed) IOP probability vector for a state basis (AS-IS).

        Mirrors the AS-IS ``_build_augmented_state``: Bayesian-mix all ``M`` IOP
        models when acting normally, or use a single level during imagination
        (``rollout_level``).
        """
        iop_trainers = self.trainers.get(trainers_key)
        uniform = np.ones(action_dim, dtype=np.float32) / action_dim
        if iop_trainers is None:
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

    def _emit_iop_trace(self, kind: str, phase: str, **payload: Any) -> None:
        """Emit one ``iop_step`` trace record (no-op while the logger is disabled).

        Used for both IOP-network inference (``kind="inference"``) and IOP training
        (``kind="update"``): the per-level state added in ``add_step`` plus the
        resulting ``update`` loss (LLR validation requirement).
        """
        logger = _get_trace_logger()
        if not logger.is_enabled():
            return
        logger.emit(
            "iop_step",
            agent_name=getattr(self, "name", None),
            kind=kind,
            phase=phase,
            **payload,
        )

    def _augment_obs(self, obs: Dict[str, Any], state_basis: np.ndarray) -> Dict[str, Any]:
        """Fold IOP probs into the packed obs (per-candidate cand col + progress).

        The IOP propose distribution (one prob per candidate) becomes an extra
        column on ``cand_feats``; the IOP respond distribution (2 values) is
        appended to ``progress_feats``. No formula changes — these are just extra
        input features carried through the mode machinery (``02-04``).
        """
        obs = dict(obs)
        iop_propose = self._iop_probs(
            state_basis, "iop_trainers_propose", self.iop_propose_action_dim, self.bayesian_mixer_propose
        )
        iop_respond = self._iop_probs(
            state_basis, "iop_trainers_respond", self.iop_respond_action_dim, self.bayesian_mixer_respond
        )
        if self.rollout_level is None:
            self._emit_iop_trace(
                "inference", "propose",
                state_basis=state_basis, mixed_probs=iop_propose,
            )
            self._emit_iop_trace(
                "inference", "respond",
                state_basis=state_basis, mixed_probs=iop_respond,
            )
        cand = obs.get("cand_feats")
        if cand is not None:
            cand = np.asarray(cand, dtype=np.float32)
            k = cand.shape[0]
            col = np.zeros((k, 1), dtype=np.float32)
            m = min(k, iop_propose.shape[0])
            col[:m, 0] = iop_propose[:m]
            obs["cand_feats"] = np.concatenate([cand, col], axis=1)
        progress = np.asarray(obs.get("progress_feats", np.zeros(0)), dtype=np.float32).reshape(-1)
        obs["progress_feats"] = np.concatenate([progress, iop_respond.astype(np.float32)], axis=0)
        return obs

    # ------------------------------------------------------------------
    # build_state — packed obs + IOP augmentation
    # ------------------------------------------------------------------

    def build_state(self, state: Any, offer: Any = None) -> Dict[str, Any]:
        """Build the augmented packed obs for the discrete policy (C4 + IOP block)."""
        if isinstance(state, dict) and ("scalar_feats" in state or "progress_feats" in state):
            # Already-packed obs (trainer rebuild / rollout pass-through).
            return state
        if isinstance(state, np.ndarray):
            # A bare flat state basis (rollout value bootstrap) -> minimal obs.
            base = state.astype(np.float32)
            obs = self.pack_step_obs(self._raw_obs, base, self._cand_feats, self._cand_mask)
            return self._augment_obs(obs, base[: self.state_dim] if base.shape[0] >= self.state_dim else base)
        obs = self.build_step_obs(state, offer)
        state_basis = self._state_basis_for_iop(state, offer)
        return self._augment_obs(obs, state_basis)

    # ------------------------------------------------------------------
    # Sampling (discrete actor over augmented packed obs)
    # ------------------------------------------------------------------

    def _sample(self, obs: Dict[str, Any], trainer_key: str) -> Tuple[int, float, Optional[float]]:
        """Forward the phase actor / critic over a B=1 batch and sample."""
        trainer = self.trainers.get(trainer_key)
        if trainer is None:
            raise ValueError(f"No {trainer_key} attached. Use attach_trainer.")
        policy = trainer.models.get("policy", {}).get("model")
        if policy is None:
            raise ValueError(f"No 'policy' model in {trainer_key}.models.")
        value_entry = trainer.models.get("value")
        batch = obs_to_batch(obs, device=self.device)
        value: Optional[float] = None
        with torch.no_grad():
            logits = policy(batch)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if value_entry is not None:
                value = float(value_entry["model"](batch).view(-1)[0].item())
            if self.rollout_level is None:
                self._trace_actor_output = {
                    "trainer_key": trainer_key,
                    "logits": logits.view(-1),
                    "probs": torch.softmax(logits, dim=-1).view(-1),
                    "sampled_action": int(action.item()),
                    "log_prob": float(log_prob.item()),
                    "value": value,
                }
        return int(action.item()), float(log_prob.item()), value

    def _record_plumbing(self, obs: Dict[str, Any]) -> Any:
        """Per-step candidate reconstruction plumbing for the mode."""
        if self.mode == MODE_PADDED:
            return action_mask_from_obs(obs, self.k_max)
        return candidate_features_from_obs(obs)

    def sample_action_propose(self, obs: Dict[str, Any]) -> Tuple[int, float, Optional[float], Any]:
        """Propose sampling path: index into ``action_library`` (+ mask plumbing)."""
        action_idx, log_prob, value = self._sample(obs, "propose_trainer")
        return action_idx, log_prob, value, self._record_plumbing(obs)

    def sample_action_respond(self, obs: Dict[str, Any]) -> Tuple[int, float, Optional[float], Any]:
        """Respond sampling path: action in ``{0, 1}`` (no candidate plumbing)."""
        action_idx, log_prob, value = self._sample(obs, "respond_trainer")
        return action_idx, log_prob, value, None

    # ------------------------------------------------------------------
    # Recursive imagination + bayesian mixer (AS-IS)
    # ------------------------------------------------------------------

    def _warn_iop_failure(self, where: str, error: Exception) -> None:
        """Log kegagalan pipeline IOP sekali per (lokasi, jenis error) — FIX E-3.

        Kegagalan imajinasi tidak boleh mematikan alur negosiasi/pembelajaran,
        tetapi juga tidak boleh sunyi (dulu exception ditelan retry NegMAS dan
        meninggalkan agen dalam "mode zombie" tanpa jejak).
        """
        key = (type(self).__name__, where, type(error).__name__)
        seen = getattr(MBOMAgent, "_iop_failures_warned", None)
        if seen is None:
            seen = set()
            MBOMAgent._iop_failures_warned = seen
        if key not in seen:
            seen.add(key)
            print(f"[WARN] {key[0]}: IOP pipeline gagal di {where}: "
                  f"{type(error).__name__}: {error} (negosiasi tetap berjalan)")
        self._emit_iop_trace("error", where, error=f"{type(error).__name__}: {error}")

    def _run_recursive_imagination(self, engine, trainers_key: str, action_dim: int, state_basis: np.ndarray, relative_time: float) -> None:
        """Recursive imagination for one phase's IOP pipeline (AS-IS formulas).

        FIX E-3: seluruh loop dibungkus ``try/finally`` sehingga ``rollout_level``
        SELALU kembali ``None`` — dulu exception di tengah imajinasi meninggalkan
        ``rollout_level`` menggantung dan retry NegMAS berjalan dalam "mode
        zombie" (tanpa perekaman memory / payload episode).
        """
        iop_trainers = self.trainers.get(trainers_key)
        if iop_trainers is None:
            return
        try:
            self._run_recursive_imagination_inner(
                engine, trainers_key, action_dim, state_basis, relative_time, iop_trainers
            )
        finally:
            self.rollout_level = None

    def _run_recursive_imagination_inner(self, engine, trainers_key: str, action_dim: int, state_basis: np.ndarray, relative_time: float, iop_trainers) -> None:
        """Isi loop imajinasi rekursif (dipisah agar ``finally`` di atas rapi)."""
        for m in range(1, self.M):
            self.rollout_level = m - 1
            iop_trainers[m].copy_iop_policy_weight(
                iop_trainers[m - 1].models.get("policy").get("model")
            )

            def opponent_value_fn(sv: np.ndarray, _phase=trainers_key) -> float:
                key = "propose_trainer" if "propose" in _phase else "respond_trainer"
                trainer = self.trainers.get(key)
                if trainer is None:
                    return 0.0
                value_entry = trainer.models.get("value")
                if value_entry is None:
                    return 0.0
                aug = self.build_state(sv)
                batch = obs_to_batch(aug, device=self.device)
                with torch.no_grad():
                    v = value_entry["model"](batch).view(-1)[0]
                return -float(v.item())

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
            _loss = iop_trainers[m].update()
            iop_trainers[m].reset_step()
            self._emit_iop_trace(
                "update",
                "propose" if "propose" in trainers_key else "respond",
                level=m, state_basis=state_basis, action=best_response_idx,
                loss=_loss, source="recursive_imagination",
            )
        self.rollout_level = None

    def _update_bayesian_mixer(self, trainers_key: str, action_dim: int, mixer: BayesianMixer, state_basis: np.ndarray, real_action_idx: int) -> None:
        """Update one phase's Bayesian mixer from the real opponent action (AS-IS)."""
        iop_trainers = self.trainers.get(trainers_key)
        if iop_trainers is None:
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
    # Lifecycle
    # ------------------------------------------------------------------

    def on_negotiation_start(self, state: Any) -> None:
        """Reset memories at episode start (only at the top imagination level)."""
        if self.rollout_level is None:
            super().on_negotiation_start(state)
            self.opponent_state = None

    def respond(self, state: Any, source: Any = None) -> Any:
        """Respond flow (AS-IS): IOP pipeline + augmented-obs sample + record step."""
        offer = self._state_offer(state)

        if self.rollout_level is None:
            self._on_real_offer_received(offer)
            self.flush_pending_propose(state)
            self.flush_pending_respond(state)

        if offer is None:
            return ResponseType.REJECT_OFFER

        # IOP propose pipeline (we know the opponent's offer here).
        if self.rollout_level is None and self.opponent_state is not None:
            opponent_propose_idx = self._iop_offer_to_propose_idx(offer)
            state_vec = self._state_basis_for_iop(self.opponent_state)
            rt = self._relative_time(self.opponent_state)
            iop_propose = self.trainers.get("iop_trainers_propose")
            if iop_propose is not None:
                try:  
                    iop_propose[0].add_step(Step(state=state_vec, action=opponent_propose_idx))
                    _loss = iop_propose[0].update()
                    iop_propose[0].reset_step()
                    self._emit_iop_trace(
                        "update", "propose", level=0, state_basis=state_vec,
                        action=opponent_propose_idx, loss=_loss, source="real_offer",
                    )
                    self._run_recursive_imagination(
                        self.rollout_engine_propose, "iop_trainers_propose",
                        self.iop_propose_action_dim, state_vec, rt,
                    )
                    self._update_bayesian_mixer(
                        "iop_trainers_propose", self.iop_propose_action_dim,
                        self.bayesian_mixer_propose, state_vec, opponent_propose_idx,
                    )
                except Exception as error:
                    self._warn_iop_failure("respond/iop_propose", error)
            iop_respond = self.trainers.get("iop_trainers_respond")
            if iop_respond is not None:
                try:
                    state_vec = self._state_basis_for_iop(self.opponent_state)
                    iop_respond[0].add_step(Step(state=state_vec, action=1))
                    _loss = iop_respond[0].update()
                    iop_respond[0].reset_step()
                    self._emit_iop_trace(
                        "update", "respond", level=0, state_basis=state_vec,
                        action=1, loss=_loss, source="real_offer",
                    )
                    self._run_recursive_imagination(
                        self.rollout_engine_respond, "iop_trainers_respond",
                        self.iop_respond_action_dim, state_vec, rt,
                    )
                    self._update_bayesian_mixer(
                        "iop_trainers_respond", self.iop_respond_action_dim,
                        self.bayesian_mixer_respond, state_vec, 1,
                    )
                except Exception as error:
                    self._warn_iop_failure("respond/iop_respond", error)
            self.opponent_state = None

        obs = self.build_state(state, offer)
        action_id, log_prob, value, _ = self.sample_action_respond(obs)
        current_u = self._offer_utility(offer)
        if self.rollout_level is None:
            self.current_state_data_respond = {
                "state": obs, "action": action_id, "utility": current_u,
                "value": value, "log_prob": log_prob,
                "action_mask": None, "candidate_features": None,
            }
        self._emit_trace_step(
            "respond", state, obs, action_id,
            extra={"decision": "accept" if action_id == RESPOND_ACCEPT else "reject",
                   "utility": current_u},
        )
        if action_id == RESPOND_ACCEPT:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: Any, dest: Any = None) -> Any:
        """Propose flow (AS-IS): augmented-obs sample -> ``(idx,)`` + record step."""
        offer_context = self._state_offer(state)
        obs = self.build_state(state, offer_context)
        action_id, log_prob, value, mask = self.sample_action_propose(obs)

        offer = self.make_offer(action_id)
        current_u = self._offer_utility(offer)
        if self.rollout_level is None:
            self.current_state_data_propose = {
                "state": obs, "action": action_id, "utility": current_u,
                "value": value, "log_prob": log_prob,
                "action_mask": mask if self.mode == MODE_PADDED else None,
                "candidate_features": mask if self.mode == MODE_ENCODER else None,
            }
        self._emit_trace_step(
            "propose", state, obs, action_id,
            extra={"offer": offer, "utility": current_u},
        )
        self.opponent_state = self._record_opponent_state_after_propose(state, offer)
        return offer

    def on_negotiation_end(self, state: Any) -> None:
        """IOP respond pipeline for the final accept + flush + store two payloads."""
        if self.rollout_level is not None:
            return

        if self.opponent_state is not None:
            try:  # FIX E-3: payload episode HARUS tetap tersimpan di bawah.
                iop_respond = self.trainers.get("iop_trainers_respond")
                state_vec = self._state_basis_for_iop(self.opponent_state)
                rt = self._relative_time(self.opponent_state)
                if iop_respond is not None:
                    iop_respond[0].add_step(Step(state=state_vec, action=0))
                    _loss = iop_respond[0].update()
                    iop_respond[0].reset_step()
                    self._emit_iop_trace(
                        "update", "respond", level=0, state_basis=state_vec,
                        action=0, loss=_loss, source="final_accept",
                    )
                    self._run_recursive_imagination(
                        self.rollout_engine_respond, "iop_trainers_respond",
                        self.iop_respond_action_dim, state_vec, rt,
                    )
                    self._update_bayesian_mixer(
                        "iop_trainers_respond", self.iop_respond_action_dim,
                        self.bayesian_mixer_respond, state_vec, 0,
                    )
            except Exception as error:
                self._warn_iop_failure("on_negotiation_end/iop_respond", error)
            finally:
                self.opponent_state = None

        self.flush_pending_propose(state)
        self.flush_pending_respond(state)
        agreement = getattr(state, "agreement", None)
        self._store_payloads(agreement)

    # ------------------------------------------------------------------
    # Episode payloads
    # ------------------------------------------------------------------

    def _build_payload(self, memory: List[Step], agreement: Any, phase: str) -> Episode:
        """Build a finalized :class:`Episode` from a phase memory (+ MBOM meta)."""
        steps = deepcopy(memory)
        mixer = self.bayesian_mixer_propose if phase == "propose" else self.bayesian_mixer_respond
        meta = {
            "agent_name": self.name,
            "mbom_M": self.M,
            "mbom_rollout_k": self.rollout_k,
            f"bayesian_alpha_{phase}": mixer.get_alpha().tolist(),
        }
        episode = Episode(
            agent_name=self.name,
            phase=phase,
            terminal_utility=self._offer_utility(agreement) if agreement is not None else -1.0,
            agreement=agreement,
            steps=steps,
            meta=meta,
        )
        return episode.finalize()

    def build_episode_payload_propose(self, agreement: Any) -> Episode:
        """Build the propose-phase :class:`Episode`."""
        return self._build_payload(self.memory_propose, agreement, "propose")

    def build_episode_payload_respond(self, agreement: Any) -> Episode:
        """Build the respond-phase :class:`Episode`."""
        return self._build_payload(self.memory_respond, agreement, "respond")

    # ------------------------------------------------------------------
    # Reset / clone
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Clear memories + pending + opponent state."""
        super().reset_episode()
        self.opponent_state = None

    def clone_for_rollout(self) -> "MBOMAgent":
        """Return a clean clone sharing trainers/context for imagined rollouts (AS-IS)."""
        clone = self.__class__(
            name=self.name + "_rollout_clone",
            ufun=self.ufun,
            domain_spec=self.domain_spec,
            state_dim=self.state_dim,
            M=self.M,
            rollout_k=self.rollout_k,
            gamma=self.gamma,
            iop_batch_size=self.iop_batch_size,
            iop_update_steps=self.iop_update_steps,
            n_rollout_samples=self.n_rollout_samples,
            max_steps=self.max_steps,
            mode=self.mode,
            k_max=self.k_max,
        )
        clone.trainers = self.trainers
        clone.rollout_level = self.rollout_level
        clone._raw_obs = self._raw_obs
        clone._cand_feats = self._cand_feats
        clone._cand_mask = self._cand_mask
        clone._world_ufun = self._world_ufun
        clone.action_library = self.action_library
        clone.utility_min = self.utility_min
        clone.utility_max = self.utility_max
        clone.memory_propose = []
        clone.memory_respond = []
        clone.current_state_data_propose = None
        clone.current_state_data_respond = None
        clone.opponent_state = None
        return clone
