"""Base negotiator interface (``02-02``) — propose & respond FULLY separated.

This is the foundation shared by every agent in :mod:`negotiation.agents`. It
ports the AS-IS :class:`BaseNegotiator` from the paper code and adapts it to the
new architecture (``00-MULAI-DI-SINI`` rule #5 / Invariant #1):

* **Separate memories** — ``memory_propose`` / ``memory_respond`` (DLST adds a
  third, ``memory_threshold_utility``, in its own subclass).
* **Separate pending state** — ``current_state_data_propose`` /
  ``current_state_data_respond`` hold a step until its reward / next-state is
  known (flushed on the next move or at ``on_negotiation_end``).
* **Separate trainers** — ``trainers["propose_trainer"]`` /
  ``trainers["respond_trainer"]`` (DLST + ``threshold_trainer``); a single agent
  builds **two** episode payloads per negotiation (DLST: three) stored to
  separate trainers. **No state / model / trainer / memory is ever shared
  between propose and respond.**
* **Index-direct decode** — :meth:`decode_action` returns ``action_library[idx]``
  (Gotcha G1; never ``idx - 1``).

New-architecture additions vs. the old ``BaseNegotiator``:

* :meth:`configure_negotiation` (Contract C3) injects the bridge-built world
  context (``raw_obs``, ``cand_feats``, ``cand_mask``, ``action_library``,
  ``world_ufun``, ``utility_min`` / ``utility_max``) **before** each
  ``NegMASBilateralEnv.run``. Without it the agent has no world context.
* :meth:`pack_step_obs` (Contract C4) builds the per-step packed observation dict
  ``{taxi/port/area_feats(+mask), scalar_feats, progress_feats, cand_feats,
  cand_mask}`` from the injected ``raw_obs`` plus a per-step progress vector. This
  dict is what the mode-aware networks consume and what every :class:`Step`
  stores in ``state`` so the action distribution can be reconstructed identically
  at update time (``03-05``).
* Utility helpers operate on the **NegMAS bundle-index outcome** ``(idx,)`` via
  the injected ``world_ufun`` (``BundleUFun`` already evaluates ``(idx,)``); there
  is no static ``LinearAdditiveUtilityFunction`` any more (``02-02`` final note).

The ``negotiation`` package depends only on torch / numpy / negmas + other
``negotiation`` modules (never ``simulation.world``).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

try:  # negmas is always present at runtime; keep parse resilient.
    from negmas import ResponseType, SAONegotiator
except Exception:  # pragma: no cover
    ResponseType = None  # type: ignore
    SAONegotiator = object  # type: ignore

from negotiation.data_type.step import Step
from negotiation.trace_logger import get_logger as _get_trace_logger


# Respond action space (mode-independent): 0 = accept, 1 = reject.
RESPOND_ACCEPT: int = 0
RESPOND_REJECT: int = 1


# =============================================================================
# Interface (AS-IS ABC)
# =============================================================================

class NegotiatorFunctionality(ABC):
    """Abstract contract every negotiator must implement (``02-02``).

    Mirrors the AS-IS interface; the two phases (propose / respond) have their
    own sampling path and their own episode payload builder.
    """

    @abstractmethod
    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """Attach trainers per role (``propose_trainer`` / ``respond_trainer`` ...)."""

    @abstractmethod
    def build_state(self, state: Any) -> Any:
        """Build the network input (packed obs dict / flat vector) for a step."""

    @abstractmethod
    def on_negotiation_start(self, state: Any) -> None:
        """Reset per-episode memory at the start of a negotiation."""

    @abstractmethod
    def on_negotiation_end(self, state: Any) -> None:
        """Flush pending steps and store the (two/three) episode payloads."""

    @abstractmethod
    def respond(self, state: Any, source: Any = None) -> Any:
        """Return a NegMAS :class:`ResponseType` (0 = accept, 1 = reject space)."""

    @abstractmethod
    def propose(self, state: Any) -> Any:
        """Return a NegMAS outcome ``(idx,)`` chosen by the propose policy."""

    @abstractmethod
    def build_episode_payload_propose(self, agreement: Any) -> Any:
        """Build the propose-phase :class:`Episode` from ``memory_propose``."""

    @abstractmethod
    def build_episode_payload_respond(self, agreement: Any) -> Any:
        """Build the respond-phase :class:`Episode` from ``memory_respond``."""

    @abstractmethod
    def reset_episode(self) -> None:
        """Clear both memories and pending step buffers."""

    @abstractmethod
    def sample_action_propose(self, obs: Any) -> Any:
        """Propose-phase sampling path: ``(action_idx, log_prob, value, mask)``."""

    @abstractmethod
    def sample_action_respond(self, obs: Any) -> Any:
        """Respond-phase sampling path: ``(action_idx, log_prob, value, mask)``."""

    @abstractmethod
    def decode_action(self, action_id: int) -> Any:
        """Index -> bundle, ``action_library[idx]`` (index-direct, G1)."""

    @abstractmethod
    def _offer_to_action_idx(self, offer: Any) -> int:
        """Map a NegMAS outcome back to a library index (0 if unknown)."""

    @abstractmethod
    def _offer_utility(self, offer: Any) -> float:
        """World utility of an outcome from this agent's viewpoint."""

    @abstractmethod
    def _mask_logits(self, logits: Any, allow_accept: bool) -> Any:
        """Optionally mask the accept logit (respond head helper)."""


# =============================================================================
# BaseNegotiator
# =============================================================================

class BaseNegotiator(SAONegotiator, NegotiatorFunctionality):  # type: ignore[misc]
    """Base class for every project negotiator (propose / respond separated).

    Parameters
    ----------
    name:
        Negotiator name.
    ufun:
        NegMAS utility function attached to this side (a ``BundleUFun`` wrapping
        the bridge-built ``world_ufun``); evaluated on outcomes ``(idx,)``.
    domain_spec:
        The :class:`~negotiation.environments.negmas_env.DomainSpec` (or a dict)
        carrying ``action_library`` and the per-side utility bounds.
    trainers:
        Optional initial ``{role: trainer}`` mapping.

    Example:
        >>> # Construction validated structurally (needs a NegMAS ufun to run).
        >>> BaseNegotiator.__name__
        'BaseNegotiator'
    """

    def __init__(
        self,
        name: str,
        ufun: Any,
        domain_spec: Any,
        trainers: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, ufun=ufun, **kwargs)
        self.name = name
        self.domain_spec = domain_spec
        self.ufun = ufun

        # ── Separated memories / pending step buffers (Invariant #1). ──
        self.memory_propose: List[Step] = []
        self.memory_respond: List[Step] = []
        self.current_state_data_propose: Optional[Dict[str, Any]] = None
        self.current_state_data_respond: Optional[Dict[str, Any]] = None

        self.trainers: Dict[str, Any] = trainers if trainers is not None else {}

        # ── Per-side world context injected by the bridge (Contract C3). ──
        self.action_library: List[Any] = self._domain_attr("action_library", []) or []
        self.utility_min: float = float(self._domain_attr("utility_min_self", 0.0) or 0.0)
        self.utility_max: float = float(self._domain_attr("utility_max_self", 1.0) or 1.0)
        self._world_ufun: Any = None
        self._raw_obs: Optional[Dict[str, Any]] = None
        self._cand_feats: Any = None
        self._cand_mask: Any = None

        # ── Validation trace: last raw actor output captured during sampling. ──
        # Set by the concrete agent's ``_sample`` / ``_sample_actor`` so that
        # ``_emit_trace_step`` can attach it to the step record. Cleared after use.
        self._trace_actor_output: Any = None

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------

    def _domain_attr(self, key: str, default: Any = None) -> Any:
        """Read ``key`` from ``domain_spec`` (dict or object)."""
        if isinstance(self.domain_spec, dict):
            return self.domain_spec.get(key, default)
        return getattr(self.domain_spec, key, default)

    # ------------------------------------------------------------------
    # Contract C3 — world-context injection (called BEFORE run)
    # ------------------------------------------------------------------

    def configure_negotiation(
        self,
        raw_obs: Dict[str, Any],
        cand_feats: Any,
        cand_mask: Any,
        action_library: List[Any],
        world_ufun: Any,
        utility_min: float,
        utility_max: float,
    ) -> None:
        """Inject the bridge-built world context for this negotiation (C3).

        Stores the per-side ``RawObservation`` blocks (``raw_obs``), the
        per-candidate feature matrix / mask, the candidate ``action_library``,
        the world utility callable (``Bundle -> float``), and the per-negotiation
        reward-scaling bounds. Must be called for each side before
        ``NegMASBilateralEnv.run`` (without it the agent has no world context).

        Example:
            >>> # Stores fields; exercised end-to-end by the bridge / smoke test.
            >>> hasattr(BaseNegotiator, "configure_negotiation")
            True
        """
        self._raw_obs = raw_obs
        self._cand_feats = cand_feats
        self._cand_mask = cand_mask
        self.action_library = list(action_library)
        self._world_ufun = world_ufun
        self.utility_min = float(utility_min)
        self.utility_max = float(utility_max)

    # ------------------------------------------------------------------
    # Contract C4 — per-step packed observation
    # ------------------------------------------------------------------

    def progress_vector(self, state: Any, offer: Any) -> np.ndarray:
        """Return the negotiation-progress block (``03-03``) for one step.

        Block (always present, both sides): ``[current_u_norm, relative_time,
        has_offer]`` — the AS-IS dim-3 basis, kept and now fused with the world
        context via the packed obs dict (``02-03`` adaptation note).

        Example:
            >>> import numpy as np
            >>> # Pure helper over scalars; values come from the live state.
            >>> isinstance(np.zeros(3), np.ndarray)
            True
        """
        relative_time = self._relative_time(state)
        if offer is None:
            offer = self._state_offer(state)
        current_u = self._offer_utility(offer)
        current_norm = self._normalize_utility(current_u)
        has_offer = 1.0 if offer is not None else 0.0
        return np.array([current_norm, relative_time, has_offer], dtype=np.float32)

    def pack_step_obs(
        self,
        raw_obs: Optional[Dict[str, Any]],
        progress_feats: np.ndarray,
        cand_feats: Any,
        cand_mask: Any,
    ) -> Dict[str, Any]:
        """Assemble one packed observation dict for the mode-aware networks (C4).

        Combines the injected ``raw_obs`` set/scalar blocks with a per-step
        ``progress_feats`` block and the candidate feature matrix / mask. The
        returned dict is stored in :attr:`Step.state` so it can be re-collated and
        re-forwarded identically at update time (``03-05``).

        Example:
            >>> import numpy as np
            >>> obs = BaseNegotiator.pack_step_obs(
            ...     object.__new__(BaseNegotiator),
            ...     {"scalar_feats": np.zeros(2)}, np.zeros(3),
            ...     np.zeros((4, 5)), np.ones(4))
            >>> sorted(obs)
            ['cand_feats', 'cand_mask', 'progress_feats', 'scalar_feats']
        """
        obs: Dict[str, Any] = {}
        if raw_obs:
            for key, value in raw_obs.items():
                obs[key] = value
        obs["progress_feats"] = np.asarray(progress_feats, dtype=np.float32)
        if cand_feats is not None:
            obs["cand_feats"] = np.asarray(cand_feats, dtype=np.float32)
        if cand_mask is not None:
            obs["cand_mask"] = np.asarray(cand_mask, dtype=np.float32)
        return obs

    def build_step_obs(self, state: Any, offer: Any) -> Dict[str, Any]:
        """Build the packed obs for the current step from the injected context.

        Example:
            >>> hasattr(BaseNegotiator, "build_step_obs")
            True
        """
        progress = self.progress_vector(state, offer)
        return self.pack_step_obs(
            self._raw_obs, progress, self._cand_feats, self._cand_mask
        )

    # ------------------------------------------------------------------
    # Validation trace (no-op unless the global TraceLogger is configured)
    # ------------------------------------------------------------------

    def _trace_state_info(self, state: Any) -> Dict[str, Any]:
        """Summarize the NegMAS state received in ``propose`` / ``respond``.

        Captures the few fields that describe one negotiation step: the step
        index, relative time, the current offer (as a library index too), and the
        agreement if the negotiation has concluded.
        """
        offer = self._state_offer(state)
        return {
            "step": getattr(state, "step", None) if not isinstance(state, dict)
            else state.get("step"),
            "relative_time": self._relative_time(state),
            "current_offer": offer,
            "current_offer_idx": self._offer_to_action_idx(offer)
            if offer is not None else None,
            "agreement": getattr(state, "agreement", None) if not isinstance(state, dict)
            else state.get("agreement"),
        }

    # ------------------------------------------------------------------
    # Inference-time instrumentation (fleet-side metric)
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs) -> None:
        """Wrap subclass ``propose``/``respond`` with a wall-clock timer.

        Only methods defined in the subclass' own ``__dict__`` are wrapped
        (inherited ones are already wrapped on the parent), guarded by the
        ``_inf_timed`` marker against double-wrapping. Imagined (rollout) calls
        are excluded — hanya langkah negosiasi NYATA yang diukur.
        """
        super().__init_subclass__(**kwargs)
        import functools
        import time as _time
        for _kind in ("propose", "respond"):
            fn = cls.__dict__.get(_kind)
            if fn is None or getattr(fn, "_inf_timed", False):
                continue

            def _make(fn, _kind):
                @functools.wraps(fn)
                def timed(self, *args, **kw):
                    if getattr(self, "rollout_level", None) is not None:
                        return fn(self, *args, **kw)
                    self._inference_t0 = _time.perf_counter()
                    self._inference_kind = _kind
                    out = fn(self, *args, **kw)
                    ms = (_time.perf_counter() - self._inference_t0) * 1000.0
                    log = getattr(self, "_inference_log", None)
                    if log is None:
                        log = {"propose": [], "respond": []}
                        self._inference_log = log
                    log[_kind].append(ms)
                    return out
                timed._inf_timed = True
                return timed

            setattr(cls, _kind, _make(fn, _kind))

    def inference_summary(self) -> Dict[str, Any]:
        """Ringkasan inference time agen ini (ms): avg per kind + overall + n."""
        log = getattr(self, "_inference_log", None) or {"propose": [], "respond": []}
        p, r = log["propose"], log["respond"]
        allv = p + r
        avg = lambda v: (sum(v) / len(v)) if v else None
        return {
            "avg_inference_ms_propose": avg(p),
            "avg_inference_ms_respond": avg(r),
            "avg_inference_ms_overall": avg(allv),
            "n_inference_propose": len(p),
            "n_inference_respond": len(r),
        }

    def _emit_trace_step(
        self,
        kind: str,
        state: Any,
        obs: Any,
        action: Any,
        actor_output: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit one ``negotiation_step`` trace record for this acting agent.

        A cheap no-op while the global :class:`~negotiation.trace_logger.TraceLogger`
        is disabled. Imagined (rollout) moves — where ``rollout_level`` is set — are
        skipped so only the real negotiation steps are recorded. ``obs`` is the
        exact observation fed to the network (a dict for the discrete agents, flat
        vectors for DLST); ``actor_output`` defaults to the buffer captured by the
        agent's sampler.
        """
        logger = _get_trace_logger()
        if not logger.is_enabled():
            return
        if getattr(self, "rollout_level", None) is not None:
            return
        # inference_ms: waktu sejak entry propose/respond nyata terakhir (fleet saja).
        inference_ms = None
        if getattr(self, "name", None) == "agent_fleet" and \
           getattr(self, "_inference_t0", None) is not None and \
           getattr(self, "_inference_kind", None) == kind:
            import time as _time
            inference_ms = round((_time.perf_counter() - self._inference_t0) * 1000.0, 3)
        payload: Dict[str, Any] = {
            "agent_name": getattr(self, "name", None),
            "agent_type": type(self).__name__,
            "side": getattr(self, "side", None),
            "kind": kind,
            "inference_ms": inference_ms,
            "negmas_state": self._trace_state_info(state),
            "obs": obs,
            "action": action,
            "actor_output": actor_output if actor_output is not None
            else getattr(self, "_trace_actor_output", None),
        }
        if extra:
            payload.update(extra)
        logger.emit("negotiation_step", **payload)
        self._trace_actor_output = None

    # ------------------------------------------------------------------
    # State / offer reading helpers (SAOState or dict)
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_time(state: Any) -> float:
        """Return ``relative_time`` from a SAOState / dict (0.0 fallback)."""
        if isinstance(state, dict):
            return float(state.get("relative_time", 0.0))
        return float(getattr(state, "relative_time", 0.0))

    @staticmethod
    def _state_offer(state: Any) -> Any:
        """Return ``current_offer`` from a SAOState / dict (None fallback)."""
        if isinstance(state, dict):
            return state.get("current_offer", None)
        return getattr(state, "current_offer", None)

    @staticmethod
    def state_to_dict(state: Any) -> Dict[str, Any]:
        """Serialize the few SAOState fields used downstream into a plain dict.

        Replaces the old ``object_to_dict``; only the fields the reward shaper /
        DLST opponent model read are kept (``current_offer`` / ``relative_time`` /
        ``agreement``).

        Example:
            >>> class S:  # minimal SAOState stand-in
            ...     current_offer = (1,)
            ...     relative_time = 0.5
            ...     agreement = None
            >>> BaseNegotiator.state_to_dict(S())
            {'current_offer': (1,), 'relative_time': 0.5, 'agreement': None}
        """
        if isinstance(state, dict):
            return dict(state)
        return {
            "current_offer": getattr(state, "current_offer", None),
            "relative_time": float(getattr(state, "relative_time", 0.0)),
            "agreement": getattr(state, "agreement", None),
        }

    # ------------------------------------------------------------------
    # Utility helpers (operate on the bundle-index outcome (idx,))
    # ------------------------------------------------------------------

    def _normalize_utility(self, u: float) -> float:
        """Min-max scale a utility with the per-negotiation bounds (C6/G10)."""
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    def _offer_utility(self, offer: Any) -> float:
        """World utility of a NegMAS outcome ``(idx,)`` / dict / int / None.

        Uses the attached ``ufun`` (a ``BundleUFun`` evaluating ``(idx,)``) when
        present, else the injected ``world_ufun`` over the decoded bundle.

        Example:
            >>> a = object.__new__(BaseNegotiator)
            >>> a.ufun = lambda o: 0.7
            >>> a._world_ufun = None
            >>> a.action_library = ["x"]
            >>> round(a._offer_utility((0,)), 3)
            0.7
        """
        if offer is None:
            return 0.0
        ufun = getattr(self, "ufun", None)
        if ufun is not None:
            try:
                return float(ufun(offer))
            except Exception:
                pass
        if self._world_ufun is not None:
            idx = self._offer_to_action_idx(offer)
            if 0 <= idx < len(self.action_library):
                return float(self._world_ufun(self.action_library[idx]))
        return 0.0

    def _bundle_utility(self, bundle: Any) -> float:
        """World utility of a RAW library bundle (bukan outcome NegMAS ``(idx,)``).

        Rollout MBOM memanggil estimator utility dengan elemen ``action_library``
        mentah (objek Bundle); jalur :meth:`_offer_utility` dirancang untuk outcome
        ``(idx,)`` dan akan salah rute (fallback indeks 0) bila diberi Bundle.
        Helper ini mengevaluasi bundle mentah langsung lewat ``world_ufun``.

        Example:
            >>> a = object.__new__(BaseNegotiator)
            >>> a.ufun = None; a._world_ufun = lambda b: 0.4; a.action_library = []
            >>> class B: pass
            >>> round(a._bundle_utility(B()), 2)
            0.4
        """
        if bundle is None:
            return 0.0
        if isinstance(bundle, (tuple, list, dict, int)):
            return self._offer_utility(bundle)
        if self._world_ufun is not None:
            try:
                return float(self._world_ufun(bundle))
            except Exception:
                return 0.0
        return 0.0

    def _bundle_to_outcome(self, offer: Any) -> Any:
        """Map a raw library bundle back to the NegMAS outcome ``(idx,)`` bila perlu.

        Outcome tuple / dict / int diteruskan apa adanya; objek bundle mentah
        dicari indeksnya di ``action_library`` (identity dulu, lalu equality).

        Example:
            >>> a = object.__new__(BaseNegotiator)
            >>> a.action_library = ["x", "y"]
            >>> a._bundle_to_outcome("y")
            (1,)
            >>> a._bundle_to_outcome((0,))
            (0,)
        """
        if offer is None or isinstance(offer, (tuple, list, dict, int)):
            return offer
        for i, b in enumerate(self.action_library):
            if b is offer:
                return (i,)
        try:
            return (self.action_library.index(offer),)
        except (ValueError, TypeError):
            return offer

    # ------------------------------------------------------------------
    # decode / encode action (index-direct, G1)
    # ------------------------------------------------------------------

    def decode_action(self, action_id: int) -> Any:
        """Return ``action_library[idx]`` directly (G1 — never ``idx - 1``).

        Example:
            >>> a = object.__new__(BaseNegotiator)
            >>> a.action_library = ["a", "b", "c"]
            >>> a.decode_action(2)
            'c'
        """
        if not self.action_library:
            return None
        idx = min(int(action_id), len(self.action_library) - 1)
        return self.action_library[idx]

    def _offer_to_action_idx(self, offer: Any) -> int:
        """Map a NegMAS outcome ``(idx,)`` / dict / int back to a library index.

        Example:
            >>> a = object.__new__(BaseNegotiator)
            >>> a.action_library = ["a", "b"]
            >>> a._offer_to_action_idx((1,))
            1
            >>> a._offer_to_action_idx(None)
            0
        """
        if offer is None:
            return 0
        if isinstance(offer, dict):
            if not offer:
                return 0
            return int(next(iter(offer.values())))
        if isinstance(offer, (tuple, list)):
            if len(offer) == 0:
                return 0
            return int(offer[0])
        try:
            return int(offer)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def make_offer(idx: int) -> tuple:
        """Wrap a library index into a NegMAS single-issue outcome ``(idx,)``.

        Example:
            >>> BaseNegotiator.make_offer(3)
            (3,)
        """
        return (int(idx),)

    def _mask_logits(self, logits: Any, allow_accept: bool) -> Any:
        """Set the accept logit to ``-1e9`` when accept is disallowed (AS-IS)."""
        if allow_accept:
            return logits
        masked = logits.clone()
        masked[..., RESPOND_ACCEPT] = -1e9
        return masked

    # ------------------------------------------------------------------
    # Trainer attach (default; overridden by some agents)
    # ------------------------------------------------------------------

    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """Attach trainers per role into ``self.trainers``.

        Example:
            >>> a = object.__new__(BaseNegotiator)
            >>> a.trainers = {}
            >>> a.attach_trainer({"propose_trainer": "P", "respond_trainer": "R"})
            >>> sorted(a.trainers)
            ['propose_trainer', 'respond_trainer']
        """
        for key, trainer in dict_of_trainer.items():
            self.trainers[key] = trainer

    # ------------------------------------------------------------------
    # Memory recording / flush
    # ------------------------------------------------------------------

    def record_step_propose(self, **fields: Any) -> None:
        """Append one propose :class:`Step` to ``memory_propose``."""
        self.memory_propose.append(Step(**fields))

    def record_step_respond(self, **fields: Any) -> None:
        """Append one respond :class:`Step` to ``memory_respond``."""
        self.memory_respond.append(Step(**fields))

    def _flush_pending(self, pending: Optional[Dict[str, Any]], record_fn, state: Any) -> None:
        """Flush one pending step dict via ``record_fn`` with the next state."""
        if pending:
            next_state = self.state_to_dict(state)
            record_fn(
                state=pending.get("state"),
                action=pending.get("action"),
                utility=pending.get("utility"),
                value=pending.get("value"),
                q_value=pending.get("q_value"),
                log_prob=pending.get("log_prob"),
                next_state=next_state,
                action_mask=pending.get("action_mask"),
                candidate_features=pending.get("candidate_features"),
            )

    def flush_pending_propose(self, state: Any) -> None:
        """Flush the pending propose step (if any) into ``memory_propose``."""
        self._flush_pending(self.current_state_data_propose, self.record_step_propose, state)
        self.current_state_data_propose = None

    def flush_pending_respond(self, state: Any) -> None:
        """Flush the pending respond step (if any) into ``memory_respond``."""
        self._flush_pending(self.current_state_data_respond, self.record_step_respond, state)
        self.current_state_data_respond = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_negotiation_start(self, state: Any) -> None:
        """Reset both memories and pending buffers at episode start (AS-IS)."""
        self.memory_propose = []
        self.memory_respond = []
        self.current_state_data_propose = None
        self.current_state_data_respond = None

    def reset_episode(self) -> None:
        """Clear both memories and pending step buffers."""
        self.memory_propose = []
        self.memory_respond = []
        self.current_state_data_propose = None
        self.current_state_data_respond = None

    def _store_payloads(self, agreement: Any) -> None:
        """Build the two episode payloads and store to the separate trainers.

        Subclasses with a third phase (DLST) override / extend this.
        """
        if not self.trainers:
            return
        payload_propose = self.build_episode_payload_propose(agreement)
        payload_respond = self.build_episode_payload_respond(agreement)
        propose_trainer = self.trainers.get("propose_trainer")
        respond_trainer = self.trainers.get("respond_trainer")
        if propose_trainer is not None:
            propose_trainer.store_episode(payload_propose)
        if respond_trainer is not None:
            respond_trainer.store_episode(payload_respond)

    def on_negotiation_end(self, state: Any) -> None:
        """Flush pending steps, build the two payloads, store to trainers.

        DLST overrides this to also flush / store the threshold phase (three
        payloads). Subclasses keep propose and respond fully separate.
        """
        self.flush_pending_propose(state)
        self.flush_pending_respond(state)
        agreement = getattr(state, "agreement", None)
        self._store_payloads(agreement)

    # ------------------------------------------------------------------
    # Abstract methods that concrete agents must define
    # ------------------------------------------------------------------

    def build_state(self, state: Any) -> Any:  # pragma: no cover - overridden
        """Build the network input for a step (overridden per agent)."""
        raise NotImplementedError

    def sample_action_propose(self, obs: Any) -> Any:  # pragma: no cover
        """Propose-phase sampling path (overridden per agent)."""
        raise NotImplementedError

    def sample_action_respond(self, obs: Any) -> Any:  # pragma: no cover
        """Respond-phase sampling path (overridden per agent)."""
        raise NotImplementedError

    def respond(self, state: Any, source: Any = None) -> Any:  # pragma: no cover
        """Return a NegMAS ResponseType (overridden per agent)."""
        raise NotImplementedError

    def propose(self, state: Any) -> Any:  # pragma: no cover
        """Return a NegMAS outcome ``(idx,)`` (overridden per agent)."""
        raise NotImplementedError

    def build_episode_payload_propose(self, agreement: Any) -> Any:  # pragma: no cover
        """Build the propose-phase episode payload (overridden per agent)."""
        raise NotImplementedError

    def build_episode_payload_respond(self, agreement: Any) -> Any:  # pragma: no cover
        """Build the respond-phase episode payload (overridden per agent)."""
        raise NotImplementedError
