"""Training orchestration: on-the-fly + offline, concurrency policy (``06-01``).

:class:`Orchestrator` owns the two negotiators (fleet + SPKLU) for a run and the
training *policy* layered on top of their per-phase trainers. It is the Tahap H
piece that plugs into the already-built scheduler **without changing it**:

* it builds (or loads-from-registry) both sides' agents via
  :func:`negotiation.agents.agent_factory.build_agent` (selection is fixed at
  init and never changes mid-run, G14 / ``06-01``);
* its :meth:`on_train_tick` (alias :meth:`train`) matches the scheduler's
  ``training_callback`` seam (``callable(scheduler) -> None``) and applies the
  configured **concurrency** policy on each on-the-fly trigger;
* it can also train **offline** from a saved one-day history payload
  (``training.history_path``) with no world running — checking the history's
  signature against each agent's signature first (Contract C7 / ``06-02``).

Concurrency (``training.training_concurrency``, ``06-01``):

* ``SIMULTANEOUS`` — self-play: both sides' trainers update every trigger.
* ``ALTERNATING`` — freeze one side while the other trains; the trainable side
  flips each trigger window (mitigates non-stationarity).
* ``FIXED_OPPONENT`` — only the designated learner side trains; the opponent is a
  fixed (typically baseline) non-learner used purely as a sparring partner.

Trainer update semantics mirror the scheduler's default on-the-fly path: each
eligible trainer ``update()``s then ``reset_episodes()`` (no-op when no episodes
were stored that window).

This module imports :mod:`simulation.config`,
:mod:`simulation.registry`, and :mod:`negotiation.agents` /
:mod:`negotiation.trainers`. It does **not** import the world or the bridge
(agents are run by the scheduler).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from negotiation.training_logger import (
    get_offline_logger,
    get_online_logger,
)
from negotiation.trace_logger import to_jsonable
from simulation.metrics.loss_history import (
    PHASE_PROPOSE,
    PHASE_RESPOND,
    PHASE_THRESHOLD,
)
from simulation.registry.agent_registry import AgentRegistry
from simulation.registry.signature import (
    Signature,
    SignatureMismatchError,
    build_signature,
)

# Concurrency mode names (``06-01``).
CONCURRENCY_SIMULTANEOUS: str = "SIMULTANEOUS"
CONCURRENCY_ALTERNATING: str = "ALTERNATING"
CONCURRENCY_FIXED_OPPONENT: str = "FIXED_OPPONENT"
_VALID_CONCURRENCY: Tuple[str, ...] = (
    CONCURRENCY_SIMULTANEOUS,
    CONCURRENCY_ALTERNATING,
    CONCURRENCY_FIXED_OPPONENT,
)
_SIDES: Tuple[str, ...] = ("fleet", "spklu")
# History file schema marker.
_HISTORY_SCHEMA_KEY: str = "schema"
_HISTORY_SIGNATURE_KEY: str = "signatures"
_HISTORY_EPISODES_KEY: str = "episodes"

# Priority order of loss keys looked up in ``trainer.update()``'s returned dict
# (LLR-M4.1). The concrete trainers report (PG/IOP) ``loss``, (PPO)
# ``policy_loss``/``value_loss``/``entropy_loss``, (DDPG)
# ``actor_loss``/``critic_loss``/``steps``; this order picks the most
# representative scalar without computing anything new (INV-3).
_LOSS_KEY_PRIORITY: Tuple[str, ...] = (
    "loss",
    "total_loss",
    "policy_loss",
    "actor_loss",
    "value_loss",
)

# Map an agent's ``trainers`` role key to a loss phase name (LLR-M4.2). Only the
# scalar per-phase trainers are tracked for loss-per-phase; the MBOM/IOP list
# roles (``iop_trainers_*``) intentionally have no entry and yield ``None``.
_TRAINER_ROLE_TO_PHASE: Dict[str, str] = {
    "propose_trainer": PHASE_PROPOSE,
    "respond_trainer": PHASE_RESPOND,
    "threshold_trainer": PHASE_THRESHOLD,
}


def _is_finite_number(value: Any) -> bool:
    """Return ``True`` when ``value`` is a finite real number (not None/NaN/inf).

    Example:
        >>> _is_finite_number(0.5), _is_finite_number(None), _is_finite_number(float("nan"))
        (True, False, False)
    """

    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _extract_loss(result: Any, trainer: Any) -> Optional[float]:
    """Extract a single scalar loss from a ``trainer.update()`` result (LLR-M4.1).

    The loss is only **read** from what the trainer already returns/exposes — no
    new loss computation is added (INV-3). Resolution order:

    * ``dict`` -> the first key in :data:`_LOSS_KEY_PRIORITY` whose value is a
      finite float;
    * ``int`` / ``float`` -> used directly when finite;
    * otherwise -> ``None``.

    As a fallback, when the result yields nothing usable, a finite ``last_loss``
    attribute on ``trainer`` (if any) is used. ``None`` means "no loss to record"
    (a valid empty series per KR-8).

    Example:
        >>> class T: pass
        >>> _extract_loss({"loss": 0.5, "steps": 4}, T())
        0.5
        >>> _extract_loss({"policy_loss": 0.2}, T())
        0.2
        >>> _extract_loss({"steps": 0}, T()) is None
        True
        >>> _extract_loss(0.7, T())
        0.7
    """

    if isinstance(result, dict):
        for key in _LOSS_KEY_PRIORITY:
            if key in result and _is_finite_number(result[key]):
                return float(result[key])
    elif _is_finite_number(result):
        return float(result)

    last_loss = getattr(trainer, "last_loss", None)
    if _is_finite_number(last_loss):
        return float(last_loss)
    return None


def _update_and_capture(trainer: Any) -> Tuple[bool, Optional[float]]:
    """Run ``update()`` + ``reset_episodes()`` and capture the reported loss (LLR-M4.1).

    Behaves like :func:`_update_and_reset` (updates only when episodes exist,
    then resets), but additionally returns the scalar loss extracted from the
    update result via :func:`_extract_loss` — WITHOUT changing any loss formula
    (INV-3). The whole update is wrapped in ``try``/``except`` so a faulty
    trainer never tears down the run.

    Returns ``(ran, loss_or_None)``: ``ran`` is ``True`` only when an update
    actually executed; ``loss_or_None`` is ``None`` when the trainer exposes no
    usable loss (a valid empty series per KR-8).

    Example:
        >>> calls = []
        >>> class T:
        ...     episodes = ["e"]
        ...     def update(self): calls.append("u"); return {"loss": 0.5}
        ...     def reset_episodes(self): calls.append("r")
        >>> _update_and_capture(T()), calls
        ((True, 0.5), ['u', 'r'])
        >>> class Empty:
        ...     episodes = []
        >>> _update_and_capture(Empty())
        (False, None)
    """

    episodes = getattr(trainer, "episodes", None)
    if not episodes:
        return (False, None)
    update = getattr(trainer, "update", None)
    reset = getattr(trainer, "reset_episodes", None)
    try:
        result = update() if callable(update) else None
        loss = _extract_loss(result, trainer)
        if callable(reset):
            reset()
        return (True, loss)
    except Exception:
        return (False, None)


def _serialize_step(step: Any) -> Any:
    """Serialize one :class:`Step`-like object into a JSON-native structure.

    Uses ``vars()`` (the dataclass ``__dict__``) so every field — state, action,
    masks, log_prob, value, reward — is captured, then routes through
    :func:`negotiation.trace_logger.to_jsonable` (tensors/numpy/nan-safe).

    Example:
        >>> class S:
        ...     def __init__(self): self.action = 3; self.reward = 0.5
        >>> _serialize_step(S())
        {'action': 3, 'reward': 0.5}
    """

    if hasattr(step, "__dict__"):
        return to_jsonable(vars(step))
    return to_jsonable(step)


def _serialize_episode(episode: Any) -> Any:
    """Serialize one :class:`Episode`-like object (including its steps) fully.

    Example:
        >>> class E:
        ...     def __init__(self): self.phase = "propose"; self.steps = []
        >>> _serialize_episode(E())["phase"]
        'propose'
    """

    if hasattr(episode, "__dict__"):
        data = dict(vars(episode))
        steps = data.get("steps")
        if isinstance(steps, (list, tuple)):
            data["steps"] = [_serialize_step(step) for step in steps]
        return to_jsonable(data)
    return to_jsonable(episode)


def _memory_snapshot(trainer: Any) -> Dict[str, Any]:
    """Serialize a trainer's *full* memory contents for the training log.

    Reads the episode buffer (PG/PPO/DDPG ``trainer.episodes``) or the flat step
    memory (IOP ``trainer.memory``) and serializes every episode / step. Returns a
    small envelope carrying the kind, counts, and the full serialized payload.

    Example:
        >>> class T: episodes = []; memory = []
        >>> _memory_snapshot(T())["kind"]
        'empty'
    """

    episodes = getattr(trainer, "episodes", None)
    memory = getattr(trainer, "memory", None)
    if episodes:
        return {
            "kind": "episodes",
            "n_episodes": len(episodes),
            "n_steps": sum(len(getattr(ep, "steps", []) or []) for ep in episodes),
            "episodes": [_serialize_episode(ep) for ep in episodes],
        }
    if memory:
        return {
            "kind": "steps",
            "n_episodes": 0,
            "n_steps": len(memory),
            "steps": [_serialize_step(step) for step in memory],
        }
    return {"kind": "empty", "n_episodes": 0, "n_steps": 0}


def _update_and_capture_full(
    trainer: Any,
) -> Tuple[bool, Optional[float], Optional[Dict[str, Any]]]:
    """Like :func:`_update_and_capture` but also returns the full ``update()`` result.

    Gates on the episode buffer exactly like :func:`_update_and_capture` (so the
    on-the-fly training behaviour is unchanged — IOP step memory is not triggered
    here), runs ``update()`` + ``reset_episodes()``, and returns
    ``(ran, loss, result_dict)`` where ``result_dict`` carries the trainer's full
    report (including the new ``epoch_losses`` key). Wrapped in try/except so a
    faulty trainer never tears down the run.

    Example:
        >>> class T:
        ...     episodes = ["e"]
        ...     def update(self): return {"loss": 0.5, "epoch_losses": [0.6, 0.5]}
        ...     def reset_episodes(self): pass
        >>> ran, loss, res = _update_and_capture_full(T())
        >>> ran, loss, res["epoch_losses"]
        (True, 0.5, [0.6, 0.5])
    """

    episodes = getattr(trainer, "episodes", None)
    if not episodes:
        return (False, None, None)
    update = getattr(trainer, "update", None)
    reset = getattr(trainer, "reset_episodes", None)
    try:
        result = update() if callable(update) else None
        loss = _extract_loss(result, trainer)
        if callable(reset):
            reset()
        result_dict = result if isinstance(result, dict) else {"loss": loss}
        return (True, loss, result_dict)
    except Exception:
        return (False, None, None)


def _phase_of_trainer(agent: Any, trainer: Any) -> Optional[str]:
    """Return the loss phase ("propose"/"respond"/"utility_threshold") of a trainer.

    Resolves the phase by matching ``trainer`` against the agent's ``trainers``
    role mapping (LLR-M4.2). List-valued MBOM/IOP roles (``iop_trainers_*``) are
    NOT mapped to a per-phase loss and return ``None`` (they are still updated
    elsewhere, just not recorded to loss-per-phase). Returns ``None`` when the
    trainer cannot be located.

    Example:
        >>> class A:
        ...     trainers = {"propose_trainer": "P", "respond_trainer": "R"}
        >>> _phase_of_trainer(A(), "P")
        'propose'
        >>> _phase_of_trainer(A(), "R")
        'respond'
        >>> _phase_of_trainer(A(), "X") is None
        True
    """

    if agent is None or trainer is None:
        return None
    trainers_map = getattr(agent, "trainers", None)
    if not trainers_map:
        return None
    for role, value in trainers_map.items():
        if isinstance(value, (list, tuple)):
            continue
        if value is trainer:
            return _TRAINER_ROLE_TO_PHASE.get(role)
    return None


def _iter_trainers(agent: Any) -> List[Any]:
    """Return every distinct trainer attached to an agent (per-phase + IOP).

    Mirrors the scheduler's helper: flattens list-valued roles (the MBOM
    ``iop_trainers_*`` lists) and de-duplicates. Baseline agents (no trainers)
    yield ``[]``.

    Example:
        >>> class A: trainers = {"propose_trainer": 1, "respond_trainer": 2}
        >>> sorted(_iter_trainers(A()))
        [1, 2]
    """

    if agent is None:
        return []
    trainers_map = getattr(agent, "trainers", None)
    if not trainers_map:
        return []
    out: List[Any] = []
    seen: set = set()
    for value in trainers_map.values():
        items = value if isinstance(value, (list, tuple)) else [value]
        for trainer in items:
            if trainer is None or id(trainer) in seen:
                continue
            seen.add(id(trainer))
            out.append(trainer)
    return out


def _update_and_reset(trainer: Any) -> bool:
    """Run ``update()`` then ``reset_episodes()`` if the trainer has episodes.

    Returns ``True`` when an update actually ran (episodes were present).

    Example:
        >>> calls = []
        >>> class T:
        ...     episodes = ["e"]
        ...     def update(self): calls.append("u"); return {}
        ...     def reset_episodes(self): calls.append("r")
        >>> _update_and_reset(T()), calls
        (True, ['u', 'r'])
    """

    episodes = getattr(trainer, "episodes", None)
    if not episodes:
        return False
    update = getattr(trainer, "update", None)
    reset = getattr(trainer, "reset_episodes", None)
    ran = False
    if callable(update):
        update()
        ran = True
    if callable(reset):
        reset()
    return ran


def _load_state_model_only(trainer: Any, checkpoint: Dict[str, Any], device: str) -> None:
    """Load ONLY model state_dict from checkpoint into trainer — skip optimizers & targets (Q6).

    Implements the weights-only load decision (Q6): optimizer state is NOT
    restored (fresh start for continued training). Target models are also skipped.

    Args:
        trainer: An already-built trainer with attached ``models``.
        checkpoint: The dict produced by the trainer's ``save_checkpoint``.
        device: Torch device string for ``map_location`` consistency.

    Example:
        >>> _load_state_model_only.__name__
        '_load_state_model_only'
    """
    models = getattr(trainer, "models", None) or {}
    for key, info in (checkpoint.get("models", {}) or {}).items():
        entry = models.get(key)
        state = info.get("state_dict") if isinstance(info, dict) else None
        if entry is not None and state is not None:
            entry["model"].load_state_dict(state)
    # SKIP: checkpoint["optimizers"] — NOT loading optimizer state (Q6)
    # SKIP: checkpoint["target_models"] — NOT loading target network state (Q6)


class Orchestrator:
    """Own the two agents + apply the training concurrency policy (``06-01``).

    Parameters
    ----------
    agent_fleet, agent_spklu:
        The two negotiators (already built via the factory, or loaded from the
        registry). Fixed for the whole run (G14).
    config:
        The :class:`~simulation.config.config.SimulationConfig` (read once).
    metrics_collector:
        Optional :class:`~simulation.metrics.collector.MetricsCollector`; the
        orchestrator emits ``negotiation_utility`` / ``negotiation_payload`` into
        it so the negotiation-group metrics (``06-03``) are complete.
    registry:
        Optional :class:`~simulation.registry.agent_registry.AgentRegistry`
        (defaults to one rooted at ``training.registry_path``).
    fixed_opponent_side:
        For ``FIXED_OPPONENT``, the side that is the **fixed** (non-learner)
        sparring partner; the *other* side is the learner. Defaults to
        ``"spklu"`` (fleet learns against a fixed SPKLU).

    Example:
        >>> Orchestrator.__name__
        'Orchestrator'
    """

    def __init__(
        self,
        agent_fleet: Any,
        agent_spklu: Any,
        config: Any,
        metrics_collector: Any = None,
        registry: Optional[AgentRegistry] = None,
        fixed_opponent_side: str = "spklu",
    ) -> None:
        """Bind agents + config + policy; validate the concurrency mode.

        Example:
            >>> Orchestrator.__init__.__qualname__
            'Orchestrator.__init__'
        """

        self.agents: Dict[str, Any] = {"fleet": agent_fleet, "spklu": agent_spklu}
        self.config = config
        self.metrics_collector = metrics_collector
        self.registry = registry if registry is not None else AgentRegistry(
            registry_path=config.training.registry_path
        )

        self.concurrency: str = str(config.training.training_concurrency)
        if self.concurrency not in _VALID_CONCURRENCY:
            raise ValueError(
                f"training_concurrency must be one of {_VALID_CONCURRENCY}, "
                f"got {self.concurrency!r}."
            )
        if fixed_opponent_side not in _SIDES:
            raise ValueError(f"fixed_opponent_side must be one of {_SIDES}.")
        self.fixed_opponent_side = fixed_opponent_side
        self.learner_side = "fleet" if fixed_opponent_side == "spklu" else "spklu"

        # Trigger counter drives the ALTERNATING flip.
        self._train_tick_index: int = 0

        # Online training-log accumulators (for the end-of-run aggregate summary).
        # Full per-(side, phase) epoch-loss series across every on-the-fly update,
        # plus the aggregate-loss series and per-(side, phase) update counts. These
        # are always maintained (cheap); they are only *emitted* when the online
        # training logger is enabled (LLR training-log).
        self._online_epoch_series: Dict[Tuple[str, str], List[Any]] = {}
        self._online_agg_series: Dict[Tuple[str, str], List[float]] = {}
        self._online_update_counts: Dict[Tuple[str, str], int] = {}
        self._online_total_steps: int = 0

    # ------------------------------------------------------------------
    # Accessors for the scheduler
    # ------------------------------------------------------------------

    @property
    def agent_fleet(self) -> Any:
        """The fleet-side negotiator.

        Example:
            >>> isinstance(Orchestrator.agent_fleet, property)
            True
        """

        return self.agents["fleet"]

    @property
    def agent_spklu(self) -> Any:
        """The SPKLU-side negotiator.

        Example:
            >>> isinstance(Orchestrator.agent_spklu, property)
            True
        """

        return self.agents["spklu"]

    def trainers_for(self, side: str) -> List[Any]:
        """Return the distinct trainers attached to one side's agent.

        Example:
            >>> Orchestrator.trainers_for.__qualname__
            'Orchestrator.trainers_for'
        """

        return _iter_trainers(self.agents.get(side))

    # ------------------------------------------------------------------
    # Concurrency policy: which sides train on this trigger (``06-01``)
    # ------------------------------------------------------------------

    def _trainable_sides(self) -> List[str]:
        """Return the side(s) whose trainers should update on this trigger.

        * ``SIMULTANEOUS`` -> both sides.
        * ``ALTERNATING`` -> one side, flipping each trigger (fleet on even
          triggers, SPKLU on odd).
        * ``FIXED_OPPONENT`` -> only the learner side (opponent frozen).

        Example:
            >>> import types
            >>> o = object.__new__(Orchestrator)
            >>> o.concurrency = "SIMULTANEOUS"; o._train_tick_index = 0
            >>> o._trainable_sides()
            ['fleet', 'spklu']
            >>> o.concurrency = "ALTERNATING"; o._train_tick_index = 1
            >>> o._trainable_sides()
            ['spklu']
            >>> o.concurrency = "FIXED_OPPONENT"; o.learner_side = "fleet"
            >>> o._trainable_sides()
            ['fleet']
        """

        if self.concurrency == CONCURRENCY_SIMULTANEOUS:
            return ["fleet", "spklu"]
        if self.concurrency == CONCURRENCY_ALTERNATING:
            return ["fleet"] if (self._train_tick_index % 2 == 0) else ["spklu"]
        # FIXED_OPPONENT: only the learner trains.
        return [self.learner_side]

    # ------------------------------------------------------------------
    # On-the-fly training callback (scheduler seam)
    # ------------------------------------------------------------------

    def on_train_tick(self, scheduler: Any = None) -> Dict[str, Any]:
        """Apply the concurrency policy for one on-the-fly trigger (``06-01``).

        Matches the scheduler's ``training_callback`` seam
        (``callable(scheduler) -> None``): when the scheduler crosses
        ``train_every_n_episodes`` it calls this with itself. The orchestrator
        selects the trainable side(s) per :meth:`_trainable_sides`, runs each of
        their trainers' ``update()`` + ``reset_episodes()`` via
        :func:`_update_and_capture`, advances the alternating counter, and emits a
        ``training_step`` metric **carrying the captured per-phase losses**
        (LLR-M4.3). No loss formula is touched — losses are only *read* from what
        each trainer already returns/exposes (INV-3).

        The emitted (and returned) ``losses`` is a list of ``(side, phase, loss)``
        tuples (LLR-M4.3 / LLR-M5.2 wire format), with one tuple per trainer that
        actually updated, has a resolvable per-phase mapping
        (:func:`_phase_of_trainer`), and reported a finite loss. Trainers that
        reported no loss, or list-valued MBOM/IOP trainers whose phase resolves to
        ``None``, are still updated but contribute no loss tuple (an empty
        ``losses`` list is valid per KR-8). The downstream collector
        (:meth:`simulation.metrics.collector.MetricsCollector._on_training_step`)
        consumes exactly this same shape.

        Frozen sides have their stored episodes left intact for now (they keep
        accumulating until that side is trainable again) so no data is wasted.

        Args:
            scheduler: The calling scheduler (unused; part of the seam signature).

        Returns:
            A small report
            ``{"sides": [...], "updated": {side: n_trainers},
            "losses": [(side, phase, loss), ...]}``.

        Example:
            >>> Orchestrator.on_train_tick.__qualname__
            'Orchestrator.on_train_tick'
        """

        online_logger = get_online_logger()
        log_enabled = online_logger.is_enabled()
        if log_enabled:
            online_logger.mark_train_tick(self._train_tick_index)

        sides = self._trainable_sides()
        losses: List[Tuple[str, str, float]] = []
        tick_updates: List[Dict[str, Any]] = []
        report: Dict[str, Any] = {"sides": list(sides), "updated": {}}
        for side in sides:
            agent = self.agents.get(side)
            n_ran = 0
            for trainer in _iter_trainers(agent):
                phase = _phase_of_trainer(agent, trainer)
                # Snapshot the FULL memory contents BEFORE the update (the buffer
                # is cleared by reset_episodes() inside the capture) — only when
                # logging, to avoid the serialization cost otherwise.
                memory = _memory_snapshot(trainer) if log_enabled else None
                ran, loss, result = _update_and_capture_full(trainer)
                if not ran:
                    continue
                n_ran += 1
                if phase is not None and loss is not None:
                    losses.append((side, phase, float(loss)))
                # Always accumulate the per-(side, phase) series for the aggregate
                # summary (cheap; independent of whether the logger is enabled).
                if phase is not None:
                    key = (side, phase)
                    epoch_losses = (result or {}).get("epoch_losses", [])
                    self._online_epoch_series.setdefault(key, []).append(epoch_losses)
                    if loss is not None:
                        self._online_agg_series.setdefault(key, []).append(float(loss))
                    self._online_update_counts[key] = (
                        self._online_update_counts.get(key, 0) + 1
                    )
                    n_steps = int((memory or {}).get("n_steps", 0)) if memory else 0
                    self._online_total_steps += n_steps
                if log_enabled:
                    online_logger.emit(
                        "online_update",
                        side=side,
                        phase=phase,
                        trainer_type=type(trainer).__name__,
                        concurrency=self.concurrency,
                        aggregate_loss=loss,
                        epoch_losses=(result or {}).get("epoch_losses", []),
                        result=result,
                        n_episodes=(memory or {}).get("n_episodes", 0),
                        n_steps=(memory or {}).get("n_steps", 0),
                        memory=memory,
                    )
                    tick_updates.append({
                        "side": side, "phase": phase,
                        "trainer_type": type(trainer).__name__,
                        "aggregate_loss": loss,
                    })
            report["updated"][side] = n_ran
        report["losses"] = losses
        if log_enabled:
            online_logger.emit(
                "online_update_tick",
                concurrency=self.concurrency,
                sides=list(sides),
                updated=report["updated"],
                updates=tick_updates,
            )
        self._train_tick_index += 1
        self._emit(
            "training_step",
            concurrency=self.concurrency,
            sides=list(sides),
            losses=losses,
        )
        return report

    def train(self, scheduler: Any = None) -> Dict[str, Any]:
        """Alias of :meth:`on_train_tick` (the ``training_callback`` name).

        Example:
            >>> Orchestrator.train.__qualname__
            'Orchestrator.train'
        """

        return self.on_train_tick(scheduler)

    def finalize_online_log(self) -> Dict[str, Any]:
        """Emit the aggregate online-training summary for the whole run (LLR training-log).

        Reads the per-``(side, phase)`` series accumulated across every on-the-fly
        :meth:`on_train_tick` — the full epoch-loss series, the aggregate-loss
        series, and update counts — and emits one ``online_training_summary``
        event to the online training logger (no-op when the logger is disabled).
        Returns the same summary dict so callers can print/inspect it.

        Example:
            >>> Orchestrator.finalize_online_log.__qualname__
            'Orchestrator.finalize_online_log'
        """

        per_series = {
            f"{side}:{phase}": {
                "n_updates": self._online_update_counts.get((side, phase), 0),
                "aggregate_loss_series": self._online_agg_series.get((side, phase), []),
                "epoch_loss_series": self._online_epoch_series.get((side, phase), []),
            }
            for (side, phase) in sorted(
                set(self._online_update_counts)
                | set(self._online_agg_series)
                | set(self._online_epoch_series)
            )
        }
        summary: Dict[str, Any] = {
            "concurrency": self.concurrency,
            "total_train_ticks": self._train_tick_index,
            "total_steps_trained": self._online_total_steps,
            "series": per_series,
        }
        logger = get_online_logger()
        if logger.is_enabled():
            logger.emit("online_training_summary", **summary)
        return summary

    # ------------------------------------------------------------------
    # Per-negotiation hooks (metric completeness, ``06-03``)
    # ------------------------------------------------------------------

    def record_negotiation_payloads(self, agreement: Any = None) -> None:
        """Emit each side's episode payloads + realized utility to the collector.

        Reads the agents' separated per-phase payload builders
        (``build_episode_payload_propose`` / ``_respond`` / DLST threshold) and
        forwards each finalized :class:`Episode` as a ``negotiation_payload``
        event, plus the propose-phase ``terminal_utility`` as a per-side
        ``negotiation_utility`` event. Best-effort: agents without the builders
        (baselines) are skipped.

        Args:
            agreement: The negotiation agreement outcome ``(idx,)`` / ``None``.

        Example:
            >>> Orchestrator.record_negotiation_payloads.__qualname__
            'Orchestrator.record_negotiation_payloads'
        """

        if self.metrics_collector is None:
            return
        for side in _SIDES:
            agent = self.agents.get(side)
            if agent is None:
                continue
            for phase, builder_name in (
                ("propose", "build_episode_payload_propose"),
                ("respond", "build_episode_payload_respond"),
                ("utility_threshold", "build_episode_payload_threshold"),
            ):
                builder = getattr(agent, builder_name, None)
                if not callable(builder):
                    continue
                try:
                    episode = builder(agreement)
                except Exception:
                    continue
                if episode is None:
                    continue
                self._emit("negotiation_payload", side=side, phase=phase, episode=episode)
                if phase == "propose":
                    utility = getattr(episode, "terminal_utility", None)
                    if utility is not None:
                        self._emit("negotiation_utility", side=side, utility=utility)

    def _emit(self, event: str, **payload: Any) -> None:
        """Forward an event to the metrics collector if one is attached.

        Example:
            >>> o = object.__new__(Orchestrator)
            >>> o.metrics_collector = None
            >>> o._emit("training_step") is None
            True
        """

        collector = self.metrics_collector
        if collector is None:
            return
        record = getattr(collector, "record", None)
        if callable(record):
            record(event, **payload)
        elif callable(collector):
            collector(event, **payload)

    # ------------------------------------------------------------------
    # Offline training from a saved one-day history payload (``06-01``)
    # ------------------------------------------------------------------

    def write_history(
        self, episodes_by_side: Dict[str, List[Any]], path: Optional[str] = None
    ) -> str:
        """Write a versioned one-day history payload + per-side signatures (``02-10``).

        The history file embeds each side's :class:`Signature` so offline
        training can cross-check it (Contract C7 / ``06-02``). ``episodes_by_side``
        maps a side to its list of ``(phase, serializable_episode)`` pairs (the
        serialized form is the caller's responsibility — typically the collector's
        ``negotiation_payload_history``).

        Args:
            episodes_by_side: ``{side: [(phase, episode_dict), ...]}``.
            path: Output path; defaults to ``<history_path>/history.json``.

        Returns:
            The written file path.

        Example:
            >>> Orchestrator.write_history.__qualname__
            'Orchestrator.write_history'
        """

        path = path or os.path.join(self.config.training.history_path, "history.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        signatures = {
            side: self.signature_for(side).to_dict() for side in _SIDES
        }
        document = {
            _HISTORY_SCHEMA_KEY: int(self.config.training.schema_version),
            _HISTORY_SIGNATURE_KEY: signatures,
            _HISTORY_EPISODES_KEY: episodes_by_side,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(document, fh, indent=2, default=str)
        return path

    def train_offline(
        self,
        episodes_by_side: Dict[str, List[Any]],
        history_signatures: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Train offline from in-memory payloads, checking signatures first (``06-01``).

        For each side, the history signature (if provided) is compared against the
        agent's live signature and a :class:`SignatureMismatchError` is raised
        naming the field on any mismatch (Contract C7 / ``06-02`` check #2). Then
        each ``(phase, episode)`` payload is pushed into the matching per-phase
        trainer (``store_episode``) and, after loading the whole day, every
        trainer ``update()``s once (respecting the concurrency policy: frozen
        sides are not updated). The world is **not** run.

        Args:
            episodes_by_side: ``{side: [(phase, episode), ...]}`` payloads.
            history_signatures: Optional ``{side: signature_dict}`` to validate.

        Returns:
            A report ``{"updated": {side: n_trainers}}``.

        Raises:
            SignatureMismatchError: If a history signature differs from the agent.

        Example:
            >>> Orchestrator.train_offline.__qualname__
            'Orchestrator.train_offline'
        """

        # ── Contract C7: validate history signature vs. agent signature. ──
        if history_signatures:
            for side in _SIDES:
                stored = history_signatures.get(side)
                if stored is None:
                    continue
                Signature.from_dict(stored).assert_compatible(self.signature_for(side))

        trainable = set(self._trainable_sides_for_offline())
        # ── Load the day's episodes into each phase's trainer. ──
        for side, payloads in episodes_by_side.items():
            agent = self.agents.get(side)
            if agent is None:
                continue
            for phase, episode in payloads:
                trainer = self._trainer_for_phase(agent, phase)
                if trainer is not None and hasattr(trainer, "store_episode"):
                    trainer.store_episode(episode)

        # ── Update every loaded trainer once (per concurrency policy). ──
        report: Dict[str, Any] = {"updated": {}}
        for side in _SIDES:
            if side not in trainable:
                continue
            n_ran = 0
            for trainer in _iter_trainers(self.agents.get(side)):
                if _update_and_reset(trainer):
                    n_ran += 1
            report["updated"][side] = n_ran
        return report

    def train_offline_with_loss(
        self,
        episodes_by_side: Dict[str, List[Any]],
        history_signatures: Optional[Dict[str, Dict[str, Any]]] = None,
        n_iterations: int = 1,
    ) -> Tuple[Dict[str, Any], Any]:
        """Offline-train while recording a per-(side, phase) loss curve (``06-01``).

        Same contract as :meth:`train_offline` (signatures are validated up front,
        the concurrency policy is honoured, the world is not run), but instead of
        a single ``update()`` per trainer it runs ``n_iterations`` passes over the
        data and **captures the loss reported by each ``update()``** into a
        :class:`~simulation.metrics.loss_history.LossHistory`. Each pass re-stores
        the episodes into each phase trainer, runs ``update()`` (loss read via
        :func:`_extract_loss` — no loss formula is added, INV-3), then
        ``reset_episodes()``. The resulting ``n_iterations``-long series per
        ``(side, phase)`` lets an offline session be plotted exactly like the
        on-the-fly run's loss charts (``plot_losses``).

        List-valued MBOM/IOP trainers (whose phase resolves to ``None``) are still
        updated but contribute no loss point — identical to the on-the-fly path
        (KR-8/9).

        Args:
            episodes_by_side: ``{side: [(phase, episode), ...]}`` payloads.
            history_signatures: Optional ``{side: signature_dict}`` to validate.
            n_iterations: Number of offline passes (>= 1); the length of each
                recorded loss series.

        Returns:
            ``(report, loss_history)`` where ``report`` is
            ``{"updated": {side: n_trainers}, "iterations": n}`` and
            ``loss_history`` is a populated :class:`LossHistory`.

        Raises:
            SignatureMismatchError: If a history signature differs from an agent.

        Example:
            >>> Orchestrator.train_offline_with_loss.__qualname__
            'Orchestrator.train_offline_with_loss'
        """

        from simulation.metrics.loss_history import LossHistory

        # ── Contract C7: validate history signature vs. agent signature. ──
        if history_signatures:
            for side in _SIDES:
                stored = history_signatures.get(side)
                if stored is None:
                    continue
                Signature.from_dict(stored).assert_compatible(self.signature_for(side))

        offline_logger = get_offline_logger()
        log_enabled = offline_logger.is_enabled()

        trainable = set(self._trainable_sides_for_offline())
        loss_history = LossHistory()
        passes = max(1, int(n_iterations))
        report: Dict[str, Any] = {"updated": {}, "iterations": passes}

        # ── Log the FULL data used for offline training (per side, per phase). ──
        if log_enabled:
            for side, payloads in episodes_by_side.items():
                if side not in trainable:
                    continue
                by_phase: Dict[str, List[Any]] = {}
                for phase, episode in payloads:
                    by_phase.setdefault(phase, []).append(_serialize_episode(episode))
                for phase, episodes in by_phase.items():
                    n_steps = sum(len(ep.get("steps", []) or []) for ep in episodes)
                    offline_logger.emit(
                        "offline_data",
                        side=side,
                        phase=phase,
                        n_episodes=len(episodes),
                        n_steps=n_steps,
                        episodes=episodes,
                    )

        for pass_idx in range(passes):
            if log_enabled:
                offline_logger.mark_iteration(pass_idx)
            for side, payloads in episodes_by_side.items():
                if side not in trainable:
                    continue
                agent = self.agents.get(side)
                if agent is None:
                    continue
                # Re-store this pass's episodes into each phase's trainer.
                for phase, episode in payloads:
                    trainer = self._trainer_for_phase(agent, phase)
                    if trainer is not None and hasattr(trainer, "store_episode"):
                        trainer.store_episode(episode)
                # Update every trainer once, capturing the per-phase loss.
                n_ran = 0
                for trainer in _iter_trainers(agent):
                    phase = _phase_of_trainer(agent, trainer)
                    memory = _memory_snapshot(trainer) if log_enabled else None
                    ran, loss, result = _update_and_capture_full(trainer)
                    if not ran:
                        continue
                    n_ran += 1
                    if phase is not None and loss is not None:
                        loss_history.record(side, phase, float(loss))
                    if log_enabled:
                        offline_logger.emit(
                            "offline_update",
                            side=side,
                            phase=phase,
                            trainer_type=type(trainer).__name__,
                            iteration=pass_idx,
                            aggregate_loss=loss,
                            epoch_losses=(result or {}).get("epoch_losses", []),
                            result=result,
                            n_episodes=(memory or {}).get("n_episodes", 0),
                            n_steps=(memory or {}).get("n_steps", 0),
                            memory=memory,
                        )
                report["updated"][side] = n_ran

        if log_enabled:
            offline_logger.emit(
                "offline_training_summary",
                concurrency=self.concurrency,
                iterations=passes,
                updated=report["updated"],
                loss_series=loss_history.to_dict(),
            )

        return report, loss_history

    def train_offline_from_file(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load a history file and train offline from it (signature-checked).

        Args:
            path: History file; defaults to ``<history_path>/history.json``.

        Returns:
            The :meth:`train_offline` report.

        Example:
            >>> Orchestrator.train_offline_from_file.__qualname__
            'Orchestrator.train_offline_from_file'
        """

        path = path or os.path.join(self.config.training.history_path, "history.json")
        with open(path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
        episodes_by_side = document.get(_HISTORY_EPISODES_KEY, {}) or {}
        signatures = document.get(_HISTORY_SIGNATURE_KEY, {}) or {}
        return self.train_offline(episodes_by_side, history_signatures=signatures)

    def _trainable_sides_for_offline(self) -> List[str]:
        """Sides eligible to update in offline mode (same policy, no flip state).

        ``ALTERNATING`` offline trains both sides (each in its own pass is a
        caller concern); ``FIXED_OPPONENT`` still freezes the opponent;
        ``SIMULTANEOUS`` trains both.

        Example:
            >>> o = object.__new__(Orchestrator)
            >>> o.concurrency = "FIXED_OPPONENT"; o.learner_side = "fleet"
            >>> o._trainable_sides_for_offline()
            ['fleet']
        """

        if self.concurrency == CONCURRENCY_FIXED_OPPONENT:
            return [self.learner_side]
        return ["fleet", "spklu"]

    @staticmethod
    def _trainer_for_phase(agent: Any, phase: str) -> Any:
        """Return the agent's trainer for a payload phase (``06-01`` mapping).

        Example:
            >>> class A: trainers = {"propose_trainer": "P", "respond_trainer": "R"}
            >>> Orchestrator._trainer_for_phase(A(), "propose")
            'P'
        """

        trainers_map = getattr(agent, "trainers", None) or {}
        key = {
            "propose": "propose_trainer",
            "respond": "respond_trainer",
            "utility_threshold": "threshold_trainer",
        }.get(phase)
        return trainers_map.get(key) if key else None

    # ------------------------------------------------------------------
    # Signatures + registry (save/load — ``06-02``)
    # ------------------------------------------------------------------

    def signature_for(self, side: str) -> Signature:
        """Build the live :class:`Signature` for one side from the config + dims.

        Example:
            >>> Orchestrator.signature_for.__qualname__
            'Orchestrator.signature_for'
        """

        from simulation.bridge.feature_builder import feature_dims

        return build_signature(self.config, side, feature_dims(self.config, side))

    def save_all(self, version: Optional[str] = None) -> Dict[str, str]:
        """Persist both sides to the registry (signature + trainer state) (``06-02``).

        Args:
            version: Optional shared version label.

        Returns:
            ``{side: artifact_dir}``.

        Example:
            >>> Orchestrator.save_all.__qualname__
            'Orchestrator.save_all'
        """

        out: Dict[str, str] = {}
        for side in _SIDES:
            out[side] = self.registry.save_side(
                self.agents[side], self.config, side, version=version
            )
        return out

    def load_all(self, version: str) -> None:
        """Load both sides from the registry, validating signatures (``06-02``).

        Args:
            version: The artifact version to load for both sides.

        Example:
            >>> Orchestrator.load_all.__qualname__
            'Orchestrator.load_all'
        """

        for side in _SIDES:
            self.registry.load_into_agent(
                self.agents[side], side, version, self.signature_for(side)
            )

    # ------------------------------------------------------------------
    # Agent bundle: save + load (LLR-2.3, Q2 dual-write, Q6 weights-only)
    # ------------------------------------------------------------------

    def save_agent_bundle(
        self,
        agent: Any,
        config: Any,
        side: str,
        run_paths: Any,
        version: Optional[str] = None,
    ) -> str:
        """Persist agent bundle to run folder + registry (dual-write, Q2 / LLR-2.3).

        A "bundle" is a folder containing one ``*.pt`` file per trainer
        (written by ``trainer.save_checkpoint``), a ``signature.json``, and an
        ``agent_meta.json`` (with ``parameter_count`` from
        :func:`count_agent_parameters`).

        Dual-write (Q2):
          (a) Run folder: ``<run_dir>/agents/<side>/`` via ``run_paths.agent_bundle_dir``.
          (b) Registry: ``data/registry/<side>/<version>/`` via ``self.registry.save_side``.

        Args:
            agent: The negotiator to persist.
            config: :class:`~simulation.config.config.SimulationConfig`.
            side: ``"fleet"`` or ``"spklu"``.
            run_paths: :class:`~simulation.metrics.run_paths.RunPaths` for (a).
            version: Optional version label; UTC timestamp used if omitted.

        Returns:
            Absolute path to the run-folder bundle directory.

        Example:
            >>> Orchestrator.save_agent_bundle.__qualname__
            'Orchestrator.save_agent_bundle'
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise RuntimeError("torch is required to save agent checkpoints.")

        from simulation.registry.agent_registry import _enumerate_trainers
        from simulation.registry.signature import build_signature

        version = version or time.strftime("%Y%m%dT%H%M%S", time.gmtime())

        # (a) Run folder destination.
        run_folder = run_paths.agent_bundle_dir(side)

        # Save each trainer checkpoint to run folder.
        trainer_keys: List[str] = []
        for key, trainer in _enumerate_trainers(agent):
            save = getattr(trainer, "save_checkpoint", None)
            if not callable(save):
                continue
            save(os.path.join(run_folder, f"{key}.pt"))
            trainer_keys.append(key)

        # Build and write signature.json.
        from simulation.bridge.feature_builder import feature_dims as _feature_dims
        sig = build_signature(config, side, _feature_dims(config, side))
        sig_path = os.path.join(run_folder, "signature.json")
        with open(sig_path, "w", encoding="utf-8") as fh:
            json.dump(sig.to_dict(), fh, indent=2)

        # Count parameters (Q4).
        from negotiation.agents.agent_factory import count_agent_parameters
        param_count = count_agent_parameters(agent)

        # Write agent_meta.json.
        meta: Dict[str, Any] = {
            "agent_id": side,
            "version": version,
            "agent_type": type(agent).__name__,
            "mode": getattr(agent, "mode", None),
            "trainer_keys": trainer_keys,
            "parameter_count": param_count,
        }
        meta_path = os.path.join(run_folder, "agent_meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        # (b) Dual-write to registry.
        self.registry.save_side(agent, config, side, version=version)

        return os.path.abspath(run_folder)

    def load_agent_bundle(
        self,
        agent: Any,
        bundle_dir: str,
        current_signature: "Signature",
        device: str = "cpu",
    ) -> "Signature":
        """Load agent weights from a run-folder bundle (weights-only, Q6 / LLR-2.3).

        Validates the stored signature against ``current_signature`` (Contract C7),
        then loads ONLY model ``state_dict``s — optimizer state is NOT restored
        (Q6: fresh optimizer for continued training).

        Args:
            agent: The already-built agent (same type + dims as when saved).
            bundle_dir: Path to the run-folder bundle directory.
            current_signature: The live agent's :class:`Signature`.
            device: Torch device string for ``map_location``.

        Returns:
            The stored :class:`Signature` (now validated).

        Raises:
            SignatureMismatchError: If stored and current signatures differ.

        Example:
            >>> Orchestrator.load_agent_bundle.__qualname__
            'Orchestrator.load_agent_bundle'
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise RuntimeError("torch is required to load agent checkpoints.")

        from simulation.registry.agent_registry import _enumerate_trainers
        from simulation.registry.signature import Signature, SignatureMismatchError

        # Load stored signature and validate (Contract C7 / LLR-2.4).
        sig_path = os.path.join(bundle_dir, "signature.json")
        with open(sig_path, "r", encoding="utf-8") as fh:
            stored_sig = Signature.from_dict(json.load(fh))
        stored_sig.assert_compatible(current_signature)

        # Restore each trainer's models (weights only — Q6).
        for key, trainer in _enumerate_trainers(agent):
            checkpoint_path = os.path.join(bundle_dir, f"{key}.pt")
            if not os.path.exists(checkpoint_path):
                continue
            checkpoint = torch.load(checkpoint_path, map_location=device)
            _load_state_model_only(trainer, checkpoint, device)

        return stored_sig

    # ------------------------------------------------------------------
    # Partial / full training from external payload (LLR-2.4, LLR-2.5)
    # ------------------------------------------------------------------

    def train_from_payload(
        self,
        agent: Any,
        payload_dict: Dict[str, List[Any]],
        side: str,
        current_signature: "Signature",
        trainer_filter: Optional[List[str]] = None,
    ) -> Dict[str, Optional[float]]:
        """Train an agent's trainers from an external payload dict (LLR-2.5).

        Validates the payload signature against the agent's live signature first
        (LLR-2.4 / Contract C7). Then, for each trainer (optionally filtered),
        injects episodes from ``payload_dict`` and runs one update cycle.

        The injection API differs per trainer kind (Q3 decision):

        * **Group A** (PG / PPO / DDPG): ``store_episode(ep)`` -> ``update()``
          -> ``reset_episodes()``
        * **Group B** (IOPTrainer, detected via ``hasattr(trainer, 'add_step')``):
          iterate steps ``add_step(step)`` per step -> ``update()``
          -> ``reset_step()``

        Phase-key mapping (trainer_key -> payload_dict key):
          ``propose_trainer``        -> ``"propose"``
          ``respond_trainer``        -> ``"respond"``
          ``threshold_trainer``      -> ``"utility_threshold"``
          ``iop_trainers_propose__*`` -> ``"propose"`` (Group B)
          ``iop_trainers_respond__*`` -> ``"respond"`` (Group B)

        Args:
            agent: The negotiator whose trainers will be updated.
            payload_dict: ``{"propose": [Episode, ...], "respond": [...], ...}``.
            side: ``"fleet"`` or ``"spklu"`` (used for signature comparison).
            current_signature: The live agent's :class:`Signature`.
            trainer_filter: Optional list of trainer keys to restrict training to;
                if ``None``, all trainers are trained.

        Returns:
            ``{trainer_key: loss_or_None}`` — one entry per trainer (``None`` if
            no payload was available or the trainer produced no loss).

        Raises:
            SignatureMismatchError: If ``payload_dict`` contains a ``"signature"``
                key that mismatches ``current_signature``.

        Example:
            >>> Orchestrator.train_from_payload.__qualname__
            'Orchestrator.train_from_payload'
        """
        from simulation.registry.agent_registry import _enumerate_trainers
        from simulation.registry.signature import Signature, SignatureMismatchError

        # LLR-2.4: validate payload signature if embedded.
        payload_sig_raw = payload_dict.get("signature")
        if payload_sig_raw is not None and isinstance(payload_sig_raw, dict):
            payload_sig = Signature.from_dict(payload_sig_raw)
            payload_sig.assert_compatible(current_signature)

        # Trainer key -> phase key mapping.
        def _phase_key_for(key: str) -> str:
            if key == "propose_trainer":
                return "propose"
            if key == "respond_trainer":
                return "respond"
            if key == "threshold_trainer":
                return "utility_threshold"
            if key.startswith("iop_trainers_propose__"):
                return "propose"
            if key.startswith("iop_trainers_respond__"):
                return "respond"
            return key

        loss_dict: Dict[str, Optional[float]] = {}

        for key, trainer in _enumerate_trainers(agent):
            # Respect trainer_filter.
            if trainer_filter is not None and key not in trainer_filter:
                continue

            phase_key = _phase_key_for(key)
            episodes = payload_dict.get(phase_key)

            if not episodes:
                loss_dict[key] = None
                continue

            try:
                is_iop = hasattr(trainer, "add_step") and callable(trainer.add_step)

                if is_iop:
                    # Group B: per-step injection for IOPTrainer.
                    for ep in episodes:
                        for step in getattr(ep, "steps", []):
                            trainer.add_step(step)
                    result = trainer.update()
                    reset_fn = getattr(trainer, "reset_step", None)
                    if callable(reset_fn):
                        reset_fn()
                else:
                    # Group A: per-episode injection (PG / PPO / DDPG).
                    store_fn = getattr(trainer, "store_episode", None)
                    if callable(store_fn):
                        for ep in episodes:
                            store_fn(ep)
                    result = trainer.update()
                    reset_fn = getattr(trainer, "reset_episodes", None)
                    if callable(reset_fn):
                        reset_fn()

                loss_dict[key] = _extract_loss(result, trainer)

            except Exception:
                loss_dict[key] = None

        return loss_dict


if __name__ == "__main__":  # pragma: no cover - lightweight self-check
    import doctest

    failures, _ = doctest.testmod()
    if failures:
        raise SystemExit(f"{failures} doctest failure(s)")
    print("Orchestrator doctests passed")
