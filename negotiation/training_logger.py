"""Run-scoped training log writer (JSON Lines) for online & offline training.

A sibling of :mod:`negotiation.trace_logger`: whereas the *trace* logger records
the per-tick / per-negotiation **simulation** stream, this logger records the
**training** stream — every ``trainer.update()`` that happens during a run
(online) or during an offline-training session, so a user can trace exactly which
data went into an update, the per-epoch loss inside that update, and the
aggregate result across the whole session.

It deliberately lives in :mod:`negotiation` — the lowest layer — so both the
simulation orchestrator (:mod:`simulation.training.orchestrator`) and the
negotiation trainers can import it without breaking the dependency boundary
(``negotiation`` never imports ``simulation``). It reuses
:func:`negotiation.trace_logger.to_jsonable` for serialization so the two logs
share one JSON-sanitizer.

Events are written as **JSON Lines** (one self-contained JSON object per line) to
a single file per session. Two independent process-global singletons keep the
online and offline streams from ever clashing (they can be active at once in the
same process — e.g. a run that also offline-trains):

* :func:`get_online_logger` — target ``data/logs/online_training/<sim_id>.jsonl``.
* :func:`get_offline_logger` — target ``data/logs/offline_training/<offline_id>.jsonl``.

Both are **disabled by default**; an entry point (``main.py`` / ``offline_train.py``
/ ``run_experiment.py`` / the dashboard session) calls :meth:`TrainingLogger.configure`
with the target path to enable it. Every public method is a cheap no-op while
disabled, and all I/O is wrapped so a logging failure can never tear down a run.

Typical event types emitted (the ``event`` field):

* ``online_update`` — one ``trainer.update()`` on-the-fly: the acting side, phase,
  trainer type, the full serialized memory (every episode / step) fed into the
  update, the per-epoch losses, the aggregate loss, and the episode/step counts.
* ``online_update_tick`` — a small envelope per on-the-fly trigger summarising
  which sides/phases updated that tick.
* ``online_training_summary`` — the aggregate result over the whole run: the full
  per-``(side, phase)`` epoch-loss series and update counts.
* ``offline_data`` — the full data used to offline-train one trainer.
* ``offline_update`` — one offline ``update()`` pass for one trainer.
* ``offline_training_summary`` — the aggregate offline result.

Each emitted record is enriched with an ambient envelope (a monotonic sequence
number, the ``mode``, the ``run_id``, and the current ``train_tick`` / ``iteration``)
so the flat event stream can be re-correlated downstream.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

# Reuse the trace logger's JSON sanitizer (single source of truth for how
# numpy / torch / dataclass values become JSON-native structures).
from negotiation.trace_logger import to_jsonable

# Canonical sub-folder names under the logs root (kept here so entry points and
# viewers share one spelling).
LOGS_SUBDIR_SIMULATION: str = "simulation"
LOGS_SUBDIR_ONLINE: str = "online_training"
LOGS_SUBDIR_OFFLINE: str = "offline_training"


class TrainingLogger:
    """Singleton JSON-Lines writer for one training session (online or offline).

    Construction yields a *disabled* logger; call :meth:`configure` with a target
    file path (and a ``mode`` / ``run_id``) to enable it. The ambient context
    (:attr:`train_tick`, :attr:`iteration`) is advanced by the caller and attached
    to every record.

    Example:
        >>> log = TrainingLogger()
        >>> log.is_enabled()
        False
        >>> log.emit("online_update", note="ignored while disabled") is None
        True
    """

    def __init__(self) -> None:
        self._fh: Optional[Any] = None
        self._enabled: bool = False
        self.path: Optional[str] = None
        self._seq: int = 0
        self.mode: Optional[str] = None
        self.run_id: Optional[str] = None
        self.train_tick: int = -1
        self.iteration: int = -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure(
        self, path: str, mode: str = "online", run_id: Optional[str] = None
    ) -> None:
        """Open ``path`` for writing and enable the logger (closing any prior file).

        Parent directories are created as needed. A failure to open the file
        leaves the logger disabled (it never raises into the caller).

        Args:
            path: Target ``.jsonl`` file.
            mode: ``"online"`` or ``"offline"`` (stamped onto every record).
            run_id: The run identity (``sim_id`` / ``offline_id``) for correlation.
        """
        self.close()
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")
            self._enabled = True
            self.path = path
            self.mode = str(mode)
            self.run_id = run_id
            self._seq = 0
            self.train_tick = -1
            self.iteration = -1
        except Exception:
            self._fh = None
            self._enabled = False
            self.path = None

    def is_enabled(self) -> bool:
        """Return whether the logger is currently writing events."""
        return self._enabled and self._fh is not None

    def close(self) -> None:
        """Flush and close the underlying file (idempotent)."""
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._enabled = False

    # ------------------------------------------------------------------
    # Ambient context
    # ------------------------------------------------------------------

    def mark_train_tick(self, train_tick: int) -> None:
        """Set the current on-the-fly training-trigger index attached to events."""
        self.train_tick = int(train_tick)

    def mark_iteration(self, iteration: int) -> None:
        """Set the current offline-pass index attached to subsequent events."""
        self.iteration = int(iteration)

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, event: str, **payload: Any) -> None:
        """Write one JSON-Lines record of type ``event`` (no-op while disabled).

        The record carries an ambient envelope (``seq`` / ``mode`` / ``run_id`` /
        ``train_tick`` / ``iteration``) plus the JSON-sanitized ``payload``. All
        serialization / I/O is wrapped so a logging error can never propagate into
        training.
        """
        if not self.is_enabled():
            return
        try:
            record = {
                "seq": self._seq,
                "event": str(event),
                "mode": self.mode,
                "run_id": self.run_id,
                "train_tick": self.train_tick,
                "iteration": self.iteration,
                **{k: to_jsonable(v) for k, v in payload.items()},
            }
            self._seq += 1
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
            self._fh.flush()  # type: ignore[union-attr]
        except Exception:
            # Never let a logging failure break training.
            pass


# Two independent process-global singletons: online + offline can be active in
# the same process without clobbering each other's file handle / sequence.
_ONLINE_LOGGER = TrainingLogger()
_OFFLINE_LOGGER = TrainingLogger()


def get_online_logger() -> TrainingLogger:
    """Return the process-global online :class:`TrainingLogger` singleton.

    Example:
        >>> get_online_logger() is get_online_logger()
        True
    """
    return _ONLINE_LOGGER


def get_offline_logger() -> TrainingLogger:
    """Return the process-global offline :class:`TrainingLogger` singleton.

    Example:
        >>> get_offline_logger() is get_offline_logger()
        True
        >>> get_offline_logger() is get_online_logger()
        False
    """
    return _OFFLINE_LOGGER
