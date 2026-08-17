"""Run-scoped validation trace logger (JSON Lines).

A process-global singleton that components across **both** the simulation engine
(:mod:`simulation.*`) and the negotiation agents (:mod:`negotiation.*`) write
structured events to, for offline validation of the negotiation mechanism.

It deliberately lives in :mod:`negotiation` — the lowest layer — so both packages
can import it without breaking the dependency boundary (``negotiation`` never
imports ``simulation``). The module itself depends only on the standard library;
``numpy`` / ``torch`` are imported defensively (only used by the serializer when
present) so importing this module never pulls a heavy dependency.

Events are written as **JSON Lines** (one self-contained JSON object per line) to
a single file per run, named after the run's ``sim_id`` under ``data/logs/``. The
logger is **disabled by default**; an entry point (``main.py`` / the dashboard
session) calls :meth:`TraceLogger.configure` with the target path to enable it.
Every public method is a cheap no-op while disabled, and all I/O is wrapped so a
logging failure can never tear down a simulation run.

Typical event types emitted (the ``event`` field):

* ``tick`` — the full world snapshot for one engine tick.
* ``negotiation_start`` — the serialized :class:`DomainSpec` for one period
  (action library, both ufuns over the library, bounds, raw obs / candidate
  feature blocks, masks, candidate taxi ids).
* ``negotiation_step`` — one propose / respond move: acting side, the NegMAS
  state received, the observation fed to the network (dict for discrete agents,
  vectors for DLST), the chosen action, and the raw actor output.
* ``iop_step`` — for MBOM-based agents: one IOP-network inference (input state
  basis + output probabilities) or one IOP ``add_step`` + ``update`` (the step
  added and the resulting loss), tagged with the reasoning level.
* ``negotiation_end`` — the agreement / deal outcome of one period.

Each emitted record is enriched with an ambient context (a monotonic sequence
number, the current tick index, and the current negotiation-period index) so the
flat event stream can be re-correlated downstream.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

try:  # optional — only the serializer touches it
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore

try:  # optional — only the serializer touches it
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None  # type: ignore


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of an arbitrary value into a JSON-native structure.

    Handles the value shapes that flow through the negotiation stack: plain
    scalars, dicts, lists/tuples, ``numpy`` arrays / scalars, and ``torch``
    tensors. Non-finite floats (``nan`` / ``inf``) are coerced to ``None`` so the
    output is strict-JSON parseable. Anything unrecognized falls back to ``repr``.
    Recursion is depth-bounded to avoid pathological nesting.

    Example:
        >>> to_jsonable({"a": (1, 2), "b": float("nan")})
        {'a': [1, 2], 'b': None}
    """
    if _depth > 12:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return to_jsonable(obj.tolist(), _depth + 1)
        if isinstance(obj, _np.generic):
            return to_jsonable(obj.item(), _depth + 1)
    if _torch is not None and isinstance(obj, _torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist(), _depth + 1)
    # Lightweight dataclass / simple object support.
    to_dict = getattr(obj, "_asdict", None)
    if callable(to_dict):
        try:
            return to_jsonable(to_dict(), _depth + 1)
        except Exception:
            pass
    return repr(obj)


class TraceLogger:
    """Singleton JSON-Lines trace writer for one simulation run.

    Construction yields a *disabled* logger; call :meth:`configure` with a target
    file path to enable it. The ambient context (:attr:`tick_index`,
    :attr:`period_index`) is set by the engine and attached to every record.

    Example:
        >>> log = TraceLogger()
        >>> log.is_enabled()
        False
        >>> log.emit("tick", note="ignored while disabled") is None
        True
    """

    def __init__(self) -> None:
        self._fh: Optional[Any] = None
        self._enabled: bool = False
        self.path: Optional[str] = None
        self._seq: int = 0
        self.tick_index: int = 0
        self.period_index: int = -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure(self, path: str) -> None:
        """Open ``path`` for writing and enable the logger (closing any prior file).

        Parent directories are created as needed. A failure to open the file
        leaves the logger disabled (it never raises into the caller).
        """
        self.close()
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")
            self._enabled = True
            self.path = path
            self._seq = 0
            self.tick_index = 0
            self.period_index = -1
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

    def mark_tick(self, tick_index: int) -> None:
        """Set the current engine-tick index attached to subsequent events."""
        self.tick_index = int(tick_index)

    def mark_period(self, period_index: int) -> None:
        """Set the current negotiation-period index attached to subsequent events."""
        self.period_index = int(period_index)

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, event: str, **payload: Any) -> None:
        """Write one JSON-Lines record of type ``event`` (no-op while disabled).

        The record carries an ambient envelope (``seq`` / ``tick`` / ``period``)
        plus the JSON-sanitized ``payload``. All serialization / I/O is wrapped so
        a logging error can never propagate into the simulation.
        """
        if not self.is_enabled():
            return
        try:
            record = {
                "seq": self._seq,
                "event": str(event),
                "tick": self.tick_index,
                "period": self.period_index,
                **{k: to_jsonable(v) for k, v in payload.items()},
            }
            self._seq += 1
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
            self._fh.flush()  # type: ignore[union-attr]
        except Exception:
            # Never let a logging failure break the run.
            pass


# Process-global singleton.
_LOGGER = TraceLogger()


def get_logger() -> TraceLogger:
    """Return the process-global :class:`TraceLogger` singleton.

    Example:
        >>> get_logger() is get_logger()
        True
    """
    return _LOGGER
