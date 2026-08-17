"""Per-period negotiation record building & JSON serialisation (KOMPONEN N2).

This module turns one frozen negotiation period into a single, JSON-serialisable
record written to ``data/runs/<sim_id>/day_<i>/`` (one file per period, see
``RunPaths``). It covers three period outcomes:

* **deal** — the negotiation reached an agreement (an ``action_library`` index);
* **no_deal** — the negotiation ran but ended without an agreement;
* **no_negotiation** — the period was skipped before any negotiation (no candidate
  taxis, or no feasible bundle), recorded with an explicit reason code/message so
  the dashboard can show a notification (KR-4 / KR-14).

The record carries everything the dashboard and the offline analysis need (KR-5):
the serialised ``action_library``, the interleaved propose/respond step sequence
reconstructed from the two separate Episode payloads plus the run ``turn_order``
(KR-12), the final agreement, and the final utility per side.

Layer & import rules (INV-1 / INV-2):

* only stdlib (``json``, ``os``, ``typing``) plus the ``Bundle`` /
  ``Assignment`` / ``BookingShift`` value types from
  :mod:`simulation.bridge.action_library` (a sibling simulation layer, allowed);
* this module **never** imports ``negotiation/*`` or ``simulation/world``. Episode
  and Step objects are read by **duck typing** (``episode.steps``, ``step.utility``,
  ``step.action`` / ``step.value`` / ``step.log_prob``) — read-only, so the fully
  separate propose/respond data types stay untouched (INV-2).

Example:
    >>> rec = build_no_negotiation_record(
    ...     sim_id="s", day_index=0, period_index_in_day=0, sim_time_min=30.0,
    ...     hour=0, minute=30, reason_code=REASON_NO_CANDIDATES)
    >>> rec["status"]
    'no_negotiation'
    >>> rec["reason_code"]
    'no_candidate_taxis'
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

# Value types of one bundle outcome (simulation layer — allowed by INV-1).
from simulation.bridge.action_library import Assignment, Bundle, BookingShift

# =============================================================================
# Constants — period status (LLR-N2.1) and no-negotiation reasons (LLR-A6)
# =============================================================================

#: Period reached an agreement.
STATUS_DEAL: str = "deal"
#: Period ran a negotiation but ended without an agreement.
STATUS_NO_DEAL: str = "no_deal"
#: Period was skipped before any negotiation took place.
STATUS_NO_NEGOTIATION: str = "no_negotiation"

#: No candidate taxi needed / was eligible to charge this period.
REASON_NO_CANDIDATES: str = "no_candidate_taxis"
#: There were candidate taxis but the action_library was empty / had no bundle
#: satisfying the constraints, so the negotiation was skipped.
REASON_NO_FEASIBLE_BUNDLE: str = "no_feasible_bundle"

#: Human-readable message per reason code (shown by the dashboard, KR-14).
REASON_MESSAGES: Dict[str, str] = {
    REASON_NO_CANDIDATES: (
        "Tidak ada taksi kandidat yang perlu/eligible mengecas pada periode ini."
    ),
    REASON_NO_FEASIBLE_BUNDLE: (
        "Ada kandidat taksi tetapi action_library kosong / tidak ada bundle yang "
        "memenuhi constraint sehingga negosiasi dilewati."
    ),
}

# Turn kinds / side labels (must match negotiation_runner / PKG-05 turn_order).
_KIND_PROPOSE: str = "propose"
_KIND_RESPOND: str = "respond"
_SIDE_FLEET: str = "fleet"
_SIDE_SPKLU: str = "spklu"

#: Schema tag stamped onto every period record (versioned for the dashboard).
_SCHEMA: str = "ev_nego_period_v1"


def reason_message(reason_code: str) -> str:
    """Return the human-readable message for a no-negotiation reason code.

    Falls back to the raw code when it is unknown, so the record is always a
    valid, informative string (never raises).

    Args:
        reason_code: One of the ``REASON_*`` constants (LLR-A6).

    Returns:
        The Indonesian human-readable message, or the code itself if unmapped.

    Example:
        >>> reason_message(REASON_NO_CANDIDATES)[:11]
        'Tidak ada t'
        >>> reason_message("unknown_code")
        'unknown_code'
    """
    return REASON_MESSAGES.get(reason_code, reason_code)


# =============================================================================
# LLR-N2.2 — deterministic period file name
# =============================================================================

def period_filename(
    period_index_in_day: int, sim_time_min: float, status: str
) -> str:
    """Return the deterministic, lexicographically-sortable period file name.

    Zero-padding the period index and the (integer) simulation minute makes a
    lexicographic sort identical to a chronological sort. This matches the format
    produced by :meth:`simulation.metrics.run_paths.RunPaths.period_json_path`.

    Args:
        period_index_in_day: Index of the period within its day (starts at 0).
        sim_time_min: Simulation time (in minutes) at which the period occurred.
        status: Period status label (``"deal"`` / ``"no_deal"`` /
            ``"no_negotiation"``).

    Returns:
        A file name of the form
        ``period_{idx:04d}_t{int(min):07d}_{status}.json``.

    Example:
        >>> period_filename(0, 30.0, STATUS_DEAL)
        'period_0000_t0000030_deal.json'
        >>> period_filename(12, 750.0, STATUS_NO_NEGOTIATION)
        'period_0012_t0000750_no_negotiation.json'
    """
    return (
        f"period_{int(period_index_in_day):04d}"
        f"_t{int(sim_time_min):07d}_{status}.json"
    )


# =============================================================================
# LLR-N2.3 — serialise the action_library
# =============================================================================

def _serialize_shift(shift: Optional[BookingShift]) -> Optional[Dict[str, Any]]:
    """Return a JSON-serialisable dict for one booking shift, or ``None``.

    Args:
        shift: A :class:`~simulation.bridge.action_library.BookingShift`, or
            ``None`` when the assignment requires no shift.

    Returns:
        A dict mirroring the ``BookingShift`` attributes, or ``None``.

    Example:
        >>> _serialize_shift(None) is None
        True
    """
    if shift is None:
        return None
    return {
        "booking_id": shift.booking_id,
        "from_port_id": shift.from_port_id,
        "to_port_id": shift.to_port_id,
        "from_slot_start": shift.from_slot_start,
        "to_slot_start": shift.to_slot_start,
        "duration_grids": shift.duration_grids,
        "compensation": shift.compensation,
    }


def _serialize_assignment(assignment: Assignment) -> Dict[str, Any]:
    """Return a JSON-serialisable dict for one taxi assignment.

    Args:
        assignment: An :class:`~simulation.bridge.action_library.Assignment`.

    Returns:
        A dict with the assignment fields plus an ``unassigned`` flag and an
        optional ``shift`` sub-dict.

    Example:
        >>> from simulation.world.entities import NONE_PORT
        >>> a = Assignment("t0", NONE_PORT.id, 0, 0, 0.0)
        >>> _serialize_assignment(a)["unassigned"]
        True
    """
    return {
        "taxi_id": assignment.taxi_id,
        "port_id": assignment.port_id,
        "slot_start": assignment.slot_start,
        "duration_grids": assignment.duration_grids,
        "price_per_kwh": assignment.price_per_kwh,
        "unassigned": assignment.is_unassigned(),
        "shift": _serialize_shift(assignment.shift),
    }


def serialize_action_library(
    action_library: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Serialise an ordered ``action_library`` into JSON-serialisable dicts.

    Each :class:`~simulation.bridge.action_library.Bundle` becomes a dict with its
    library ``index``, the per-taxi ``assignments``, the number of assigned taxis,
    and whether it is the all-UNASSIGNED safety bundle.

    Args:
        action_library: The ordered sequence of bundles for one negotiation.

    Returns:
        A list of dicts, one per bundle, in library order. ``json.dumps`` of the
        result round-trips without error.

    Example:
        >>> from simulation.world.entities import NONE_PORT
        >>> b = Bundle((Assignment("t0", NONE_PORT.id, 0, 0, 0.0),))
        >>> out = serialize_action_library([b])
        >>> out[0]["index"], out[0]["is_all_unassigned"]
        (0, True)
    """
    serialized: List[Dict[str, Any]] = []
    for index, bundle in enumerate(action_library):
        assignments = [
            _serialize_assignment(a) for a in bundle.assignments
        ]
        serialized.append(
            {
                "index": index,
                "assignments": assignments,
                "assigned_count": bundle.assigned_count(),
                "is_all_unassigned": bundle.is_all_unassigned(),
            }
        )
    return serialized


# =============================================================================
# LLR-N2.4 — extract per-phase utilities from one Episode payload (duck typing)
# =============================================================================

def _coerce_action(action: Any) -> Any:
    """Coerce a step action into a JSON-serialisable form (``int`` / ``list`` / ``None``).

    NegMAS propose actions are library indices (ints); respond actions are 0/1.
    Anything that is not a plain ``int`` and not ``None`` is best-effort coerced to
    an ``int`` (scalar) or a ``list`` (sequence/array), else returned unchanged.

    Args:
        action: The raw ``step.action`` value (may be ``None``, an ``int``, a numpy
            scalar/array, a tensor, or a sequence).

    Returns:
        ``None``, an ``int``, or a ``list`` — a JSON-serialisable value.

    Example:
        >>> _coerce_action(None) is None
        True
        >>> _coerce_action(3)
        3
        >>> _coerce_action([1, 2])
        [1, 2]
    """
    if action is None:
        return None
    if isinstance(action, bool):
        return int(action)
    if isinstance(action, int):
        return action
    # Array-like (numpy array, tensor): convert to a plain list when possible.
    if hasattr(action, "tolist"):
        try:
            return action.tolist()
        except Exception:
            pass
    if isinstance(action, (list, tuple)):
        return [_coerce_action(item) for item in action]
    # Scalar-like (numpy/torch scalar): coerce to int.
    try:
        return int(action)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    """Coerce a numeric attribute to ``float``, or ``None`` when absent/invalid.

    Args:
        value: The raw attribute value (may be ``None`` or array/scalar-like).

    Returns:
        A ``float`` or ``None``.

    Example:
        >>> _coerce_float(None) is None
        True
        >>> _coerce_float(1)
        1.0
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_phase_utilities(episode: Any) -> List[Dict[str, Any]]:
    """Extract per-step utility / action / value / log_prob from one Episode.

    Reads ``episode.steps`` by duck typing (no ``negotiation/data_type`` import,
    INV-2). The returned list preserves the order of ``episode.steps`` (the order
    in which the agent acted in that phase).

    Args:
        episode: An Episode-like payload exposing ``.steps`` whose items expose
            ``.utility`` and optionally ``.action`` / ``.value`` / ``.log_prob``.
            ``None`` (or an episode without steps) yields an empty list.

    Returns:
        A list of ``{"utility", "action", "value", "log_prob"}`` dicts, one per
        step, in step order.

    Example:
        >>> class _S:
        ...     def __init__(self, u, a):
        ...         self.utility, self.action = u, a
        ...         self.value, self.log_prob = None, None
        >>> class _E:
        ...     steps = [_S(0.5, 3)]
        >>> extract_phase_utilities(_E())
        [{'utility': 0.5, 'action': 3, 'value': None, 'log_prob': None}]
    """
    if episode is None:
        return []
    steps = getattr(episode, "steps", None)
    if not steps:
        return []
    extracted: List[Dict[str, Any]] = []
    for step in steps:
        extracted.append(
            {
                "utility": _coerce_float(getattr(step, "utility", None)),
                "action": _coerce_action(getattr(step, "action", None)),
                "value": _coerce_float(getattr(step, "value", None)),
                "log_prob": _coerce_float(getattr(step, "log_prob", None)),
            }
        )
    return extracted


# =============================================================================
# LLR-N2.5 — reconstruct the interleaved propose/respond step sequence (KR-12)
# =============================================================================

def _fallback_step_sequence(
    fleet_propose: List[Dict[str, Any]],
    fleet_respond: List[Dict[str, Any]],
    spklu_propose: List[Dict[str, Any]],
    spklu_respond: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the canonical SAO interleaving when ``turn_order`` is unavailable.

    Canonical order (fleet initiates): fleet.propose[0], spklu.respond[0],
    spklu.propose[0], fleet.respond[0], fleet.propose[1], ... — i.e. the repeating
    pattern (A propose, B respond, B propose, A respond) — consuming each deque in
    order until all are exhausted.

    Args:
        fleet_propose: Per-step dicts for fleet's propose phase (from
            :func:`extract_phase_utilities`).
        fleet_respond: Per-step dicts for fleet's respond phase.
        spklu_propose: Per-step dicts for SPKLU's propose phase.
        spklu_respond: Per-step dicts for SPKLU's respond phase.

    Returns:
        The interleaved list of step dicts (each carries ``step``/``actor``/
        ``kind``/``utility``/``action``).

    Example:
        >>> fp = [{"utility": 0.1, "action": 0}]
        >>> sr = [{"utility": 0.2, "action": 1}]
        >>> seq = _fallback_step_sequence(fp, [], [], sr)
        >>> [(e["actor"], e["kind"]) for e in seq]
        [('fleet', 'propose'), ('spklu', 'respond')]
    """
    queues = {
        (_SIDE_FLEET, _KIND_PROPOSE): list(fleet_propose),
        (_SIDE_FLEET, _KIND_RESPOND): list(fleet_respond),
        (_SIDE_SPKLU, _KIND_PROPOSE): list(spklu_propose),
        (_SIDE_SPKLU, _KIND_RESPOND): list(spklu_respond),
    }
    pointers: Dict[Any, int] = {key: 0 for key in queues}
    # Repeating canonical pattern (A=fleet initiates, B=spklu responds).
    pattern = [
        (_SIDE_FLEET, _KIND_PROPOSE),
        (_SIDE_SPKLU, _KIND_RESPOND),
        (_SIDE_SPKLU, _KIND_PROPOSE),
        (_SIDE_FLEET, _KIND_RESPOND),
    ]
    sequence: List[Dict[str, Any]] = []
    step = 0
    while any(pointers[key] < len(queues[key]) for key in queues):
        progressed = False
        for actor, kind in pattern:
            key = (actor, kind)
            ptr = pointers[key]
            if ptr < len(queues[key]):
                item = queues[key][ptr]
                pointers[key] = ptr + 1
                sequence.append(
                    {
                        "step": step,
                        "actor": actor,
                        "kind": kind,
                        "utility": item.get("utility"),
                        "action": item.get("action"),
                    }
                )
                step += 1
                progressed = True
        if not progressed:
            break
    return sequence


def reconstruct_step_sequence(
    turn_order: Optional[List[Dict[str, Any]]],
    fleet_propose: List[Dict[str, Any]],
    fleet_respond: List[Dict[str, Any]],
    spklu_propose: List[Dict[str, Any]],
    spklu_respond: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reconstruct the interleaved propose/respond step sequence (KR-12).

    Uses ``turn_order`` (from the negotiation runner, PKG-05) as the source of the
    chronological order. For each ``{"step", "actor", "kind"}`` descriptor, the next
    item is pulled from the matching phase deque (one pointer per ``(actor, kind)``),
    so the four fully-separate propose/respond payloads (INV-2) are stitched back
    into a single ordered timeline the dashboard & plots consume.

    Degradation: if a pointer runs past its deque (a length mismatch between
    ``turn_order`` and the payloads) the entry is emitted with ``"missing": True``
    and null utility/action rather than crashing. If ``turn_order`` is empty/``None``
    the canonical SAO interleaving is used as a fallback.

    Args:
        turn_order: Ordered turn descriptors from the runner, or ``None``/empty.
        fleet_propose: Fleet propose-phase per-step dicts.
        fleet_respond: Fleet respond-phase per-step dicts.
        spklu_propose: SPKLU propose-phase per-step dicts.
        spklu_respond: SPKLU respond-phase per-step dicts.

    Returns:
        The ordered list of step dicts, each
        ``{"step", "actor", "kind", "utility", "action"[, "missing"]}``.

    Example:
        >>> to = [{"step": 0, "actor": "fleet", "kind": "propose"},
        ...       {"step": 1, "actor": "spklu", "kind": "respond"}]
        >>> fp = [{"utility": 0.7, "action": 2}]
        >>> sr = [{"utility": 0.3, "action": 0}]
        >>> seq = reconstruct_step_sequence(to, fp, [], [], sr)
        >>> [(e["step"], e["actor"], e["kind"], e["utility"]) for e in seq]
        [(0, 'fleet', 'propose', 0.7), (1, 'spklu', 'respond', 0.3)]
    """
    if not turn_order:
        return _fallback_step_sequence(
            fleet_propose, fleet_respond, spklu_propose, spklu_respond
        )

    queues = {
        (_SIDE_FLEET, _KIND_PROPOSE): list(fleet_propose),
        (_SIDE_FLEET, _KIND_RESPOND): list(fleet_respond),
        (_SIDE_SPKLU, _KIND_PROPOSE): list(spklu_propose),
        (_SIDE_SPKLU, _KIND_RESPOND): list(spklu_respond),
    }
    pointers: Dict[Any, int] = {key: 0 for key in queues}
    sequence: List[Dict[str, Any]] = []
    for entry in turn_order:
        actor = entry.get("actor")
        kind = entry.get("kind")
        step = entry.get("step")
        key = (actor, kind)
        queue = queues.get(key)
        if queue is None:
            # Unknown (actor, kind) descriptor: record as missing, do not crash.
            sequence.append(
                {
                    "step": step,
                    "actor": actor,
                    "kind": kind,
                    "utility": None,
                    "action": None,
                    "missing": True,
                }
            )
            continue
        ptr = pointers[key]
        if ptr >= len(queue):
            sequence.append(
                {
                    "step": step,
                    "actor": actor,
                    "kind": kind,
                    "utility": None,
                    "action": None,
                    "missing": True,
                }
            )
            continue
        item = queue[ptr]
        pointers[key] = ptr + 1
        sequence.append(
            {
                "step": step,
                "actor": actor,
                "kind": kind,
                "utility": item.get("utility"),
                "action": item.get("action"),
            }
        )
    return sequence


# =============================================================================
# LLR-N2.6 — build the deal / no-deal period record
# =============================================================================

def _agreement_index(agreement: Any) -> Optional[int]:
    """Return the library index encoded in a NegMAS agreement, or ``None``.

    The agreement is an outcome tuple ``(idx,)`` (or ``None`` for no-deal). The
    single element indexes the ``action_library`` (index-direct, Gotcha G1).

    Args:
        agreement: An outcome tuple ``(idx,)``, an ``int``, or ``None``.

    Returns:
        The integer index, or ``None`` when there is no agreement.

    Example:
        >>> _agreement_index((2,))
        2
        >>> _agreement_index(None) is None
        True
    """
    if agreement is None:
        return None
    if isinstance(agreement, (tuple, list)) and agreement:
        try:
            return int(agreement[0])
        except (TypeError, ValueError):
            return None
    try:
        return int(agreement)
    except (TypeError, ValueError):
        return None


def build_period_record(
    sim_id: str,
    day_index: int,
    period_index_in_day: int,
    sim_time_min: float,
    hour: int,
    minute: int,
    status: str,
    action_library: Sequence[Any],
    steps_sequence: List[Dict[str, Any]],
    agreement: Any,
    deal_bundle: Any,
    ufun_fleet_final: Optional[float],
    ufun_spklu_final: Optional[float],
    candidates_count: int,
) -> Dict[str, Any]:
    """Build the JSON-serialisable record for a deal / no-deal period (KR-5).

    Bundles everything the dashboard & offline analysis need: metadata, the
    serialised ``action_library``, the interleaved step sequence (from
    :func:`reconstruct_step_sequence`), the final agreement, the deal assignments
    (when any), and the final utility per side.

    Args:
        sim_id: Run identifier.
        day_index: Simulated day index.
        period_index_in_day: Period index within the day (starts at 0).
        sim_time_min: Simulation time (minutes) at the period.
        hour: Simulated hour of day.
        minute: Simulated minute of hour.
        status: ``"deal"`` or ``"no_deal"``.
        action_library: The ordered bundles offered this negotiation.
        steps_sequence: The reconstructed interleaved step sequence.
        agreement: The NegMAS outcome ``(idx,)`` or ``None``.
        deal_bundle: The won :class:`Bundle` (``action_library[idx]``) or ``None``.
        ufun_fleet_final: Fleet's final utility (deal: ``U(agreement)``;
            no-deal: the actual reservation/no-deal value, #C4) or ``None``.
        ufun_spklu_final: SPKLU's final utility (analogous) or ``None``.
        candidates_count: Number of candidate taxis in this negotiation.

    Returns:
        A JSON-serialisable period record (schema ``ev_nego_period_v1``).

    Example:
        >>> from simulation.world.entities import NONE_PORT
        >>> b = Bundle((Assignment("t0", NONE_PORT.id, 0, 0, 0.0),))
        >>> rec = build_period_record(
        ...     "s", 0, 0, 30.0, 0, 30, STATUS_NO_DEAL, [b], [], None, None,
        ...     None, None, 1)
        >>> rec["status"], rec["deal"], rec["agreement"]
        ('no_deal', None, None)
    """
    agreement_index = _agreement_index(agreement)
    agreement_field: Optional[Dict[str, Any]] = (
        {"index": agreement_index} if agreement_index is not None else None
    )

    deal_field: Optional[Dict[str, Any]] = None
    if status == STATUS_DEAL and deal_bundle is not None:
        deal_field = {
            "assignments": [
                _serialize_assignment(a) for a in deal_bundle.assignments
            ],
            "deal_index": agreement_index,
        }

    return {
        "schema": _SCHEMA,
        "sim_id": sim_id,
        "day_index": day_index,
        "period_index_in_day": period_index_in_day,
        "sim_time_min": sim_time_min,
        "hour": hour,
        "minute": minute,
        "status": status,
        "candidates_count": candidates_count,
        "action_library": serialize_action_library(action_library),
        "steps": steps_sequence,
        "agreement": agreement_field,
        "deal": deal_field,
        "final_utility": {
            "fleet": ufun_fleet_final,
            "spklu": ufun_spklu_final,
        },
    }


# =============================================================================
# LLR-N2.7 — build the no-negotiation record (KR-4 / KR-7)
# =============================================================================

def build_no_negotiation_record(
    sim_id: str,
    day_index: int,
    period_index_in_day: int,
    sim_time_min: float,
    hour: int,
    minute: int,
    reason_code: str,
    candidates_count: int = 0,
) -> Dict[str, Any]:
    """Build the record for a period skipped before any negotiation (KR-4).

    Records the no-negotiation status plus an explicit reason code and message so
    the dashboard can show a notification (KR-14). The record stays a valid period
    JSON with an empty ``action_library``/``steps`` and zero final utility (KR-7:
    no-negotiation periods enter the metric series with utility ``0.0``).

    Args:
        sim_id: Run identifier.
        day_index: Simulated day index.
        period_index_in_day: Period index within the day (starts at 0).
        sim_time_min: Simulation time (minutes) at the period.
        hour: Simulated hour of day.
        minute: Simulated minute of hour.
        reason_code: One of the ``REASON_*`` constants (LLR-A6).
        candidates_count: Number of candidate taxis (``0`` for no-candidates).

    Returns:
        A JSON-serialisable no-negotiation record (schema ``ev_nego_period_v1``).

    Example:
        >>> rec = build_no_negotiation_record(
        ...     "s", 0, 1, 60.0, 1, 0, REASON_NO_FEASIBLE_BUNDLE, candidates_count=2)
        >>> rec["status"], rec["reason_code"], rec["candidates_count"]
        ('no_negotiation', 'no_feasible_bundle', 2)
        >>> rec["final_utility"]
        {'fleet': 0.0, 'spklu': 0.0}
    """
    return {
        "schema": _SCHEMA,
        "sim_id": sim_id,
        "day_index": day_index,
        "period_index_in_day": period_index_in_day,
        "sim_time_min": sim_time_min,
        "hour": hour,
        "minute": minute,
        "status": STATUS_NO_NEGOTIATION,
        "reason_code": reason_code,
        "reason_message": reason_message(reason_code),
        "candidates_count": candidates_count,
        "action_library": [],
        "steps": [],
        "agreement": None,
        "deal": None,
        "final_utility": {"fleet": 0.0, "spklu": 0.0},
    }


# =============================================================================
# LLR-N2.8 — writer
# =============================================================================

def write_period_json(
    path: str, record: Dict[str, Any], indent: int = 2
) -> None:
    """Write one complete period record to ``path`` as a single JSON document.

    Creates the parent directory if needed and dumps the record once (INV-6:
    one whole file per period). The function deliberately lets any IO error
    propagate to the caller (RunReporter / scheduler wrap it in try/except, N5/M3)
    so the failure can be logged without bringing down the run.

    Args:
        path: Absolute or relative target file path (``.json``).
        record: The JSON-serialisable record (from ``build_*_record``).
        indent: JSON indentation (from ``config.reporting.json_indent`` when passed
            by the caller; this module never imports config itself).

    Returns:
        ``None``. Side effect: the file at ``path`` is created/overwritten.

    Example:
        >>> import os, tempfile, json
        >>> rec = build_no_negotiation_record(
        ...     "s", 0, 0, 0.0, 0, 0, REASON_NO_CANDIDATES)
        >>> p = os.path.join(tempfile.mkdtemp(), "period_0000_t0000000_no_negotiation.json")
        >>> write_period_json(p, rec)
        >>> json.load(open(p, encoding="utf-8"))["status"]
        'no_negotiation'
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=indent, ensure_ascii=False, default=str)
