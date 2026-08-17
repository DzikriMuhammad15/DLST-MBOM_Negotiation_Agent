"""Config serialization / override / validation for the dashboard (``07-02``).

This module is the dashboard's bridge between the JSON config editor in the
frontend and the strongly-typed :class:`~simulation.config.config.SimulationConfig`
dataclass tree (``08-01``). It provides three pure functions:

* :func:`config_to_dict` — serialize a :class:`SimulationConfig` to a plain JSON
  dict (via :func:`dataclasses.asdict`), JSON-round-trippable so it can fill the
  editor (REST ``GET /api/config/default``).
* :func:`config_from_dict` — rebuild a :class:`SimulationConfig` by applying an
  **override subset** on top of :func:`~simulation.config.default.default_config`.
  Because the override is a deep-merge of arbitrary editable subsets, **every**
  parameter of ``08-01`` is settable (a single field, a nested group, or a full
  config uploaded verbatim).
* :func:`validate_config` — return a list of **non-blocking** warning strings
  (``07-01``/``07-02``): (1) a taxi whose initial SoC is below ``critical_soc`` or
  ``B_area``; (2) an unusual visibility configuration (e.g. the fleet side cannot
  see its own taxis).

Tuples in the dataclass tree (``position``/``location``/``center`` and the
``allowed_slot_shift_range``) become JSON lists on the way out and are coerced
back to tuples on the way in, so a serialized-then-deserialized config is
structurally identical to the original.

This module imports only :mod:`simulation.config` (it is an app-layer helper).

Example
-------
>>> from simulation.config.default import default_config
>>> from dashboard.config_io import config_to_dict, config_from_dict
>>> data = config_to_dict(default_config())
>>> data["time"]["n_days"]
1
>>> cfg = config_from_dict({"time": {"n_days": 2}})
>>> cfg.time.n_days
2
>>> cfg.time.tick_resolution_min  # untouched default preserved
5
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Dict, List

from simulation.config.config import (
    SimulationConfig,
    InitialTaxiState,
    ChargingStationSpec,
    PortSpec,
    MarkingAreaSpec,
)
from simulation.config.default import default_config

# Dataclass fields whose value is a ``(min, max)`` / coordinate tuple. JSON has
# no tuple type, so these arrive as lists and are coerced back to tuples.
_TUPLE_FIELDS: frozenset = frozenset(
    {"position", "location", "center", "emergency_station_location",
     "allowed_slot_shift_range", "compensation_per_kwh_range",
     "order_reference_point"}
)


# =============================================================================
# Serialize: SimulationConfig -> JSON dict
# =============================================================================

def config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Serialize a :class:`SimulationConfig` to a JSON-native dict (``07-02``).

    Uses :func:`dataclasses.asdict` (recursive), which turns the whole nested
    dataclass tree into plain dicts / lists / scalars. Tuples (coordinates and
    ``allowed_slot_shift_range``) become lists — JSON-serializable and accepted
    back by :func:`config_from_dict`.

    Args:
        config: The configuration to serialize.

    Returns:
        A plain ``dict`` ready for ``json.dumps`` and the frontend editor.

    Example:
        >>> from simulation.config.default import default_config
        >>> d = config_to_dict(default_config())
        >>> d["battery"]["B_charge"]
        0.3
        >>> d["taxi_spec"]["initial_taxi_state"][0]["taxi_id"]
        'taxi_0'
    """

    return dataclasses.asdict(config)


# =============================================================================
# Deserialize: JSON dict (override subset) -> SimulationConfig
# =============================================================================

def config_from_dict(data: Dict[str, Any]) -> SimulationConfig:
    """Apply an editable override subset on top of the default config (``07-02``).

    The whole nested config tree is first serialized to a dict, then ``data`` is
    deep-merged into it (recursing into nested dicts, replacing lists / scalars
    wholesale), and finally the merged dict is rebuilt into a fresh
    :class:`SimulationConfig`. Passing ``{}`` returns a copy of the default
    config; passing a complete serialized config restores it exactly. Because the
    merge is recursive, **any** subset of ``08-01`` parameters — a single deep
    field or a whole group — can be overridden without restating the rest.

    Args:
        data: The override subset (or a full serialized config). Unknown keys are
            ignored by the dataclass rebuild step.

    Returns:
        A fresh :class:`SimulationConfig` with the overrides applied (the global
        defaults are never mutated).

    Example:
        >>> cfg = config_from_dict({"negotiation": {"max_steps": 40}})
        >>> cfg.negotiation.max_steps
        40
        >>> cfg.time.n_days  # default preserved
        1
    """

    base = config_to_dict(default_config())
    merged = _deep_merge(base, data or {})
    return _build_simulation_config(merged)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts recurse).

    Nested dicts are merged key-by-key; any non-dict value (scalar or list) in
    ``override`` replaces the corresponding ``base`` value wholesale. ``base`` is
    not mutated.

    Args:
        base: The default config dict.
        override: The user-supplied override subset.

    Returns:
        A new merged dict.

    Example:
        >>> _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9}})
        {'a': {'x': 1, 'y': 9}}
    """

    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _build_simulation_config(data: Dict[str, Any]) -> SimulationConfig:
    """Rebuild a :class:`SimulationConfig` from a fully-populated config dict.

    Walks the dataclass tree generically: for each dataclass field whose declared
    type is itself a dataclass it recurses; ``List[dataclass]`` fields rebuild
    each element; tuple-valued fields are coerced from JSON lists back to tuples.
    Keys not present in the target dataclass are ignored, so a slightly out-of-date
    editor payload still loads.

    Args:
        data: A config dict (typically the deep-merged default + override).

    Returns:
        The reconstructed :class:`SimulationConfig`.

    Example:
        >>> from simulation.config.default import default_config
        >>> d = config_to_dict(default_config())
        >>> _build_simulation_config(d).agent.fleet.agent_type
        'RLNegotiator'
    """

    return _from_dict(SimulationConfig, data)


def _from_dict(cls: type, data: Any) -> Any:
    """Generically reconstruct a dataclass instance ``cls`` from ``data``.

    Args:
        cls: The target dataclass type.
        data: The dict of field values (non-dict input is returned as-is).

    Returns:
        An instance of ``cls`` with each field rebuilt from ``data``.

    Example:
        >>> from simulation.config.config import TimeConfig
        >>> _from_dict(TimeConfig, {"n_days": 3}).n_days
        3
    """

    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data

    # ``from __future__ import annotations`` makes ``field.type`` a *string*, so
    # resolve the real types once per dataclass to detect nested dataclasses.
    hints = _resolved_hints(cls)
    kwargs: Dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        kwargs[field.name] = _coerce_field(field.name, hints.get(field.name), data[field.name])
    return cls(**kwargs)  # type: ignore[call-arg]


def _resolved_hints(cls: type) -> Dict[str, Any]:
    """Return ``cls``'s resolved type hints, cached per dataclass type.

    Args:
        cls: A dataclass type.

    Returns:
        Mapping of field name to its resolved (non-string) type.

    Example:
        >>> from simulation.config.config import TimeConfig
        >>> _resolved_hints(TimeConfig)["n_days"]
        <class 'int'>
    """

    cached = _HINTS_CACHE.get(cls)
    if cached is None:
        cached = typing.get_type_hints(cls)
        _HINTS_CACHE[cls] = cached
    return cached


def _coerce_field(name: str, field_type: Any, value: Any) -> Any:
    """Coerce one JSON value into a dataclass field's declared (resolved) shape.

    Handles three cases beyond plain scalars: nested dataclasses (recurse),
    ``List[...]`` of the known config element dataclasses (rebuild each), and
    tuple-valued fields (coerce a JSON list back to a tuple).

    Args:
        name: The field name (used for the tuple-field / list-element lookups).
        field_type: The field's resolved type (or ``None`` if unknown).
        value: The JSON value for this field.

    Returns:
        The coerced value ready to pass to the dataclass constructor.

    Example:
        >>> _coerce_field("n_days", int, 7)
        7
    """

    # Coordinate / range tuples: JSON list -> tuple.
    if name in _TUPLE_FIELDS and isinstance(value, list):
        return tuple(value)

    # Nested dataclass instance.
    if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
        return _from_dict(field_type, value)  # type: ignore[arg-type]

    # Known list-of-dataclass element fields (rebuild each element).
    element_cls = _LIST_ELEMENT_TYPES.get(name)
    if element_cls is not None and isinstance(value, list):
        return [_from_dict(element_cls, item) for item in value]

    return value


# Resolved-type-hint cache (avoids re-running ``get_type_hints`` per nested call).
_HINTS_CACHE: Dict[type, Dict[str, Any]] = {}


# Map ``List[...]`` config fields to their element dataclass so we can rebuild
# each element generically. (Field type annotations are strings under
# ``from __future__ import annotations``; this avoids ``typing.get_type_hints``.)
_LIST_ELEMENT_TYPES: Dict[str, type] = {
    "initial_taxi_state": InitialTaxiState,
    "stations": ChargingStationSpec,
    "ports": PortSpec,
    "areas": MarkingAreaSpec,
}


# =============================================================================
# Validate: non-blocking warnings
# =============================================================================

def validate_config(config: SimulationConfig) -> List[str]:
    """Return non-blocking warning strings for an editable config (``07-02``).

    Two checks (the user may proceed regardless — the warnings are advisory, e.g.
    when deliberately testing a stress scenario):

    1. **Critical initial SoC** — a taxi whose ``initial_taxi_state.soc`` is below
       ``battery.critical_soc`` (immediate danger, ``05-05``) or below
       ``battery.B_area`` (cannot safely commit to a marking area, ``01-04``).
    2. **Unusual visibility** — the fleet side's visibility list contains no
       ``taxi.*`` field, i.e. the fleet cannot observe its own taxis
       (``03-03``/``06-02``).

    Args:
        config: The configuration to validate.

    Returns:
        A list of human-readable warning strings (empty when nothing is unusual).

    Example:
        >>> from simulation.config.default import default_config
        >>> # default scenario: taxi_1 (0.45) and taxi_2 (0.20) are below B_area=0.50
        >>> len(validate_config(default_config()))
        2
    """

    warnings: List[str] = []
    warnings.extend(_warn_critical_initial_soc(config))
    warnings.extend(_warn_unusual_visibility(config))
    return warnings


def _warn_critical_initial_soc(config: SimulationConfig) -> List[str]:
    """Warn for any taxi whose initial SoC is below ``critical_soc`` or ``B_area``.

    Args:
        config: The configuration to inspect.

    Returns:
        One warning per offending taxi (critical takes precedence over B_area).

    Example:
        >>> from simulation.config.default import default_config
        >>> len(_warn_critical_initial_soc(default_config()))
        2
    """

    out: List[str] = []
    critical = float(config.battery.critical_soc)
    b_area = float(config.battery.B_area)
    for initial in config.taxi_spec.initial_taxi_state:
        soc = float(initial.soc)
        if soc < critical:
            out.append(
                f"Taxi {initial.taxi_id} initial SoC {soc:.2f} is below "
                f"critical_soc {critical:.2f} (immediate critical-battery danger)."
            )
        elif soc < b_area:
            out.append(
                f"Taxi {initial.taxi_id} initial SoC {soc:.2f} is below "
                f"B_area {b_area:.2f} (cannot safely commit to a marking area)."
            )
    return out


def _warn_unusual_visibility(config: SimulationConfig) -> List[str]:
    """Warn when the fleet side cannot observe any of its own taxi fields.

    Args:
        config: The configuration to inspect.

    Returns:
        A single warning if the fleet visibility list has no ``taxi.*`` field,
        otherwise an empty list.

    Example:
        >>> from simulation.config.default import default_config
        >>> _warn_unusual_visibility(default_config())
        []
    """

    fleet_visibility = config.state_feature.visibility.fleet
    sees_own_taxi = any(str(field).startswith("taxi.") for field in fleet_visibility)
    if not sees_own_taxi:
        return [
            "Unusual visibility: the fleet side cannot observe any taxi.* field "
            "(it does not see its own taxis)."
        ]
    return []
