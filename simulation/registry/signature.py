"""Agent compatibility :class:`Signature` + :class:`SignatureMismatchError` (``06-02``).

A **signature** is the versioned, structural fingerprint of an agent: the exact
feature schema, visibility regime, feature dimensions, mode, side, and critic
enrichment flags it was built with (Contract C7 / Gotcha G6). Because feature
spec and visibility are free experiment variables, an agent trained under one
configuration may be silently incompatible with another. The signature is stored
with the agent **and** embedded in the one-day history payload (``02-10``), and
checked at two points (``06-02``):

1. loading an agent from the registry into a new simulation — agent signature vs.
   the current config;
2. offline training — agent signature vs. the history-file signature.

On any mismatch a :class:`SignatureMismatchError` is raised whose message **names
the offending field and both values** (never a bare "incompatible"). Unusual but
valid visibility (e.g. the fleet not seeing its own taxis) yields a soft
``warnings.warn`` only — the user owns that choice.

The signature fields (``06-02``):

```
signature = {
  schema_version:    int,
  feature_spec:      {fleet:{taxi:[...],port:[...],...}, spklu:{...}},   # 03-03
  visibility_config: {fleet:[...], spklu:[...]},
  dims:              {n_max, k_max, p_max, a_max, f_taxi, f_port, f_area,
                      f_scalar, f_progress, f_cand, f_privileged,
                      phi_hidden_dim, pooling},
  mode:              "PADDED"|"ENCODER",
  side:              "fleet"|"spklu",
  critic_flags:      {sees_candidate_pool, privileged_info},            # 03-06
}
```

This module imports only :mod:`simulation.config` and the standard library; the
``FeatureDims`` it reads is duck-typed (any object exposing the dim attributes),
so it does not hard-depend on torch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Canonical dim attribute names captured from a ``FeatureDims`` (Contract C7).
_DIM_FIELDS: Tuple[str, ...] = (
    "f_taxi",
    "f_port",
    "f_area",
    "f_scalar",
    "f_progress",
    "f_cand",
    "f_privileged",
    "n_max",
    "p_max",
    "a_max",
    "k_max",
    "phi_hidden_dim",
    "phi_out_dim",
    "pooling",
)
# Feature-block roles compared per side (``03-03``).
_BLOCK_ROLES: Tuple[str, ...] = ("taxi", "port", "area", "booking", "scalar", "cand")
_SIDES: Tuple[str, ...] = ("fleet", "spklu")


class SignatureMismatchError(ValueError):
    """Raised when two signatures disagree on a structural field (Contract C7).

    The message always names the field (dotted path) and both differing values,
    so the user can see exactly what changed (e.g. a feature spec dropped, a mode
    flipped, or a dim widened).

    Example:
        >>> err = SignatureMismatchError("mode", "PADDED", "ENCODER")
        >>> str(err)
        "Signature mismatch on 'mode': agent has 'PADDED' but current config has 'ENCODER'."
        >>> err.field
        'mode'
    """

    def __init__(self, field_path: str, agent_value: Any, other_value: Any) -> None:
        """Build a field-naming mismatch error.

        Args:
            field_path: Dotted path of the mismatching field (e.g.
                ``"feature_spec.fleet.taxi"``).
            agent_value: The value recorded on the agent signature.
            other_value: The value found in the current config / history.

        Example:
            >>> SignatureMismatchError("side", "fleet", "spklu").other_value
            'spklu'
        """

        self.field = field_path
        self.agent_value = agent_value
        self.other_value = other_value
        message = (
            f"Signature mismatch on '{field_path}': "
            f"agent has {agent_value!r} but current config has {other_value!r}."
        )
        super().__init__(message)


@dataclass(frozen=True)
class Signature:
    """Versioned structural fingerprint of one side's agent (``06-02``).

    Parameters
    ----------
    schema_version:
        Integer version (``training.schema_version``); a bump means the layout
        itself changed.
    side:
        ``"fleet"`` or ``"spklu"`` — the side this agent negotiates.
    mode:
        ``"PADDED"`` or ``"ENCODER"`` front-end mode.
    feature_spec:
        Per-side feature-block field lists ``{side: {role: [field, ...]}}``.
    visibility_config:
        Per-side visible world-field lists ``{side: [field, ...]}``.
    dims:
        The captured ``FeatureDims`` widths / caps (see :data:`_DIM_FIELDS`).
    critic_flags:
        ``{"sees_candidate_pool": bool, "privileged_info": bool}`` (G6 / ``03-06``).

    Example:
        >>> sig = Signature(schema_version=1, side="fleet", mode="PADDED",
        ...     feature_spec={"fleet": {"taxi": ["soc"]}, "spklu": {}},
        ...     visibility_config={"fleet": ["taxi.soc"], "spklu": []},
        ...     dims={"k_max": 16, "f_taxi": 1},
        ...     critic_flags={"sees_candidate_pool": False, "privileged_info": False})
        >>> sig.mode
        'PADDED'
    """

    schema_version: int
    side: str
    mode: str
    feature_spec: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    visibility_config: Dict[str, List[str]] = field(default_factory=dict)
    dims: Dict[str, Any] = field(default_factory=dict)
    critic_flags: Dict[str, bool] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization (for the registry artifact + history payload)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of the signature (``06-02``).

        Example:
            >>> Signature(1, "fleet", "PADDED").to_dict()["side"]
            'fleet'
        """

        return {
            "schema_version": int(self.schema_version),
            "side": str(self.side),
            "mode": str(self.mode),
            "feature_spec": {
                s: {r: list(v) for r, v in blocks.items()}
                for s, blocks in self.feature_spec.items()
            },
            "visibility_config": {
                s: list(v) for s, v in self.visibility_config.items()
            },
            "dims": dict(self.dims),
            "critic_flags": dict(self.critic_flags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signature":
        """Rebuild a :class:`Signature` from its serialized dict.

        Example:
            >>> d = Signature(1, "spklu", "ENCODER").to_dict()
            >>> Signature.from_dict(d).mode
            'ENCODER'
        """

        return cls(
            schema_version=int(data.get("schema_version", 0)),
            side=str(data.get("side", "")),
            mode=str(data.get("mode", "")),
            feature_spec=data.get("feature_spec", {}) or {},
            visibility_config=data.get("visibility_config", {}) or {},
            dims=data.get("dims", {}) or {},
            critic_flags=data.get("critic_flags", {}) or {},
        )

    # ------------------------------------------------------------------
    # Mismatch checking (Contract C7 — names the field)
    # ------------------------------------------------------------------

    def assert_compatible(self, other: "Signature") -> None:
        """Raise :class:`SignatureMismatchError` if ``other`` differs structurally.

        ``self`` is the agent's stored signature; ``other`` is the
        current-config (or history) signature. The first differing field raises,
        naming the dotted field path and both values. The order checked is:
        ``schema_version`` -> ``side`` -> ``mode`` -> ``critic_flags`` ->
        ``dims`` -> ``feature_spec`` (per side/role). ``visibility_config`` is NOT
        a hard gate — an unusual-but-valid difference only emits a soft warning
        (the user owns visibility).

        Args:
            other: The signature to compare against.

        Raises:
            SignatureMismatchError: On the first structural mismatch (field named).

        Example:
            >>> a = build_signature_from_parts(side="fleet", mode="PADDED",
            ...     feature_spec={"fleet": {"taxi": ["soc"]}, "spklu": {}},
            ...     visibility_config={"fleet": [], "spklu": []},
            ...     dims={"k_max": 16}, critic_flags={}, schema_version=1)
            >>> b = build_signature_from_parts(side="fleet", mode="ENCODER",
            ...     feature_spec={"fleet": {"taxi": ["soc"]}, "spklu": {}},
            ...     visibility_config={"fleet": [], "spklu": []},
            ...     dims={"k_max": 16}, critic_flags={}, schema_version=1)
            >>> try:
            ...     a.assert_compatible(b)
            ... except SignatureMismatchError as exc:
            ...     print(exc.field, str(exc))
            mode Signature mismatch on 'mode': agent has 'PADDED' but current config has 'ENCODER'.
        """

        if int(self.schema_version) != int(other.schema_version):
            raise SignatureMismatchError(
                "schema_version", self.schema_version, other.schema_version
            )
        if str(self.side) != str(other.side):
            raise SignatureMismatchError("side", self.side, other.side)
        if str(self.mode) != str(other.mode):
            raise SignatureMismatchError("mode", self.mode, other.mode)

        # ── critic flags (G6) ─────────────────────────────────────────────
        for flag in ("sees_candidate_pool", "privileged_info"):
            a_val = bool(self.critic_flags.get(flag, False))
            o_val = bool(other.critic_flags.get(flag, False))
            if a_val != o_val:
                raise SignatureMismatchError(
                    f"critic_flags.{flag}", a_val, o_val
                )

        # ── dims (Contract C7) ────────────────────────────────────────────
        for dim in _DIM_FIELDS:
            if dim not in self.dims and dim not in other.dims:
                continue
            a_val = self.dims.get(dim)
            o_val = other.dims.get(dim)
            if a_val != o_val:
                raise SignatureMismatchError(f"dims.{dim}", a_val, o_val)

        # ── feature_spec per side/role (the canonical C7 example) ─────────
        for s in _SIDES:
            a_blocks = self.feature_spec.get(s, {}) or {}
            o_blocks = other.feature_spec.get(s, {}) or {}
            for role in _BLOCK_ROLES:
                a_list = list(a_blocks.get(role, []) or [])
                o_list = list(o_blocks.get(role, []) or [])
                if a_list != o_list:
                    raise SignatureMismatchError(
                        f"feature_spec.{s}.{role}", a_list, o_list
                    )

        # ── visibility (soft warning only — never raises) ────────────────
        self._warn_visibility(other)

    def _warn_visibility(self, other: "Signature") -> None:
        """Emit a soft warning when visibility differs (non-blocking, ``06-02``).

        Example:
            >>> import warnings
            >>> a = build_signature_from_parts(side="fleet", mode="PADDED",
            ...     feature_spec={}, visibility_config={"fleet": ["taxi.soc"]},
            ...     dims={}, critic_flags={}, schema_version=1)
            >>> b = build_signature_from_parts(side="fleet", mode="PADDED",
            ...     feature_spec={}, visibility_config={"fleet": []},
            ...     dims={}, critic_flags={}, schema_version=1)
            >>> with warnings.catch_warnings(record=True) as w:
            ...     warnings.simplefilter("always")
            ...     a._warn_visibility(b)
            ...     bool(w)
            True
        """

        for s in _SIDES:
            a_vis = list(self.visibility_config.get(s, []) or [])
            o_vis = list(other.visibility_config.get(s, []) or [])
            if a_vis != o_vis:
                warnings.warn(
                    f"visibility_config.{s} differs (agent {a_vis} vs config "
                    f"{o_vis}); proceeding (visibility is the user's choice).",
                    stacklevel=2,
                )


# =============================================================================
# Builders
# =============================================================================

def _feature_spec_to_dict(feature_spec: Any) -> Dict[str, Dict[str, List[str]]]:
    """Convert a :class:`FeatureSpecConfig` to a plain per-side/role dict.

    Example:
        >>> from simulation.config.config import FeatureSpecConfig, FeatureBlockSpec
        >>> fs = FeatureSpecConfig(fleet=FeatureBlockSpec(taxi=["soc"]))
        >>> _feature_spec_to_dict(fs)["fleet"]["taxi"]
        ['soc']
    """

    out: Dict[str, Dict[str, List[str]]] = {}
    for s in _SIDES:
        block = getattr(feature_spec, s, None)
        roles: Dict[str, List[str]] = {}
        for role in _BLOCK_ROLES:
            roles[role] = list(getattr(block, role, []) or []) if block is not None else []
        out[s] = roles
    return out


def _visibility_to_dict(visibility: Any) -> Dict[str, List[str]]:
    """Convert a :class:`VisibilityConfig` to a plain per-side dict.

    Example:
        >>> from simulation.config.config import VisibilityConfig
        >>> _visibility_to_dict(VisibilityConfig(fleet=["taxi.soc"]))["fleet"]
        ['taxi.soc']
    """

    return {s: list(getattr(visibility, s, []) or []) for s in _SIDES}


def _dims_to_dict(feature_dims: Any) -> Dict[str, Any]:
    """Capture the ``FeatureDims`` widths / caps into a plain dict (C7).

    Args:
        feature_dims: Any object exposing the :data:`_DIM_FIELDS` attributes
            (the bridge-emitted :class:`FeatureDims`).

    Example:
        >>> class D:
        ...     f_taxi = 5; f_port = 3; f_area = 3; f_scalar = 2; f_progress = 3
        ...     f_cand = 6; f_privileged = 0; n_max = 10; p_max = 2; a_max = 10
        ...     k_max = 16; phi_hidden_dim = 32; pooling = "mean"
        >>> _dims_to_dict(D())["k_max"]
        16
    """

    if feature_dims is None:
        return {}
    out: Dict[str, Any] = {}
    for dim in _DIM_FIELDS:
        value = getattr(feature_dims, dim, None)
        if value is None:
            continue
        out[dim] = value if dim == "pooling" else (
            value if isinstance(value, str) else int(value)
        )
    return out


def build_signature(
    config: Any,
    side: str,
    feature_dims: Any,
    schema_version: Optional[int] = None,
) -> Signature:
    """Build a side's :class:`Signature` from the config + bridge ``FeatureDims``.

    Pulls ``feature_spec`` / ``visibility`` from ``config.state_feature``, the
    mode + critic flags from ``config.agent.<side>``, and the dims from the
    supplied ``feature_dims`` (Contract C7). ``schema_version`` defaults to
    ``config.training.schema_version``.

    Args:
        config: The :class:`~simulation.config.config.SimulationConfig`.
        side: ``"fleet"`` or ``"spklu"``.
        feature_dims: The bridge-emitted ``FeatureDims`` for this side.
        schema_version: Optional override of the recorded schema version.

    Returns:
        The constructed :class:`Signature`.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.bridge.feature_builder import feature_dims
        >>> cfg = default_config()
        >>> sig = build_signature(cfg, "fleet", feature_dims(cfg, "fleet"))
        >>> sig.side, sig.mode
        ('fleet', 'PADDED')
    """

    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}.")
    state_feature = config.state_feature
    agent_side = getattr(config.agent, side)
    enrichment = getattr(agent_side, "critic_enrichment", None)
    version = (
        int(schema_version) if schema_version is not None
        else int(config.training.schema_version)
    )
    return Signature(
        schema_version=version,
        side=side,
        mode=str(agent_side.mode),
        feature_spec=_feature_spec_to_dict(state_feature.feature_spec),
        visibility_config=_visibility_to_dict(state_feature.visibility),
        dims=_dims_to_dict(feature_dims),
        critic_flags={
            "sees_candidate_pool": bool(
                getattr(enrichment, "critic_sees_candidate_pool", False)
            ),
            "privileged_info": bool(
                getattr(enrichment, "critic_privileged_info", False)
            ),
        },
    )


def build_signature_from_parts(
    side: str,
    mode: str,
    feature_spec: Dict[str, Dict[str, List[str]]],
    visibility_config: Dict[str, List[str]],
    dims: Dict[str, Any],
    critic_flags: Dict[str, bool],
    schema_version: int,
) -> Signature:
    """Build a :class:`Signature` straight from plain parts (tests / history).

    Example:
        >>> build_signature_from_parts("fleet", "PADDED", {}, {}, {}, {}, 1).side
        'fleet'
    """

    return Signature(
        schema_version=int(schema_version),
        side=str(side),
        mode=str(mode),
        feature_spec=feature_spec or {},
        visibility_config=visibility_config or {},
        dims=dims or {},
        critic_flags=critic_flags or {},
    )


if __name__ == "__main__":  # pragma: no cover - lightweight self-check
    import doctest

    failures, _ = doctest.testmod()
    if failures:
        raise SystemExit(f"{failures} doctest failure(s)")
    print("Signature doctests passed")
