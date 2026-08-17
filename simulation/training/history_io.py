"""Persist / reload negotiation history for offline training (end-user seam).

The :class:`~simulation.metrics.collector.MetricsCollector` keeps every
finalized :class:`~negotiation.data_type.episode.Episode` in memory under
``negotiation_payload_history`` (when ``keep_payload_history=True``). Those live
Episode objects are exactly what :meth:`Orchestrator.train_offline` consumes, but
they cannot survive a process boundary as JSON: a Step's ``state`` carries packed
observations (tensors / numpy / nested dicts) that ``json.dump(..., default=str)``
would stringify irreversibly.

This module therefore provides a **pickle**-based round-trip so an end user can:

1. run a simulation and ``save_history(...)`` the recorded episodes to disk; then
2. in a later process, ``load_history(...)`` and feed the episodes straight into
   :meth:`Orchestrator.train_offline` (which re-checks the embedded per-side
   :class:`~simulation.registry.signature.Signature` against the live agents).

The saved document is a plain dict::

    {
        "schema":     <training.schema_version:int>,
        "signatures": {"fleet": {...}, "spklu": {...}},   # Signature.to_dict()
        "episodes":   {"fleet": [(phase, Episode), ...],
                       "spklu": [(phase, Episode), ...]},
        "meta":       {"sim_id": ..., "n_episodes": ...},  # informational only
    }

It uses only the standard library plus the project's own types (no torch import
here; tensors inside the episodes are pickled by their own reducers).
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from simulation.registry.signature import Signature, SignatureMismatchError

_SIDES: Tuple[str, ...] = ("fleet", "spklu")

_SCHEMA_KEY: str = "schema"
_SIGNATURE_KEY: str = "signatures"
_EPISODES_KEY: str = "episodes"
_META_KEY: str = "meta"


def episodes_by_side_from_collector(collector: Any) -> Dict[str, List[Tuple[str, Any]]]:
    """Group a collector's recorded payloads into ``{side: [(phase, episode), ...]}``.

    Reads :attr:`MetricsCollector.negotiation_payload_history` — a flat list of
    ``{"side", "phase", "episode"}`` records — and buckets it per side in the
    exact shape :meth:`Orchestrator.train_offline` expects. Records whose
    ``episode`` is ``None`` or whose ``side`` is not a known negotiating side are
    skipped.

    Args:
        collector: A :class:`~simulation.metrics.collector.MetricsCollector` that
            was constructed with ``keep_payload_history=True``.

    Returns:
        ``{"fleet": [(phase, Episode), ...], "spklu": [...]}``.

    Example:
        >>> class _C: negotiation_payload_history = [
        ...     {"side": "fleet", "phase": "propose", "episode": object()}]
        >>> out = episodes_by_side_from_collector(_C())
        >>> list(out.keys())
        ['fleet', 'spklu']
        >>> len(out["fleet"])
        1
    """

    out: Dict[str, List[Tuple[str, Any]]] = {side: [] for side in _SIDES}
    history = getattr(collector, "negotiation_payload_history", None) or []
    for record in history:
        side = record.get("side")
        phase = record.get("phase")
        episode = record.get("episode")
        if episode is None or side not in out:
            continue
        out[side].append((phase, episode))
    return out


def save_history(
    collector: Any,
    orchestrator: Any,
    path: str,
    sim_id: Optional[str] = None,
) -> str:
    """Pickle the collector's recorded episodes + per-side signatures to ``path``.

    The signatures are taken from ``orchestrator.signature_for(side)`` so that a
    later offline-training process can verify the data was produced under a
    structurally compatible config (Contract C7). Parent directories are created
    as needed.

    Args:
        collector: The run's :class:`MetricsCollector` (``keep_payload_history``).
        orchestrator: The run's :class:`Orchestrator` (for the live signatures).
        path: Destination file path (``.pkl`` recommended).
        sim_id: Optional run id stored in the document's ``meta`` for traceability.

    Returns:
        The written file path.

    Example:
        >>> save_history.__name__
        'save_history'
    """

    episodes_by_side = episodes_by_side_from_collector(collector)
    signatures: Dict[str, Any] = {}
    for side in _SIDES:
        try:
            signatures[side] = orchestrator.signature_for(side).to_dict()
        except Exception:
            signatures[side] = None

    schema = 1
    try:
        schema = int(orchestrator.config.training.schema_version)
    except Exception:
        pass

    document = {
        _SCHEMA_KEY: schema,
        _SIGNATURE_KEY: signatures,
        _EPISODES_KEY: episodes_by_side,
        _META_KEY: {
            "sim_id": sim_id,
            "n_episodes": {side: len(eps) for side, eps in episodes_by_side.items()},
        },
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(document, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_history(path: str) -> Tuple[Dict[str, List[Tuple[str, Any]]], Dict[str, Any]]:
    """Load a pickled history document written by :func:`save_history`.

    Args:
        path: Path to the ``.pkl`` history file.

    Returns:
        ``(episodes_by_side, signatures)`` ready to pass to
        :meth:`Orchestrator.train_offline`.

    Example:
        >>> load_history.__name__
        'load_history'
    """

    with open(path, "rb") as fh:
        document = pickle.load(fh)
    episodes_by_side = document.get(_EPISODES_KEY, {}) or {}
    signatures = document.get(_SIGNATURE_KEY, {}) or {}
    return episodes_by_side, signatures


def history_summary(path: str) -> Dict[str, Any]:
    """Return a small informational summary of a history file (no training).

    Args:
        path: Path to the ``.pkl`` history file.

    Returns:
        ``{"schema", "sim_id", "n_episodes": {side: count}}``.

    Example:
        >>> history_summary.__name__
        'history_summary'
    """

    with open(path, "rb") as fh:
        document = pickle.load(fh)
    episodes = document.get(_EPISODES_KEY, {}) or {}
    meta = document.get(_META_KEY, {}) or {}
    return {
        "schema": document.get(_SCHEMA_KEY),
        "sim_id": meta.get("sim_id"),
        "n_episodes": {side: len(eps) for side, eps in episodes.items()},
    }


# =============================================================================
# Merge: combine several history files into one (signature-checked)
# =============================================================================


class HistoryMergeError(ValueError):
    """Raised when two history files cannot be merged logically/structurally.

    Distinct from :class:`~simulation.registry.signature.SignatureMismatchError`
    so callers can tell a *merge-time* incompatibility (differing schema, a side
    with episodes but no signature, or two files whose per-side signatures
    disagree) apart from an *agent-vs-config* mismatch. The message always names
    the offending files and field.

    Example:
        >>> isinstance(HistoryMergeError("x"), ValueError)
        True
    """


def split_document(document: Dict[str, Any]) -> Tuple[Dict[str, List[Tuple[str, Any]]], Dict[str, Any]]:
    """Split a history document into ``(episodes_by_side, signatures)``.

    Mirrors :func:`load_history` but for an already-loaded (e.g. merged)
    in-memory document, so the result can be passed straight to
    :meth:`Orchestrator.train_offline`.

    Args:
        document: A history document (from :func:`merge_history_files` or a
            pickle written by :func:`save_history`).

    Returns:
        ``(episodes_by_side, signatures)``.

    Example:
        >>> split_document({"episodes": {"fleet": []}, "signatures": {}})
        ({'fleet': []}, {})
    """

    episodes_by_side = document.get(_EPISODES_KEY, {}) or {}
    signatures = document.get(_SIGNATURE_KEY, {}) or {}
    return episodes_by_side, signatures


def merge_history_documents(
    docs: List[Tuple[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Merge several ``(label, history_document)`` pairs into one document.

    The merge is **only** performed when every contributing file is logically
    and structurally compatible — guaranteeing the merged file trains exactly
    like its parts would, with no silent shape/semantic drift (Contract C7):

    1. **Schema** — all documents must share one ``schema`` version; otherwise a
       :class:`HistoryMergeError` is raised naming the differing versions.
    2. **Per-side signature** — for each side, the first file that contributes
       episodes for that side sets the reference signature; every other
       contributing file's signature for that side must be
       :meth:`Signature.assert_compatible` with it (hard-gating
       ``schema_version`` / ``side`` / ``mode`` / ``critic_flags`` / ``dims`` /
       ``feature_spec``; ``visibility`` differences only warn). A file that has
       episodes for a side but **no** signature for it is rejected (cannot be
       verified). Episodes for a side are concatenated **in input order**.

    The merged document carries the reference signature per side (so it can be
    offline-trained against an agent the same way a single file would) plus a
    ``meta.merged_from`` provenance list.

    Args:
        docs: List of ``(label, document)`` pairs; ``label`` (usually the file
            path) is used only in error/provenance messages.

    Returns:
        A new merged history document (``schema`` / ``signatures`` / ``episodes``
        / ``meta``).

    Raises:
        HistoryMergeError: On any schema / signature incompatibility.

    Example:
        >>> sig = {"schema_version": 1, "side": "fleet", "mode": "PADDED"}
        >>> a = {"schema": 1, "signatures": {"fleet": sig},
        ...      "episodes": {"fleet": [("propose", 1)]}, "meta": {}}
        >>> b = {"schema": 1, "signatures": {"fleet": sig},
        ...      "episodes": {"fleet": [("propose", 2)]}, "meta": {}}
        >>> merged = merge_history_documents([("a", a), ("b", b)])
        >>> [e for _, e in merged["episodes"]["fleet"]]
        [1, 2]
    """

    if not docs:
        raise HistoryMergeError("merge needs at least one history document.")

    # ── 1. Schema must be uniform across all files. ──────────────────────────
    schemas = {
        int(doc.get(_SCHEMA_KEY))
        for _, doc in docs
        if doc.get(_SCHEMA_KEY) is not None
    }
    if len(schemas) > 1:
        labels = ", ".join(label for label, _ in docs)
        raise HistoryMergeError(
            f"Cannot merge: differing schema versions {sorted(schemas)} across "
            f"files [{labels}]. Re-run the simulations under one schema_version."
        )
    schema = next(iter(schemas)) if schemas else 1

    merged_episodes: Dict[str, List[Tuple[str, Any]]] = {side: [] for side in _SIDES}
    ref_sig: Dict[str, Optional[Signature]] = {side: None for side in _SIDES}
    ref_label: Dict[str, Optional[str]] = {side: None for side in _SIDES}
    sources: List[Dict[str, Any]] = []

    # ── 2. Per side: validate signatures, then concatenate episodes. ─────────
    for label, doc in docs:
        episodes = doc.get(_EPISODES_KEY, {}) or {}
        sigs = doc.get(_SIGNATURE_KEY, {}) or {}
        meta = doc.get(_META_KEY, {}) or {}
        per_file_counts: Dict[str, int] = {}

        for side in _SIDES:
            side_eps = episodes.get(side, []) or []
            per_file_counts[side] = len(side_eps)
            if not side_eps:
                continue  # a file that contributes nothing for this side is ignored

            sig_dict = sigs.get(side)
            if sig_dict is None:
                raise HistoryMergeError(
                    f"Cannot merge: file {label!r} has {len(side_eps)} '{side}' "
                    f"episodes but no '{side}' signature, so its compatibility "
                    f"cannot be verified. Refusing to merge unverifiable data."
                )
            sig = Signature.from_dict(sig_dict)

            if ref_sig[side] is None:
                ref_sig[side] = sig
                ref_label[side] = label
            else:
                try:
                    ref_sig[side].assert_compatible(sig)
                except SignatureMismatchError as exc:
                    raise HistoryMergeError(
                        f"Cannot merge '{side}' history: signature in {label!r} is "
                        f"incompatible with {ref_label[side]!r} on field "
                        f"'{exc.field}' ({exc.agent_value!r} vs {exc.other_value!r}). "
                        f"Both files must come from runs with the same '{side}' "
                        f"agent/feature configuration."
                    ) from exc

            merged_episodes[side].extend(side_eps)

        sources.append(
            {"label": label, "sim_id": meta.get("sim_id"), "n_episodes": per_file_counts}
        )

    signatures = {
        side: (ref_sig[side].to_dict() if ref_sig[side] is not None else None)
        for side in _SIDES
    }
    return {
        _SCHEMA_KEY: schema,
        _SIGNATURE_KEY: signatures,
        _EPISODES_KEY: merged_episodes,
        _META_KEY: {
            "merged_from": sources,
            "n_episodes": {side: len(eps) for side, eps in merged_episodes.items()},
        },
    }


def merge_history_files(
    paths: List[str], output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load several history files, merge them (signature-checked), optionally save.

    Thin file-level wrapper over :func:`merge_history_documents`: it loads each
    path, merges, and — when ``output_path`` is given — pickles the merged
    document so the combined dataset can be reused / inspected later.

    Args:
        paths: History ``.pkl`` files to merge (order is preserved).
        output_path: Optional destination for the merged ``.pkl``.

    Returns:
        The merged history document.

    Raises:
        HistoryMergeError: On any schema / signature incompatibility.

    Example:
        >>> merge_history_files.__name__
        'merge_history_files'
    """

    docs = [(path, _load_document(path)) for path in paths]
    merged = merge_history_documents(docs)
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as fh:
            pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return merged


def _load_document(path: str) -> Dict[str, Any]:
    """Load one pickled history document from ``path``.

    Example:
        >>> _load_document.__name__
        '_load_document'
    """

    with open(path, "rb") as fh:
        return pickle.load(fh)
