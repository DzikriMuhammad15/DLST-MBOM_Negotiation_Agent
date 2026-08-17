"""Aggregate experiment metrics + structured diagram bundling (``run_experiment``).

After a :func:`simulation.run_pipeline.run_once` finishes inside an experiment
folder, this module builds the ``<experiment>/metrices/`` deliverable:

* ``aggregate_metrics.json`` — the run's three-group summary
  (:meth:`~simulation.metrics.collector.MetricsCollector.summary`) plus the flat
  numeric metrics (:meth:`numeric_metrics_flat`) and a small ``aggregate`` block
  of headline numbers requested for cross-experiment comparison (average utility
  per side, deal / partial-deal rate, average episode length, average cycles
  waited, negotiation / episode counts).
* a structured **copy of the run's diagrams** mirroring the ``data/runs`` layout
  (``metrices/day_<i>/*.png`` for the per-day + per-negotiation charts, and the
  per-simulation ``metrices/*.png``). The PNGs are produced by the existing
  :class:`~simulation.metrics.run_reporter.RunReporter`; this module only *copies*
  them into ``metrices/`` (no re-rendering), so ``metrices/`` is a
  self-describing, comparable bundle.

Pure standard-library + reads the collector; it renders nothing and mutates no
world/agent state.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional


def _avg_episode_length(collector: Any) -> Optional[float]:
    """Mean number of steps over propose-phase episodes in the payload history.

    Returns ``None`` when no usable episodes were recorded.
    """
    history = getattr(collector, "negotiation_payload_history", None) or []
    lengths: List[int] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("phase") != "propose":
            continue
        episode = entry.get("episode")
        steps = getattr(episode, "steps", None)
        if steps is None and isinstance(episode, dict):
            steps = episode.get("steps")
        if steps is not None:
            lengths.append(len(steps))
    if not lengths:
        return None
    return sum(lengths) / len(lengths)


def _deal_only_utilities(collector: Any) -> Dict[str, Any]:
    """Deal-only utilities per side, computed from the collector (no ``runs/`` needed).

    Reads the propose-phase episodes in ``negotiation_payload_history``: an episode
    with a non-``None`` ``agreement`` is a **deal**, and its ``terminal_utility`` is
    the realized ``U(agreement)`` for that side (a no-deal instead carries the
    ``-1`` marker). This is the SAME number written into each period JSON's
    ``final_utility`` (verified equal), so the deal-quality signal survives even
    when per-negotiation JSON / diagrams are not written (bulk default).

    Returns ``avg_utility_*_deal`` (mean over deals), ``social_welfare_deal``
    (their sum), and ``n_deal`` / ``n_no_deal`` counts (from the fleet side).
    """
    history = getattr(collector, "negotiation_payload_history", None) or []
    deal: Dict[str, List[float]] = {"fleet": [], "spklu": []}
    nodeal: Dict[str, int] = {"fleet": 0, "spklu": 0}
    for entry in history:
        if not isinstance(entry, dict) or entry.get("phase") != "propose":
            continue
        side = entry.get("side")
        if side not in deal:
            continue
        episode = entry.get("episode")
        agreement = getattr(episode, "agreement", None)
        if agreement is None and isinstance(episode, dict):
            agreement = episode.get("agreement")
        terminal = getattr(episode, "terminal_utility", None)
        if terminal is None and isinstance(episode, dict):
            terminal = episode.get("terminal_utility")
        if agreement is not None:
            try:
                deal[side].append(float(terminal))
            except (TypeError, ValueError):
                pass
        else:
            nodeal[side] += 1

    def mean(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    uf, us = mean(deal["fleet"]), mean(deal["spklu"])
    return {
        "avg_utility_fleet_deal": uf,
        "avg_utility_spklu_deal": us,
        "social_welfare_deal": (uf + us) if (uf is not None and us is not None) else None,
        "n_deal": len(deal["fleet"]),
        "n_no_deal": nodeal["fleet"],
    }


def build_aggregate_metrics(collector: Any) -> Dict[str, Any]:
    """Assemble the aggregate metrics dict from a finished run's collector.

    Example:
        >>> from simulation.metrics.collector import MetricsCollector
        >>> m = build_aggregate_metrics(MetricsCollector())
        >>> set(["summary", "numeric_flat", "aggregate"]).issubset(m)
        True
        >>> "avg_utility_fleet_deal" in m["aggregate"]
        True
    """
    summary = collector.summary()
    flat = collector.numeric_metrics_flat()
    neg = summary.get("negotiation_system", {})
    fleet = neg.get("fleet", {})
    spklu = neg.get("spklu", {})

    aggregate = {
        "avg_utility_fleet": fleet.get("avg_utility_per_negotiation"),
        "avg_utility_spklu": spklu.get("avg_utility_per_negotiation"),
        "total_utility_fleet": fleet.get("total_utility"),
        "total_utility_spklu": spklu.get("total_utility"),
        "avg_deal_rate": neg.get("deal_success_rate"),
        "avg_partial_deal_rate": neg.get("partial_deal_rate"),
        "avg_cycles_waited": neg.get("avg_cycles_waited"),
        "avg_episode_length": _avg_episode_length(collector),
        "negotiations_total": neg.get("negotiations_total"),
        "orders_expired_at_hour_change": neg.get("orders_expired_at_hour_change"),
        # Deal-only utilities (from the collector; independent of runs/ diagrams).
        **_deal_only_utilities(collector),
    }
    return {"summary": summary, "numeric_flat": flat, "aggregate": aggregate}


def copy_run_diagrams(run_dir: str, metrices_dir: str) -> List[str]:
    """Copy every ``*.png`` under ``run_dir`` into ``metrices_dir``, mirroring layout.

    Preserves the ``day_<i>/`` sub-folder structure (per-day + per-negotiation
    charts) and the run-level PNGs, so ``metrices/`` reproduces the ``data/runs``
    diagram tree. Returns the list of destination paths written.
    """
    written: List[str] = []
    if not os.path.isdir(run_dir):
        return written
    for root, _dirs, files in os.walk(run_dir):
        rel = os.path.relpath(root, run_dir)
        for name in files:
            if not name.lower().endswith(".png"):
                continue
            dest_dir = metrices_dir if rel == "." else os.path.join(metrices_dir, rel)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, name)
            try:
                shutil.copyfile(os.path.join(root, name), dest)
                written.append(dest)
            except OSError:
                continue
    return written


def build_experiment_metrics(
    collector: Any, run_dir: str, metrices_dir: str,
    extra_aggregate: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Write ``aggregate_metrics.json`` + copy diagrams into ``metrices_dir``.

    Args:
        collector: The finished run's
            :class:`~simulation.metrics.collector.MetricsCollector`.
        run_dir: The run folder (``<experiment>/runs/<sim_id>``) holding the PNGs.
        metrices_dir: The ``<experiment>/metrices`` destination.

    Returns:
        The aggregate metrics dict (also written to disk).
    """
    os.makedirs(metrices_dir, exist_ok=True)
    metrics = build_aggregate_metrics(collector)
    if extra_aggregate:
        metrics["aggregate"].update(extra_aggregate)
    with open(os.path.join(metrices_dir, "aggregate_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    copied = copy_run_diagrams(run_dir, metrices_dir)
    metrics.setdefault("_provenance", {})["diagram_files_copied"] = len(copied)
    return metrics
