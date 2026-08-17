"""CLI entry point for a **structured experiment** run.

Where ``main.py`` runs one simulation and scatters its artefacts across the shared
``data/`` roots (``data/runs``, ``data/logs``, ``data/registry``, ``data/history``),
``run_experiment.py`` runs the *same* simulation but funnels **every** artefact
into a single self-contained experiment folder so runs can be compared and
archived:

    data/experiment/<name>/
      logs/
        simulation/<sim_id>.jsonl        # per-tick / per-negotiation trace
        online_training/<sim_id>.jsonl   # per-update training log
      runs/<sim_id>/                     # per-day + per-negotiation JSON + PNG
      registry/<side>/<version>/         # trained agents saved here
      history/<sim_id>.pkl               # negotiation history (for future training)
      metrices/                          # aggregate metrics + copied diagrams
        aggregate_metrics.json
        day_<i>/*.png, *.png
      experiment_meta.json

It reuses :func:`simulation.run_pipeline.run_once` verbatim; the only difference
from ``main.py`` is that the config's path fields are redirected into the
experiment folder (via :func:`dataclasses.replace`, so the original config object
is never mutated — G14). Starting weights are still loaded from the **global**
registry (``data/registry``) so an experiment can build on an existing compatible
agent, while the trained result is saved into the experiment's own registry.

Usage::

    python run_experiment.py --name baseline_ppo
    python run_experiment.py --name warm_start --config config_10d.json --seed 42 \
        --load-fleet v1 --load-spklu v1
    python run_experiment.py --name quick --no-training-log

Then visualise:

    python main_debug_visualize.py data/experiment/<name>/logs/simulation/<sim_id>.jsonl
    python viz_training.py        data/experiment/<name>/logs/online_training/<sim_id>.jsonl
    python viz_metrics.py         data/experiment/<name>/metrices
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Dict

from main import load_config_from_path
from simulation.metrics.experiment_metrics import build_experiment_metrics
from simulation.run_pipeline import run_once


EXPERIMENT_ROOT = os.path.join("data", "experiment")


def _make_experiment_dir(name: str, root: str = EXPERIMENT_ROOT) -> str:
    """Create ``<root>/<name>/`` (collision-suffixed) + return its path.

    ``root`` defaults to ``data/experiment`` for a single experiment; the bulk
    runner passes ``data/experiment_bulk/<bulk_name>`` so all 18 sub-experiments
    of one config land together.
    """
    base = os.path.join(root, name)
    exp_dir = base
    suffix = 2
    while os.path.exists(exp_dir):
        exp_dir = f"{base}_{suffix}"
        suffix += 1
    for sub in ("logs/simulation", "logs/online_training", "runs",
                "registry", "history", "metrices"):
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    return exp_dir


def _redirect_config_paths(cfg: Any, exp_dir: str) -> Any:
    """Return a copy of ``cfg`` with all artefact roots pointing into ``exp_dir``.

    Uses :func:`dataclasses.replace` on the nested ``reporting`` / ``training``
    configs so the original config object is left untouched (G14).
    """
    reporting = dataclasses.replace(
        cfg.reporting,
        runs_root=os.path.join(exp_dir, "runs"),
        logs_root=os.path.join(exp_dir, "logs"),
        offline_runs_root=os.path.join(exp_dir, "offline_runs"),
    )
    training = dataclasses.replace(
        cfg.training,
        registry_path=os.path.join(exp_dir, "registry"),
        history_path=os.path.join(exp_dir, "history"),
    )
    return dataclasses.replace(cfg, reporting=reporting, training=training)


def run_single_experiment(
    name: str,
    cfg: Any,
    *,
    seed: Any = None,
    load_fleet: Any = None,
    load_spklu: Any = None,
    save_version: Any = None,
    enable_trace: bool = True,
    enable_online_log: bool = True,
    render_runs: bool = True,
    exp_root: str = EXPERIMENT_ROOT,
    config_override_label: Any = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run ONE experiment (a fully-built ``cfg``) into ``<exp_root>/<name>/``.

    This is the reusable core shared by the CLI :func:`main` and the bulk runner
    ``run_experiment_bulk_per_config.py``. The caller supplies an already-built
    :class:`SimulationConfig` (with any agent / scenario overrides applied); this
    function redirects its artefact roots into the experiment folder (the original
    ``cfg`` is left untouched — G14), runs the simulation via
    :func:`simulation.run_pipeline.run_once`, saves the trained agents + history
    into the experiment, bundles ``metrices/``, and writes ``experiment_meta.json``.

    Args:
        name: Sub-folder name under ``exp_root``.
        cfg: The built :class:`SimulationConfig` (agent overrides already applied).
        seed: Effective seed (``None`` → ``cfg.seed``).
        load_fleet / load_spklu: Registry version to warm-start each side from the
            GLOBAL registry (validated against the live signature — Contract C7).
        save_version: Version label for the trained agents saved into the
            experiment registry (default: UTC timestamp).
        enable_trace / enable_online_log: Toggle the two log streams.
        render_runs: When ``True`` (default) the per-negotiation JSON + per-day /
            per-simulation PNG diagrams are written under ``runs/`` (and copied
            into ``metrices/``). When ``False`` those are skipped entirely (no
            period JSON, no matplotlib rendering) — a big compute saving for bulk
            runs. Metrics (incl. deal-only utilities) still come from the
            collector, so analysis stays valid either way.
        exp_root: Parent dir; ``data/experiment`` for a single run,
            ``data/experiment_bulk/<bulk_name>`` for a bulk.
        config_override_label: Provenance string recorded in the meta (e.g. the
            ``--config`` path used).
        verbose: Print progress.

    Returns:
        A summary dict: ``{name, exp_dir, sim_id, aggregate, negotiations, episodes}``.
    """

    def _say(msg: str) -> None:
        if verbose:
            print(msg)

    exp_dir = _make_experiment_dir(name, root=exp_root)
    _say(f"[experiment] Folder: {exp_dir}")

    global_registry = cfg.training.registry_path  # load starting weights from here
    # Optionally suppress the heavy runs/ artefacts (per-negotiation JSON + all PNG
    # rendering). Metrics still come from the collector, so this is purely an I/O +
    # matplotlib saving; the deal-only utilities remain exact.
    if not render_runs:
        cfg = dataclasses.replace(cfg, reporting=dataclasses.replace(
            cfg.reporting,
            write_period_json=False,
            render_day_png=False,
            render_sim_png=False,
        ))
    cfg = _redirect_config_paths(cfg, exp_dir)

    effective_seed = seed if seed is not None else cfg.seed
    _say(f"[experiment] Config: {cfg.time.n_days} day(s), "
         f"agent={cfg.agent.fleet.agent_type}/{cfg.agent.fleet.mode} vs "
         f"{cfg.agent.spklu.agent_type}/{cfg.agent.spklu.mode}, seed={effective_seed}")

    # ── Run (all artefacts land inside exp_dir via the redirected cfg). ────────
    result = run_once(
        cfg,
        seed=effective_seed,
        enable_trace=enable_trace,
        enable_online_log=enable_online_log,
        load_fleet=load_fleet,
        load_spklu=load_spklu,
        load_registry_path=global_registry,
        verbose=verbose,
    )
    orchestrator = result.orchestrator
    run_paths = result.run_paths

    # ── Save trained agents into the experiment registry (+ run bundle). ───────
    version = save_version or time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    for side in ("fleet", "spklu"):
        orchestrator.save_agent_bundle(
            orchestrator.agents[side], cfg, side, run_paths, version=version
        )

    # ── Save negotiation history (for future offline training). ────────────────
    from simulation.training.history_io import save_history
    hist_path = os.path.join(cfg.training.history_path, f"{run_paths.sim_id}.pkl")
    save_history(result.collector, orchestrator, hist_path, sim_id=run_paths.sim_id)

    # ── Aggregate metrics + copy diagrams into metrices/. ──────────────────────
    metrices_dir = os.path.join(exp_dir, "metrices")
    # Metrik inference time fleet (ms) — dari instrumentasi BaseAgent.
    _inf = getattr(orchestrator.agents["fleet"], "inference_summary", lambda: {})()
    metrics = build_experiment_metrics(
        result.collector, run_paths.run_dir, metrices_dir,
        extra_aggregate={f"fleet_{k}": v for k, v in _inf.items()},
    )

    # ── Provenance. ────────────────────────────────────────────────────────────
    meta = {
        "name": os.path.basename(exp_dir),
        "sim_id": run_paths.sim_id,
        "config_override": config_override_label,
        "seed": effective_seed,
        "loaded_fleet": load_fleet,
        "loaded_spklu": load_spklu,
        "saved_version": version,
        "render_runs": render_runs,
        "n_days": cfg.time.n_days,
        "mode_fleet": cfg.agent.fleet.mode,
        "mode_spklu": cfg.agent.spklu.mode,
        "agent_fleet": f"{cfg.agent.fleet.agent_type}/{cfg.agent.fleet.trainer_type}",
        "agent_spklu": f"{cfg.agent.spklu.agent_type}/{cfg.agent.spklu.trainer_type}",
        "negotiations_total": result.scheduler.negotiation_count,
        "episodes": result.scheduler.episode_count,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace_log": result.trace_log_path,
        "online_training_log": result.online_log_path,
    }
    with open(os.path.join(exp_dir, "experiment_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    _say("=" * 60)
    _say(f"[experiment] DONE -> {exp_dir}")
    _say(f"[experiment]   deal_rate: {metrics['aggregate']['avg_deal_rate']}  "
         f"avg_utility fleet/spklu: {metrics['aggregate']['avg_utility_fleet']} / "
         f"{metrics['aggregate']['avg_utility_spklu']}")
    _say(f"[experiment]   Visualise metrics: python viz_metrics.py {metrices_dir}")
    _say("=" * 60)

    return {
        "name": os.path.basename(exp_dir),
        "exp_dir": exp_dir,
        "sim_id": run_paths.sim_id,
        "aggregate": metrics["aggregate"],
        "negotiations": result.scheduler.negotiation_count,
        "episodes": result.scheduler.episode_count,
    }


def main() -> None:
    """Parse args, run one simulation into an experiment folder, bundle metrics."""

    parser = argparse.ArgumentParser(
        description="Run one simulation as a self-contained, comparable experiment."
    )
    parser.add_argument("--name", required=True,
                        help="Experiment name -> data/experiment/<name>/.")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config override (deep-merged on the default). "
                             "Omit for the default scenario.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--load-fleet", type=str, default=None, metavar="VERSION",
                        help="Load starting fleet weights from the GLOBAL registry "
                             "(data/registry/fleet/<VERSION>/) — must be compatible.")
    parser.add_argument("--load-spklu", type=str, default=None, metavar="VERSION",
                        help="Load starting spklu weights from the GLOBAL registry "
                             "(data/registry/spklu/<VERSION>/) — must be compatible.")
    parser.add_argument("--save-version", type=str, default=None, metavar="VERSION",
                        help="Version label for the trained agents saved into the "
                             "experiment registry (default: UTC timestamp).")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable the simulation trace log.")
    parser.add_argument("--no-training-log", action="store_true",
                        help="Disable the on-the-fly training log.")
    parser.add_argument("--no-runs", action="store_true",
                        help="Skip the per-negotiation JSON + PNG diagrams under "
                             "runs/ (saves compute). Metrics still computed from the "
                             "collector.")
    args = parser.parse_args()

    if args.config is not None:
        print(f"[experiment] Loading config override: {args.config}")
    cfg = load_config_from_path(args.config)

    run_single_experiment(
        args.name,
        cfg,
        seed=args.seed,
        load_fleet=args.load_fleet,
        load_spklu=args.load_spklu,
        save_version=args.save_version,
        enable_trace=not args.no_trace,
        enable_online_log=not args.no_training_log,
        render_runs=not args.no_runs,
        config_override_label=args.config,
    )


if __name__ == "__main__":
    main()
