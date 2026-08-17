"""CLI entry point for one end-to-end simulation run (Tahap I.1).

Loads :func:`~simulation.config.default.default_config` (or a config overridden
via ``--config``), assembles the world via
:func:`~simulation.world.world_assembly.build_world_from_config`, builds fleet
and SPKLU agents from the factory using :class:`FeatureDims` emitted by the
bridge (Contract C7), wires the :class:`~simulation.training.orchestrator.Orchestrator`
and :class:`~simulation.metrics.collector.MetricsCollector`, constructs the
:class:`~simulation.engine.scheduler.Scheduler`, runs to ``finished``, and
prints a concise three-group summary.

Usage::

    python main.py                    # default config
    python main.py --config path.json # future config-override path (not yet parsed)
    python main.py --seed 42          # reproducible run

Example
-------
>>> # Running as a script is the intended usage:
>>> # python main.py
>>> main.__module__
'__main__'
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from simulation.config.config import SimulationConfig
from simulation.config.default import default_config
from simulation.metrics.collector import MetricsCollector

# The build + run pipeline (shared with run_experiment.py) plus the pure agent
# construction helpers. Re-exported below under their historical private names so
# ``offline_train.py`` (and any other caller) keeps working unchanged.
from simulation.run_pipeline import (
    build_side_agent as _build_side_agent,
    run_once,
    side_agent_build_cfg as _side_agent_build_cfg,
)


# =============================================================================
# Summary printer
# =============================================================================

def _print_summary(scheduler: Any, collector: MetricsCollector) -> None:
    """Print a concise end-of-run summary covering the three metric groups.

    Args:
        scheduler: The finished :class:`~simulation.engine.scheduler.Scheduler`.
        collector: The :class:`~simulation.metrics.collector.MetricsCollector`.

    Example:
        >>> # Called internally by main() after the run completes.
        >>> _print_summary.__name__
        '_print_summary'
    """

    summary = collector.summary()
    sim_time = scheduler.world.sim_time_min

    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print(f"  Simulated time : {sim_time:.0f} min  "
          f"({sim_time / 60:.1f} h  /  "
          f"{sim_time / 1440:.2f} days)")
    print(f"  Negotiations   : {scheduler.negotiation_count}")
    print(f"  Episodes       : {scheduler.episode_count}")
    print("=" * 60)

    print()
    print("--- SPKLU Business ---")
    spklu = summary["spklu_business"]
    print(f"  total_revenue          : {spklu['total_revenue']:,.0f} IDR")
    print(f"  total_compensation_paid: {spklu['total_compensation_paid']:,.0f} IDR")
    print(f"  shifted_bookings       : {spklu['shifted_bookings_count']}")
    port_util = spklu["port_utilization"]
    print(f"  port_hours_used        : {port_util['port_hours_used']:.2f} h")
    if port_util["ratio"] is not None:
        print(f"  port_utilization_ratio : {port_util['ratio']:.3f}")
    print(f"  avg_price_per_kwh      : {spklu['avg_price_per_kwh']:.1f} IDR/kWh")

    print()
    print("--- Fleet Business ---")
    fleet = summary["fleet_business"]
    print(f"  passenger_revenue      : {fleet['passenger_revenue']:,.0f} IDR")
    print(f"  charging_cost          : {fleet['charging_cost']:,.0f} IDR")
    print(f"  unproductive_km        : {fleet['unproductive_km']:.2f} km")
    print(f"  orders_served          : {fleet['orders_served']}")
    print(f"  orders_unserved        : {fleet['orders_unserved']}")
    print(f"  taxis_died             : {fleet['taxis_died']}")
    print(f"  emergency_rescue_count : {fleet['emergency_rescue_count']}")
    print(f"  emergency_premium_cost : {fleet['emergency_premium_cost']:,.0f} IDR")
    print(f"  slot_forfeit_count     : {fleet['slot_forfeit_count']}")

    print()
    print("--- Negotiation / System ---")
    neg = summary["negotiation_system"]
    print(f"  negotiations_total     : {neg['negotiations_total']}")
    print(f"  deal_success_rate      : {neg['deal_success_rate']:.3f}")
    print(f"  partial_deal_rate      : {neg['partial_deal_rate']:.3f}")
    print(f"  orders_expired         : {neg['orders_expired_at_hour_change']}")
    print(f"  avg_cycles_waited      : {neg['avg_cycles_waited']:.2f}")
    print(f"  fleet avg_utility      : {neg['fleet']['avg_utility_per_negotiation']:.4f}")
    print(f"  spklu avg_utility      : {neg['spklu']['avg_utility_per_negotiation']:.4f}")
    print("=" * 60)
    print()


# =============================================================================
# Config loading (--config <path.json>)
# =============================================================================

def load_config_from_path(path: Optional[str]) -> SimulationConfig:
    """Load a :class:`SimulationConfig`, applying a JSON override file if given.

    When ``path`` is ``None`` the built-in :func:`default_config` scenario is
    returned. Otherwise the JSON file is read and deep-merged on top of the
    default scenario via :func:`dashboard.config_io.config_from_dict`, so the
    file only needs to contain the fields the user wants to change (a single
    field, a nested group, or a full serialized config — see
    ``panduan_penggunaan.md``). Unknown keys are ignored by the rebuild step.

    Args:
        path: Path to a ``.json`` override file, or ``None`` for the default.

    Returns:
        The reconstructed :class:`SimulationConfig`.

    Example:
        >>> isinstance(load_config_from_path(None), SimulationConfig)
        True
    """

    if path is None:
        return default_config()

    # Imported lazily so a plain CLI run does not pull in the dashboard helper
    # unless --config is actually used.
    from dashboard.config_io import config_from_dict, validate_config

    with open(path, "r", encoding="utf-8") as fh:
        override = json.load(fh)
    cfg = config_from_dict(override)

    warnings = validate_config(cfg)
    if warnings:
        print(f"[main] Config warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"        - {warning}")
    return cfg


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    """Load config, build world + agents, run scheduler to completion, print summary.

    Steps (G14 — config is built once and never mutated):

    1. Parse CLI args (``--seed`` for reproducibility).
    2. Load :func:`~simulation.config.default.default_config`.
    3. Build :class:`~simulation.world.world_state.WorldState` via
       :func:`~simulation.world.world_assembly.build_world_from_config`.
    4. Build fleet + SPKLU agents via :func:`_build_side_agent` (dims from
       the bridge, Contract C7).
    5. Create :class:`~simulation.metrics.collector.MetricsCollector`.
    6. Create :class:`~simulation.metrics.run_paths.RunPaths` (the timestamped
       ``sim_id`` is minted here, LLR-M7.1 / KR-1) and a
       :class:`~simulation.metrics.run_reporter.RunReporter` bound to the SAME
       collector instance the scheduler uses (so reporter snapshots/diffs read
       the metrics the scheduler emits — shared collector invariant).
    7. Create :class:`~simulation.training.orchestrator.Orchestrator`
       (owns both agents, applies training concurrency policy).
    8. Create :class:`~simulation.world.order_generator.OrderGenerator`.
    9. Construct :class:`~simulation.engine.scheduler.Scheduler` (wires world,
       agents, metrics, order generator, ``orchestrator.train`` as the training
       callback, and ``run_reporter`` for the runs/ artefacts).
    10. Run ``scheduler.run()`` until ``finished`` (the scheduler already calls
        ``reporter.on_run_finished()`` at the horizon, M3-E).
    11. Safety-net call to ``reporter.on_run_finished()`` (idempotent, LLR-M7.2)
        in case the normal finish path did not fire.
    12. Print the three-group summary from the collector.

    Example:
        >>> # End-to-end verification: python main.py
        >>> main.__name__
        'main'
    """

    parser = argparse.ArgumentParser(
        description="Run one end-to-end EV-taxi charging negotiation simulation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: non-deterministic).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config override file (deep-merged on top of the "
             "default scenario). Omit to use the built-in default scenario.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable the per-tick / per-negotiation validation trace log "
             "(data/logs/simulation/<sim_id>.jsonl).",
    )
    parser.add_argument(
        "--no-training-log",
        action="store_true",
        help="Disable the on-the-fly training log "
             "(data/logs/online_training/<sim_id>.jsonl).",
    )
    parser.add_argument(
        "--load-fleet",
        type=str,
        default=None,
        metavar="VERSION",
        help="Before running, load fleet-agent weights from "
             "data/registry/fleet/<VERSION>/ (must be config-compatible).",
    )
    parser.add_argument(
        "--load-spklu",
        type=str,
        default=None,
        metavar="VERSION",
        help="Before running, load spklu-agent weights from "
             "data/registry/spklu/<VERSION>/ (must be config-compatible).",
    )
    parser.add_argument(
        "--save-agents",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="VERSION",
        help="After the run, save both agents to the registry (and into the run "
             "folder). Optionally give a version label; default is a UTC timestamp.",
    )
    parser.add_argument(
        "--save-history",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="PATH",
        help="After the run, pickle the recorded negotiation history for offline "
             "training. Optional path; default data/history/<sim_id>.pkl.",
    )
    args = parser.parse_args()

    # ── 2. Load config (G14: built once, never mutated). ──────────────────────
    if args.config is not None:
        print(f"[main] Loading config override: {args.config}")
    cfg = load_config_from_path(args.config)

    # CLI --seed takes precedence over cfg.seed; cfg.seed is the config-level
    # default so one scenario can be run reproducibly without a CLI flag.
    effective_seed = args.seed if args.seed is not None else cfg.seed

    print(f"[main] Config loaded: {len(cfg.taxi_spec.initial_taxi_state)} taxis, "
          f"{len(cfg.port_station.stations)} SPKLU stations, "
          f"{len(cfg.marking_area_order.areas)} marking areas, "
          f"{cfg.time.n_days} day(s), "
          f"agent={cfg.agent.fleet.agent_type}/{cfg.agent.fleet.trainer_type}"
          f"/{cfg.agent.fleet.mode}")

    # ── 3-10. Build + run via the shared pipeline (seeding, world, agents,
    # optional registry load, orchestrator, scheduler, and the two log streams are
    # all handled inside run_once — driven purely by cfg's path fields). ─────────
    result = run_once(
        cfg,
        seed=effective_seed,
        enable_trace=not args.no_trace,
        enable_online_log=not args.no_training_log,
        load_fleet=args.load_fleet,
        load_spklu=args.load_spklu,
    )
    scheduler = result.scheduler
    collector = result.collector
    orchestrator = result.orchestrator
    run_paths = result.run_paths

    print(f"[main] Simulation finished: "
          f"{scheduler.negotiation_count} negotiations, "
          f"{scheduler.episode_count} training episodes")
    print(f"[main] Reports written to: {result.reporter.run_dir}")

    # ── 11b. Optionally persist trained agents (registry + run-folder bundle).
    if args.save_agents is not None:
        version = None if args.save_agents == "__auto__" else args.save_agents
        # Use the orchestrator's *current* agents: per-episode rebuilds replace the
        # initial agent_fleet/agent_spklu locals, but every rebuilt shell shares the
        # same persistent trainers, so this saves the trained models either way.
        for side in ("fleet", "spklu"):
            bundle = orchestrator.save_agent_bundle(
                orchestrator.agents[side], cfg, side, run_paths, version=version
            )
            print(f"[main] Saved {side} agent bundle -> {bundle}")
        latest = orchestrator.registry.latest_version("fleet")
        print(f"[main] Registry now at version '{version or latest}' "
              f"(reuse with: --load-fleet/--load-spklu <version>).")

    # ── 11c. Optionally persist the negotiation history for offline training.
    if args.save_history is not None:
        from simulation.training.history_io import save_history
        if args.save_history == "__auto__":
            import os
            hist_path = os.path.join(
                cfg.training.history_path, f"{run_paths.sim_id}.pkl"
            )
        else:
            hist_path = args.save_history
        written = save_history(collector, orchestrator, hist_path,
                               sim_id=run_paths.sim_id)
        n_fleet = len(collector.negotiation_payload_history)
        print(f"[main] Saved negotiation history ({n_fleet} payload records) "
              f"-> {written}")
        print(f"[main] Offline-train later with: python offline_train.py "
              f"--history {written}")

    # ── 12. Summary. ──────────────────────────────────────────────────────────
    _print_summary(scheduler, collector)


if __name__ == "__main__":
    main()
