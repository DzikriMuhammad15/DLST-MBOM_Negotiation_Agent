"""Reusable end-to-end simulation-run pipeline (shared by ``main.py`` + experiments).

This module factors the "build a world + agents, wire the orchestrator /
collector / reporter, run the scheduler to the horizon" flow out of
``main.main()`` so it can be reused verbatim by both the plain CLI entry point
(``main.py``) and the structured experiment runner (``run_experiment.py``) without
duplicating the wiring — the only thing that changes between the two is *where*
the artefacts land, which is driven entirely by the config's path fields
(``reporting.runs_root`` / ``reporting.logs_root`` / ``training.registry_path`` /
``training.history_path``).

It also owns the small, pure agent-construction helpers (dims from the bridge
feature builder — Contract C7) that ``main.py`` and ``offline_train.py`` previously
imported from ``main``; they live here now so nothing needs to import the CLI
module. ``main.py`` re-exports them for backwards compatibility.

Log routing (the new three-stream layout):

* simulation trace   -> ``<logs_root>/simulation/<sim_id>.jsonl``
* online training log -> ``<logs_root>/online_training/<sim_id>.jsonl``

Both are opt-out via the ``enable_trace`` / ``enable_online_log`` flags.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, fields as _dc_fields, replace as _dc_replace
from typing import Any, Dict, Optional

from negotiation.agents.agent_factory import (
    AgentBuildConfig,
    TrainerHyperparams,
    build_agent,
    count_agent_parameters,
    rebuild_agent_shell_with_trainers,
)
from negotiation.agents.dlst_mbom_directed_agent import DirectedConfig
from negotiation.training_logger import (
    LOGS_SUBDIR_ONLINE,
    LOGS_SUBDIR_SIMULATION,
    get_online_logger,
)
from simulation.bridge.feature_builder import feature_dims as _feature_dims
from simulation.config.config import SimulationConfig
from simulation.metrics.collector import MetricsCollector
from simulation.metrics.run_paths import RunPaths
from simulation.metrics.run_reporter import RunReporter
from simulation.training.orchestrator import Orchestrator
from simulation.world.order_generator import OrderGenerator
from simulation.world.world_assembly import build_world_from_config


# =============================================================================
# Agent construction helpers (C7 dims flow: feature_builder -> factory)
# =============================================================================

def _hyperparams_from_config(cfg: SimulationConfig, side: str) -> TrainerHyperparams:
    """Convert one side's config hyperparameters into a :class:`TrainerHyperparams`.

    Example:
        >>> from simulation.config.default import default_config
        >>> _hyperparams_from_config(default_config(), "fleet").gamma
        0.99
    """

    side_cfg = cfg.agent.fleet if side == "fleet" else cfg.agent.spklu
    hp_cfg = side_cfg.hyperparameters
    return TrainerHyperparams(
        gamma=hp_cfg.gamma,
        gae_lambda=hp_cfg.gae_lambda,
        clip_ratio=hp_cfg.clip_ratio,
        value_coef=hp_cfg.value_coef,
        entropy_coef=hp_cfg.entropy_coef,
        gradient_clip_norm=hp_cfg.gradient_clip_norm,
        epoch=hp_cfg.epoch,
        batch_size=hp_cfg.batch_size,
        learning_rate=hp_cfg.learning_rate,
        tau=hp_cfg.tau,
        hidden_dim=hp_cfg.hidden_dim,
        hidden_layers=hp_cfg.hidden_layers,
    )


def _apply_directed_env_overrides(base: DirectedConfig) -> DirectedConfig:
    """ADITIF (tuning): override field ``DirectedConfig`` dari env ``DIRECTED_HP`` (JSON dict).

    Tanpa env var → objek dikembalikan APA ADANYA, sehingga seluruh run/korpus lama
    tetap bit-identik. Key tak dikenal → error keras (gagal cepat, bukan diam-diam
    mengabaikan setelan tuning). Dipakai oleh directed_impl/main_nego_directed.py --hp.
    """
    raw = os.environ.get("DIRECTED_HP")
    if not raw:
        return base
    try:
        ov = json.loads(raw)
    except Exception as exc:  # pragma: no cover - jalur error CLI
        raise SystemExit(f"DIRECTED_HP bukan JSON valid: {exc}")
    if not isinstance(ov, dict):
        raise SystemExit("DIRECTED_HP harus objek JSON {param: nilai}")
    valid = {f.name for f in _dc_fields(DirectedConfig)}
    unknown = sorted(set(ov) - valid)
    if unknown:
        raise SystemExit(f"DIRECTED_HP memuat param tak dikenal: {unknown}")
    return _dc_replace(base, **ov)


def side_agent_build_cfg(cfg: SimulationConfig, side: str) -> AgentBuildConfig:
    """Assemble the :class:`AgentBuildConfig` for one side (no agent built yet).

    Example:
        >>> from simulation.config.default import default_config
        >>> side_agent_build_cfg(default_config(), "fleet").side
        'fleet'
    """
    side_cfg = cfg.agent.fleet if side == "fleet" else cfg.agent.spklu
    dims = _feature_dims(cfg, side)
    hp = _hyperparams_from_config(cfg, side)

    hp_cfg = side_cfg.hyperparameters
    # DLSTMBOMDirected: aktifkan SEMUA layer anti-kolaps (L1+persist+rescue+handover+L3+L5).
    # Tipe lain → DirectedConfig default (semua OFF) → tak berpengaruh (bit-identik).
    directed_cfg = DirectedConfig()
    if side_cfg.agent_type == "DLSTMBOMDirected":
        directed_cfg = _apply_directed_env_overrides(DirectedConfig(
            l1=True, persistent_freq=True, stuck_rescue=True,
            handover=True, shaping=True, hygiene=True,
        ))
    return AgentBuildConfig(
        name=f"agent_{side}",
        side=side,
        agent_type=side_cfg.agent_type,
        trainer=side_cfg.trainer_type,
        mode=side_cfg.mode,
        feature_dims=dims,
        world_ufun=None,        # injected per negotiation via configure_negotiation
        ufun=None,              # injected per negotiation via configure_negotiation
        domain_spec=None,       # injected per negotiation via configure_negotiation
        k_max=dims.k_max,
        hyperparams=hp,
        critic_sees_candidate_pool=side_cfg.critic_enrichment.critic_sees_candidate_pool,
        critic_privileged_info=side_cfg.critic_enrichment.critic_privileged_info,
        propose_reward_mode=side_cfg.reward_mode.propose_mode,
        respond_reward_mode=side_cfg.reward_mode.respond_mode,
        threshold_reward_mode=side_cfg.reward_mode.threshold_mode,
        device="cpu",
        # MBOM / DLST-MBOM hyperparameters — kini dibaca dari config (dulu selalu
        # default factory; lihat audit E.6). ``mbom_M`` = jumlah level IOP
        # (M=2/M=3 = kedalaman recursive imagination). ``max_steps`` di sini adalah
        # horizon negosiasi KHAYALAN (rollout), sengaja DECOUPLED & kecil
        # (``rollout_max_steps``) dari horizon negosiasi nyata (negotiation.max_steps)
        # — lever utama runtime MBOM (imajinasi hanya pakai rollout_k langkah).
        mbom_M=hp_cfg.mbom_M,
        rollout_k=hp_cfg.rollout_k,
        n_rollout_samples=hp_cfg.n_rollout_samples,
        max_steps=(hp_cfg.rollout_max_steps
                   if hp_cfg.rollout_max_steps is not None
                   else cfg.negotiation.max_steps),
        directed=directed_cfg,
    )


def build_side_agent(cfg: SimulationConfig, side: str) -> Any:
    """Build one side's negotiator with dims from the bridge feature builder (C7).

    Example:
        >>> from simulation.config.default import default_config
        >>> hasattr(build_side_agent(default_config(), "fleet"), "configure_negotiation")
        True
    """

    return build_agent(side_agent_build_cfg(cfg, side))


# =============================================================================
# Log-path helpers
# =============================================================================

def simulation_log_path(cfg: SimulationConfig, sim_id: str) -> str:
    """Return ``<logs_root>/simulation/<sim_id>.jsonl``.

    Example:
        >>> from simulation.config.default import default_config
        >>> simulation_log_path(default_config(), "x").replace("\\\\", "/")
        'data/logs/simulation/x.jsonl'
    """
    return os.path.join(cfg.reporting.logs_root, LOGS_SUBDIR_SIMULATION, f"{sim_id}.jsonl")


def online_training_log_path(cfg: SimulationConfig, sim_id: str) -> str:
    """Return ``<logs_root>/online_training/<sim_id>.jsonl``.

    Example:
        >>> from simulation.config.default import default_config
        >>> online_training_log_path(default_config(), "x").replace("\\\\", "/")
        'data/logs/online_training/x.jsonl'
    """
    return os.path.join(cfg.reporting.logs_root, LOGS_SUBDIR_ONLINE, f"{sim_id}.jsonl")


# =============================================================================
# RunResult
# =============================================================================

@dataclass
class RunResult:
    """Everything a caller might need after :func:`run_once` completes.

    Attributes:
        scheduler: The finished :class:`~simulation.engine.scheduler.Scheduler`.
        collector: The :class:`~simulation.metrics.collector.MetricsCollector`.
        orchestrator: The :class:`~simulation.training.orchestrator.Orchestrator`
            (its ``agents`` hold the trained trainers).
        run_paths: The :class:`~simulation.metrics.run_paths.RunPaths` used.
        reporter: The :class:`~simulation.metrics.run_reporter.RunReporter`.
        config: The :class:`SimulationConfig` actually run.
        rng: The seeded :class:`random.Random` used for the run.
        trace_log_path: Where the simulation trace was written (or ``None``).
        online_log_path: Where the online training log was written (or ``None``).
    """

    scheduler: Any
    collector: MetricsCollector
    orchestrator: Orchestrator
    run_paths: RunPaths
    reporter: RunReporter
    config: SimulationConfig
    rng: random.Random
    trace_log_path: Optional[str]
    online_log_path: Optional[str]


# =============================================================================
# The pipeline
# =============================================================================

def _seed_everything(seed: Optional[int]) -> random.Random:
    """Seed python/numpy/torch global RNGs (no-op parts when ``seed is None``).

    Returns a fresh :class:`random.Random` for the world/orders (shared for
    reproducibility). Mirrors the seeding ``main.main()`` used to do.
    """
    rng = random.Random(seed)
    if seed is not None:
        import numpy as _np
        import torch as _torch
        random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)
    return rng


def run_once(
    cfg: SimulationConfig,
    *,
    seed: Optional[int] = None,
    enable_trace: bool = True,
    enable_online_log: bool = True,
    load_fleet: Optional[str] = None,
    load_spklu: Optional[str] = None,
    load_registry_path: Optional[str] = None,
    run_paths: Optional[RunPaths] = None,
    verbose: bool = True,
) -> RunResult:
    """Build a world + agents, run the scheduler to the horizon, return a RunResult.

    All artefact locations are taken from ``cfg`` (``reporting.runs_root`` /
    ``reporting.logs_root`` / ``training.registry_path``), so redirecting them
    into an experiment folder is purely a config concern — this function does not
    special-case experiments.

    Args:
        cfg: The (already path-configured) :class:`SimulationConfig`.
        seed: Effective seed; when ``None`` uses ``cfg.seed``.
        enable_trace: Write the simulation trace log.
        enable_online_log: Write the on-the-fly training log.
        load_fleet / load_spklu: Registry version to load into each side before the
            run (validated against the live signature — Contract C7).
        load_registry_path: Registry root to LOAD the starting weights from
            (defaults to ``cfg.training.registry_path``). Lets an experiment load a
            compatible agent from the global registry while still *saving* results
            into its own experiment registry.
        run_paths: Optionally provide a pre-made :class:`RunPaths` (else minted).
        verbose: Print progress lines (matches the old ``main`` output).

    Returns:
        A populated :class:`RunResult`.
    """

    def _say(msg: str) -> None:
        if verbose:
            print(msg)

    effective_seed = seed if seed is not None else cfg.seed
    rng = _seed_everything(effective_seed)

    # ── World ────────────────────────────────────────────────────────────────
    _say("[run] Building world ...")
    world = build_world_from_config(cfg, rng=rng)
    _say(f"[run] World ready: {len(world.taxis)} taxis, {len(world.stations)} stations, "
         f"{len(world.areas)} areas, {len(world.bookings)} initial bookings")

    # ── Agents (C7 dims) ──────────────────────────────────────────────────────
    _say("[run] Building agents ...")
    agent_fleet = build_side_agent(cfg, "fleet")
    agent_spklu = build_side_agent(cfg, "spklu")
    param_fleet = count_agent_parameters(agent_fleet)
    param_spklu = count_agent_parameters(agent_spklu)
    _say(f"[run] Agents ready: fleet={type(agent_fleet).__name__} "
         f"({param_fleet['total']} params), spklu={type(agent_spklu).__name__} "
         f"({param_spklu['total']} params)")

    # ── Optional load from registry (signature-checked). ──────────────────────
    if load_fleet is not None or load_spklu is not None:
        from simulation.registry.agent_registry import AgentRegistry
        from simulation.registry.signature import build_signature
        registry = AgentRegistry(load_registry_path or cfg.training.registry_path)
        if load_fleet is not None:
            sig = build_signature(cfg, "fleet", _feature_dims(cfg, "fleet"))
            registry.load_into_agent(agent_fleet, "fleet", load_fleet, sig)
            _say(f"[run] Loaded fleet weights from registry version '{load_fleet}'.")
        if load_spklu is not None:
            sig = build_signature(cfg, "spklu", _feature_dims(cfg, "spklu"))
            registry.load_into_agent(agent_spklu, "spklu", load_spklu, sig)
            _say(f"[run] Loaded spklu weights from registry version '{load_spklu}'.")

    # ── Metrics + run paths + reporter. ───────────────────────────────────────
    collector = MetricsCollector(config=cfg, keep_payload_history=True)
    if run_paths is None:
        run_paths = RunPaths(cfg.reporting.runs_root)
    reporter = RunReporter(run_paths, cfg, collector=collector)
    _say(f"[run] Run ID: {run_paths.sim_id} -> {run_paths.run_dir}")

    # ── Log paths (the new three-stream layout). ──────────────────────────────
    trace_log_path: Optional[str] = None
    if enable_trace:
        trace_log_path = simulation_log_path(cfg, run_paths.sim_id)
        _say(f"[run] Trace log: {trace_log_path}")

    online_log_path: Optional[str] = None
    if enable_online_log:
        online_log_path = online_training_log_path(cfg, run_paths.sim_id)
        get_online_logger().configure(
            online_log_path, mode="online", run_id=run_paths.sim_id
        )
        _say(f"[run] Online training log: {online_log_path}")

    # ── Orchestrator + order gen + per-episode rebuilder. ─────────────────────
    orchestrator = Orchestrator(
        agent_fleet=agent_fleet,
        agent_spklu=agent_spklu,
        config=cfg,
        metrics_collector=collector,
    )
    order_gen = OrderGenerator(world, rng=rng)

    build_cfgs = {"fleet": side_agent_build_cfg(cfg, "fleet"),
                  "spklu": side_agent_build_cfg(cfg, "spklu")}

    def _rebuild_agents() -> Dict[str, Any]:
        fresh = {
            side: rebuild_agent_shell_with_trainers(
                build_cfgs[side], orchestrator.agents[side].trainers
            )
            for side in ("fleet", "spklu")
        }
        orchestrator.agents.update(fresh)
        return fresh

    # ── Scheduler. ────────────────────────────────────────────────────────────
    from simulation.engine.scheduler import Scheduler  # local import avoids circular

    scheduler = Scheduler(
        world=world,
        agent_fleet=agent_fleet,
        agent_spklu=agent_spklu,
        config=cfg,
        order_generator=order_gen,
        metrics_collector=collector,
        training_callback=orchestrator.train,
        negotiation_recorder=orchestrator.record_negotiation_payloads,
        run_reporter=reporter,
        rng=rng,
        trace_log_path=trace_log_path,
        agent_rebuilder=_rebuild_agents,
    )
    _say(f"[run] Scheduler ready — running {cfg.time.n_days} day(s) simulation ...")

    # ── Run to completion. ────────────────────────────────────────────────────
    scheduler.run()
    _say(f"[run] Simulation finished: {scheduler.negotiation_count} negotiations, "
         f"{scheduler.episode_count} training episodes")

    # Safety-net (idempotent) + aggregate training-log summary, then close it.
    reporter.on_run_finished()
    orchestrator.finalize_online_log()
    get_online_logger().close()
    _say(f"[run] Reports written to: {reporter.run_dir}")

    return RunResult(
        scheduler=scheduler,
        collector=collector,
        orchestrator=orchestrator,
        run_paths=run_paths,
        reporter=reporter,
        config=cfg,
        rng=rng,
        trace_log_path=trace_log_path,
        online_log_path=online_log_path,
    )
