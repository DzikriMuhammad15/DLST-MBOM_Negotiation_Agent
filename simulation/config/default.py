"""Default runnable scenario configuration.

Provides :func:`default_config`, which builds a small but fully runnable
:class:`~simulation.config.config.SimulationConfig` scenario: 3 taxis, 2
charging stations (SPKLU) with multiple ports, 3 marking areas, and both
negotiating sides using ``RLNegotiator`` + ``PPO`` + ``PADDED`` mode. This
scenario is intentionally small so it can be used directly in smoke tests
and as a starting point for the dashboard's configuration editor.

Per ``.claude/project_instruction/08-konfigurasi/08-01-daftar-konfigurasi.md``,
the specification only mandates the *existence and meaning* of each
parameter; concrete default values are an implementation decision made here,
chosen to be internally consistent (e.g. ``B_charge < B_area``, feasible
geography, ``initial_taxi_state`` covering all 3 taxis).

Example
-------
>>> from simulation.config.default import default_config
>>> cfg = default_config()
>>> len(cfg.taxi_spec.initial_taxi_state)
3
>>> len(cfg.port_station.stations)
2
>>> cfg.agent.fleet.agent_type, cfg.agent.fleet.trainer_type, cfg.agent.fleet.mode
('RLNegotiator', 'PPO', 'PADDED')
"""

from __future__ import annotations

from simulation.config.config import (
    AgentConfig,
    AgentSideConfig,
    AppUserBookingConfig,
    BatteryThresholdConfig,
    ChargingStationSpec,
    CriticEnrichmentConfig,
    DashboardConfig,
    DeadBatteryConfig,
    EmergencyStationConfig,
    EncoderDimsConfig,
    FallbackDistributionConfig,
    FareConfig,
    FeatureBlockSpec,
    FeatureSpecConfig,
    FleetUfunReferences,
    FleetUfunWeights,
    FutureSlotConfig,
    InitialTaxiState,
    MarkingAreaAndOrderConfig,
    MarkingAreaSpec,
    NegotiationConfig,
    OsmConfig,
    PortAndStationConfig,
    PortSpec,
    ReportingConfig,
    RewardModeConfig,
    SimulationConfig,
    SpkluUfunReferences,
    SpkluUfunWeights,
    StateFeatureConfig,
    TaxiSpecConfig,
    TimeConfig,
    TrainerHyperparameters,
    TrainingConfig,
    UfunConfig,
    UfunShapeConfig,
    VisibilityConfig,
)
from simulation.engine.clock import real_seconds_per_tick


def _build_time_config() -> TimeConfig:
    """Build a short-horizon time configuration suitable for smoke testing.

    Example:
        >>> _build_time_config().interval_min
        30
        >>> _build_time_config().speed_mode
        'fast'
    """

    return TimeConfig(n_days=1, tick_resolution_min=5, interval_min=30, speed_mode="fast")


def _build_negotiation_config() -> NegotiationConfig:
    """Build negotiation caps and discrete price levels for the default scenario.

    Example:
        >>> _build_negotiation_config().max_action_library
        16
    """

    return NegotiationConfig(
        max_action_library=16,
        max_steps=20,
        price_levels=[2000.0, 3000.0, 4000.0],
    )


def _build_taxi_spec_config() -> TaxiSpecConfig:
    """Build the uniform taxi specification and 3 explicit initial taxi states.

    Example:
        >>> spec = _build_taxi_spec_config()
        >>> spec.initial_taxi_state[0].taxi_id
        'taxi_0'
    """

    initial_states = [
        InitialTaxiState(taxi_id="taxi_0", position=(-6.9025, 107.6041), soc=0.80),
        InitialTaxiState(taxi_id="taxi_1", position=(-6.9125, 107.6241), soc=0.45),
        InitialTaxiState(taxi_id="taxi_2", position=(-6.9325, 107.6441), soc=0.20),
    ]
    return TaxiSpecConfig(
        battery_capacity_kwh=50.0,
        consumption_kwh_per_km=0.15,
        speed_kmh=30.0,
        initial_taxi_state=initial_states,
    )


def _build_battery_config() -> BatteryThresholdConfig:
    """Build three-zone SoC thresholds with ``B_charge < B_area`` enforced.

    Example:
        >>> cfg = _build_battery_config()
        >>> cfg.B_charge < cfg.B_area
        True
    """

    return BatteryThresholdConfig(B_charge=0.30, B_area=0.50, critical_soc=0.10, SoC_target=1.0)


def _build_port_station_config() -> PortAndStationConfig:
    """Build 2 charging stations with a total of 4 ports for the default scenario.

    Example:
        >>> cfg = _build_port_station_config()
        >>> len(cfg.stations)
        2
    """

    station_0 = ChargingStationSpec(
        station_id="spklu_0",
        location=(-6.9075, 107.6091),
        ports=[
            PortSpec(port_id="spklu_0_port_0", power_kw=50.0, base_price_per_kwh=2500.0),
            PortSpec(port_id="spklu_0_port_1", power_kw=50.0, base_price_per_kwh=2500.0),
        ],
    )
    station_1 = ChargingStationSpec(
        station_id="spklu_1",
        location=(-6.9275, 107.6391),
        ports=[
            PortSpec(port_id="spklu_1_port_0", power_kw=22.0, base_price_per_kwh=3000.0),
            PortSpec(port_id="spklu_1_port_1", power_kw=22.0, base_price_per_kwh=3000.0),
        ],
    )
    return PortAndStationConfig(stations=[station_0, station_1])


def _build_emergency_config() -> EmergencyStationConfig:
    """Build the special emergency charging station configuration.

    Example:
        >>> _build_emergency_config().emergency_price_per_kwh
        10000.0
    """

    return EmergencyStationConfig(
        emergency_station_location=(-6.9175, 107.6191),
        emergency_price_per_kwh=10000.0,
        emergency_power_kw=60.0,
    )


def _build_marking_area_order_config() -> MarkingAreaAndOrderConfig:
    """Build 3 marking areas and an hourly order generation table (busy evening hours).

    Example:
        >>> cfg = _build_marking_area_order_config()
        >>> len(cfg.areas)
        3
    """

    areas = [
        MarkingAreaSpec(area_id="area_0", center=(-6.9025, 107.6041)),
        MarkingAreaSpec(area_id="area_1", center=(-6.9175, 107.6191)),
        MarkingAreaSpec(area_id="area_2", center=(-6.9325, 107.6341)),
    ]
    time_weight = {hour: 0.3 for hour in range(24)}
    for busy_hour in (7, 8, 17, 18, 19):
        time_weight[busy_hour] = 1.0
    for moderate_hour in (6, 9, 12, 13, 16, 20):
        time_weight[moderate_hour] = 0.6
    return MarkingAreaAndOrderConfig(
        areas=areas,
        order_min=1,
        order_max=5,
        time_weight=time_weight,
        order_claim_mode="lock",
        # "Point 0" reference (Bandung center) + radius bounding every order
        # destination; sampled destinations beyond it are projected onto the circle.
        order_reference_point=(-6.9175, 107.6191),
        order_destination_radius_km=8.0,
    )


def _build_fare_config() -> FareConfig:
    """Build a pure per-km fare configuration.

    Example:
        >>> _build_fare_config().fare_mode
        'PURE'
    """

    return FareConfig(fare_mode="PURE", price_per_km=3000.0, base_fare=0.0)


def _build_future_slot_config() -> FutureSlotConfig:
    """Build future-slot departure-gate configuration with a strict guard policy.

    Example:
        >>> _build_future_slot_config().late_arrival_mode
        'STRICT_GUARD'
    """

    return FutureSlotConfig(departure_margin_min=10, late_arrival_mode="STRICT_GUARD")


def _build_dead_battery_config() -> DeadBatteryConfig:
    """Build the default dead-battery handling policy (sikap A: LET_DIE).

    Example:
        >>> _build_dead_battery_config().dead_battery_mode
        'LET_DIE'
    """

    return DeadBatteryConfig(dead_battery_mode="LET_DIE")


def _build_ufun_config() -> UfunConfig:
    """Build default ufun weights, intrinsic REF_* references, and shape parameters.

    Example:
        >>> _build_ufun_config().shape.ufun_distance_mode
        'HYBRID'
    """

    return UfunConfig(
        fleet_weights=FleetUfunWeights(
            w_ready=1.0, w_price=1.0, w_travel=1.0, w_detour=1.0, w_oppty=1.0, w_unassigned=1.0
        ),
        spklu_weights=SpkluUfunWeights(w_rev=1.0, w_util=1.0, w_shift=1.0),
        fleet_references=FleetUfunReferences(
            REF_cost=100000.0, REF_energy=10.0, REF_dist=10.0, REF_earn=100000.0
        ),
        spklu_references=SpkluUfunReferences(
            REF_rev=100000.0, REF_porthours=10.0, REF_comp=50000.0
        ),
        shape=UfunShapeConfig(d0=2.0, g_max=1.0, tau=5.0, ufun_distance_mode="HYBRID"),
    )


def _build_app_user_booking_config() -> AppUserBookingConfig:
    """Build the initial app-user booking count and single-fallback parameters.

    Example:
        >>> _build_app_user_booking_config().initial_booking_count
        2
    """

    fallback = FallbackDistributionConfig(
        allowed_slot_shift_range=(1, 4),
        compensation_per_kwh_range=(0.0, 1000.0),
    )
    return AppUserBookingConfig(
        initial_booking_count=2,
        fallback=fallback,
    )


def _build_state_feature_config() -> StateFeatureConfig:
    """Build feature_spec, visibility, and encoder dims consistent with each other.

    Fleet sees taxi SoC/position/cycles_waited/urgency, port capacity/price,
    and area order-rate/time_weight. SPKLU sees only its own ports/bookings
    (no taxi fields at all), per the default asymmetric visibility described
    in ``.claude/project_instruction/03-adaptasi-baru/03-03-state-blok-fitur.md``.

    Example:
        >>> cfg = _build_state_feature_config()
        >>> "soc" in cfg.feature_spec.fleet.taxi
        True
        >>> "soc" in cfg.feature_spec.spklu.taxi
        False
    """

    fleet_feature_spec = FeatureBlockSpec(
        taxi=["soc", "dist_to_area", "cycles_waited", "urgency", "energy_need"],
        port=["power_kw", "free_grids", "base_price"],
        area=["order_rate_now", "order_rate_slot", "time_weight"],
        scalar=["hour", "day_progress"],
        cand=["assigned_count", "total_charge_cost", "avg_readiness", "total_distance"],
    )
    spklu_feature_spec = FeatureBlockSpec(
        port=["power_kw", "free_grids", "base_price", "n_bookings"],
        booking=["slot", "is_shiftable", "compensation"],
        scalar=["hour", "day_progress"],
        cand=["revenue", "porthours", "n_shifted", "total_compensation"],
    )
    feature_spec = FeatureSpecConfig(fleet=fleet_feature_spec, spklu=spklu_feature_spec)

    visibility = VisibilityConfig(
        fleet=[
            "taxi.soc",
            "taxi.position",
            "taxi.cycles_waited",
            "taxi.urgency",
            "port.power_kw",
            "port.free_grids",
            "port.base_price_per_kwh",
            "area.order_rate",
            "area.time_weight",
            "scalar.hour",
        ],
        spklu=[
            "port.power_kw",
            "port.free_grids",
            "port.base_price_per_kwh",
            "port.bookings",
            "booking.slot",
            "booking.is_shiftable",
            "booking.compensation",
            "scalar.hour",
        ],
    )

    encoder_dims = EncoderDimsConfig(N_max=10, K_max=16, phi_hidden_dim=32, pooling="mean")

    return StateFeatureConfig(
        feature_spec=feature_spec, visibility=visibility, encoder_dims=encoder_dims
    )


def _build_agent_side_config() -> AgentSideConfig:
    """Build one side's agent configuration: RLNegotiator + PPO + PADDED mode.

    Both fleet and SPKLU use an identical structure (RLNegotiator/PPO/PADDED)
    in the default scenario, with critic enrichment left at its default-off
    setting.

    Example:
        >>> _build_agent_side_config().agent_type
        'RLNegotiator'
    """

    return AgentSideConfig(
        agent_type="RLNegotiator",
        trainer_type="PPO",
        mode="PADDED",
        hyperparameters=TrainerHyperparameters(
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            gradient_clip_norm=0.5,
            epoch=4,
            batch_size=32,
            learning_rate=3e-4,
            tau=0.005,
            hidden_layers=2,
        ),
        reward_mode=RewardModeConfig(
            propose_mode="mode1",
            respond_mode="mode1",
            threshold_mode="mode1",
            offer_avg_window=10,
            utility_scale=True,
        ),
        critic_enrichment=CriticEnrichmentConfig(
            critic_sees_candidate_pool=False, critic_privileged_info=False
        ),
        load_from_registry=None,
    )


def _build_agent_config() -> AgentConfig:
    """Build the per-side agent configuration container for both fleet and SPKLU.

    Example:
        >>> cfg = _build_agent_config()
        >>> cfg.fleet.mode, cfg.spklu.mode
        ('PADDED', 'PADDED')
    """

    return AgentConfig(fleet=_build_agent_side_config(), spklu=_build_agent_side_config())


def _build_training_config() -> TrainingConfig:
    """Build training orchestration configuration (on-the-fly, simultaneous concurrency).

    Example:
        >>> _build_training_config().training_concurrency
        'SIMULTANEOUS'
    """

    return TrainingConfig(
        training_concurrency="SIMULTANEOUS",
        train_every_n_episodes=5,
        on_the_fly=True,
        schema_version=1,
        registry_path="data/registry",
        history_path="data/history",
    )


def _build_osm_config() -> OsmConfig:
    """Build the default OSM region and routing-cache configuration.

    Example:
        >>> _build_osm_config().network_type
        'drive'
    """

    return OsmConfig(
        osm_region="Bandung, Indonesia",
        network_type="drive",
        use_cache=True,
        cache_path="data/history/distance_matrix_cache.json",
    )


def _build_dashboard_config(time_config: TimeConfig) -> DashboardConfig:
    """Build the default dashboard throttle/playback-speed configuration.

    ``tick_delay_s`` is derived from ``time_config.speed_mode``:

    - ``"fast"`` (default): ``tick_delay_s = 0.0`` — no real-time sleep between
      ticks; the simulation runs as fast as possible.
    - ``"slow"``: ``tick_delay_s = real_seconds_per_tick(time_config)`` — honours
      the fixed 1:60 world:real time scale so one simulated day takes ~24 real
      minutes (1440 simulated minutes / 60).

    Example:
        >>> from simulation.config.config import TimeConfig
        >>> _build_dashboard_config(TimeConfig(n_days=1, tick_resolution_min=5,
        ...                                    interval_min=30,
        ...                                    speed_mode='fast')).tick_delay_s
        0.0
        >>> _build_dashboard_config(TimeConfig(n_days=1, tick_resolution_min=5,
        ...                                    interval_min=30,
        ...                                    speed_mode='slow')).tick_delay_s
        5.0
    """

    if time_config.speed_mode == "slow":
        tick_delay = real_seconds_per_tick(time_config)
    else:
        # "fast" mode (default): no sleep between ticks
        tick_delay = 0.0

    return DashboardConfig(
        websocket_throttle_s=0.1,
        tick_delay_s=tick_delay,
        map_renderer="leaflet",
    )


def _build_reporting_config() -> ReportingConfig:
    """Build the default reporting/runs configuration (runs root ``data/runs``).

    All fields keep their :class:`ReportingConfig` defaults: the reporting layer
    is enabled, real-time per-period JSON plus per-day and per-simulation PNG
    charts are written under ``data/runs/<sim_id>/day_<index>/``.

    Example:
        >>> _build_reporting_config().runs_root
        'data/runs'
        >>> _build_reporting_config().enabled
        True
    """

    return ReportingConfig(
        enabled=True,
        runs_root="data/runs",
        write_period_json=True,
        render_day_png=True,
        render_sim_png=True,
        png_dpi=130,
        figure_width_in=14.0,
        figure_height_in=8.0,
        cri_star_theta=0.1,
        min_points_for_tanh_fit=4,
        json_indent=2,
        keep_period_payloads_in_memory=True,
        max_periods_in_memory=5000,
    )


def default_config() -> SimulationConfig:
    """Build the default small-but-runnable :class:`SimulationConfig` scenario.

    The scenario has 3 taxis, 2 charging stations (4 ports total), 3 marking
    areas, both negotiating sides as ``RLNegotiator`` + ``PPO`` + ``PADDED``,
    on-the-fly simultaneous training, and a short 1-day horizon — suitable for
    smoke testing and as the dashboard's initial configuration.

    Returns:
        A fully populated, internally consistent :class:`SimulationConfig`.

    Example:
        >>> cfg = default_config()
        >>> cfg.time.n_days
        1
        >>> len(cfg.taxi_spec.initial_taxi_state)
        3
        >>> len(cfg.marking_area_order.areas)
        3
        >>> cfg.agent.fleet.trainer_type
        'PPO'
    """

    time_config = _build_time_config()
    return SimulationConfig(
        time=time_config,
        negotiation=_build_negotiation_config(),
        taxi_spec=_build_taxi_spec_config(),
        battery=_build_battery_config(),
        port_station=_build_port_station_config(),
        emergency=_build_emergency_config(),
        marking_area_order=_build_marking_area_order_config(),
        fare=_build_fare_config(),
        future_slot=_build_future_slot_config(),
        dead_battery=_build_dead_battery_config(),
        ufun=_build_ufun_config(),
        app_user_booking=_build_app_user_booking_config(),
        state_feature=_build_state_feature_config(),
        agent=_build_agent_config(),
        training=_build_training_config(),
        osm=_build_osm_config(),
        dashboard=_build_dashboard_config(time_config),
        reporting=_build_reporting_config(),
        seed=42,
    )
