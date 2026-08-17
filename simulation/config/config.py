"""Single source-of-truth configuration tree for one simulation run.

This module defines :class:`SimulationConfig`, one dataclass tree that holds
**every** parameter listed in
``.claude/project_instruction/08-konfigurasi/08-01-daftar-konfigurasi.md``
(time, negotiation, taxi spec, battery thresholds, port/station, emergency,
marking area/order, fare, future slot, dead-battery handling, ufun weights +
REF_* + shape, app-user booking & fallback, state/feature/visibility, agent
type/trainer/mode/hyperparameters/critic flags, training/registry, OSM/map,
and dashboard).

Configuration is immutable for the duration of one run (see
``.claude/project_instruction/00-MULAI-DI-SINI.md`` rule #6 and Gotcha G14 in
``91-PLAYBOOK-IMPLEMENTASI.md``): nothing here is mutated after a
:class:`SimulationConfig` instance is constructed and handed to the engine.

Two normalization layers must never be confused (Kontrak C6 / Gotcha G10):

- ``UfunWeightsConfig`` + ``UfunReferenceConfig`` (``REF_*``) make each side's
  utility function *intrinsic* (independent of what else is in the
  ``action_library`` menu for that negotiation).
- ``utility_min``/``utility_max`` are computed *per negotiation* from the
  realized ``action_library`` (see ``simulation.bridge.domain_builder``) and
  are used **only** to scale RL reward (``negotiation.reward_shaper``). They
  are not part of this configuration tree because they are derived at
  negotiation time, not configured ahead of time.

Example
-------
>>> from simulation.config.config import SimulationConfig, TimeConfig
>>> cfg = SimulationConfig(time=TimeConfig(n_days=1, tick_resolution_min=5,
...                                        interval_min=30))
>>> cfg.time.n_days
1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# A. Time & duration
# ---------------------------------------------------------------------------


@dataclass
class TimeConfig:
    """Simulation time scale, tick resolution, and negotiation trigger interval.

    The world:real time scale is fixed at 1:60 (1 simulated minute = 1 real
    second) and is therefore not a configurable field — it is a structural
    constant of the simulation, not a variable parameter (see
    ``01-dunia-simulasi/01-02-waktu-dan-tick.md``).

    Attributes:
        n_days: Number of simulated days to run.
        tick_resolution_min: Simulated minutes advanced per engine tick.
        interval_min: Simulated minutes between negotiation-trigger checks.
            Usually a multiple of ``tick_resolution_min``.
        speed_mode: Simulation speed mode. ``"fast"`` (default) runs ticks
            without any real-time sleep between them. ``"slow"`` inserts a
            ``real_seconds_per_tick``-second sleep between ticks to honour the
            fixed 1:60 world:real time scale for visualisation/debug purposes.
            The freeze/resume mechanism (negotiation pause) is unaffected by
            this field and operates identically in both modes.

    Example:
        >>> TimeConfig(n_days=1, tick_resolution_min=5, interval_min=30).interval_min
        30
    """

    n_days: int = 1
    tick_resolution_min: int = 5
    interval_min: int = 30
    speed_mode: str = "fast"  # "fast" | "slow"


# ---------------------------------------------------------------------------
# B. Negotiation
# ---------------------------------------------------------------------------


@dataclass
class NegotiationConfig:
    """Caps and discrete price levels governing one bilateral SAO negotiation.

    Every IDLE_WAITING taxi that can reach at least one port becomes a candidate
    (there is no cap on the number of candidate taxis — see
    :func:`simulation.bridge.snapshot.collect_charge_candidates`); the only size
    bound is ``max_action_library`` (``K_max``) on the number of bundles.

    Attributes:
        max_action_library: Upper bound on the number of candidate bundles
            (``K_max``) generated for one negotiation. When the cartesian
            combination of per-taxi assignments exceeds this, the library is
            truncated to the ``K_max`` bundles with the highest *fleet* utility.
        max_steps: Maximum number of SAO mechanism steps before declaring
            no-deal.
        price_levels: Discrete set of admissible price-per-kWh values
            (Rupiah) that bundle generation and ufun pricing draw from.
        prune_floor_price_on_shift: When ``True``, any assignment that displaces
            an app-user booking (``shift is not None``) may only be offered at a
            price **>= that port's base price** — the below-base ("floor") price
            levels are pruned for shifted assignments. Rationale (varian
            ``simulasi_mahal_kompen``): a booking shift already costs the SPKLU a
            compensation payment, so allowing a floor (discounted) charge on top
            double-penalises the SPKLU. Default ``False`` (no pruning; every price
            level is admissible for every assignment).

    Example:
        >>> NegotiationConfig(max_steps=20, price_levels=[2000, 3000, 4000]).max_steps
        20
        >>> NegotiationConfig().prune_floor_price_on_shift
        False
    """

    max_action_library: int = 16
    max_steps: int = 20
    price_levels: List[float] = field(default_factory=lambda: [2000.0, 3000.0, 4000.0])
    prune_floor_price_on_shift: bool = False


# ---------------------------------------------------------------------------
# C. Taxi specification (uniform across fleet)
# ---------------------------------------------------------------------------


@dataclass
class InitialTaxiState:
    """Explicit initial position and state of charge for one taxi.

    Attributes:
        taxi_id: Identifier of the taxi this initial state belongs to.
        position: ``(lat, lng)`` initial coordinate.
        soc: Initial state of charge in ``[0, 1]``.

    Example:
        >>> InitialTaxiState(taxi_id="taxi_0", position=(-6.2, 106.8), soc=0.8).soc
        0.8
    """

    taxi_id: str
    position: Tuple[float, float]
    soc: float


@dataclass
class TaxiSpecConfig:
    """Uniform taxi specification shared by every taxi in the fleet.

    Attributes:
        battery_capacity_kwh: Battery capacity in kWh, identical for all taxis.
        consumption_kwh_per_km: Energy consumption per km traveled.
        speed_kmh: Uniform travel speed in km/h.
        initial_taxi_state: Explicit list of initial (position, SoC) per taxi
            (see ``01-dunia-simulasi/01-02-waktu-dan-tick.md`` "Inisialisasi").

    Example:
        >>> spec = TaxiSpecConfig(battery_capacity_kwh=50.0,
        ...                       consumption_kwh_per_km=0.15, speed_kmh=30.0,
        ...                       initial_taxi_state=[])
        >>> spec.speed_kmh
        30.0
    """

    battery_capacity_kwh: float = 50.0
    consumption_kwh_per_km: float = 0.15
    speed_kmh: float = 30.0
    initial_taxi_state: List[InitialTaxiState] = field(default_factory=list)


# ---------------------------------------------------------------------------
# D. Battery thresholds
# ---------------------------------------------------------------------------


@dataclass
class BatteryThresholdConfig:
    """Three-zone SoC thresholds governing idle taxi decisions and ufun shaping.

    Invariant: ``B_charge < B_area`` (see
    ``01-dunia-simulasi/01-04-logika-taksi-idle.md``).

    Attributes:
        B_charge: SoC threshold below which a taxi needs charging (critical zone).
        B_area: Minimum SoC a taxi must still have on arrival to commit to
            (re)positioning to a marking area. An idle taxi that cannot reach
            any area while keeping ``SoC >= B_area`` stays put and becomes a
            charge candidate at the next negotiation.
        critical_soc: Lower SoC reference used by the order generator to bound a
            trip's "soft-reachable" destination distance. (Emergency handling is
            **not** triggered by this threshold any more — it is triggered when an
            idle, waiting taxi can reach no port at all; see
            ``simulation.bridge.emergency``.)
        SoC_target: Target fill level when charging (default full = 1.0).

    Example:
        >>> BatteryThresholdConfig(B_charge=0.3, B_area=0.5).B_charge
        0.3
    """

    B_charge: float = 0.30
    B_area: float = 0.50
    critical_soc: float = 0.10
    SoC_target: float = 1.0


# ---------------------------------------------------------------------------
# E. SPKLU & ports
# ---------------------------------------------------------------------------


@dataclass
class PortSpec:
    """Configuration of one charging port at a station.

    Attributes:
        port_id: Identifier of the port, unique within its station.
        power_kw: Charging power in kW (determines charge speed and energy
            delivered per grid slot).
        base_price_per_kwh: Base price per kWh for this port (part of the
            SPKLU's specification, used by ufun and bundle generation).

    Example:
        >>> PortSpec(port_id="p0", power_kw=50.0, base_price_per_kwh=2500.0).power_kw
        50.0
    """

    port_id: str
    power_kw: float
    base_price_per_kwh: float


@dataclass
class ChargingStationSpec:
    """Configuration of one charging station (SPKLU) and its ports.

    Attributes:
        station_id: Identifier of the station.
        location: ``(lat, lng)`` fixed location of the station.
        ports: List of :class:`PortSpec` belonging to this station.

    Example:
        >>> station = ChargingStationSpec(station_id="s0", location=(-6.2, 106.8),
        ...                                ports=[PortSpec("p0", 50.0, 2500.0)])
        >>> len(station.ports)
        1
    """

    station_id: str
    location: Tuple[float, float]
    ports: List[PortSpec] = field(default_factory=list)


@dataclass
class PortAndStationConfig:
    """Collection of all regular (non-emergency) charging stations.

    There is no station "service radius": a taxi can be paired with **any** port
    it has enough battery to reach (the former ``port_radius`` "SPKLU area"
    gating has been removed). Each ``(taxi, port)`` pairing also yields exactly
    one candidate slot (the start grid is the taxi's exact arrival grid), so
    there is no per-port slot cap either.

    Attributes:
        stations: List of :class:`ChargingStationSpec` for every regular SPKLU.

    Example:
        >>> PortAndStationConfig(stations=[]).stations
        []
    """

    stations: List[ChargingStationSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# F. Emergency SPKLU (sikap B)
# ---------------------------------------------------------------------------


@dataclass
class EmergencyStationConfig:
    """Configuration of the special emergency charging station (rescue, sikap B).

    The emergency station has unlimited ports (structural property, not a
    configurable field) and is always invisible to negotiation / never
    enters ``action_library`` (Gotcha G8).

    Attributes:
        emergency_station_location: Fixed ``(lat, lng)`` location of the
            emergency station.
        emergency_price_per_kwh: Premium price per kWh charged at the
            emergency station.
        emergency_power_kw: Charging power (kW) of the emergency port. Determines
            how long an emergency (HARD_FLOOR) rescue charge takes (see ``05-05``).
            ``None`` means "derive from a representative regular-port power" (the
            historical fallback); set an explicit value to control rescue charge
            speed as a configurable experiment variable.

    Example:
        >>> EmergencyStationConfig(emergency_station_location=(-6.3, 106.9),
        ...                        emergency_price_per_kwh=10000.0).emergency_price_per_kwh
        10000.0
        >>> EmergencyStationConfig(emergency_power_kw=60.0).emergency_power_kw
        60.0
    """

    emergency_station_location: Tuple[float, float] = (0.0, 0.0)
    emergency_price_per_kwh: float = 10000.0
    emergency_power_kw: Optional[float] = None


# ---------------------------------------------------------------------------
# G. Marking area & order
# ---------------------------------------------------------------------------


@dataclass
class MarkingAreaSpec:
    """Configuration of one marking area (where passenger orders may appear).

    Attributes:
        area_id: Identifier of the marking area.
        center: ``(lat, lng)`` center point of the area (also used as the
            geometry reference for fixed-point distance matrix precomputation).

    Example:
        >>> MarkingAreaSpec(area_id="a0", center=(-6.21, 106.81)).area_id
        'a0'
    """

    area_id: str
    center: Tuple[float, float]


@dataclass
class MarkingAreaAndOrderConfig:
    """Marking areas and the per-hour order generation parameters.

    Attributes:
        areas: List of :class:`MarkingAreaSpec` defining every marking area.
        order_min: Minimum number of orders spawned per area per hour.
        order_max: Maximum number of orders spawned per area per hour.
        time_weight: Table mapping hour-of-day (0..23) to a weight value used
            both to bias order volume and as the "right timing" factor in the
            fleet ufun (single source of truth, see ``01-03``/``04-02``).
        order_claim_mode: ``"lock"`` (claim a specific order before departing)
            or ``"steal"`` (claim only on arrival, greedy-nearest).
        order_reference_point: ``(lat, lng)`` "point 0" — the reference all order
            destinations are bounded around. ``None`` disables the bound (legacy
            behaviour). See ``order_destination_radius_km``.
        order_destination_radius_km: Maximum distance (km) an order destination
            may lie from ``order_reference_point``. Any sampled destination farther
            than this is projected back onto the circle, so destinations **never**
            leave the radius. ``None`` (or non-positive) disables the bound.

    Example:
        >>> cfg = MarkingAreaAndOrderConfig(areas=[], order_min=1, order_max=5,
        ...                                 time_weight={h: 1.0 for h in range(24)},
        ...                                 order_claim_mode="lock")
        >>> cfg.order_claim_mode
        'lock'
        >>> MarkingAreaAndOrderConfig(
        ...     order_reference_point=(-6.9175, 107.6191),
        ...     order_destination_radius_km=8.0).order_destination_radius_km
        8.0
    """

    areas: List[MarkingAreaSpec] = field(default_factory=list)
    order_min: int = 1
    order_max: int = 5
    time_weight: Dict[int, float] = field(
        default_factory=lambda: {hour: 1.0 for hour in range(24)}
    )
    order_claim_mode: str = "lock"  # "lock" | "steal"
    order_reference_point: Optional[Tuple[float, float]] = None
    order_destination_radius_km: Optional[float] = None


# ---------------------------------------------------------------------------
# H. Fare
# ---------------------------------------------------------------------------


@dataclass
class FareConfig:
    """Passenger fare configuration.

    Attributes:
        fare_mode: ``"PURE"`` (``price_per_km * productive_km``) or
            ``"WITH_BASE_FARE"`` (``base_fare + price_per_km * productive_km``).
        price_per_km: Fare per productive km, assumed constant across time.
        base_fare: Flat fare component, only used when ``fare_mode`` is
            ``"WITH_BASE_FARE"``.

    Example:
        >>> FareConfig(fare_mode="PURE", price_per_km=3000.0).fare_mode
        'PURE'
    """

    fare_mode: str = "PURE"  # "PURE" | "WITH_BASE_FARE"
    price_per_km: float = 3000.0
    base_fare: float = 0.0


# ---------------------------------------------------------------------------
# I. Future slot
# ---------------------------------------------------------------------------


@dataclass
class FutureSlotConfig:
    """Future charging slot booking and departure-gate configuration.

    The time grid itself is fixed at 30 minutes (structural constant, not a
    configurable field, see ``01-dunia-simulasi/01-08-slot-masa-depan.md``).

    Attributes:
        departure_margin_min: Safety margin (minutes) subtracted from
            ``slot_start - travel_time(position -> port)`` to compute
            ``departure_time``.
        late_arrival_mode: ``"STRICT_GUARD"`` (conservative order acceptance
            so a taxi never arrives late) or ``"SLOT_FORFEIT"`` (more
            aggressive order-taking; occasional late arrivals forfeit the
            slot, recorded as a metric).

    Example:
        >>> FutureSlotConfig(departure_margin_min=10,
        ...                  late_arrival_mode="STRICT_GUARD").late_arrival_mode
        'STRICT_GUARD'
    """

    departure_margin_min: int = 10
    late_arrival_mode: str = "STRICT_GUARD"  # "STRICT_GUARD" | "SLOT_FORFEIT"


# ---------------------------------------------------------------------------
# J. Dead battery handling
# ---------------------------------------------------------------------------


@dataclass
class DeadBatteryConfig:
    """Critical/dead battery handling policy.

    Attributes:
        dead_battery_mode: ``"LET_DIE"`` (sikap A: taxi becomes ``STRANDED``,
            recorded as a failure metric) or ``"HARD_FLOOR"`` (sikap B:
            rescued via teleport to the emergency station).

    Example:
        >>> DeadBatteryConfig(dead_battery_mode="LET_DIE").dead_battery_mode
        'LET_DIE'
    """

    dead_battery_mode: str = "LET_DIE"  # "LET_DIE" | "HARD_FLOOR"


# ---------------------------------------------------------------------------
# K & L & M. Utility function — weights, references, shape
# ---------------------------------------------------------------------------


@dataclass
class FleetUfunWeights:
    """Weights for each term of the fleet utility function (see ``04-02``).

    Attributes:
        w_ready: Weight on ``readiness`` (energy_frac * time_weight * demand_weight).
        w_price: Weight (penalty) on normalized charging cost.
        w_travel: Weight (penalty) on normalized travel energy to reach the port.
        w_detour: Weight (penalty) on normalized port-to-area detour distance.
        w_oppty: Weight (penalty) on normalized opportunity cost.
        w_unassigned: Weight (penalty) on the saturating ``cycles_waited`` penalty
            for UNASSIGNED taxis.

    Example:
        >>> FleetUfunWeights().w_ready
        1.0
    """

    w_ready: float = 1.0
    w_price: float = 1.0
    w_travel: float = 1.0
    w_detour: float = 1.0
    w_oppty: float = 1.0
    w_unassigned: float = 1.0


@dataclass
class SpkluUfunWeights:
    """Weights for each term of the SPKLU utility function (see ``04-03``).

    Attributes:
        w_rev: Weight on normalized revenue (``price * energy``).
        w_util: Weight on normalized port-hours utilization.
        w_shift: Weight (penalty) on normalized booking-shift compensation.

    Example:
        >>> SpkluUfunWeights().w_rev
        1.0
    """

    w_rev: float = 1.0
    w_util: float = 1.0
    w_shift: float = 1.0


@dataclass
class FleetUfunReferences:
    """Fixed reference values (``REF_*``) that make the fleet ufun intrinsic (see ``04-04``).

    These are NOT min/max over the action_library (that is a separate,
    per-negotiation scaling used only for reward, see Gotcha G10 / Kontrak C6).

    Attributes:
        REF_cost: Reference Rupiah cost used to normalize ``charge_cost``.
        REF_energy: Reference kWh used to normalize travel energy.
        REF_dist: Reference km used to normalize detour distance.
        REF_earn: Reference Rupiah earning used to normalize opportunity cost.

    Example:
        >>> FleetUfunReferences().REF_cost
        100000.0
    """

    REF_cost: float = 100000.0
    REF_energy: float = 10.0
    REF_dist: float = 10.0
    REF_earn: float = 100000.0


@dataclass
class SpkluUfunReferences:
    """Fixed reference values (``REF_*``) that make the SPKLU ufun intrinsic (see ``04-04``).

    Attributes:
        REF_rev: Reference Rupiah revenue used to normalize ``price * energy``.
        REF_porthours: Reference port-hours used to normalize utilization.
        REF_comp: Reference Rupiah compensation used to normalize shift cost.

    Example:
        >>> SpkluUfunReferences().REF_rev
        100000.0
    """

    REF_rev: float = 100000.0
    REF_porthours: float = 10.0
    REF_comp: float = 50000.0


@dataclass
class UfunShapeConfig:
    """Shape parameters for non-linear terms shared by ufun formulas (see ``04-04``).

    Attributes:
        d0: Distance-decay scale used in ``dist_decay(d) = exp(-d / d0)``
            (demand_weight term of fleet readiness).
        g_max: Saturation cap for the UNASSIGNED penalty
            ``g(c) = g_max * (1 - exp(-c / tau))``.
        tau: Saturation rate constant for the same UNASSIGNED penalty.
        ufun_distance_mode: ``"HYBRID"`` (precomputed matrix inside area,
            on-demand OSM route outside) or ``"SNAP_NEAREST_AREA"`` (Euclidean
            snap to nearest area for out-of-area taxis, cheaper but biased).

    Example:
        >>> UfunShapeConfig(d0=2.0, g_max=1.0, tau=5.0).g_max
        1.0
    """

    d0: float = 2.0
    g_max: float = 1.0
    tau: float = 5.0
    ufun_distance_mode: str = "HYBRID"  # "HYBRID" | "SNAP_NEAREST_AREA"


@dataclass
class UfunConfig:
    """Aggregate of all utility-function configuration (weights, references, shape).

    Attributes:
        fleet_weights: Fleet-side ufun term weights.
        spklu_weights: SPKLU-side ufun term weights.
        fleet_references: Fleet-side fixed REF_* normalization constants.
        spklu_references: SPKLU-side fixed REF_* normalization constants.
        shape: Shared non-linear shape parameters (d0, g_max, tau, distance mode).

    Example:
        >>> UfunConfig().shape.ufun_distance_mode
        'HYBRID'
    """

    fleet_weights: FleetUfunWeights = field(default_factory=FleetUfunWeights)
    spklu_weights: SpkluUfunWeights = field(default_factory=SpkluUfunWeights)
    fleet_references: FleetUfunReferences = field(default_factory=FleetUfunReferences)
    spklu_references: SpkluUfunReferences = field(default_factory=SpkluUfunReferences)
    shape: UfunShapeConfig = field(default_factory=UfunShapeConfig)


# ---------------------------------------------------------------------------
# N. App-user booking & fallback
# ---------------------------------------------------------------------------


@dataclass
class FallbackDistributionConfig:
    """Distributional parameters for app-user booking fallback plans (see ``01-01``).

    Every app-user booking has **exactly one** fallback plan: a single concrete
    alternative placement ``(to_port, to_slot)`` of the **same duration** as the
    original booking, picked once at simulation start. Once that fallback is used
    (the booking is shifted), the booking has no fallback left and can never be
    shifted again.

    Attributes:
        allowed_slot_shift_range: ``(min, max)`` magnitude (in half-hour grids,
            inclusive) of the time-shift used when choosing the booking's single
            fallback slot. The sign (earlier/later) is chosen randomly.
        compensation_per_kwh_range: ``(min, max)`` Rupiah-per-kWh range the
            random shift compensation rate is drawn from. The **total**
            compensation owed for shifting a booking is energy-based:
            ``rate_per_kwh * power_kw * duration_grids * 0.5`` (0.5 h per grid).

    Example:
        >>> cfg = FallbackDistributionConfig(allowed_slot_shift_range=(1, 4),
        ...                                  compensation_per_kwh_range=(0.0, 1000.0))
        >>> cfg.compensation_per_kwh_range
        (0.0, 1000.0)
    """

    allowed_slot_shift_range: Tuple[int, int] = (1, 4)
    compensation_per_kwh_range: Tuple[float, float] = (0.0, 1000.0)


@dataclass
class AppUserBookingConfig:
    """Initial app-user booking configuration (see ``01-01``).

    All app-user bookings are created **once at simulation start**, spread across
    the whole simulation's grid timeline; there is no per-hour spawning. Every
    booking has exactly one fallback plan (see :class:`FallbackDistributionConfig`).

    Attributes:
        initial_booking_count: Number of app-user bookings randomly placed at
            simulation start (a target; a booking is skipped if its random slot
            is already occupied).
        fallback: Parameters for each booking's single fallback plan.

    Example:
        >>> AppUserBookingConfig(initial_booking_count=2).initial_booking_count
        2
    """

    initial_booking_count: int = 2
    fallback: FallbackDistributionConfig = field(default_factory=FallbackDistributionConfig)


# ---------------------------------------------------------------------------
# O. State / feature / visibility
# ---------------------------------------------------------------------------


@dataclass
class FeatureBlockSpec:
    """Field selection for one feature block (taxi/port/area/scalar/cand/booking) for one side.

    Attributes:
        taxi: Field names included in the taxi feature block.
        port: Field names included in the port feature block.
        area: Field names included in the area feature block.
        booking: Field names included in the booking feature block.
        scalar: Field names included in the scalar feature block.
        cand: Field names included in the per-candidate-bundle feature block.

    Example:
        >>> FeatureBlockSpec(taxi=["soc", "cycles_waited"]).taxi
        ['soc', 'cycles_waited']
    """

    taxi: List[str] = field(default_factory=list)
    port: List[str] = field(default_factory=list)
    area: List[str] = field(default_factory=list)
    booking: List[str] = field(default_factory=list)
    scalar: List[str] = field(default_factory=list)
    cand: List[str] = field(default_factory=list)


@dataclass
class FeatureSpecConfig:
    """Per-side feature_spec: which world fields populate which feature block (see ``03-03``).

    Attributes:
        fleet: Feature block specification for the fleet side.
        spklu: Feature block specification for the SPKLU side.

    Example:
        >>> spec = FeatureSpecConfig(
        ...     fleet=FeatureBlockSpec(taxi=["soc"]),
        ...     spklu=FeatureBlockSpec(port=["power_kw"]),
        ... )
        >>> spec.fleet.taxi
        ['soc']
    """

    fleet: FeatureBlockSpec = field(default_factory=FeatureBlockSpec)
    spklu: FeatureBlockSpec = field(default_factory=FeatureBlockSpec)


@dataclass
class VisibilityConfig:
    """Per-side list of world fields each side is allowed to observe (see ``03-03``).

    Attributes:
        fleet: List of world field names visible to the fleet side.
        spklu: List of world field names visible to the SPKLU side.

    Example:
        >>> VisibilityConfig(fleet=["taxi.soc"], spklu=["port.power_kw"]).fleet
        ['taxi.soc']
    """

    fleet: List[str] = field(default_factory=list)
    spklu: List[str] = field(default_factory=list)


@dataclass
class EncoderDimsConfig:
    """Dimensions used by both PADDED and ENCODER front-ends (see ``03-01``/``03-02``).

    There are four independent padding/encoder caps (per the locked design,
    chat-history config enumeration): ``N_max`` (taxis), ``P_max`` (ports),
    ``A_max`` (areas), and ``K_max`` (candidate bundles). ``P_max``/``A_max``
    default to ``None``, meaning the feature builder derives them (port count /
    ``N_max`` respectively) for backward compatibility; set them explicitly to
    control the port/area padding caps as experiment variables.

    Attributes:
        N_max: Maximum number of taxis represented (padding cap / encoder
            informational cap).
        P_max: Maximum number of ports represented. ``None`` -> derive from the
            number of configured regular ports.
        A_max: Maximum number of marking areas represented. ``None`` -> derive
            from ``N_max``.
        K_max: Maximum number of candidate bundles represented (also equals
            ``negotiation.max_action_library``).
        phi_hidden_dim: Hidden dimension of the per-element encoder MLP (phi)
            used by the ENCODER/PADDED set-encoders.
        phi_out_dim: Output (embedding) dimension of phi per element. ``None``
            -> equals ``phi_hidden_dim`` (the historical behaviour). Set it to
            decouple the per-element embedding width from the hidden width.
        pooling: Pooling operation for set-encoders, ``"mean"`` or ``"max"``.

    Example:
        >>> EncoderDimsConfig(N_max=10, K_max=16, phi_hidden_dim=32,
        ...                   pooling="mean").pooling
        'mean'
        >>> EncoderDimsConfig(P_max=8, A_max=6, phi_out_dim=16).phi_out_dim
        16
    """

    N_max: int = 10
    P_max: Optional[int] = None
    A_max: Optional[int] = None
    K_max: int = 16
    phi_hidden_dim: int = 32
    phi_out_dim: Optional[int] = None
    pooling: str = "mean"  # "mean" | "max"


@dataclass
class StateFeatureConfig:
    """Aggregate of feature_spec, visibility, and encoder/padding dimensions.

    Attributes:
        feature_spec: Per-side feature block field selection.
        visibility: Per-side visible world field list.
        encoder_dims: Shared encoder/padding dimensions (N_max, K_max, phi, pooling).

    Example:
        >>> StateFeatureConfig().encoder_dims.K_max
        16
    """

    feature_spec: FeatureSpecConfig = field(default_factory=FeatureSpecConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    encoder_dims: EncoderDimsConfig = field(default_factory=EncoderDimsConfig)


# ---------------------------------------------------------------------------
# P. Agent / trainer / mode (per side)
# ---------------------------------------------------------------------------


@dataclass
class TrainerHyperparameters:
    """Hyperparameters for one trainer instance (PG/PPO/DDPG/IOP), per role.

    Not every field is used by every trainer type; unused fields are simply
    ignored by that trainer (e.g. ``clip_ratio`` is PPO-only, ``tau`` is
    DDPG-only). See ``02-negosiasi-asis/02-08-trainers.md``.

    Attributes:
        gamma: Discount factor.
        gae_lambda: GAE lambda (PPO only).
        clip_ratio: PPO clipped-surrogate clip range.
        value_coef: PPO value-loss coefficient.
        entropy_coef: PPO entropy-bonus coefficient.
        gradient_clip_norm: Gradient clipping norm applied during ``update()``.
        epoch: Number of epochs per ``update()`` call.
        batch_size: Mini-batch size used by the trainer's DataLoader.
        learning_rate: Optimizer learning rate.
        tau: DDPG soft target-update rate.
        hidden_dim: Hidden-layer width of the agent's networks (policy/value/
            DDPG/IOP). Configurable experiment variable (default 128).
        mbom_M: Number of MBOM Imagined-Opponent-Policy reasoning levels
            (Bayesian-mixer size). ``M`` levels means recursion depth ``M-1``
            (``M=2`` -> level 0+1 = 1 recursion; ``M=3`` -> level 0+1+2). Only
            used by MBOM / DLST-MBOM agents; ignored otherwise. Wired to
            :class:`AgentBuildConfig.mbom_M` (default 3).
        rollout_k: MBOM recursive-imagination rollout depth (steps collected per
            imagined negotiation). Only used by MBOM / DLST-MBOM (default 2).
        n_rollout_samples: MBOM uniform-sample count per imagined candidate
            (default 1). Only used by MBOM / DLST-MBOM.
        rollout_max_steps: Horizon (max SAO steps) of each **imagined** negotiation
            inside recursive imagination — **decoupled** from ``negotiation.max_steps``
            (the real negotiation horizon). ``None`` (default) = faithful: use
            ``negotiation.max_steps``. Set a small explicit cap to bound MBOM/DLST-MBOM
            runtime when the real horizon is large (e.g. real-case ``max_steps=100``).
            NOTE: capping makes imagination more myopic and can shift outcomes toward
            more no-deal loops (which, via closed-loop feedback, may *increase* total
            negotiations) — so a too-small cap can be counterproductive; tune per case.

    Example:
        >>> TrainerHyperparameters(gamma=0.99, clip_ratio=0.2).clip_ratio
        0.2
        >>> TrainerHyperparameters(hidden_dim=256).hidden_dim
        256
        >>> TrainerHyperparameters(mbom_M=2).mbom_M
        2
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gradient_clip_norm: float = 0.5
    epoch: int = 4
    batch_size: int = 32
    learning_rate: float = 3e-4
    tau: float = 0.005
    hidden_dim: int = 128
    mbom_M: int = 3
    rollout_k: int = 2
    n_rollout_samples: int = 1
    rollout_max_steps: Optional[int] = None
    hidden_layers: int = 2


@dataclass
class RewardModeConfig:
    """Reward-shaping mode selection per role (see ``02-09``).

    Attributes:
        propose_mode: Reward mode for the propose role (e.g. ``"mode1"``).
        respond_mode: Reward mode for the respond role (e.g. ``"mode1"`` or
            ``"dlst"``).
        threshold_mode: Reward mode for the DLST utility-threshold role
            (e.g. ``"mode1"``).
        offer_avg_window: Number of most recent opponent offers averaged for
            offer_variant ``"c"`` in the respond reward (default 10).
        utility_scale: Whether utility values are scaled by
            ``utility_min``/``utility_max`` before reward computation.

    Example:
        >>> RewardModeConfig(respond_mode="dlst").respond_mode
        'dlst'
    """

    propose_mode: str = "mode1"
    respond_mode: str = "mode1"
    threshold_mode: str = "mode1"
    offer_avg_window: int = 10
    utility_scale: bool = True


@dataclass
class CriticEnrichmentConfig:
    """Two independent opt-in flags enriching only the PPO V(s) critic (see ``03-06``).

    Both flags default to off (behavior equivalent to the as-is mechanism).
    Never applies to DDPG/DLST's Q(s,a) critic.

    Attributes:
        critic_sees_candidate_pool: If True, the critic additionally receives
            a pooled summary of ``cand_feats`` (the available bundle menu).
        critic_privileged_info: If True, the critic additionally receives
            ground-truth fields not visible to the actor (asymmetric
            actor-critic), only during training.

    Example:
        >>> CriticEnrichmentConfig().critic_sees_candidate_pool
        False
    """

    critic_sees_candidate_pool: bool = False
    critic_privileged_info: bool = False


@dataclass
class AgentSideConfig:
    """Full agent/trainer/mode configuration for one negotiating side.

    Attributes:
        agent_type: One of ``"RLNegotiator"``, ``"MBOM"``,
            ``"MBOMNonZeroSum"``, ``"MBOMNewMechanism"``, ``"DLST"``,
            ``"DLSTMBOM"``, ``"DLSTMBOMNonZeroSum"``, ``"DLSTMBOMNewMechanism"``,
            or ``"RandomIOP"`` (baseline).
        trainer_type: One of ``"PG"``, ``"PPO"``, ``"DDPG"``, ``"IOP"``;
            must be compatible with ``agent_type`` (discrete agents use
            PG/PPO, DLST family uses DDPG (+IOP for hybrids), RandomIOP uses
            none).
        mode: ``"PADDED"`` or ``"ENCODER"`` front-end input representation
            (see ``03-01``). Ignored by the DLST head itself, but still
            governs DLST's front-end input encoding.
        hyperparameters: Trainer hyperparameters for this side's trainer(s).
        reward_mode: Reward-shaping mode selection for this side.
        critic_enrichment: Optional PPO V(s) critic enrichment flags.
        load_from_registry: If set, the agent is loaded from this registry
            id/version instead of being constructed fresh.

    Example:
        >>> AgentSideConfig(agent_type="RLNegotiator", trainer_type="PPO",
        ...                 mode="PADDED").mode
        'PADDED'
    """

    agent_type: str = "RLNegotiator"
    trainer_type: str = "PPO"
    mode: str = "PADDED"  # "PADDED" | "ENCODER"
    hyperparameters: TrainerHyperparameters = field(default_factory=TrainerHyperparameters)
    reward_mode: RewardModeConfig = field(default_factory=RewardModeConfig)
    critic_enrichment: CriticEnrichmentConfig = field(default_factory=CriticEnrichmentConfig)
    load_from_registry: Optional[str] = None


@dataclass
class AgentConfig:
    """Per-side agent configuration container (fleet and SPKLU).

    Attributes:
        fleet: Agent/trainer/mode configuration for the fleet side.
        spklu: Agent/trainer/mode configuration for the SPKLU side.

    Example:
        >>> AgentConfig().fleet.agent_type
        'RLNegotiator'
    """

    fleet: AgentSideConfig = field(default_factory=AgentSideConfig)
    spklu: AgentSideConfig = field(default_factory=AgentSideConfig)


# ---------------------------------------------------------------------------
# Q. Training & registry
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Training orchestration, concurrency, and artifact path configuration (see ``06-01``/``06-02``).

    Attributes:
        training_concurrency: ``"SIMULTANEOUS"``, ``"ALTERNATING"``, or
            ``"FIXED_OPPONENT"``.
        train_every_n_episodes: Number of negotiation episodes between
            on-the-fly ``trainer.update()`` calls.
        on_the_fly: If True, train while the simulation runs; if False, only
            offline training (from history files) is performed.
        schema_version: Integer version recorded in agent/history signatures
            for compatibility checking.
        registry_path: Directory path where trained agent artifacts are saved/loaded.
        history_path: Directory path where one-day negotiation payload
            history files are written/read.

    Example:
        >>> TrainingConfig(training_concurrency="SIMULTANEOUS").training_concurrency
        'SIMULTANEOUS'
    """

    training_concurrency: str = "SIMULTANEOUS"  # SIMULTANEOUS | ALTERNATING | FIXED_OPPONENT
    train_every_n_episodes: int = 5
    on_the_fly: bool = True
    schema_version: int = 1
    registry_path: str = "data/registry"
    history_path: str = "data/history"


# ---------------------------------------------------------------------------
# R. Map / OSM
# ---------------------------------------------------------------------------


@dataclass
class OsmConfig:
    """OSM road network retrieval and caching configuration (see ``01-05``).

    Attributes:
        osm_region: Region query string passed to ``osmnx`` graph retrieval
            (e.g. a place name or bounding box descriptor).
        network_type: ``osmnx`` network type filter (e.g. ``"drive"``).
        use_cache: If True, cache the precomputed distance matrix between
            fixed points (stations/areas/emergency station) to disk.
        cache_path: File path used to store/load the cached distance matrix.

    Example:
        >>> OsmConfig(osm_region="Jakarta, Indonesia").osm_region
        'Jakarta, Indonesia'
    """

    osm_region: str = "Jakarta, Indonesia"
    network_type: str = "drive"
    use_cache: bool = True
    cache_path: str = "data/history/distance_matrix_cache.json"


# ---------------------------------------------------------------------------
# S. Dashboard
# ---------------------------------------------------------------------------


@dataclass
class DashboardConfig:
    """Dashboard backend behavior configuration (see ``07-01``/``07-02``).

    Attributes:
        websocket_throttle_s: Minimum interval, in seconds, between
            consecutive WebSocket snapshot pushes.
        tick_delay_s: Real-time delay, in seconds, inserted between
            simulated ticks when running the background simulation loop
            (controls perceived playback speed). The default ``5.0`` matches
            the fixed 1:60 world:real time scale for the default 5-minute tick
            (5 simulated minutes -> 5 real seconds), so one simulated day takes
            ~24 real minutes; override for fast-forward / slow-motion.
            :func:`simulation.config.default.default_config` derives this from
            the actual ``tick_resolution_min`` via
            :func:`simulation.engine.clock.real_seconds_per_tick`.
        map_renderer: Preferred map rendering backend, e.g. ``"leaflet"``,
            ``"maplibre"``, or ``"deckgl"``.

    Example:
        >>> DashboardConfig(websocket_throttle_s=0.1).websocket_throttle_s
        0.1
    """

    websocket_throttle_s: float = 0.1
    tick_delay_s: float = 5.0
    map_renderer: str = "leaflet"


# ---------------------------------------------------------------------------
# T. Reporting / runs
# ---------------------------------------------------------------------------


@dataclass
class ReportingConfig:
    """Reporting, run-persistence, and metric-plot configuration (group T).

    Controls the "runs" persistence layer (real-time per-period negotiation
    JSON logs, no-negotiation records, and per-day / per-simulation PNG metric
    charts written under ``runs_root/<sim_id>/day_<index>/``). Every tunable —
    the runs root path, per-output enable flags, figure DPI/size, the CRI*
    ``theta``, the minimum number of points required to fit the tanh model, the
    JSON indent, and the in-memory period-payload caps — is a configuration
    field here so that no magic numbers are scattered across the metrics layer
    (see ``08-konfigurasi/08-01-daftar-konfigurasi.md`` and INV-4).

    This configuration only governs how results are *captured, stored, and
    visualized*; it never changes any reward/loss/ufun formula (the reporting
    layer is purely read-only with respect to the world and agents).

    Attributes:
        enabled: Master switch for the whole reporting/runs layer. When False,
            no run folder, JSON, or PNG is produced.
        runs_root: Root folder under which each run's ``<sim_id>`` directory is
            created (e.g. ``runs_root/<sim_id>/day_0/...``).
        offline_runs_root: Root folder under which each **offline training**
            session writes its own ``<offline_id>`` directory (loss PNGs +
            ``losses.json`` + ``meta.json``). Used by ``offline_train.py`` so
            offline-training loss curves are tracked separately from live runs.
        logs_root: Root folder for the JSON-Lines log streams. The simulation
            trace is written under ``logs_root/simulation/<sim_id>.jsonl``, the
            on-the-fly training log under ``logs_root/online_training/<sim_id>.jsonl``,
            and the offline training log under
            ``logs_root/offline_training/<offline_id>.jsonl``.
        write_period_json: If True, write one JSON file per negotiation period
            (including no-negotiation records).
        render_day_png: If True, render per-day metric PNG charts into each
            ``day_<index>`` folder.
        render_sim_png: If True, render per-simulation (cumulative) metric PNG
            charts into the run folder.
        png_dpi: Dots-per-inch resolution used when saving PNG figures.
        figure_width_in: Figure width in inches for rendered charts.
        figure_height_in: Figure height in inches for rendered charts.
        cri_star_theta: ``theta`` threshold used by the CRI* concession metric
            (fraction of the tanh concession profile considered "active").
        min_points_for_tanh_fit: Minimum number of propose-utility points
            required before attempting to fit the tanh concession model.
        json_indent: Indentation (number of spaces) used when serializing the
            period JSON files.
        keep_period_payloads_in_memory: If True, keep per-period payloads in
            memory for regenerating charts; set False to bound memory use.
        max_periods_in_memory: Upper bound on the number of period payloads
            kept in memory (performance/memory cap when retaining payloads).

    Example:
        >>> ReportingConfig().runs_root
        'data/runs'
        >>> ReportingConfig(png_dpi=200).png_dpi
        200
    """

    enabled: bool = True
    runs_root: str = "data/runs"
    offline_runs_root: str = "data/offline_runs"
    logs_root: str = "data/logs"
    write_period_json: bool = True
    render_day_png: bool = True
    render_sim_png: bool = True
    png_dpi: int = 130
    figure_width_in: float = 14.0
    figure_height_in: float = 8.0
    cri_star_theta: float = 0.1
    min_points_for_tanh_fit: int = 4
    json_indent: int = 2
    keep_period_payloads_in_memory: bool = True
    max_periods_in_memory: int = 5000


# ---------------------------------------------------------------------------
# Top-level SimulationConfig
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """One immutable configuration tree describing an entire simulation run.

    This is the single source of truth referenced throughout the codebase
    (Kontrak/Gotcha G14: configuration never changes mid-run). It aggregates
    every group of parameters listed in
    ``.claude/project_instruction/08-konfigurasi/08-01-daftar-konfigurasi.md``:
    time (A), negotiation (B), taxi spec (C), battery thresholds (D),
    port/station (E), emergency station (F), marking area/order (G), fare (H),
    future slot (I), dead-battery handling (J), ufun weights/REF_*/shape
    (K, L, M), app-user booking & fallback (N), state/feature/visibility (O),
    agent/trainer/mode (P), training & registry (Q), OSM/map (R),
    dashboard (S), and reporting / runs (T).

    Attributes:
        time: Time, tick, and negotiation-interval configuration (group A).
        negotiation: Negotiation caps and price levels (group B).
        taxi_spec: Uniform taxi specification (group C).
        battery: Battery zone thresholds (group D).
        port_station: Charging stations and ports (group E).
        emergency: Emergency station configuration (group F).
        marking_area_order: Marking areas and order generation (group G).
        fare: Passenger fare configuration (group H).
        future_slot: Future slot booking and departure-gate configuration (group I).
        dead_battery: Dead-battery handling policy (group J).
        ufun: Utility function weights, references, and shape (groups K, L, M).
        app_user_booking: App-user booking and fallback configuration (group N).
        state_feature: feature_spec, visibility, and encoder dims (group O).
        agent: Per-side agent/trainer/mode configuration (group P).
        training: Training orchestration and registry paths (group Q).
        osm: OSM region and routing cache configuration (group R).
        dashboard: Dashboard backend behavior configuration (group S).
        reporting: Reporting/runs persistence and metric-plot configuration
            (group T).
        seed: Optional global RNG seed for reproducible world assembly + order
            matching. ``None`` -> non-deterministic. An explicit seed passed to
            ``main.py``/``SimSession`` takes precedence over this field, so one
            config scenario can be run under several seeds.

    Example:
        >>> cfg = SimulationConfig()
        >>> cfg.time.n_days
        1
        >>> cfg.agent.fleet.mode
        'PADDED'
        >>> SimulationConfig(seed=42).seed
        42
    """

    time: TimeConfig = field(default_factory=TimeConfig)
    negotiation: NegotiationConfig = field(default_factory=NegotiationConfig)
    taxi_spec: TaxiSpecConfig = field(default_factory=TaxiSpecConfig)
    battery: BatteryThresholdConfig = field(default_factory=BatteryThresholdConfig)
    port_station: PortAndStationConfig = field(default_factory=PortAndStationConfig)
    emergency: EmergencyStationConfig = field(default_factory=EmergencyStationConfig)
    marking_area_order: MarkingAreaAndOrderConfig = field(default_factory=MarkingAreaAndOrderConfig)
    fare: FareConfig = field(default_factory=FareConfig)
    future_slot: FutureSlotConfig = field(default_factory=FutureSlotConfig)
    dead_battery: DeadBatteryConfig = field(default_factory=DeadBatteryConfig)
    ufun: UfunConfig = field(default_factory=UfunConfig)
    app_user_booking: AppUserBookingConfig = field(default_factory=AppUserBookingConfig)
    state_feature: StateFeatureConfig = field(default_factory=StateFeatureConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    osm: OsmConfig = field(default_factory=OsmConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    seed: Optional[int] = None
