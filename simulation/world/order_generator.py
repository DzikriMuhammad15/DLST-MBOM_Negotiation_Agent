"""Per-hour passenger order generation (see ``01-03``).

At the start of every simulated hour the world is refreshed: any order still
``PENDING`` from the previous hour **expires** (recorded as a metric), then a
fresh batch of orders is spawned per marking area. The per-area count is a
random integer in ``[order_min, order_max]`` biased upward by the hour's
``time_weight`` — the single source of truth that both drives order volume and
acts as the fleet ufun's "right timing" factor (see ``01-03`` / ``04-02``).

Each order's ``pickup_point`` is a random point inside the area; its
``destination`` is a random point on the map with a **soft tendency** to be
reachable by a taxi departing from the area at SoC ``B_area`` and arriving with
SoC still ``>= critical_soc`` (not a hard constraint, see ``01-03``).

A separate **hard bound** is also applied: when ``order_reference_point``
("point 0") and a positive ``order_destination_radius_km`` are configured, every
destination is clamped onto that disk, so an order destination never leaves the
radius around the reference point.

The generator takes an injectable :class:`random.Random` so the engine can make
runs reproducible. It depends only on :mod:`simulation.config` and
:mod:`simulation.world.entities`; it never imports ``negotiation``.

Example
-------
>>> import random
>>> from simulation.config.default import default_config
>>> from simulation.world.routing import Router
>>> from simulation.world.world_state import WorldState
>>> from simulation.world.entities import MarkingArea
>>> cfg = default_config()
>>> world = WorldState(config=cfg, router=Router(speed_kmh=30.0))
>>> world.areas = {"a0": MarkingArea(id="a0", center=(-6.2, 106.8))}
>>> gen = OrderGenerator(world, rng=random.Random(0))
>>> expired = gen.refresh_hour()
>>> all(o.area_id == "a0" for o in world.areas["a0"].current_orders)
True
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from simulation.world.entities import MarkingArea, Order, OrderState
from simulation.world.world_state import WorldState

# Approximate km per degree of latitude (used to size random spreads in
# coordinate space). Longitude is scaled by cos(latitude).
_KM_PER_DEG_LAT: float = 111.0
# Radius (km) of the disk around an area center used to sample pickup points.
_PICKUP_RADIUS_KM: float = 0.5


class OrderGenerator:
    """Spawns and expires passenger orders on hour boundaries (see ``01-03``).

    Attributes:
        world: The :class:`WorldState` whose marking areas receive orders.
        rng: The random source used for counts, pickups, and destinations.

    Example:
        >>> import random
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.world_state import WorldState
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> OrderGenerator(w, rng=random.Random(1)).world is w
        True
    """

    def __init__(self, world: WorldState, rng: Optional[random.Random] = None) -> None:
        """Initialize the generator over a world with an optional RNG.

        Args:
            world: The :class:`WorldState` to populate.
            rng: Random source; a fresh :class:`random.Random` is created if
                omitted.

        Example:
            >>> import random
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> g = OrderGenerator(w, rng=random.Random(0))
            >>> g._next_order_index
            0
        """

        self.world = world
        self.rng = rng if rng is not None else random.Random()
        self._next_order_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh_hour(self) -> List[Order]:
        """Expire stale PENDING orders, then spawn the new hour's orders.

        Call this once at every hour rollover (see the tick loop in ``01-02``).

        Returns:
            The list of orders that EXPIRED on this refresh (for metrics).

        Example:
            >>> import random
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> from simulation.world.entities import MarkingArea
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> w.areas = {"a0": MarkingArea(id="a0", center=(-6.2, 106.8))}
            >>> _ = OrderGenerator(w, rng=random.Random(0)).refresh_hour()
            >>> isinstance(w.areas["a0"].current_orders, list)
            True
        """

        expired = self._expire_pending_orders()
        self._spawn_orders()
        return expired

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------
    def _expire_pending_orders(self) -> List[Order]:
        """Mark every still-PENDING order EXPIRED and clear each area's list.

        Orders that were CLAIMED / IN_PROGRESS / COMPLETED are not present in
        ``current_orders`` of the *new* hour: ``current_orders`` only ever holds
        the orders generated for the hour just ended, so we expire the PENDING
        remainder and reset the list for fresh spawning.

        Returns:
            The orders transitioned to :attr:`OrderState.EXPIRED`.

        Example:
            >>> import random
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> from simulation.world.entities import MarkingArea, Order
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> area = MarkingArea(id="a0", center=(-6.2, 106.8))
            >>> area.current_orders = [Order("o0", "a0", (-6.2, 106.8),
            ...                              (-6.3, 106.9), 8)]
            >>> w.areas = {"a0": area}
            >>> expired = OrderGenerator(w, rng=random.Random(0))._expire_pending_orders()
            >>> expired[0].state.name
            'EXPIRED'
        """

        expired: List[Order] = []
        for area in self.world.areas.values():
            for order in area.current_orders:
                if order.state is OrderState.PENDING:
                    order.state = OrderState.EXPIRED
                    expired.append(order)
            area.current_orders = []
        return expired

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------
    def _spawn_orders(self) -> None:
        """Spawn this hour's orders for every area, weighted by ``time_weight``.

        The per-area count is ``order_min`` plus a binomial-like share of the
        ``[order_min, order_max]`` span scaled by the clamped hour weight, then
        jittered randomly — high-weight hours skew toward ``order_max`` while
        remaining random (see ``01-03``).

        Example:
            >>> import random
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> from simulation.world.entities import MarkingArea
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> w.areas = {"a0": MarkingArea(id="a0", center=(-6.2, 106.8))}
            >>> OrderGenerator(w, rng=random.Random(0))._spawn_orders()
            >>> isinstance(w.areas["a0"].current_orders, list)
            True
        """

        cfg = self.world.config
        order_cfg = cfg.marking_area_order
        hour = self.world.hour()
        weight = float(order_cfg.time_weight.get(hour, 1.0))
        # Clamp weight to [0, 1] for biasing the count toward order_max.
        weight_clamped = max(0.0, min(1.0, weight))

        for area in self.world.areas.values():
            count = self._sample_count(order_cfg.order_min, order_cfg.order_max, weight_clamped)
            for _ in range(count):
                order = self._make_order(area, hour)
                area.current_orders.append(order)

    def _sample_count(self, order_min: int, order_max: int, weight_clamped: float) -> int:
        """Sample an order count in ``[order_min, order_max]`` biased by weight.

        Args:
            order_min: Minimum orders per area per hour.
            order_max: Maximum orders per area per hour.
            weight_clamped: Hour weight clamped to ``[0, 1]``; higher means the
                count skews toward ``order_max``.

        Returns:
            An integer in ``[order_min, order_max]``.

        Example:
            >>> import random
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> g = OrderGenerator(w, rng=random.Random(0))
            >>> g._sample_count(2, 2, 1.0)
            2
        """

        if order_max <= order_min:
            return order_min
        span = order_max - order_min
        # Random uniform draw, then nudged upward proportionally to the weight.
        base = self.rng.random()
        biased = base * (1.0 - weight_clamped) + weight_clamped
        extra = int(round(biased * span))
        return order_min + max(0, min(span, extra))

    def _make_order(self, area: MarkingArea, hour: int) -> Order:
        """Build a single PENDING order for an area at the given hour.

        Args:
            area: The marking area the order belongs to.
            hour: Spawn hour-of-day.

        Returns:
            A new PENDING :class:`Order` with a random pickup inside the area
            and a soft-reachable random destination.

        Example:
            >>> import random
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> from simulation.world.entities import MarkingArea
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> area = MarkingArea(id="a0", center=(-6.2, 106.8))
            >>> g = OrderGenerator(w, rng=random.Random(0))
            >>> g._make_order(area, 8).state.name
            'PENDING'
        """

        pickup = self._random_point_near(area.center, _PICKUP_RADIUS_KM)
        destination = self._sample_destination(pickup)
        order_id = f"order_{self._next_order_index}"
        self._next_order_index += 1
        return Order(
            id=order_id,
            area_id=area.id,
            pickup_point=pickup,
            destination=destination,
            spawn_hour=hour,
            state=OrderState.PENDING,
            assigned_taxi=None,
        )

    def _sample_destination(self, pickup: tuple) -> tuple:
        """Sample a destination with a soft tendency to be reachable.

        Reachability tendency (see ``01-03``): a taxi departing an area at SoC
        ``B_area`` should usually finish the trip and arrive with SoC still
        ``>= critical_soc``. The energy budget for the productive leg is
        ``(B_area - critical_soc) * battery_capacity`` kWh; dividing by
        consumption gives a soft maximum productive distance. The destination is
        drawn within that radius of the pickup. It remains a tendency, so trips
        may occasionally cross ``B_charge`` — that is allowed.

        Args:
            pickup: The ``(lat, lng)`` pickup point.

        Returns:
            A ``(lat, lng)`` destination point.

        Example:
            >>> import random
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> g = OrderGenerator(w, rng=random.Random(0))
            >>> dest = g._sample_destination((-6.2, 106.8))
            >>> len(dest)
            2
        """

        cfg = self.world.config
        battery = cfg.battery
        spec = cfg.taxi_spec
        usable_soc = max(0.0, battery.B_area - battery.critical_soc)
        usable_kwh = usable_soc * spec.battery_capacity_kwh
        if spec.consumption_kwh_per_km > 0.0:
            soft_max_km = usable_kwh / spec.consumption_kwh_per_km
        else:
            soft_max_km = _PICKUP_RADIUS_KM
        # Soft tendency: most destinations within the budget radius, occasional
        # overshoot allowed by sampling beyond it with low probability.
        if soft_max_km <= 0.0:
            soft_max_km = _PICKUP_RADIUS_KM
        radius_km = self.rng.random() * soft_max_km
        destination = self._random_point_near(pickup, radius_km)
        # Hard bound: clamp inside the reference-point radius so a destination
        # never leaves it (see ``order_reference_point`` / ``order_destination_radius_km``).
        return self._clamp_to_reference(destination)

    def _clamp_to_reference(self, point: tuple) -> tuple:
        """Project ``point`` back onto the order reference disk if it escapes it.

        If ``order_reference_point`` and a positive ``order_destination_radius_km``
        are configured, any destination farther than the radius from "point 0" is
        moved radially inward onto the circle boundary (planar km approximation,
        consistent with :meth:`_random_point_near`). When unconfigured the point is
        returned unchanged (legacy behaviour).

        Args:
            point: The ``(lat, lng)`` destination to bound.

        Returns:
            A ``(lat, lng)`` point within ``order_destination_radius_km`` of the
            reference point.

        Example:
            >>> import random
            >>> from dataclasses import replace
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> base = default_config()
            >>> mo = replace(base.marking_area_order,
            ...              order_reference_point=(0.0, 0.0),
            ...              order_destination_radius_km=1.0)
            >>> cfg = replace(base, marking_area_order=mo)
            >>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
            >>> g = OrderGenerator(w, rng=random.Random(0))
            >>> p = g._clamp_to_reference((5.0, 0.0))   # ~555 km north -> clamped
            >>> round(p[0] * 111.0, 3)                  # ~1 km in latitude
            1.0
        """

        order_cfg = self.world.config.marking_area_order
        ref = order_cfg.order_reference_point
        radius_km = order_cfg.order_destination_radius_km
        if ref is None or radius_km is None or radius_km <= 0.0:
            return point

        ref_lat, ref_lng = ref
        cos_lat = math.cos(math.radians(ref_lat))
        scale_lng = _KM_PER_DEG_LAT * (cos_lat if abs(cos_lat) > 1e-9 else 1e-9)
        d_lat_km = (point[0] - ref_lat) * _KM_PER_DEG_LAT
        d_lng_km = (point[1] - ref_lng) * scale_lng
        dist_km = math.hypot(d_lat_km, d_lng_km)
        if dist_km <= radius_km:
            return point
        # Project radially onto the circle boundary.
        factor = radius_km / dist_km
        new_lat = ref_lat + (d_lat_km * factor) / _KM_PER_DEG_LAT
        new_lng = ref_lng + (d_lng_km * factor) / scale_lng
        return (new_lat, new_lng)

    def _random_point_near(self, center: tuple, radius_km: float) -> tuple:
        """Sample a uniform random point within ``radius_km`` of ``center``.

        Args:
            center: The ``(lat, lng)`` center point.
            radius_km: Maximum distance from the center, in km (>= 0).

        Returns:
            A ``(lat, lng)`` point within the disk.

        Example:
            >>> import random
            >>> from simulation.config.default import default_config
            >>> from simulation.world.routing import Router
            >>> from simulation.world.world_state import WorldState
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> g = OrderGenerator(w, rng=random.Random(0))
            >>> p = g._random_point_near((-6.2, 106.8), 0.0)
            >>> p
            (-6.2, 106.8)
        """

        lat, lng = center
        if radius_km <= 0.0:
            return (lat, lng)
        # Uniform sampling within a disk: radius ~ sqrt(U), angle ~ U.
        angle = self.rng.random() * 2.0 * math.pi
        sampled_km = radius_km * math.sqrt(self.rng.random())
        d_lat = (sampled_km / _KM_PER_DEG_LAT) * math.cos(angle)
        cos_lat = math.cos(math.radians(lat))
        denom = _KM_PER_DEG_LAT * (cos_lat if abs(cos_lat) > 1e-9 else 1e-9)
        d_lng = (sampled_km / denom) * math.sin(angle)
        return (lat + d_lat, lng + d_lng)
