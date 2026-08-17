"""Distance and travel-time routing for the simulation world (see ``01-05``).

The :class:`Router` answers two kinds of distance query at two precisions:

1. **Actual movement & unproductive-km metrics** -> always precise
   point-to-point distance (:meth:`Router.distance`, cached).
2. **Ufun distances** (taxi->port, port->area) -> area granularity governed by
   ``ufun_distance_mode`` (:meth:`Router.ufun_distance`):
   ``HYBRID`` (matrix inside area, on-demand precise outside) or
   ``SNAP_NEAREST_AREA`` (Euclidean snap to nearest area, then matrix).

Fixed points (stations, area centers, the emergency station) are registered
once and their pairwise distances precomputed into a matrix
(:meth:`Router.precompute_matrix`). Distances involving random points (taxi
positions, pickups, destinations) are computed on demand and cached.

The underlying metric is a great-circle (haversine) Euclidean proxy. The class
is **OSM-swappable**: the only geometry primitive is :meth:`Router._raw_distance`,
which can be replaced by an ``osmnx`` + ``networkx`` shortest-path lookup
without changing any caller. ``register_fixed`` / ``precompute_matrix`` are the
hooks where an OSM graph and a fixed-point node mapping would be wired in.

Example
-------
>>> r = Router(speed_kmh=30.0)
>>> r.register_fixed("a0", (-6.20, 106.80))
>>> r.register_fixed("a1", (-6.20, 106.81))
>>> r.precompute_matrix()
>>> round(r.fixed_distance("a0", "a1"), 2) > 0
True
>>> r.travel_time_min(0.0)
0.0
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

Coordinate = Tuple[float, float]

# Mean Earth radius in km, used by the haversine great-circle proxy.
_EARTH_RADIUS_KM: float = 6371.0088

# Minutes per hour, for travel-time conversion.
_MINUTES_PER_HOUR: float = 60.0


class Router:
    """Cached distance / travel-time provider over fixed and random points.

    Attributes:
        speed_kmh: Uniform travel speed used to convert km to minutes.

    Example:
        >>> r = Router(speed_kmh=60.0)
        >>> r.travel_time_min(30.0)
        30.0
    """

    def __init__(self, speed_kmh: float) -> None:
        """Initialize a router with a uniform travel speed.

        Args:
            speed_kmh: Uniform speed (km/h) shared by all taxis (> 0).

        Example:
            >>> Router(speed_kmh=30.0).speed_kmh
            30.0
        """

        self.speed_kmh: float = speed_kmh
        self._fixed_points: Dict[str, Coordinate] = {}
        self._matrix: Dict[Tuple[str, str], float] = {}
        self._cache: Dict[Tuple[Coordinate, Coordinate], float] = {}

    # ------------------------------------------------------------------
    # Geometry primitive (OSM-swappable seam)
    # ------------------------------------------------------------------
    def _raw_distance(self, a: Coordinate, b: Coordinate) -> float:
        """Return the proxy road distance (km) between two ``(lat, lng)`` points.

        This is the single geometry primitive of the router. Replacing its body
        with an ``osmnx``/``networkx`` shortest-path lookup upgrades the whole
        router to true road distance without touching any caller.

        Args:
            a: First ``(lat, lng)`` coordinate.
            b: Second ``(lat, lng)`` coordinate.

        Returns:
            Great-circle (haversine) distance in km.

        Example:
            >>> r = Router(speed_kmh=30.0)
            >>> r._raw_distance((-6.2, 106.8), (-6.2, 106.8))
            0.0
        """

        lat1, lng1 = a
        lat2, lng2 = b
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lng2 - lng1)
        sin_phi = math.sin(d_phi / 2.0)
        sin_lambda = math.sin(d_lambda / 2.0)
        h = sin_phi * sin_phi + math.cos(phi1) * math.cos(phi2) * sin_lambda * sin_lambda
        return 2.0 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))

    @staticmethod
    def _euclidean(a: Coordinate, b: Coordinate) -> float:
        """Return a cheap planar distance in degrees, used only for area snapping.

        ``SNAP_NEAREST_AREA`` selects the nearest area by this cheap metric
        (see ``01-05``); it is intentionally not a real distance.

        Args:
            a: First ``(lat, lng)`` coordinate.
            b: Second ``(lat, lng)`` coordinate.

        Returns:
            Planar Euclidean distance in coordinate-degrees.

        Example:
            >>> Router._euclidean((0.0, 0.0), (3.0, 4.0))
            5.0
        """

        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ------------------------------------------------------------------
    # Fixed points & precomputed matrix
    # ------------------------------------------------------------------
    def register_fixed(self, key: str, position: Coordinate) -> None:
        """Register a fixed point (station / area center / emergency station).

        Args:
            key: Unique identifier of the fixed point (e.g. an area or station id).
            position: ``(lat, lng)`` coordinate of the fixed point.

        Example:
            >>> r = Router(speed_kmh=30.0)
            >>> r.register_fixed("a0", (-6.2, 106.8))
            >>> r._fixed_points["a0"]
            (-6.2, 106.8)
        """

        self._fixed_points[key] = position

    def precompute_matrix(self) -> None:
        """Precompute pairwise distances between all registered fixed points.

        Builds the area<->station / area<->area / station<->station matrix once
        (see ``01-05`` "Cache jarak"). Idempotent: safe to call again after more
        fixed points have been registered.

        Example:
            >>> r = Router(speed_kmh=30.0)
            >>> r.register_fixed("x", (0.0, 0.0))
            >>> r.register_fixed("y", (0.0, 1.0))
            >>> r.precompute_matrix()
            >>> r.fixed_distance("x", "y") == r.fixed_distance("y", "x")
            True
        """

        keys = list(self._fixed_points.keys())
        for i, key_a in enumerate(keys):
            for key_b in keys[i:]:
                dist = self._raw_distance(self._fixed_points[key_a], self._fixed_points[key_b])
                self._matrix[(key_a, key_b)] = dist
                self._matrix[(key_b, key_a)] = dist

    def fixed_distance(self, key_a: str, key_b: str) -> float:
        """Return the precomputed distance (km) between two fixed points.

        Falls back to an on-the-fly computation (and stores it) if the matrix
        has not yet been precomputed for the pair.

        Args:
            key_a: Identifier of the first fixed point.
            key_b: Identifier of the second fixed point.

        Returns:
            Distance in km between the two fixed points.

        Raises:
            KeyError: If either key was never registered.

        Example:
            >>> r = Router(speed_kmh=30.0)
            >>> r.register_fixed("x", (0.0, 0.0))
            >>> r.register_fixed("y", (0.0, 0.0))
            >>> r.precompute_matrix()
            >>> r.fixed_distance("x", "y")
            0.0
        """

        if (key_a, key_b) in self._matrix:
            return self._matrix[(key_a, key_b)]
        dist = self._raw_distance(self._fixed_points[key_a], self._fixed_points[key_b])
        self._matrix[(key_a, key_b)] = dist
        self._matrix[(key_b, key_a)] = dist
        return dist

    # ------------------------------------------------------------------
    # Precise point-to-point distance (cached) & travel time
    # ------------------------------------------------------------------
    def distance(self, a: Coordinate, b: Coordinate) -> float:
        """Return the precise road-proxy distance (km) between two points, cached.

        Used for actual movement and unproductive-km accounting (always precise,
        see ``01-05``).

        Args:
            a: Origin ``(lat, lng)`` coordinate.
            b: Destination ``(lat, lng)`` coordinate.

        Returns:
            Distance in km, memoized per ordered pair.

        Example:
            >>> r = Router(speed_kmh=30.0)
            >>> r.distance((-6.2, 106.8), (-6.2, 106.8))
            0.0
        """

        cache_key = (a, b)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        dist = self._raw_distance(a, b)
        self._cache[cache_key] = dist
        return dist

    def travel_time_min(self, distance_km: float) -> float:
        """Convert a distance to travel time in simulated minutes.

        ``time_min = distance_km / speed_kmh * 60``. Because the world:real
        scale is 1:60, one simulated minute is one real second, but all internal
        accounting is in simulated minutes (see ``01-02`` / ``01-05``).

        Args:
            distance_km: Distance to traverse, in km (>= 0).

        Returns:
            Travel time in simulated minutes.

        Example:
            >>> Router(speed_kmh=60.0).travel_time_min(30.0)
            30.0
        """

        if self.speed_kmh <= 0.0:
            return 0.0
        return distance_km / self.speed_kmh * _MINUTES_PER_HOUR

    # ------------------------------------------------------------------
    # Nearest fixed point
    # ------------------------------------------------------------------
    def nearest_fixed(
        self, position: Coordinate, keys: Optional[List[str]] = None
    ) -> Optional[Tuple[str, float]]:
        """Return the nearest registered fixed point to ``position`` (precise).

        Args:
            position: ``(lat, lng)`` query point.
            keys: Optional subset of fixed-point keys to consider; defaults to
                all registered fixed points.

        Returns:
            ``(key, distance_km)`` of the nearest fixed point, or ``None`` if
            there are no candidate fixed points.

        Example:
            >>> r = Router(speed_kmh=30.0)
            >>> r.register_fixed("a", (0.0, 0.0))
            >>> r.register_fixed("b", (10.0, 10.0))
            >>> r.nearest_fixed((0.1, 0.1))[0]
            'a'
        """

        candidate_keys = keys if keys is not None else list(self._fixed_points.keys())
        best: Optional[Tuple[str, float]] = None
        for key in candidate_keys:
            point = self._fixed_points.get(key)
            if point is None:
                continue
            dist = self.distance(position, point)
            if best is None or dist < best[1]:
                best = (key, dist)
        return best

    def nearest_fixed_euclidean(
        self, position: Coordinate, keys: Optional[List[str]] = None
    ) -> Optional[Tuple[str, float]]:
        """Return the nearest fixed point by the cheap Euclidean snap metric.

        Used only by ``SNAP_NEAREST_AREA`` ufun distance (see ``01-05``); the
        returned distance is in coordinate-degrees, not km.

        Args:
            position: ``(lat, lng)`` query point.
            keys: Optional subset of fixed-point keys to consider.

        Returns:
            ``(key, euclidean_degrees)`` of the nearest fixed point, or ``None``.

        Example:
            >>> r = Router(speed_kmh=30.0)
            >>> r.register_fixed("a", (0.0, 0.0))
            >>> r.nearest_fixed_euclidean((0.1, 0.0))[0]
            'a'
        """

        candidate_keys = keys if keys is not None else list(self._fixed_points.keys())
        best: Optional[Tuple[str, float]] = None
        for key in candidate_keys:
            point = self._fixed_points.get(key)
            if point is None:
                continue
            dist = self._euclidean(position, point)
            if best is None or dist < best[1]:
                best = (key, dist)
        return best

    # ------------------------------------------------------------------
    # Ufun distance (area granularity, mode-governed)
    # ------------------------------------------------------------------
    def ufun_distance(
        self,
        taxi_position: Coordinate,
        taxi_area_id: Optional[str],
        fixed_key: str,
        area_keys: List[str],
        mode: str,
    ) -> float:
        """Return the ufun-side distance (km) from a taxi to a fixed point.

        Implements ``ufun_distance_mode`` (see ``01-05`` / ``04-04``):

        - If the taxi is inside an area (``taxi_area_id`` set), read the
          precomputed matrix between that area and ``fixed_key`` (both fixed).
        - ``HYBRID``: an out-of-area taxi gets a precise on-demand route to the
          fixed point.
        - ``SNAP_NEAREST_AREA``: an out-of-area taxi is snapped (Euclidean) to
          the nearest area, then the matrix is read. May under-estimate for
          taxis far from all areas (a chosen approximation, not a bug).

        Args:
            taxi_position: Current ``(lat, lng)`` of the taxi.
            taxi_area_id: Area id the taxi is inside, or ``None`` if out of area.
            fixed_key: Fixed-point key to measure to (e.g. a station/port-station
                id or an area id).
            area_keys: Keys of all marking areas (snap candidates).
            mode: ``"HYBRID"`` or ``"SNAP_NEAREST_AREA"``.

        Returns:
            Distance in km used by the ufun for this taxi -> fixed pair.

        Example:
            >>> r = Router(speed_kmh=30.0)
            >>> r.register_fixed("a0", (0.0, 0.0))
            >>> r.register_fixed("s0", (0.0, 1.0))
            >>> r.precompute_matrix()
            >>> d = r.ufun_distance((5.0, 5.0), "a0", "s0", ["a0"], "HYBRID")
            >>> d == r.fixed_distance("a0", "s0")
            True
        """

        if taxi_area_id is not None:
            return self.fixed_distance(taxi_area_id, fixed_key)

        if mode == "SNAP_NEAREST_AREA":
            nearest = self.nearest_fixed_euclidean(taxi_position, area_keys)
            if nearest is not None:
                return self.fixed_distance(nearest[0], fixed_key)
            # No areas to snap to; fall through to precise route.

        # HYBRID (or SNAP fallback): precise on-demand route from the taxi to
        # the fixed point's coordinate.
        fixed_point = self._fixed_points[fixed_key]
        return self.distance(taxi_position, fixed_point)
