"""Standalone simulation clock (time scale, tick advance, freeze/resume).

This module implements :class:`Clock`, the single authority for *simulated*
world time. Per
``.claude/project_instruction/01-dunia-simulasi/01-02-waktu-dan-tick.md``:

- The world:real time scale is fixed at **1:60** (1 simulated minute = 1 real
  second). This is a *structural constant* of the simulation, not a
  configurable field, and is therefore exposed here as the module-level
  constant :data:`WORLD_TO_REAL_TIME_SCALE` rather than read from config.
- One tick advances simulated time by ``tick_resolution_min`` simulated
  minutes (a configured value, see :class:`simulation.config.config.TimeConfig`).
- While a negotiation runs the world is **frozen**: simulated time does not
  advance (negotiation happens "outside of time", instantaneous from the
  world's point of view). :meth:`Clock.freeze` stops advance,
  :meth:`Clock.resume` re-enables it.
- The run lasts ``n_days`` simulated days; :attr:`Clock.finished` reports when
  that horizon is reached.

The clock measures elapsed simulated minutes in :attr:`Clock.sim_time_min`,
matching the convention used by :class:`simulation.world.world_state.WorldState`
(which derives ``hour``/``day``/``current_grid`` from the same quantity). The
scheduler is expected to copy :attr:`Clock.sim_time_min` onto the world each
tick so both views stay consistent.

This module depends only on :mod:`simulation.config`. It does not import
``negotiation`` or ``simulation.world`` (it is a leaf component, buildable
early; see ``91-PLAYBOOK-IMPLEMENTASI.md`` Tahap D item 13).

Example:
    >>> from simulation.config.config import TimeConfig
    >>> clock = Clock(TimeConfig(n_days=1, tick_resolution_min=30, interval_min=30))
    >>> clock.hour, clock.day
    (0, 0)
    >>> clock.tick()
    30.0
    >>> clock.hour
    0
    >>> clock.tick()
    60.0
    >>> clock.hour
    1
"""

from __future__ import annotations

from simulation.config.config import TimeConfig

# ---------------------------------------------------------------------------
# Structural constants (NOT configurable — see 01-02).
# ---------------------------------------------------------------------------

#: Fixed world:real time scale. 1 simulated minute elapses per real second,
#: i.e. simulated time runs 60x faster than real time. Structural constant of
#: the simulation, never a config field (01-02 "Skala waktu").
WORLD_TO_REAL_TIME_SCALE: int = 60

#: Number of seconds in one minute of simulated time.
_SECONDS_PER_MINUTE: int = 60

#: Number of minutes in one hour of simulated time.
_MINUTES_PER_HOUR: int = 60

#: Number of hours in one simulated day.
_HOURS_PER_DAY: int = 24

#: Number of simulated minutes in one simulated day (24 * 60).
_MINUTES_PER_DAY: int = _HOURS_PER_DAY * _MINUTES_PER_HOUR


class Clock:
    """Authoritative source of simulated world time with freeze/resume support.

    The clock tracks elapsed simulated minutes (:attr:`sim_time_min`) and
    advances it by ``tick_resolution_min`` on every :meth:`tick`, except while
    frozen. Hour-of-day, day index, and run completion are all derived from the
    elapsed minutes. The world:real scale (1:60) is fixed and exposed via the
    module constant :data:`WORLD_TO_REAL_TIME_SCALE`; it does not affect the
    arithmetic here (which is entirely in simulated minutes) but documents the
    rate at which simulated minutes should map to real wall-clock seconds for
    playback.

    Attributes:
        time_config: The :class:`~simulation.config.config.TimeConfig` whose
            ``tick_resolution_min`` sets the advance per tick and whose
            ``n_days`` sets the run horizon. Held by reference and never
            mutated (Gotcha G14: config is immutable mid-run).
        sim_time_min: Elapsed simulated time in minutes since run start.
        tick_count: Number of completed (non-frozen) ticks since run start.
        frozen: Whether the world is currently frozen (negotiation in progress).

    Example:
        >>> from simulation.config.config import TimeConfig
        >>> c = Clock(TimeConfig(n_days=2, tick_resolution_min=5, interval_min=30))
        >>> c.freeze()
        >>> c.tick()          # no advance while frozen
        0.0
        >>> c.resume()
        >>> c.tick()
        5.0
    """

    def __init__(self, time_config: TimeConfig) -> None:
        """Initialize the clock at simulated time zero, unfrozen.

        Args:
            time_config: Time configuration providing ``tick_resolution_min``
                (simulated minutes advanced per tick) and ``n_days`` (run
                horizon). The ``interval_min`` field is consulted by the
                scheduler, not by the clock itself.

        Example:
            >>> from simulation.config.config import TimeConfig
            >>> c = Clock(TimeConfig(n_days=1, tick_resolution_min=5, interval_min=30))
            >>> c.sim_time_min
            0.0
            >>> c.frozen
            False
        """
        self.time_config: TimeConfig = time_config
        self.sim_time_min: float = 0.0
        self.tick_count: int = 0
        self.frozen: bool = False

    # ------------------------------------------------------------------
    # Tick advance
    # ------------------------------------------------------------------

    def tick(self) -> float:
        """Advance simulated time by one tick, unless the world is frozen.

        While frozen (during a negotiation) simulated time does not move:
        the call is a no-op and the unchanged :attr:`sim_time_min` is returned
        (01-02 "FREEZE"). Otherwise simulated time advances by
        ``time_config.tick_resolution_min`` minutes and :attr:`tick_count` is
        incremented.

        Returns:
            The simulated time in minutes (:attr:`sim_time_min`) after the
            (possibly skipped) advance.

        Example:
            >>> from simulation.config.config import TimeConfig
            >>> c = Clock(TimeConfig(n_days=1, tick_resolution_min=15, interval_min=30))
            >>> c.tick()
            15.0
            >>> c.tick()
            30.0
        """
        if self.frozen:
            return self.sim_time_min
        self.sim_time_min += float(self.time_config.tick_resolution_min)
        self.tick_count += 1
        return self.sim_time_min

    # ------------------------------------------------------------------
    # Freeze / resume (negotiation pause)
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze the world so that subsequent ticks do not advance time.

        Called by the scheduler immediately before running one negotiation so
        the negotiation operates on a single consistent snapshot
        (01-02 "FREEZE"). Idempotent: freezing an already-frozen clock has no
        additional effect.

        Example:
            >>> from simulation.config.config import TimeConfig
            >>> c = Clock(TimeConfig(n_days=1, tick_resolution_min=5, interval_min=30))
            >>> c.freeze()
            >>> c.frozen
            True
        """
        self.frozen = True

    def resume(self) -> None:
        """Resume the world so that subsequent ticks advance time again.

        Called by the scheduler after a negotiation has been applied
        (apply/no-deal) to let simulated time continue (01-02 "FREEZE").
        Idempotent: resuming an already-running clock has no additional effect.

        Example:
            >>> from simulation.config.config import TimeConfig
            >>> c = Clock(TimeConfig(n_days=1, tick_resolution_min=5, interval_min=30))
            >>> c.freeze()
            >>> c.resume()
            >>> c.frozen
            False
        """
        self.frozen = False

    # ------------------------------------------------------------------
    # Derived calendar quantities
    # ------------------------------------------------------------------

    @property
    def hour(self) -> int:
        """Return the current hour-of-day in ``0..23``.

        Derived from :attr:`sim_time_min` exactly as
        :meth:`simulation.world.world_state.WorldState.hour` does, so the
        clock and world agree. Used (via :meth:`hour_changed_since`) to trigger
        per-hour order refresh.

        Returns:
            The hour-of-day, ``0..23``.

        Example:
            >>> from simulation.config.config import TimeConfig
            >>> c = Clock(TimeConfig(n_days=1, tick_resolution_min=60, interval_min=60))
            >>> c.tick()
            60.0
            >>> c.hour
            1
        """
        return int(self.sim_time_min // _MINUTES_PER_HOUR) % _HOURS_PER_DAY

    @property
    def day(self) -> int:
        """Return the current day index (0-based).

        Returns:
            The number of whole simulated days elapsed since run start.

        Example:
            >>> from simulation.config.config import TimeConfig
            >>> c = Clock(TimeConfig(n_days=2, tick_resolution_min=60, interval_min=60))
            >>> c.sim_time_min = 25 * 60.0
            >>> c.day
            1
        """
        return int(self.sim_time_min // _MINUTES_PER_DAY)

    @property
    def finished(self) -> bool:
        """Return whether the configured run horizon (``n_days``) is reached.

        The run spans ``n_days`` simulated days. It is finished once elapsed
        simulated time reaches (or exceeds) ``n_days`` full days
        (01-02 "Hari & durasi run").

        Returns:
            ``True`` when :attr:`sim_time_min` has reached ``n_days`` days.

        Example:
            >>> from simulation.config.config import TimeConfig
            >>> c = Clock(TimeConfig(n_days=1, tick_resolution_min=60, interval_min=60))
            >>> c.finished
            False
            >>> c.sim_time_min = 24 * 60.0
            >>> c.finished
            True
        """
        return self.sim_time_min >= self.time_config.n_days * _MINUTES_PER_DAY

    # ------------------------------------------------------------------
    # Hour-rollover detection
    # ------------------------------------------------------------------

    def hour_changed_since(self, previous_hour: int) -> bool:
        """Return whether the hour-of-day differs from ``previous_hour``.

        The scheduler records the hour at the end of each tick and passes it
        here on the next tick; a ``True`` result means the simulated clock
        crossed into a new hour, which is the trigger for refreshing orders
        (01-02 loop "if hour_changed: refresh_orders()", see also 01-03).

        Args:
            previous_hour: The hour-of-day (``0..23``) observed on the previous
                tick (or run start).

        Returns:
            ``True`` if :attr:`hour` differs from ``previous_hour``.

        Example:
            >>> from simulation.config.config import TimeConfig
            >>> c = Clock(TimeConfig(n_days=1, tick_resolution_min=60, interval_min=60))
            >>> prev = c.hour
            >>> c.tick()
            60.0
            >>> c.hour_changed_since(prev)
            True
            >>> c.hour_changed_since(c.hour)
            False
        """
        return self.hour != previous_hour


def real_seconds_per_tick(time_config: TimeConfig) -> float:
    """Return the real-world seconds one tick should take at the fixed 1:60 scale.

    Derives the wall-clock pacing of one engine tick from the structural
    :data:`WORLD_TO_REAL_TIME_SCALE` (60: simulated time runs 60x faster than
    real time, i.e. 1 simulated minute elapses per real second). One tick
    advances ``tick_resolution_min`` simulated minutes, so at true 1:60 it must
    occupy ``tick_resolution_min`` real seconds. Used by the dashboard auto-run
    loop to make 1 simulated day take ~24 real minutes (1440 simulated minutes
    / 60), honouring the scale rather than fast-forwarding.

    Args:
        time_config: Time configuration providing ``tick_resolution_min``.

    Returns:
        Real seconds one tick should occupy at the 1:60 scale.

    Example:
        >>> from simulation.config.config import TimeConfig
        >>> real_seconds_per_tick(TimeConfig(n_days=1, tick_resolution_min=5,
        ...                                  interval_min=30))
        5.0
    """
    sim_seconds_per_tick = time_config.tick_resolution_min * _SECONDS_PER_MINUTE
    return float(sim_seconds_per_tick) / float(WORLD_TO_REAL_TIME_SCALE)
