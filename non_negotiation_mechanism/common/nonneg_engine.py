# -*- coding: utf-8 -*-
"""Engine simulasi NON-NEGOSIASI (as_is & booking) — ADDITIVE, tidak mengubah kode as-is.

Me-reuse SELURUH dunia dari ``simulation/`` (tick, taksi, SoC, order, booking app-user,
emergency) via subclass :class:`Scheduler`, dan HANYA mengganti langkah negosiasi dengan
alokasi non-negosiasi:

* **AsIsScheduler** — "siapa cepat dia dapat": taksi menuju port TERDEKAT; bila span
  kedatangannya bebas (tanpa perlu menggeser booking siapa pun) → charging harga base;
  bila terisi → MENUNGGU (aging cycles_waited) dan coba lagi di cadence berikutnya.
* **BookingScheduler** — booking statis tanpa negosiasi: fleet mem-book opsi TERBAIK yang
  bebas-konflik (``shift is None`` — tidak ada displacement) dengan harga base; bila semua
  opsi relevan terblokir booking app-user → GAGAL (coba lagi cadence berikutnya).

Kunci kesetiaan pada dunia yang sama: kandidat, reachability, span kedatangan, dan
feasibility diambil dari :func:`build_action_library` YANG SAMA dengan rezim negosiasi;
bundle ber-``shift`` (butuh menggeser booking) diperlakukan sebagai "terisi" karena kedua
mekanisme ini tidak mengenal displacement. Utilitas uF/uS dievaluasi dengan ufun builder
YANG SAMA → metrik lintas-rezim apple-to-apple.

Log: event ``tick`` bawaan Scheduler + event alokasi (charge_request / charge_wait /
booking_fail / booking_success / charge_start — lihat ../SCHEMA.md) via TraceLogger yang
sama → dapat dibuka main_debug_visualize.py. Metrik → metrices/aggregate_metrics.json
(summary + numeric_flat dari MetricsCollector; blok aggregate dipetakan per SCHEMA).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from simulation.bridge.action_library import Assignment, Bundle, build_action_library
from simulation.bridge.apply import apply_deal, handle_no_deal
from simulation.bridge.emergency import handle_emergency_taxis
from simulation.bridge.snapshot import collect_charge_candidates
from simulation.bridge.ufun_builder import build_ufun_fleet, build_ufun_spklu
from simulation.engine.scheduler import Scheduler
from simulation.metrics.collector import MetricsCollector
from simulation.world.order_generator import OrderGenerator
from simulation.world.world_assembly import build_world_from_config


# ---------------------------------------------------------------------------
# Scheduler non-negosiasi
# ---------------------------------------------------------------------------

class _NonNegScheduler(Scheduler):
    """Base: ganti siklus negosiasi dengan alokasi mekanisme non-negosiasi."""

    MECHANISM = "base"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.alloc_records: List[Dict[str, Any]] = []
        self._interval_min = float(self.config.time.interval_min)
        self._ufun_fleet = None
        self._ufun_spklu = None

    # -- helper ------------------------------------------------------------
    def _single_options(self, taxi: Any) -> List[Bundle]:
        """Bundle satu-taksi untuk ``taxi`` dari action-library builder yang sama."""
        lib = build_action_library(self.world, [taxi], self.config)
        out = []
        for b in lib:
            assigned = [a for a in b.assignments if not a.is_unassigned()]
            if len(assigned) == 1 and assigned[0].taxi_id == taxi.id:
                out.append(b)
        return out

    def _rebase_price(self, bundle: Bundle) -> Bundle:
        """Salin bundle dgn harga = base price port (mekanisme non-negosiasi = harga base)."""
        new = []
        for a in bundle.assignments:
            if a.is_unassigned():
                new.append(a)
                continue
            port = self.world.get_port(a.port_id)
            base = float(getattr(port, "base_price_per_kwh", a.price_per_kwh))
            new.append(Assignment(a.taxi_id, a.port_id, a.slot_start,
                                  a.duration_grids, base, None))
        return Bundle(tuple(new))

    def _dist_km(self, taxi: Any, port_id: str) -> float:
        try:
            port = self.world.get_port(port_id)
            station = self.world.stations.get(getattr(port, "station_id", ""), None)
            loc = getattr(station, "location", None)
            if loc is None:
                return 1e9
            # FIX T4 (2026-07-07): Router tidak punya method `distance_km` —
            # panggilan lama AttributeError & tertelan except di bawah, sehingga
            # SEMUA port berjarak 1e9 dan "port terdekat" jatuh ke urutan hash
            # (arbitrer). API yang benar: Router.distance(a, b) -> km.
            return float(self.world.router.distance(tuple(taxi.position), tuple(loc)))
        except Exception:
            return 1e9

    # -- keputusan per mekanisme (override) ---------------------------------
    def _decide(self, taxi: Any, options: List[Bundle]) -> Tuple[Optional[Bundle], str]:
        raise NotImplementedError

    # -- siklus utama --------------------------------------------------------
    def _run_negotiation_cycle(self) -> None:  # nama dipertahankan (dipanggil step_tick)
        _min = float(self.clock.sim_time_min)
        _day = int(self.clock.day)
        handle_emergency_taxis(self.world, self.config.dead_battery.dead_battery_mode)
        candidates = collect_charge_candidates(self.world)
        if not candidates:
            return
        self.clock.freeze()
        self.in_negotiation = True
        try:
            self._ufun_fleet = build_ufun_fleet(self.world)
            self._ufun_spklu = build_ufun_spklu(self.world)
            for taxi in candidates:
                period = int(self.negotiation_count)
                if self.trace_logger.is_enabled():
                    self.trace_logger.mark_period(period)
                    self.trace_logger.emit(
                        "charge_request", sim_time_min=_min, day=_day,
                        taxi_id=str(taxi.id), soc=float(taxi.soc),
                        cycles_waited=int(getattr(taxi, "cycles_waited", 0)),
                        mechanism=self.MECHANISM,
                    )
                options = self._single_options(taxi)
                bundle, fail_event = self._decide(taxi, options)
                wait_min = float(getattr(taxi, "cycles_waited", 0)) * self._interval_min
                if bundle is not None:
                    bundle = self._rebase_price(bundle)
                    uf = float(self._ufun_fleet(bundle))
                    us = float(self._ufun_spklu(bundle))
                    a = [x for x in bundle.assignments if not x.is_unassigned()][0]
                    apply_deal(self.world, bundle)
                    if self.trace_logger.is_enabled():
                        if self.MECHANISM == "booking":
                            self.trace_logger.emit(
                                "booking_success", taxi_id=str(taxi.id),
                                port_id=str(a.port_id), slot_start=int(a.slot_start),
                                duration_grids=int(a.duration_grids),
                                price_per_kwh=float(a.price_per_kwh), price_mode="base",
                                utility_fleet=round(uf, 4), utility_spklu=round(us, 4),
                            )
                        self.trace_logger.emit(
                            "charge_start", taxi_id=str(taxi.id), port_id=str(a.port_id),
                            slot_start=int(a.slot_start), duration_grids=int(a.duration_grids),
                            price_per_kwh=float(a.price_per_kwh), price_mode="base",
                            wait_min=round(wait_min, 1),
                            utility_fleet=round(uf, 4), utility_spklu=round(us, 4),
                        )
                    self.alloc_records.append(dict(
                        period=period, success=True, uF=uf, uS=us, wait_min=wait_min,
                        duration_grids=int(a.duration_grids), port_id=str(a.port_id),
                        day=_day,
                    ))
                else:
                    handle_no_deal(self.world, [taxi])
                    if self.trace_logger.is_enabled():
                        self.trace_logger.emit(
                            fail_event, taxi_id=str(taxi.id),
                            wait_min=round(wait_min + self._interval_min, 1),
                            n_options=len(options), mechanism=self.MECHANISM,
                        )
                    self.alloc_records.append(dict(
                        period=period, success=False, uF=None, uS=None,
                        wait_min=wait_min + self._interval_min, duration_grids=0,
                        port_id=None, day=_day,
                    ))
                self.negotiation_count += 1
                self.episode_count += 1
                self.metrics_hook(
                    "negotiation_completed", deal=bundle is not None,
                    candidate_taxis=1, mechanism=self.MECHANISM,
                )
        finally:
            self.in_negotiation = False
            self.clock.resume()


class AsIsScheduler(_NonNegScheduler):
    """Siapa cepat dia dapat: port TERDEKAT; terisi → menunggu."""

    MECHANISM = "as_is"

    def _decide(self, taxi: Any, options: List[Bundle]):
        if not options:
            return None, "charge_wait"
        # port terdekat menurut jarak tempuh (taksi tidak tahu okupansi di depan)
        def a_of(b):
            return [x for x in b.assignments if not x.is_unassigned()][0]
        # sorted(): tie-break jarak deterministik (id port), bukan urutan hash set.
        nearest_port = min(sorted({a_of(b).port_id for b in options}),
                           key=lambda pid: self._dist_km(taxi, pid))
        free_here = [b for b in options
                     if a_of(b).port_id == nearest_port
                     and all(x.shift is None for x in b.assignments)]
        if free_here:
            # span kedatangan bebas → langsung isi (pilih terbaik utk fleet di port itu)
            return max(free_here, key=lambda b: self._ufun_fleet(b)), ""
        return None, "charge_wait"   # port terdekat terisi/terblokir → tunggu


class BookingScheduler(_NonNegScheduler):
    """Booking statis GREEDY-SEKARANG: pesan slot PADA WAKTU KEDATANGAN taksi di
    port bebas terdekat; harga base; tak ada port yang kedatangannya bebas → gagal.

    TANPA fungsi utilitas dan TANPA memesan masa depan — pustaka aksi memang
    hanya menawarkan SATU slot per port, yaitu grid kedatangan taksi
    (``_start_grid_for_arrival``); span kedatangan yang terblokir tidak pernah
    menjadi opsi. Paralel dengan as_is yang juga hanya beroperasi pada waktu
    kedatangan; beda semantiknya tinggal dua: (a) jaminan slot — pemesan tidak
    pernah mengantre, dan (b) boleh menilik semua port sehingga dapat memesan di
    port terdekat-berikutnya bila span kedatangan port terdekat terisi.
    """

    MECHANISM = "booking"

    def _decide(self, taxi: Any, options: List[Bundle]):
        free = [b for b in options if all(x.shift is None for x in b.assignments)]
        if not free:
            return None, "booking_fail"  # tak ada port dgn span-kedatangan bebas

        def a_of(b):
            return [x for x in b.assignments if not x.is_unassigned()][0]

        # Greedy deterministik: (1) jarak tempuh ke port; (2) waktu-kedatangan
        # (= slot_start; relevan hanya sbg pemecah seri jarak); (3) id port;
        # (4) durasi terpanjang (isi penuh bila masih seri).
        return min(free, key=lambda b: (self._dist_km(taxi, a_of(b).port_id),
                                        a_of(b).slot_start,
                                        str(a_of(b).port_id),
                                        -a_of(b).duration_grids)), ""


# ---------------------------------------------------------------------------
# Runner satu simulasi
# ---------------------------------------------------------------------------

def run_nonneg_once(cfg: Any, mechanism: str, run_dir: str, seed: Optional[int],
                    config_label: str, verbose: bool = True,
                    enable_trace: bool = True) -> Dict[str, Any]:
    """Jalankan satu simulasi non-negosiasi; tulis artefak per ../SCHEMA.md.

    ``enable_trace=False`` mematikan trace JSONL per-alokasi (hemat disk utk
    horizon panjang; agregat metrik tetap ditulis)."""
    t0 = time.time()
    rng = random.Random(seed)
    if seed is not None:
        import numpy as _np
        random.seed(seed)
        _np.random.seed(seed)

    for sub in ("logs/simulation", "metrices", "registry", "runs", "history"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    sim_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    trace_path = (os.path.join(run_dir, "logs", "simulation", f"{sim_id}.jsonl")
                  if enable_trace else None)

    world = build_world_from_config(cfg, rng=rng)
    collector = MetricsCollector(config=cfg, keep_payload_history=False)
    order_gen = OrderGenerator(world, rng=rng)
    klass = AsIsScheduler if mechanism == "as_is" else BookingScheduler
    sched = klass(
        world=world, agent_fleet=None, agent_spklu=None, config=cfg,
        order_generator=order_gen, metrics_collector=collector, rng=rng,
        trace_log_path=trace_path,
    )
    if verbose:
        print(f"[nonneg:{mechanism}] {config_label}: {len(world.taxis)} taksi, "
              f"{sum(len(s.ports) for s in world.stations.values())} port, "
              f"{cfg.time.n_days} hari -> {run_dir}")
    sched.run()

    # ---- agregasi ----
    rec = sched.alloc_records
    n = len(rec)
    succ = [r for r in rec if r["success"]]
    avg = lambda v: (sum(v) / len(v)) if v else None
    uF = avg([r["uF"] for r in succ])
    uS = avg([r["uS"] for r in succ])
    n_ports = sum(len(s.ports) for s in world.stations.values())
    horizon_h = float(cfg.time.n_days) * 24.0
    util = (sum(r["duration_grids"] for r in succ) * 0.5) / max(1e-9, n_ports * horizon_h)
    summary = collector.summary() if collector is not None else {}
    flat = collector.numeric_metrics_flat() if collector is not None else {}
    neg = (summary or {}).get("negotiation_system", {}) or {}
    aggregate = {
        "avg_utility_fleet": ((sum(r["uF"] for r in succ) - 1.0 * (n - len(succ))) / n) if n else None,
        "avg_utility_spklu": ((sum(r["uS"] for r in succ) - 0.1 * (n - len(succ))) / n) if n else None,
        "avg_deal_rate": (len(succ) / n) if n else None,
        "avg_partial_deal_rate": 0.0,
        "avg_cycles_waited": neg.get("avg_cycles_waited"),
        "avg_episode_length": avg([1.0 if r["wait_min"] > 0 else 0.0 for r in rec]) or 0.0,
        "negotiations_total": n,
        "orders_expired_at_hour_change": neg.get("orders_expired_at_hour_change"),
        "avg_utility_fleet_deal": uF,
        "avg_utility_spklu_deal": uS,
        "social_welfare_deal": (uF + uS) if (uF is not None and uS is not None) else None,
        # --- metrik bisnis (SCHEMA.md) ---
        "allocation_success_rate": (len(succ) / n) if n else None,
        "avg_wait_time_min": avg([r["wait_min"] for r in rec]) or 0.0,
        "port_utilization": round(util, 4),
        "compensation_paid": 0.0,
        "mechanism": mechanism,
    }
    with open(os.path.join(run_dir, "metrices", "aggregate_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"summary": summary, "numeric_flat": flat, "aggregate": aggregate},
                  fh, indent=2, default=str)
    meta = {
        "name": os.path.basename(run_dir), "sim_id": sim_id, "mechanism": mechanism,
        "config_override": config_label, "seed": seed, "n_days": cfg.time.n_days,
        "n_taxi": len(world.taxis), "n_port": n_ports,
        "allocation_events": n, "allocation_success": len(succ),
        "duration_s": round(time.time() - t0, 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": "non_negotiation_mechanism/common/nonneg_engine.py",
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    if verbose:
        print(f"[nonneg:{mechanism}] done: alloc={len(succ)}/{n} "
              f"uF={None if uF is None else round(uF,3)} "
              f"wait={round(aggregate['avg_wait_time_min'] or 0,1)}mnt "
              f"({meta['duration_s']}s)")
    return meta
