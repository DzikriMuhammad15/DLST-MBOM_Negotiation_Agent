# -*- coding: utf-8 -*-
"""Override ADITIF untuk dunia real-case Bandung (real_case_common) — kode inti TIDAK diubah.

Dua fungsi publik:

1. :func:`install_v2_bookings` — monkeypatch
   ``simulation.world.world_assembly._build_initial_bookings`` dengan versi v2
   yang menyampel (kapan/berapa-lama/di-mana) dari distribusi transaksi nyata
   via :class:`booking_sampler.EmpiricalBookingSampler`. Semantik lain
   dipertahankan IDENTIK engine (desain v2 §2.5):
   - skip diam-diam bila span asal terisi / tak ada fallback (N = target);
   - fallback plan tunggal via logika engine ``_choose_fallback_destination``
     (geser waktu ±shift_range di port sama → port lain slot sama; harus kosong);
   - kompensasi energy-based ``rate × power_kw × duration_grids × 0,5`` dengan
     ``rate ~ U(compensation_per_kwh_range)`` (v2: degenerate (986.72, 986.72));
   - alokasi port dikunci (``add_allocation(..., "booking")``).
   Kedua pipeline (negosiasi ``run_pipeline.run_once`` dan non-negosiasi
   ``nonneg_engine.run_nonneg_once``) memanggil ``build_world_from_config`` yang
   merujuk simbol modul ``_build_initial_bookings`` — patch berlaku utk keduanya.

2. :func:`normalize_time_weight` — koreksi key ``time_weight`` hasil load JSON.
   JSON memaksa key dict menjadi string ("0".."23") sementara konsumen engine
   (``order_generator``, ``ufun_builder._time_weight_at``) membaca key int
   (``dict.get(hour_int, 1.0)``) — tanpa normalisasi ini bobot override DIAM-DIAM
   diabaikan (fallback 1.0). Fungsi ini membangun ulang dict {int: float}
   dengan key string (override kita) diprioritaskan atas key int (default).
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any, Dict, List, Optional

from simulation.config.config import SimulationConfig
from simulation.world import world_assembly
from simulation.world.entities import AppUserBooking, ChargingStation, FallbackPlan

from booking_sampler import EmpiricalBookingSampler

# Statistik pemasangan booking terakhir (utk log/laporan N terpasang aktual).
# Diisi ulang setiap kali dunia dirakit; dibaca oleh booking_report.py.
LAST_INSTALL_STATS: Dict[str, Any] = {}

_original_build_initial_bookings = world_assembly._build_initial_bookings


def normalize_time_weight(cfg: SimulationConfig) -> SimulationConfig:
    """Kembalikan config dgn ``marking_area_order.time_weight`` ber-key int.

    Key string (dari JSON override) menang atas key int (default 1.0) —
    lihat docstring modul. Aman dipanggil pada config apa pun (idempoten).
    """
    tw = cfg.marking_area_order.time_weight
    fixed: Dict[int, float] = {}
    for hour in range(24):
        if str(hour) in tw:
            fixed[hour] = float(tw[str(hour)])
        elif hour in tw:
            fixed[hour] = float(tw[hour])
        else:
            fixed[hour] = 1.0
    mao = dataclasses.replace(cfg.marking_area_order, time_weight=fixed)
    return dataclasses.replace(cfg, marking_area_order=mao)


def _build_initial_bookings_v2_factory(sampler: EmpiricalBookingSampler,
                                        geser_maks: Optional[int] = None):
    """Bangun fungsi pengganti ``_build_initial_bookings`` (signature identik).

    Kebijakan penempatan: **permintaan diundi sekali, lalu yang dicari hanyalah
    tempatnya.** Durasi TIDAK PERNAH diundi ulang.

    1. Coba port dan jam persis seperti yang diundi.
    2. Masih pada jam yang sama, coba port lain menurut urutan acak berbobot
       volume stasiun.
    3. Geser jam, besaran kecil dulu, sampai ``geser_maks`` grid ke depan atau
       ke belakang. Pada tiap jam baru seluruh port dicoba lagi.
    4. Gagal semua, booking ditolak. Tidak ada undian permintaan kedua.

    Mengapa durasi tidak diundi ulang. Cara lama mengundi ulang port, durasi,
    dan jam sekaligus, sehingga permintaan yang gagal lenyap dan digantikan
    permintaan lain yang biasanya lebih pendek. Booking panjang paling sering
    bentrok, jadi booking panjang menghilang diam-diam dari dunia simulasi.
    Terukur: dengan cara lama sebaran durasi terpasang menyimpang 12,35 persen
    dari sebaran permintaan, dengan cara ini hanya 3,54 persen, dan penolakan
    turun dari 20,5 menjadi 7,4 persen pada N = 366.

    Tafsir perilakunya juga lebih masuk akal. Pengemudi yang menemukan port
    penuh tidak mengubah berapa lama ia perlu mengecas; ia pindah stasiun, dan
    kalau terpaksa menggeser jam.

    ``geser_maks`` bawaan mengikuti batas atas ``allowed_slot_shift_range`` pada
    config, sehingga sejalan dengan pergeseran fallback.
    """

    def _build_initial_bookings_v2(
        config: SimulationConfig,
        stations: Dict[str, ChargingStation],
        rng: random.Random,
    ) -> Dict[str, AppUserBooking]:
        all_ports = world_assembly._all_port_objects(stations)
        if not all_ports:
            return {}
        port_by_id = {p.id: p for p in all_ports}

        booking_cfg = config.app_user_booking
        fallback_cfg = booking_cfg.fallback
        n_days = int(config.time.n_days)
        total_grids = max(1, n_days * world_assembly._GRIDS_PER_DAY)
        comp_min, comp_max = fallback_cfg.compensation_per_kwh_range
        maks_geser = int(geser_maks if geser_maks is not None
                         else fallback_cfg.allowed_slot_shift_range[1])
        maks_geser = max(1, maks_geser)

        bookings: Dict[str, AppUserBooking] = {}
        n_langsung = n_pindah_port = n_geser_jam = 0
        n_tolak_kapasitas = n_tolak_horizon = n_tolak_tanpa_fallback = 0
        durasi_terpasang: List[int] = []
        jam_terpasang: List[int] = []
        durasi_diminta: List[int] = []
        jam_diminta: List[int] = []

        for i in range(booking_cfg.initial_booking_count):
            # ── PERMINTAAN: diundi SEKALI, tidak pernah diubah ───────────────
            port_awal = port_by_id[sampler.sample_port(rng)]
            durasi = sampler.sample_duration_grids(rng)
            slot_awal = sampler.sample_start_grid(rng, n_days)
            durasi_diminta.append(int(durasi))
            jam_diminta.append(int(slot_awal % world_assembly._GRIDS_PER_DAY))

            urutan = [port_awal] + [port_by_id[pid] for pid in sampler.urutan_port(rng)
                                    if pid != port_awal.id]

            pilihan = None
            jenis = ""
            ada_slot_sah = False

            # (1) dan (2) jam yang sama, port awal lalu port lain.
            if 0 <= slot_awal and slot_awal + durasi <= total_grids:
                ada_slot_sah = True
                for p in urutan:
                    if p.is_free(slot_awal, durasi):
                        pilihan = (p, slot_awal)
                        jenis = "langsung" if p.id == port_awal.id else "pindah_port"
                        break

            # (3) geser jam, besaran kecil dulu, arah diacak.
            if pilihan is None:
                for besaran in range(1, maks_geser + 1):
                    for tanda in rng.sample((-1, 1), 2):
                        slot = slot_awal + tanda * besaran
                        if slot < 0 or slot + durasi > total_grids:
                            continue
                        ada_slot_sah = True
                        for p in urutan:
                            if p.is_free(slot, durasi):
                                pilihan = (p, slot)
                                jenis = "geser_jam"
                                break
                        if pilihan is not None:
                            break
                    if pilihan is not None:
                        break

            # (4) ditolak.
            if pilihan is None:
                if ada_slot_sah:
                    n_tolak_kapasitas += 1
                else:
                    n_tolak_horizon += 1
                continue
            port, slot_start = pilihan

            # Fallback tunggal via LOGIKA ENGINE (bukan re-implementasi).
            destination = world_assembly._choose_fallback_destination(
                all_ports, port, slot_start, durasi, total_grids,
                fallback_cfg.allowed_slot_shift_range, rng,
            )
            if destination is None:
                n_tolak_tanpa_fallback += 1
                continue
            to_port_id, to_slot = destination

            # Kompensasi energy-based (rumus engine; v2: rate fix 986.72).
            rate_per_kwh = rng.uniform(float(comp_min), float(comp_max))
            compensation = (rate_per_kwh * port.power_kw * durasi
                            * world_assembly._GRID_HOURS)

            booking_id = f"booking_{i}"
            bookings[booking_id] = AppUserBooking(
                id=booking_id,
                port_id=port.id,
                slot_start=slot_start,
                duration_grids=durasi,
                is_shiftable=True,
                fallback_plan=FallbackPlan(
                    to_port_id=to_port_id,
                    to_slot_start=to_slot,
                    compensation=compensation,
                ),
                compensation=0.0,
            )
            port.add_allocation(slot_start, durasi, booking_id, "booking")
            durasi_terpasang.append(int(durasi))
            jam_terpasang.append(int(slot_start % world_assembly._GRIDS_PER_DAY))
            if jenis == "langsung":
                n_langsung += 1
            elif jenis == "pindah_port":
                n_pindah_port += 1
            else:
                n_geser_jam += 1

        target = int(booking_cfg.initial_booking_count)
        terpasang = len(bookings)
        ditolak = n_tolak_kapasitas + n_tolak_horizon + n_tolak_tanpa_fallback
        port_jam = sum(b.duration_grids for b in bookings.values()) * world_assembly._GRID_HOURS
        kapasitas = len(all_ports) * n_days * 24.0

        LAST_INSTALL_STATS.clear()
        LAST_INSTALL_STATS.update({
            "permintaan": target,
            "terpasang": terpasang,
            "tanpa_menggeser": n_langsung,
            "menggeser_pindah_port": n_pindah_port,
            "menggeser_geser_jam": n_geser_jam,
            "menggeser_total": n_pindah_port + n_geser_jam,
            "ditolak_total": ditolak,
            "ditolak_kapasitas_penuh": n_tolak_kapasitas,
            "ditolak_di_luar_horizon": n_tolak_horizon,
            "ditolak_tanpa_fallback": n_tolak_tanpa_fallback,
            "persen_terpasang": round(terpasang / max(1, target) * 100.0, 4),
            "persen_ditolak": round(ditolak / max(1, target) * 100.0, 4),
            "okupansi_persen": round(port_jam / max(1e-9, kapasitas) * 100.0, 4),
            "geser_maks_grid": maks_geser,
            "sampler": sampler.provenance().get("sampler", "empiris"),
            # Sebaran mentah untuk laporan pembanding (lihat booking_report.py).
            "durasi_diminta_grid": durasi_diminta,
            "jam_diminta_grid": jam_diminta,
            "durasi_terpasang_grid": durasi_terpasang,
            "jam_terpasang_grid": jam_terpasang,
        })
        print(f"[v2-bookings] permintaan={target} terpasang={terpasang} "
              f"(langsung={n_langsung} pindah_port={n_pindah_port} "
              f"geser_jam={n_geser_jam}) ditolak={ditolak} "
              f"(kapasitas={n_tolak_kapasitas} horizon={n_tolak_horizon} "
              f"tanpa_fallback={n_tolak_tanpa_fallback}) "
              f"okupansi={LAST_INSTALL_STATS['okupansi_persen']:.2f}%")
        return bookings

    return _build_initial_bookings_v2


def install_v2_bookings(sampler: EmpiricalBookingSampler,
                        geser_maks: Optional[int] = None) -> None:
    """Pasang override booking v2 (monkeypatch modul world_assembly).

    ``geser_maks`` membatasi pergeseran jam saat penempatan; ``None`` berarti
    mengikuti batas atas ``allowed_slot_shift_range`` pada config.
    """
    world_assembly._build_initial_bookings = _build_initial_bookings_v2_factory(
        sampler, geser_maks=geser_maks)
    jenis = sampler.provenance().get("sampler", "empiris")
    print(f"[v2-bookings] override _build_initial_bookings TERPASANG "
          f"(sampler {jenis}, penempatan: durasi tetap, port lalu jam)")


def uninstall_v2_bookings() -> None:
    """Kembalikan ``_build_initial_bookings`` asli (utk kebersihan/test)."""
    world_assembly._build_initial_bookings = _original_build_initial_bookings
