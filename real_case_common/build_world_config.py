# -*- coding: utf-8 -*-
"""Generator config dunia real-case Bandung (desain v2 §2 — data-grounded).

Membaca SEMUA kuantitas dunia yang tersedia di data dari file sumber
(prinsip grounding §2.0 — tidak ada angka dunia yang di-hardcode di sini
kecuali yang berstatus KEPUTUSAN/ASUMSI terdokumentasi):

- 13 SPKLU PLN ber-port-tersedikit (lat/lon/kW/port) ← `simulation_data/_derived/bandung_spklu.json`
- harga base = median price_per_kwh nyata               ← `bandung_trx.csv`
- bobot waktu order 24 jam                              ← `simulation_data/sumber_data_traffic_kendaraanUmum/time_weight_24.json` (T1, TomTom Bandung)
- N booking = target okupansi 22,5% ÷ E[durasi tercap]  ← `bandung_only_trx_summary.json` (via EmpiricalBookingSampler)

Yang berstatus KEPUTUSAN peneliti (bukan data — dokumentasi di
DOKUMENTASI_SAMPLING_DAN_PROVENANCE.md §F): 18 marking area (§7 desain,
data peneliti), 50 taksi U(bbox kota)+SoC U(0.35,0.90), okupansi 22,5%,
price_levels [0.8×base, base, 1.4×base, 4000], kompensasi fix 40%×base,
fare 3000/km, n_days=3, seed run1/2/3 = 42/7/123.
**RALAT GERBANG RUNTIME (keputusan peneliti 2026-07-08):** max_steps 100→15,
K (max_action_library=K_max) 50→30, mbom_M 3→2, rollout_max_steps 20→None
(faithful = ikut max_steps 15) — smoke DLSTMBOM K50/steps100/M3 terbukti
>>6 jam/exp (54% hari dlm 7,3 jam-CPU). Deviasi didokumentasikan
(DOKUMENTASI §D.3/§F + desain v2).

Pakai:  python real_case_common/build_world_config.py
Output: configs/run{1,2,3}.json + configs/v2_world_meta.json (audit lengkap)

Catatan (clean_code): file run{1,2,3}.json + v2_world_meta.json hasil generator ini
SUDAH disertakan langsung di configs/ (disalin dari korpus asal), jadi TIDAK WAJIB
dijalankan ulang kecuali ingin regenerasi from-scratch dari data mentah untuk verifikasi.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from booking_sampler import DERIVED_DIR, EmpiricalBookingSampler  # noqa: E402

CFG_DIR = os.path.join(_HERE, "configs")
TRAFFIC_DIR = os.path.join(_ROOT, "simulation_data", "sumber_data_traffic_kendaraanUmum")

# ── KEPUTUSAN SKENARIO (bukan data — lihat docstring) ────────────────────────
N_STATIONS = 13            # SPKLU PLN ber-port-tersedikit (desain §2.1)
N_TAXIS = 50               # desain §2.2
N_DAYS = 3
OCCUPANCY_TARGET = 0.225   # ralat peneliti 2026-07-07 (N bukan data-derived)
SOC_RANGE = (0.35, 0.90)
# BBox Kota Bandung utk posisi taksi uniform (desain §2.2).
BBOX_LAT = (-6.98, -6.84)
BBOX_LON = (107.55, 107.72)
RUN_SEEDS = {1: 42, 2: 7, 3: 123}
TAXI_SEED_OFFSET = 10_000  # rng taksi = Random(TAXI_SEED_OFFSET + run_seed)
PRICE_CEILING = 4000.0     # level harga tertinggi (keputusan peneliti)
COMP_RATE_FRACTION = 0.4   # kompensasi = 40% × harga base per kWh
# Spec BYD e6 (ikut rc lama; data peneliti "1 kWh per 5 km").
BATTERY_KWH = 61.0
CONSUMPTION_KWH_PER_KM = 0.2
SPEED_KMH = 30.0
ORDER_MIN, ORDER_MAX = 0, 50  # global (schema engine), desain §7

# 18 marking area NYATA (data peneliti; desain v2 §7 — 4 dari 22 dihapus
# 2026-07-07: Padalarang, Dusun Bambu, The Lodge Maribaya, Soreang).
MARKING_AREAS: List[Dict] = [
    {"nama": "d'Botanica Pasteur",            "lat": -6.8812, "lon": 107.5800},
    {"nama": "Cihampelas Walk (Ciwalk)",      "lat": -6.8938, "lon": 107.6052},
    {"nama": "Paris Van Java (PVJ)",          "lat": -6.8895, "lon": 107.5957},
    {"nama": "Bandung Indah Plaza (BIP)",     "lat": -6.9112, "lon": 107.6097},
    {"nama": "Summarecon Mall Bandung",       "lat": -6.9600, "lon": 107.7170},
    {"nama": "Trans Studio Mall Bandung",     "lat": -6.9276, "lon": 107.6364},
    {"nama": "Stasiun Bandung",               "lat": -6.9175, "lon": 107.6030},
    {"nama": "Stasiun Tegalluar Whoosh",      "lat": -6.9850, "lon": 107.7400},
    {"nama": "Terminal Leuwipanjang",         "lat": -6.9333, "lon": 107.5742},
    {"nama": "Kertajati International Airport", "lat": -6.5569, "lon": 108.2314},
    {"nama": "Cititrans Dipatiukur",          "lat": -6.8840, "lon": 107.6186},
    {"nama": "Shuttle Drop Off Pasteur",      "lat": -6.8855, "lon": 107.5779},
    {"nama": "Majesty Apartement",            "lat": -6.8858, "lon": 107.5790},
    {"nama": "Kota Baru Parahyangan",         "lat": -6.8655, "lon": 107.4750},
    {"nama": "W Super Club",                  "lat": -6.9180, "lon": 107.6150},
    {"nama": "MOD Pool and Club",             "lat": -6.9205, "lon": 107.6162},
    {"nama": "Lembang Park Zoo",              "lat": -6.8247, "lon": 107.6133},
    {"nama": "Rest Area Alun-Alun Lembang",   "lat": -6.8180, "lon": 107.6169},
]


def _slug(nama: str) -> str:
    s = nama.upper().replace("SPKLU PLN", "").strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return f"spklu_{s}"


def _kw(value: str) -> float:
    m = re.search(r"([\d.]+)", str(value))
    if not m:
        raise ValueError(f"kW tidak terbaca: {value!r}")
    return float(m.group(1))


def select_stations() -> List[Dict]:
    """13 SPKLU PLN ber-port-tersedikit dari bandung_spklu.json (deterministik).

    Urut: jumlah port menaik, lalu nama (tie-break stabil). Port = total
    konektor semua chargebox; kW port = kW chargebox induknya (apa adanya).
    """
    with open(os.path.join(DERIVED_DIR, "bandung_spklu.json"), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    pln = [s for s in data if "PLN" in s["nama"].upper()]
    pln.sort(key=lambda s: (sum(cb["connectors"] for cb in s["chargeboxes"]), s["nama"]))
    out = []
    for s in pln[:N_STATIONS]:
        sid = _slug(s["nama"])
        ports = []
        for cb in s["chargeboxes"]:
            for _ in range(int(cb["connectors"])):
                ports.append({"port_id": f"{sid}_p{len(ports)}", "power_kw": _kw(cb["kw"])})
        out.append({
            "station_id": sid, "csv_id": str(s["id"]), "nama": s["nama"],
            "lat": float(s["lat"]), "lon": float(s["lon"]), "ports": ports,
        })
    return out


def load_time_weight() -> Dict[str, float]:
    """24 bobot jam hasil T1 (TomTom Bandung) — key string utk JSON config."""
    with open(os.path.join(TRAFFIC_DIR, "time_weight_24.json"), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    tw = {str(h): float(data["time_weight"][str(h)]) for h in range(24)}
    return tw


def sample_taxis(run_seed: int) -> List[Dict]:
    """50 taksi: posisi ~ U(bbox kota), SoC ~ U(0.35, 0.90) — per run seed."""
    rng = random.Random(TAXI_SEED_OFFSET + run_seed)
    out = []
    for i in range(N_TAXIS):
        out.append({
            "taxi_id": f"taxi_{i}",
            "position": [round(rng.uniform(*BBOX_LAT), 6), round(rng.uniform(*BBOX_LON), 6)],
            "soc": round(rng.uniform(*SOC_RANGE), 3),
        })
    return out


def main() -> None:
    os.makedirs(CFG_DIR, exist_ok=True)
    stations = select_stations()
    n_ports = sum(len(s["ports"]) for s in stations)

    sampler = EmpiricalBookingSampler(
        station_ports={s["station_id"]: [p["port_id"] for p in s["ports"]] for s in stations},
        station_csv_ids={s["station_id"]: s["csv_id"] for s in stations},
    )
    base_price = EmpiricalBookingSampler.real_base_price()
    price_levels = [round(0.8 * base_price, 2), base_price,
                    round(1.4 * base_price, 2), PRICE_CEILING]
    comp_rate = round(COMP_RATE_FRACTION * base_price, 2)

    e_dur_h = sampler.expected_capped_duration_hours()
    capacity_port_hours = n_ports * N_DAYS * 24
    n_booking = sampler.n_target_bookings(n_ports, N_DAYS, OCCUPANCY_TARGET)

    time_weight = load_time_weight()

    station_specs = [{
        "station_id": s["station_id"],
        "location": [s["lat"], s["lon"]],
        "ports": [{"port_id": p["port_id"], "power_kw": p["power_kw"],
                   "base_price_per_kwh": base_price} for p in s["ports"]],
    } for s in stations]

    area_specs = [{"area_id": f"area_{i:02d}", "center": [a["lat"], a["lon"]]}
                  for i, a in enumerate(MARKING_AREAS)]

    for run, seed in RUN_SEEDS.items():
        cfg = {
            "time": {"n_days": N_DAYS},
            "seed": seed,
            "training": {"train_every_n_episodes": 1},
            # RALAT gerbang runtime 2026-07-08 (peneliti): K 50->30, steps 100->15.
            "negotiation": {"max_action_library": 30, "max_steps": 15,
                            "price_levels": price_levels},
            "state_feature": {"encoder_dims": {"K_max": 30, "N_max": N_TAXIS,
                                               "P_max": n_ports,
                                               "A_max": len(MARKING_AREAS)}},
            "taxi_spec": {"battery_capacity_kwh": BATTERY_KWH,
                          "consumption_kwh_per_km": CONSUMPTION_KWH_PER_KM,
                          "speed_kmh": SPEED_KMH,
                          "initial_taxi_state": sample_taxis(seed)},
            "port_station": {"stations": station_specs},
            "marking_area_order": {"areas": area_specs,
                                   "order_min": ORDER_MIN, "order_max": ORDER_MAX,
                                   "time_weight": time_weight,
                                   "order_claim_mode": "lock"},
            "fare": {"fare_mode": "PURE", "price_per_km": 3000.0},
            "app_user_booking": {"initial_booking_count": n_booking,
                                 "fallback": {"allowed_slot_shift_range": [1, 4],
                                              "compensation_per_kwh_range": [comp_rate, comp_rate]}},
            # RALAT 2026-07-08: mbom_M 3->2; rollout faithful (None = ikut max_steps 15).
            "agent": {"fleet": {"hyperparameters": {"mbom_M": 2, "rollout_max_steps": None}},
                      "spklu": {"hyperparameters": {"rollout_max_steps": None}}},
        }
        path = os.path.join(CFG_DIR, f"run{run}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)
        print(f"config: run{run}.json (seed={seed}, N_booking={n_booking})")

    meta = {
        "dibuat": "build_world_config.py (real_case_common)",
        "stations_13_pln": [{k: s[k] for k in ("station_id", "csv_id", "nama", "lat", "lon")}
                            | {"ports": s["ports"]} for s in stations],
        "n_ports_total": n_ports,
        "harga": {
            "base_price_per_kwh": base_price,
            "sumber": "median price_per_kwh bandung_trx.csv (tarif PLN nyata, flat)",
            "price_levels": price_levels,
            "rumus_levels": "[0.8*base, base, 1.4*base, 4000.0]",
            "kompensasi_per_kwh": comp_rate,
            "rumus_kompensasi": f"{COMP_RATE_FRACTION} * base = {COMP_RATE_FRACTION} * {base_price} = {comp_rate}",
        },
        "n_booking": {
            "okupansi_target": OCCUPANCY_TARGET,
            "kapasitas_port_jam": capacity_port_hours,
            "rumus_kapasitas": f"{n_ports} port x {N_DAYS} hari x 24 jam = {capacity_port_hours}",
            "durasi_efektif_jam": round(e_dur_h, 6),
            "durasi_efektif_grid": round(sampler.expected_capped_duration_grids(), 6),
            "N": n_booking,
            "rumus_N": (f"round({OCCUPANCY_TARGET} x {capacity_port_hours} / {round(e_dur_h, 6)}) "
                        f"= round({round(OCCUPANCY_TARGET * capacity_port_hours, 4)} / {round(e_dur_h, 6)}) = {n_booking}"),
        },
        "sampler_provenance": sampler.provenance(),
        "time_weight": {"sumber": "simulation_data/sumber_data_traffic_kendaraanUmum/time_weight_24.json (TomTom Traffic Index Bandung, data 2025)",
                        "bobot": time_weight},
        "taksi": {"n": N_TAXIS, "battery_kwh": BATTERY_KWH,
                  "konsumsi_kwh_per_km": CONSUMPTION_KWH_PER_KM, "speed_kmh": SPEED_KMH,
                  "posisi": f"Uniform bbox lat {BBOX_LAT} lon {BBOX_LON}",
                  "soc": f"Uniform{SOC_RANGE}",
                  "seed_taksi": {r: TAXI_SEED_OFFSET + s for r, s in RUN_SEEDS.items()}},
        "run_seeds": RUN_SEEDS,
        "marking_areas_18": MARKING_AREAS,
        "kontensi": {"taksi_per_port": round(N_TAXIS / n_ports, 4),
                     "rumus": f"{N_TAXIS} / {n_ports} = {round(N_TAXIS / n_ports, 4)}"},
    }
    with open(os.path.join(CFG_DIR, "v2_world_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)

    print(f"\nRingkasan dunia v2:")
    print(f"  13 SPKLU PLN, {n_ports} port; kontensi {N_TAXIS}/{n_ports} = {N_TAXIS/n_ports:.2f}")
    print(f"  base_price = {base_price} (median data); levels = {price_levels}; komp = {comp_rate}")
    print(f"  E[durasi tercap] = {e_dur_h:.4f} jam; N_booking = {n_booking} "
          f"(= {OCCUPANCY_TARGET} x {capacity_port_hours} / {e_dur_h:.4f})")
    print(f"  meta audit: configs/v2_world_meta.json")


if __name__ == "__main__":
    main()
