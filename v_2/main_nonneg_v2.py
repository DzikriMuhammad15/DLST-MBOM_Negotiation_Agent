# -*- coding: utf-8 -*-
"""Entry NON-NEGOSIASI (booking / as_is) v_2 — dunia real-case v2 fixed_mechanism.

Dunia dan config identik dengan main_nego.py. Mekanisme statis memakai engine
yang sama, run_nonneg_once; yang berubah oleh fixed_mechanism hanya kalibrasi
fungsi utilitas yang dicatat, keputusan mekanismenya sendiri tidak dilemahkan.

Empat keluarga skenario, seluruhnya dibuat satu generator ``v_2/gen_configs.py``
yang menulis ``configs/run{1..40}_<skenario>.json``:

===========================  ==========  =========================
Skenario                     Permintaan  Harga lantai saat digeser
===========================  ==========  =========================
medium_occupancy                    185  dibiarkan
high_occupancy                      366  dibiarkan
mahalkompen_medium_occupancy        185  dilarang
mahalkompen_high_occupancy          366  dilarang
===========================  ==========  =========================

Angka 185 dan 366 adalah rata-rata dan kuantil 95 jumlah transaksi harian
seluruh SPKLU Bandung 2024, dikali horizon tiga hari. Yang dikendalikan adalah
PERMINTAAN booking; okupansi adalah keluaran, dicatat per run pada
``metrices/booking_generation.json``.

Sampler booking bawaan menarik durasi dan jam mulai dari distribusi teoretis
hasil pencocokan. Tambahkan ``--empiris`` untuk memakai histogram data seperti
cara lama. Kebijakan penempatan booking sama pada kedua mode.

Pakai:
    python v_2/main_nonneg_v2.py --mech booking --run 1
    python v_2/main_nonneg_v2.py --mech as_is   --run 1 --skenario high_occupancy

Output: v_2/data_<skenario>/non_negotiation/real_case_v2/<mech>/run{n}/
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
V2_DIR = os.path.join(ROOT, "real_case_common")
sys.path.insert(0, V2_DIR)
from v2_common import build_sampler, fix_sys_path  # noqa: E402

fix_sys_path()

from main import load_config_from_path                                       # noqa: E402
from non_negotiation_mechanism.common.nonneg_engine import run_nonneg_once   # noqa: E402

from overrides import install_v2_bookings, normalize_time_weight             # noqa: E402
from booking_report import tulis_laporan                                     # noqa: E402

# Empat keluarga skenario. Yang dikendalikan adalah PERMINTAAN booking, bukan
# okupansi: 185 untuk permintaan menengah dan 366 untuk permintaan tinggi, yaitu
# rata-rata dan kuantil 95 jumlah transaksi harian SPKLU Bandung 2024 dikali tiga
# hari. Okupansi yang dihasilkan dicatat per run pada booking_generation.json.
SKENARIO = (
    "medium_occupancy",
    "high_occupancy",
    "mahalkompen_medium_occupancy",
    "mahalkompen_high_occupancy",
)


def folder_data(skenario: str) -> str:
    """Satu folder keluaran tersendiri per skenario."""
    return f"data_{skenario}"


def cli() -> None:
    ap = argparse.ArgumentParser(description="Non-nego (booking/as_is) fixed_mechanism v2.")
    ap.add_argument("--mech", required=True, choices=("booking", "as_is"))
    ap.add_argument("--run", type=int, default=1, choices=tuple(range(1, 41)),
                    help="Run replikasi 1-40 (config run{n}_<skenario>.json).")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--skenario", default="medium_occupancy", choices=SKENARIO,
                    help="Keluarga skenario. medium/high mengatur banyaknya "
                         "permintaan booking (185 atau 366); awalan mahalkompen "
                         "mengaktifkan larangan harga lantai pada penugasan "
                         "ber-shift. Semuanya sudah dibakukan di config.")
    ap.add_argument("--empiris", action="store_true",
                    help="Pakai sampler EMPIRIS lama untuk durasi & jam mulai "
                         "booking (undian berbobot histogram, dipotong 2 jam). "
                         "Bawaan tanpa flag ini: sampling dari distribusi "
                         "teoretis hasil fit.")
    ap.add_argument("--lean", action="store_true",
                    help="Matikan trace JSONL (hemat disk); agregat tetap ditulis.")
    ap.add_argument("--seed-override", type=int, default=None,
                    help="ADITIF 2026-07-29: timpa seed+armada awal (sinkronisasi ke seed "
                         "yang sudah dipakai negosiasi pada run yang sama). Dunia lain identik.")
    args = ap.parse_args()

    scenario = args.skenario
    cfg_path = os.path.join(HERE, "configs", f"run{args.run}_{scenario}.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"Config tidak ada: {os.path.relpath(cfg_path, ROOT)} — "
                         f"jalankan `python v_2/gen_configs.py` dulu.")
    if args.seed_override is not None:
        from seed_override_util import resolve_config_path
        cfg_path = resolve_config_path(cfg_path, args.seed_override)
    cfg = normalize_time_weight(load_config_from_path(cfg_path))
    sampler = build_sampler(empiris=args.empiris)
    install_v2_bookings(sampler)

    out_root = args.out_root or os.path.join(
        HERE, folder_data(scenario), "non_negotiation", "real_case_v2")
    run_dir = os.path.join(out_root, args.mech, f"run{args.run}")
    print(f"[v_2-nonneg] mech={args.mech} seed={cfg.seed} "
          f"(skenario={scenario}) -> {run_dir}")
    run_nonneg_once(cfg, args.mech, run_dir, cfg.seed,
                    config_label=os.path.relpath(cfg_path, ROOT),
                    enable_trace=not args.lean)
    tulis_laporan(run_dir, sampler, n_days=cfg.time.n_days, label=scenario)


if __name__ == "__main__":
    cli()
