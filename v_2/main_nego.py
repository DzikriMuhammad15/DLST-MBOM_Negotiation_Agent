# -*- coding: utf-8 -*-
"""Entry NEGOSIASI v_2 (real-case fixed_mechanism) — 1 invokasi = 1 eksperimen.

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

Mekanisme yang DIPERBAIKI ada di kode global (negotiation/ + simulation/):
  1b   penskalaan reward C6 per-negosiasi (min menu -> 0, max menu -> 1);
  2c/d action library full-assignment (semua kandidat yang termuat fisik
       di-assign; tak ada parsial by-choice);
  2a/b kalibrasi w_unassigned/g_max via config.

Pakai (dari root repo clean_code/):
    python v_2/gen_configs.py                                   # sekali saja
    python v_2/main_nego.py --agent-fleet DLST --run 1
    python v_2/main_nego.py --agent-fleet DLST --run 1 --skenario high_occupancy
    python v_2/main_nego.py --agent-fleet DLST --run 1 --skenario mahalkompen_high_occupancy
    python v_2/main_nego.py --agent-fleet DLST --run 1 --empiris
    python v_2/main_nego.py --agent-fleet DLST --run 1 --smoke

Output: v_2/data_<skenario>/negotiation/real_case_v2/run{n}_padded_<agen>/
Lengkap bila ada metrices/aggregate_metrics.json, metrices/booking_generation.json,
serta dua gambar pembanding sebaran booking.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
V2_DIR = os.path.join(ROOT, "real_case_common")
sys.path.insert(0, V2_DIR)
from v2_common import apply_agents, build_sampler, fix_sys_path  # noqa: E402

fix_sys_path()

from main import load_config_from_path            # noqa: E402  (main.py ROOT)
from run_experiment import run_single_experiment  # noqa: E402

from overrides import install_v2_bookings, normalize_time_weight  # noqa: E402
from booking_report import tulis_laporan  # noqa: E402

AGENT_SLUGS = {
    "DLST": "dlst",
    "DLSTMBOM": "dlstmbom",
    "DLSTMBOMNewMechanism": "dlstmbomnew",
    # 2026-07-22: Directed di network base 128x2 (dunia run1-30 IDENTIK korpus).
    # Switch anti-kolaps diaktifkan otomatis oleh run_pipeline.side_agent_build_cfg
    # (keyed agent_type=="DLSTMBOMDirected"); tak perlu wiring tambahan di sini.
    "DLSTMBOMDirected": "directed",
}

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


SMOKE_EXP_ROOT = os.path.join(HERE, "_smoke")
N_DAYS_FULL = 3


def exp_root_skenario(skenario: str) -> str:
    return os.path.join(HERE, folder_data(skenario), "negotiation", "real_case_v2")


def cli() -> None:
    ap = argparse.ArgumentParser(description="Nego fixed_mechanism real-case v2 (1 exp/invokasi).")
    ap.add_argument("--agent-fleet", required=True, choices=sorted(AGENT_SLUGS))
    ap.add_argument("--mode", default="PADDED", choices=("PADDED", "ENCODER"))
    ap.add_argument("--run", type=int, default=1, choices=tuple(range(1, 41)),
                    help="Run replikasi 1-40 (seed = 7 + 104729*n, lihat gen_configs.py).")
    ap.add_argument("--name", default=None)
    ap.add_argument("--exp-root", default=None)
    ap.add_argument("--skenario", default="medium_occupancy", choices=SKENARIO,
                    help="Keluarga skenario. medium/high mengatur banyaknya "
                         "permintaan booking (185 atau 366); awalan mahalkompen "
                         "mengaktifkan larangan harga lantai pada penugasan "
                         "ber-shift. Semuanya sudah dibakukan di config.")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke: n_days=1, N booking diskala 1/3, output _smoke/.")
    ap.add_argument("--empiris", action="store_true",
                    help="Pakai sampler EMPIRIS lama untuk durasi & jam mulai "
                         "booking (undian berbobot histogram, dipotong 2 jam). "
                         "Bawaan tanpa flag ini: sampling dari distribusi "
                         "teoretis hasil fit.")
    ap.add_argument("--lean", action="store_true",
                    help="Matikan trace & online-log (hemat disk; aggregate_metrics tetap ditulis).")
    ap.add_argument("--online-log", action="store_true",
                    help="Nyalakan log online training MESKI --lean (trace tetap "
                         "mengikuti --lean; hemat disk + kurva loss tetap terekam).")
    ap.add_argument("--online-log-full", action="store_true",
                    help="Seperti --online-log TAPI snapshot memori TETAP DITULIS "
                         "(state+action per step -> bisa tracing output aktor, bukan "
                         "hanya kurva loss). BOROS DISK. Menang atas --online-log.")
    args = ap.parse_args()

    scenario = args.skenario
    cfg_path = os.path.join(HERE, "configs", f"run{args.run}_{scenario}.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"Config tidak ada: {os.path.relpath(cfg_path, ROOT)} — "
                         f"jalankan `python v_2/gen_configs.py` dulu.")
    cfg = load_config_from_path(cfg_path)
    cfg = normalize_time_weight(cfg)
    cfg = apply_agents(cfg, args.agent_fleet, args.mode)

    if args.smoke:
        n_book = int(round(cfg.app_user_booking.initial_booking_count / N_DAYS_FULL))
        cfg = dataclasses.replace(
            cfg,
            time=dataclasses.replace(cfg.time, n_days=1),
            app_user_booking=dataclasses.replace(cfg.app_user_booking,
                                                 initial_booking_count=n_book),
        )
        print(f"[v_2] SMOKE: n_days=1, N booking -> {n_book}")

    sampler = build_sampler(empiris=args.empiris)
    install_v2_bookings(sampler)

    name = args.name or f"run{args.run}_{args.mode.lower()}_{AGENT_SLUGS[args.agent_fleet]}"
    exp_root = args.exp_root or (SMOKE_EXP_ROOT if args.smoke else exp_root_skenario(scenario))
    os.makedirs(exp_root, exist_ok=True)

    print(f"[v_2] exp={name} agen={args.agent_fleet}/{args.mode} vs RLNegotiator "
          f"seed={cfg.seed} (skenario={scenario}) -> {exp_root}")
    if args.online_log and not args.online_log_full:
        # Log online RAMPING: tanpa snapshot memori per-update (hemat disk;
        # deret loss tetap terekam utuh).
        import simulation.training.orchestrator as _orch
        _orch._memory_snapshot = lambda trainer: None
        print("[v_2] online-log RAMPING (tanpa snapshot memori)")
    elif args.online_log_full:
        # PENUH: snapshot memori dipertahankan -> state+action per step ikut tercatat,
        # sehingga output aktor bisa di-trace (bukan cuma kurva loss). BOROS DISK.
        print("[v_2] online-log PENUH (snapshot memori AKTIF — boros disk)")
    run_single_experiment(
        name,
        cfg,
        seed=cfg.seed,
        render_runs=False,
        enable_trace=not args.lean,
        enable_online_log=(not args.lean) or args.online_log or args.online_log_full,
        exp_root=exp_root,
        config_override_label=os.path.relpath(cfg_path, ROOT),
    )
    tulis_laporan(os.path.join(exp_root, name), sampler,
                  n_days=cfg.time.n_days, label=scenario)


if __name__ == "__main__":
    cli()
