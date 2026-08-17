# -*- coding: utf-8 -*-
"""Entry GREEDY-vs-GREEDY v_2 (real-case fixed_mechanism) — 1 invokasi = 1 exp.

Baseline metaheuristic pada dunia real-case v2 terkalibrasi.

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
    python v_2/main_greedy.py --run 1
    python v_2/main_greedy.py --run 1 --skenario high_occupancy
    python v_2/main_greedy.py --run 1 --skenario mahalkompen_medium_occupancy

Pasangan agen: fleet GreedyMH vs spklu GreedyMH, mode PADDED (greedy tanpa
jaringan sehingga mode-invariant).

Output: v_2/data_<skenario>/negotiation/real_case_v2/run{n}_padded_greedy/
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
V2_DIR = os.path.join(ROOT, "real_case_common")
MH_DIR = os.path.join(ROOT, "metaheuristic")
sys.path.insert(0, MH_DIR)
sys.path.insert(0, V2_DIR)
from v2_common import build_sampler, fix_sys_path  # noqa: E402

fix_sys_path()

from main import load_config_from_path            # noqa: E402  (main.py ROOT)
from run_experiment import run_single_experiment  # noqa: E402

from overrides import install_v2_bookings, normalize_time_weight  # noqa: E402
from booking_report import tulis_laporan  # noqa: E402
from registry_patch import install_greedy_agent_type  # noqa: E402 (metaheuristic)
from greedy_agent import AGENT_GREEDY                 # noqa: E402

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


def _apply_greedy_pair(cfg, mode: str = "PADDED"):
    fleet = dataclasses.replace(cfg.agent.fleet, agent_type=AGENT_GREEDY, mode=mode)
    spklu = dataclasses.replace(cfg.agent.spklu, agent_type=AGENT_GREEDY, mode=mode)
    return dataclasses.replace(cfg, agent=dataclasses.replace(cfg.agent, fleet=fleet, spklu=spklu))


def cli() -> None:
    ap = argparse.ArgumentParser(description="Greedy-vs-greedy fixed_mechanism rc v2 (1 exp).")
    ap.add_argument("--run", type=int, default=1, choices=tuple(range(1, 41)),
                    help="Run replikasi 1-40 (config run{n}_<skenario>.json).")
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
    ap.add_argument("--lean", action="store_true", help="Matikan trace & online-log (hemat disk).")
    ap.add_argument("--exp-root", default=None)
    ap.add_argument("--seed-override", type=int, default=None,
                    help="ADITIF 2026-07-29: timpa seed+armada awal (sinkronisasi ke seed "
                         "yang sudah dipakai negosiasi pada run yang sama). Dunia lain identik.")
    args = ap.parse_args()

    install_greedy_agent_type()

    scenario = args.skenario
    cfg_path = os.path.join(HERE, "configs", f"run{args.run}_{scenario}.json")
    if args.seed_override is not None:
        from seed_override_util import resolve_config_path
        cfg_path = resolve_config_path(cfg_path, args.seed_override)
    cfg = load_config_from_path(cfg_path)
    cfg = normalize_time_weight(cfg)
    cfg = _apply_greedy_pair(cfg)

    sampler = build_sampler(empiris=args.empiris)
    install_v2_bookings(sampler)

    exp_root = args.exp_root or os.path.join(
        HERE, folder_data(scenario), "negotiation", "real_case_v2")
    os.makedirs(exp_root, exist_ok=True)

    name = f"run{args.run}_padded_greedy"
    print(f"[v_2] exp={name} GreedyMH vs GreedyMH seed={cfg.seed} "
          f"(skenario={scenario}) -> {exp_root}")
    run_single_experiment(
        name,
        cfg,
        seed=cfg.seed,
        render_runs=False,
        enable_trace=not args.lean,
        enable_online_log=not args.lean,
        exp_root=exp_root,
        config_override_label=os.path.relpath(cfg_path, ROOT),
    )
    tulis_laporan(os.path.join(exp_root, name), sampler,
                  n_days=cfg.time.n_days, label=scenario)


if __name__ == "__main__":
    cli()
