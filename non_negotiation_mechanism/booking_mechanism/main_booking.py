# -*- coding: utf-8 -*-
"""Main simulasi mekanisme BOOKING statis (tanpa negosiasi) — 1 run = 1 simulasi.

Pakai:
    python non_negotiation_mechanism/booking_mechanism/main_booking.py \
        --name ref_n10_upd1_s42 --config desain_eksperiment/configs/ref_n10_upd1_s42.json --seed 42

Output: non_negotiation_mechanism/booking_mechanism/data/<name>/ (format: ../SCHEMA.md).
"""
import argparse, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from run_experiment import load_config_from_path                      # noqa: E402
from non_negotiation_mechanism.common.nonneg_engine import run_nonneg_once  # noqa: E402

MECH = "booking"

def main() -> None:
    ap = argparse.ArgumentParser(description="Simulasi non-negosiasi: booking statis.")
    ap.add_argument("--name", required=True, help="Nama run (folder output).")
    ap.add_argument("--config", default=None, help="JSON config override (deep-merge).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    a = ap.parse_args()
    cfg = load_config_from_path(a.config) if a.config else load_config_from_path(None)
    run_dir = os.path.join(a.out_root, a.name)
    run_nonneg_once(cfg, MECH, run_dir, a.seed if a.seed is not None else cfg.seed,
                    config_label=a.config or "(default)")

if __name__ == "__main__":
    main()
