# -*- coding: utf-8 -*-
"""Titik masuk v_1 untuk mekanisme non-negosiasi, memakai dunia korpus lama.

Mekanisme tanpa-pemesanan dan pemesanan-statis tidak memakai reinforcement
learning sama sekali, sehingga hasilnya deterministik dan menjadi alat ukur
paling bersih untuk memastikan dunianya benar. Pada dunia korpus lama, mekanisme
pemesanan statis terbukti mereproduksi Tabel V.1 laporan persis pada seluruh
lima kolom untuk empat dunia.

Pakai dari root project::

    python v_1/main_nonneg_v1.py --mech booking --name ref_n10_upd1_s42 \\
        --config v_1/configs/ref_n10_upd1_s42.json --seed 42 \\
        --out-root v_1/data/booking
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overrides import pasang_dunia_lama  # noqa: E402

pasang_dunia_lama()

from run_experiment import load_config_from_path  # noqa: E402
from non_negotiation_mechanism.common.nonneg_engine import run_nonneg_once  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Non-negosiasi v_1 pada dunia korpus lama.")
    ap.add_argument("--mech", required=True, choices=("as_is", "booking"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()

    cfg = load_config_from_path(a.config)
    run_dir = os.path.join(a.out_root, a.name)
    run_nonneg_once(cfg, a.mech, run_dir, a.seed, config_label=a.config)


if __name__ == "__main__":
    main()
