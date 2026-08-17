# -*- coding: utf-8 -*-
"""Generator config eksperimen network_scaling (TERISOLASI dari kode inti).

Menyalin config real-case v2 lalu menyuntikkan lebar dan kedalaman jaringan SISI
FLEET (``agent.fleet.hyperparameters.hidden_dim`` / ``hidden_layers``). Sisi
SPKLU sengaja TIDAK diubah supaya lingkungan dan lawan identik dengan eksperimen
utama; satu-satunya variabel yang berubah adalah kapasitas jaringan fleet.

Dua keluarga skenario, sama dengan v_2 tetapi tanpa varian mahalkompen:

* ``medium_occupancy`` — permintaan 185 booking, okupansi sekitar 26 persen.
* ``high_occupancy``  — permintaan 366 booking, okupansi sekitar 47 persen.

Sumbernya ``v_2/configs/run{n}_<skenario>.json``, jadi ``python v_2/gen_configs.py``
harus dijalankan lebih dulu.

Pakai:
    python network_scaling/gen_configs.py --hidden 512 --layers 3
    python network_scaling/gen_configs.py --hidden 512 --layers 3 --variant high_occupancy

Keluaran: ``network_scaling/configs/run{n}_<skenario>_h{H}x{L}.json``.
Tidak ada config yang di-pre-generate di sini karena arsitektur mana yang dipakai
masih keputusan terbuka.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BASE_CFG_DIR = os.path.join(ROOT, "v_2", "configs")
OUT_DIR = os.path.join(HERE, "configs")


# Varian skenario: (config sumber korpus utama, override tambahan).
# Dua keluarga skenario, mengikuti v_2 tetapi TANPA varian mahalkompen: yang
# diuji di sini adalah ukuran jaringan agen, bukan aturan kompensasi. Nama dan
# angka permintaannya sama persis dengan v_2 supaya hasilnya sebanding.
VARIANTS = {
    "medium_occupancy": "run{n}_medium_occupancy.json",
    "high_occupancy": "run{n}_high_occupancy.json",
}

# Ulangan mengikuti v_2 yang menyediakan run1 sampai run40. Di sini dipakai
# lima ulangan pertama karena satu kombinasi ukuran jaringan sudah mahal.
RUNS = (1, 2, 3, 4, 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--variant", default="medium_occupancy",
                    choices=sorted(VARIANTS))
    ap.add_argument("--opp-hidden", type=int, default=None,
                    help="hidden_dim sisi SPKLU (lawan); default: tidak diubah (128x2)")
    ap.add_argument("--opp-layers", type=int, default=None,
                    help="hidden_layers sisi SPKLU (lawan)")
    args = ap.parse_args()
    if (args.opp_hidden is None) != (args.opp_layers is None):
        raise SystemExit("--opp-hidden dan --opp-layers harus dipakai bersamaan")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"h{args.hidden}x{args.layers}"
    if args.opp_hidden is not None:
        tag += f"_opp{args.opp_hidden}x{args.opp_layers}"
    src_tpl = VARIANTS[args.variant]
    vpref = f"{args.variant}_"
    for n in RUNS:
        src = os.path.join(BASE_CFG_DIR, src_tpl.format(n=n))
        if not os.path.exists(src):
            raise SystemExit(f"Config sumber tidak ada: {src} — "
                             f"jalankan `python v_2/gen_configs.py` dulu.")
        with open(src, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        agent = cfg.setdefault("agent", {})
        fleet = agent.setdefault("fleet", {})
        hp = fleet.setdefault("hyperparameters", {})
        hp["hidden_dim"] = args.hidden
        hp["hidden_layers"] = args.layers
        if args.opp_hidden is not None:
            shp = agent.setdefault("spklu", {}).setdefault("hyperparameters", {})
            shp["hidden_dim"] = args.opp_hidden
            shp["hidden_layers"] = args.opp_layers
        dst = os.path.join(OUT_DIR, f"run{n}_{vpref}{tag}.json")
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)
        opp_note = ("spklu tetap default" if args.opp_hidden is None
                    else f"spklu hidden {args.opp_hidden} x {args.opp_layers}")
        print(f"OK run{n} [{args.variant}] -> {os.path.relpath(dst, ROOT)} "
              f"(fleet hidden {args.hidden} x {args.layers}; {opp_note}; "
              f"N={cfg['app_user_booking']['initial_booking_count']})")


if __name__ == "__main__":
    main()
