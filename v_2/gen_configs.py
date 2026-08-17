# -*- coding: utf-8 -*-
"""Generator config real-case v2 — EMPAT keluarga skenario, 40 run masing-masing.

    python v_2/gen_configs.py            # tulis seluruh 160 config
    python v_2/gen_configs.py --skenario medium_occupancy

Menghasilkan ``configs/run{1..40}_<skenario>.json`` untuk keempat skenario:

===========================  ==========  ==========================  ===================
Skenario                     Permintaan  Harga lantai saat digeser   Okupansi diharapkan
===========================  ==========  ==========================  ===================
medium_occupancy                    185  dibiarkan                   sekitar 26 persen
high_occupancy                      366  dibiarkan                   sekitar 47 persen
mahalkompen_medium_occupancy        185  dilarang (prune)            sekitar 26 persen
mahalkompen_high_occupancy          366  dilarang (prune)            sekitar 47 persen
===========================  ==========  ==========================  ===================

Dari mana angka 185 dan 366
---------------------------
Keduanya statistik dari sebaran **jumlah transaksi per hari seluruh SPKLU
Bandung sepanjang 2024**, yaitu 366 nilai harian dari 22.514 transaksi, lalu
dikalikan panjang horizon 3 hari:

* **185** = rata-rata harian 61,51 transaksi x 3 hari. Beban rata-rata.
* **366** = kuantil 95 harian 122 transaksi x 3 hari. Beban puncak.

Kedua angka itu dihitung ulang kapan saja dengan::

    python reference_data_simulation/preprocessing/laju_permintaan.py

yang menulis ``laju_permintaan.json`` beserta gambar laju hariannya. Perlu
dicatat, volume seluruh Bandung dimampatkan ke 14 port dunia simulasi, yaitu
16,857 kali kepadatan nyata; pemampatan itu disengaja dan harus disebut.

Yang dikendalikan adalah **permintaan**, bukan okupansi. Dalam pemodelan sistem
berkapasitas, yang ditetapkan adalah proses permintaan sedangkan tingkat
keterpakaian adalah keluaran. Memaksa okupansi mencapai angka tertentu berarti
membengkokkan permintaan demi mengejar keluaran. Karena itu kolom "okupansi
diharapkan" di atas adalah **perkiraan hasil**, bukan sasaran; angka yang
sebenarnya dicatat per run pada ``metrices/booking_generation.json``.

Nama skenario karena itu menyebut tingkat permintaan, dan istilah okupansi 22,5
persen maupun 50 persen pada rancangan lama sudah tidak dipakai lagi.

Yang berbeda antar skenario hanya dua saklar
--------------------------------------------
* ``app_user_booking.initial_booking_count`` — 185 atau 366.
* ``negotiation.prune_floor_price_on_shift`` — hanya diset ``true`` pada kedua
  skenario mahalkompen.

Seluruh field dunia lainnya, yaitu 13 SPKLU, 18 marking area, price_levels,
tarif 6.300 per km, dan kalibrasi ``g_max``/``tau``, diambil apa adanya dari
``configs/_template_dunia.json``.

Seed
----
``seed(run n) = 7 + 104729 * n``. Aturan sederhana yang dapat diperiksa ulang,
dengan pengali bilangan prima supaya jaraknya lebar. Seed **sama untuk run yang
sama lintas skenario**, sehingga keempat skenario berpasangan dan dapat
dibandingkan pada dunia yang seed-nya identik. ``taxi_spec.initial_taxi_state``
diturunkan deterministik dari seed lewat ``sample_taxis``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CFG_DIR = os.path.join(HERE, "configs")
TEMPLATE_PATH = os.path.join(CFG_DIR, "_template_dunia.json")

sys.path.insert(0, os.path.join(ROOT, "real_case_common"))
from build_world_config import sample_taxis  # noqa: E402

N_RUN = 40

# Permintaan booking per skenario (lihat docstring untuk asal angkanya).
PERMINTAAN = {
    "medium_occupancy": 185,
    "high_occupancy": 366,
    "mahalkompen_medium_occupancy": 185,
    "mahalkompen_high_occupancy": 366,
}
MAHALKOMPEN = {
    "medium_occupancy": False,
    "high_occupancy": False,
    "mahalkompen_medium_occupancy": True,
    "mahalkompen_high_occupancy": True,
}
SKENARIO = tuple(PERMINTAAN)


def seed_run(n: int) -> int:
    """Seed run ke-n; sama lintas skenario supaya keempatnya berpasangan."""
    return 7 + 104729 * int(n)


def tulis(skenario: str) -> int:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as fh:
        template = json.load(fh)

    os.makedirs(CFG_DIR, exist_ok=True)
    for n in range(1, N_RUN + 1):
        seed = seed_run(n)
        cfg = copy.deepcopy(template)
        cfg["seed"] = seed
        cfg["taxi_spec"]["initial_taxi_state"] = sample_taxis(seed)
        cfg["app_user_booking"]["initial_booking_count"] = PERMINTAAN[skenario]
        if MAHALKOMPEN[skenario]:
            cfg["negotiation"]["prune_floor_price_on_shift"] = True
        else:
            cfg["negotiation"].pop("prune_floor_price_on_shift", None)
        with open(os.path.join(CFG_DIR, f"run{n}_{skenario}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)
    print(f"[{skenario}] {N_RUN} config ditulis "
          f"(permintaan={PERMINTAAN[skenario]}, "
          f"mahalkompen={'ya' if MAHALKOMPEN[skenario] else 'tidak'})")
    return N_RUN


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skenario", default="semua",
                    choices=("semua",) + SKENARIO)
    args = ap.parse_args()

    daftar = SKENARIO if args.skenario == "semua" else (args.skenario,)
    total = sum(tulis(s) for s in daftar)
    print(f"TOTAL {total} config di {os.path.relpath(CFG_DIR, ROOT)}")


if __name__ == "__main__":
    main()
