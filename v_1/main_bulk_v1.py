# -*- coding: utf-8 -*-
"""Titik masuk v_1 untuk agen negosiasi, memakai dunia korpus lama.

Argumennya sama persis dengan ``run_experiment_bulk_per_config.py`` di root,
sebab skrip itu yang dipanggil setelah tambalan terpasang. Yang berubah hanya
perakitan dunia, dan hanya selama proses ini hidup. Tidak ada kode di luar
``v_1/`` yang disentuh.


Pakai dari root project::

    python v_1/main_bulk_v1.py --name ref_n10_upd1_s42 \\
        --config v_1/configs/ref_n10_upd1_s42.json \\
        --only 6 8 9 --modes PADDED --out-root v_1/data/experiment_bulk
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overrides import pasang_dunia_lama  # noqa: E402

pasang_dunia_lama()

from run_experiment_bulk_per_config import main  # noqa: E402

if __name__ == "__main__":
    main()
