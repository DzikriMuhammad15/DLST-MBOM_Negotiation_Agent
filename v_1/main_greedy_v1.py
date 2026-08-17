# -*- coding: utf-8 -*-
"""Titik masuk v_1 untuk baseline greedy lawan greedy, memakai dunia lama.

Tambalan dipasang lebih dulu supaya dunia yang dirakit sama dengan korpus
non-*real-case* lama, lalu ``metaheuristic/run_greedy.py`` dipanggil apa adanya.
Tidak ada kode di luar ``v_1/`` yang disentuh.

Pakai dari root project::

    python v_1/main_greedy_v1.py --config v_1/configs/ref_n10_upd1_s42.json \\
        --name ref_n10_upd1_s42 --exp-root v_1/data/greedy
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

sys.path.insert(0, os.path.join(_ROOT, "metaheuristic"))
from metaheuristic.run_greedy import cli  # noqa: E402

if __name__ == "__main__":
    cli()
