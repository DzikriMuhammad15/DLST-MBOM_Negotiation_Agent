# -*- coding: utf-8 -*-
"""Helper bersama real-case Bandung (dipakai oleh v_2/ dan network_scaling/) — path & sampler & agen.

Folder ini (``real_case_common/``) TIDAK punya entry-point ``main.py`` sendiri
(berbeda dari versi asalnya di proyek ``nyoba_general``), jadi tidak ada
tabrakan nama modul dengan ``main.py`` di root ``clean_code/``.
:func:`fix_sys_path` dipertahankan (idempoten, tanpa efek berbahaya) supaya
pola pemanggilan di ``v_2/`` dan ``network_scaling/`` tetap identik dengan versi
asalnya — modul lokal ``booking_sampler``/``overrides``/``v2_common`` tetap
ter-import lewat ``sys.path.insert(0, real_case_common_dir)`` di skrip pemanggil.
"""
from __future__ import annotations

import json
import os
import sys

V2_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(V2_DIR, ".."))


def fix_sys_path() -> None:
    """ROOT dulu, lalu folder v2 (lihat docstring modul). Idempoten."""
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) not in (V2_DIR, ROOT)]
    sys.path.insert(0, V2_DIR)
    sys.path.insert(0, ROOT)


def build_sampler(empiris: bool = False):
    """Sampler booking dari mapping stasiun `configs/v2_world_meta.json`.

    Args:
        empiris: bila ``True``, pakai sampler EMPIRIS lama (durasi dan jam mulai
            diundi dari histogram data dengan bobot, lalu dibulatkan ke grid
            terdekat dan dipotong pada 2 jam). Bila ``False`` (bawaan), pakai
            sampler DISTRIBUSI HASIL FIT: durasi dan jam mulai ditarik dari
            keluarga distribusi juara di
            ``reference_data_simulation/fitting/best_parameters.json``.
            Sampling stasiun dan port identik pada kedua pilihan.
    """
    from booking_sampler import EmpiricalBookingSampler  # import setelah fix path
    meta_path = os.path.join(V2_DIR, "configs", "v2_world_meta.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    stations = meta["stations_13_pln"]
    argumen = dict(
        station_ports={s["station_id"]: [p["port_id"] for p in s["ports"]] for s in stations},
        station_csv_ids={s["station_id"]: s["csv_id"] for s in stations},
    )
    if empiris:
        print("[sampler] EMPIRIS (--empiris): durasi & jam mulai dari histogram data")
        return EmpiricalBookingSampler(**argumen)
    from fitted_booking_sampler import FittedBookingSampler
    sampler = FittedBookingSampler(**argumen)
    print(f"[sampler] DISTRIBUSI HASIL FIT (bawaan): "
          f"durasi ~ {sampler.durasi_keluarga}, jam mulai ~ {sampler.jam_keluarga}")
    return sampler


def apply_agents(cfg, fleet_type: str, mode: str):
    """Fleet <fleet_type> vs SPKLU RLNegotiator, mode sama dua sisi (pola bulk lama)."""
    import dataclasses
    fleet = dataclasses.replace(cfg.agent.fleet, agent_type=fleet_type, mode=mode)
    spklu = dataclasses.replace(cfg.agent.spklu, agent_type="RLNegotiator", mode=mode)
    return dataclasses.replace(cfg, agent=dataclasses.replace(cfg.agent, fleet=fleet, spklu=spklu))
