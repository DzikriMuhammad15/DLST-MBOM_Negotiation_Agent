# -*- coding: utf-8 -*-
"""Util ADITIF (2026-07-29): override seed+armada awal pada config sebelum load.

Dipakai HANYA oleh main_greedy.py & main_nonneg_v2.py untuk menyinkronkan seed
Greedy/As-is/Booking terhadap seed yang SUDAH DIPAKAI negosiasi pada nomor run
yang sama (dibaca dari experiment_meta.json hasil nego, bukan dari config).

TIDAK dipakai/diimpor oleh main_nego.py — mekanisme negosiasi tidak disentuh.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "real_case_common"))
from build_world_config import sample_taxis  # noqa: E402


def resolve_config_path(cfg_path: str, seed_override: "int | None") -> str:
    """Return `cfg_path` unchanged bila `seed_override is None`; else tulis config
    sementara dengan seed & armada awal ditimpa (dunia lain identik) dan return
    path file sementara itu.
    """
    if seed_override is None:
        return cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["seed"] = seed_override
    raw["taxi_spec"]["initial_taxi_state"] = sample_taxis(seed_override)
    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="seed_override_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=1)
    return tmp_path
