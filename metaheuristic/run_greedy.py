# -*- coding: utf-8 -*-
"""Runner baseline METAHEURISTIC greedy-vs-greedy — 1 invokasi = 1 eksperimen.

Dua moda:

1. **Skenario non-RC** (16 config desain lama):
   python metaheuristic/run_greedy.py --config desain_eksperiment/configs/ref_n10_upd1_s42.json --name ref_n10_upd1_s42
   → metaheuristic/data/non_rc/<name>/ (format identik eksperimen negosiasi:
     experiment_meta.json + logs/ + metrices/aggregate_metrics.json)

2. **Dunia real-case v2** (3 run):
   python metaheuristic/run_greedy.py --v2-run 1
   → metaheuristic/data/real_case_v2/run{n}_padded_greedy/
   (memasang override booking empiris v2 + normalisasi time_weight — dunia
   identik eksperimen nego/non-neg v2)

Pasangan agen: fleet GreedyMH vs spklu GreedyMH, mode PADDED (greedy tanpa
jaringan → mode-invariant; 1 pasangan per skenario, bukan 18 — desain v2 §5).
Marker lengkap: experiment_meta.json (ditulis paling akhir).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_V2 = os.path.join(_ROOT, "real_case_common")
# ROOT dulu (nama bentrok "main" = root), lalu folder ini & v2 (modul lokal).
sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or os.getcwd()) not in (_HERE, _ROOT, _V2)]
sys.path.insert(0, _V2)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from main import load_config_from_path            # noqa: E402 (main.py ROOT)
from run_experiment import run_single_experiment  # noqa: E402

from registry_patch import install_greedy_agent_type  # noqa: E402
from greedy_agent import AGENT_GREEDY                 # noqa: E402


def _apply_greedy_pair(cfg, mode: str = "PADDED"):
    fleet = dataclasses.replace(cfg.agent.fleet, agent_type=AGENT_GREEDY, mode=mode)
    spklu = dataclasses.replace(cfg.agent.spklu, agent_type=AGENT_GREEDY, mode=mode)
    return dataclasses.replace(cfg, agent=dataclasses.replace(cfg.agent, fleet=fleet, spklu=spklu))


def cli() -> None:
    ap = argparse.ArgumentParser(description="Baseline metaheuristic greedy-vs-greedy (1 exp).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="Path config JSON skenario non-RC (desain lama).")
    g.add_argument("--v2-run", type=int, choices=(1, 2, 3),
                   help="Dunia real-case v2 run{n} (config + override booking v2).")
    ap.add_argument("--name", default=None, help="Nama exp (wajib utk --config).")
    ap.add_argument("--exp-root", default=None)
    args = ap.parse_args()

    install_greedy_agent_type()

    if args.v2_run is not None:
        from overrides import install_v2_bookings, normalize_time_weight  # noqa: E402 (modul v2)
        from v2_common import build_sampler                               # noqa: E402
        cfg_path = os.path.join(_V2, "configs", f"run{args.v2_run}.json")
        cfg = load_config_from_path(cfg_path)
        cfg = normalize_time_weight(cfg)
        install_v2_bookings(build_sampler())
        name = args.name or f"run{args.v2_run}_padded_greedy"
        exp_root = args.exp_root or os.path.join(_HERE, "data", "real_case_v2")
    else:
        if not args.name:
            ap.error("--name wajib bila memakai --config")
        cfg_path = args.config
        cfg = load_config_from_path(cfg_path)
        name = args.name
        exp_root = args.exp_root or os.path.join(_HERE, "data", "non_rc")

    cfg = _apply_greedy_pair(cfg)
    os.makedirs(exp_root, exist_ok=True)

    print(f"[metaheuristic] exp={name} GreedyMH-vs-GreedyMH seed={cfg.seed} -> {exp_root}")
    run_single_experiment(
        name,
        cfg,
        seed=cfg.seed,
        render_runs=False,
        exp_root=exp_root,
        config_override_label=os.path.relpath(cfg_path, _ROOT),
    )


if __name__ == "__main__":
    cli()
