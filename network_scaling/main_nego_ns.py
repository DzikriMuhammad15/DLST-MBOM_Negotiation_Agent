# -*- coding: utf-8 -*-
"""Entry NEGOSIASI eksperimen network_scaling (korpus #12) — 1 invokasi = 1 eksperimen.

Salinan terpangkas dari v_2/main_nego.py, dengan tiga perbedaan yang disengaja:
  1. Config dibaca dari network_scaling/configs/run{n}_<skenario>_h{H}x{L}.json
     (hasil gen_configs.py) — berisi override hidden_dim/hidden_layers SISI
     FLEET; sisi SPKLU tetap default agar lawannya identik.
  2. Output ke network_scaling/data/<skenario>_h{H}x{L}/negotiation/real_case_v2/
     run{n}_padded_{agen}/ — terisolasi dari korpus utama (v_2/).
  3. Hanya DUA keluarga skenario, yaitu medium_occupancy dan high_occupancy;
     varian mahalkompen tidak dipakai karena yang diuji di sini ukuran jaringan
     agen, bukan aturan kompensasi.

Seluruh mekanisme lain (dunia Bandung terkalibrasi, injeksi booking v2,
normalisasi time_weight, fixed_mechanism) identik dengan eksperimen utama.

Pakai (dari root repo clean_code/):
    python network_scaling/main_nego_ns.py --agent-fleet DLST --hidden 512 --layers 3 --run 1 --lean
    # skenario permintaan tinggi + log online training (kurva loss) tanpa trace:
    python network_scaling/main_nego_ns.py --agent-fleet DLSTMBOM --hidden 512 --layers 3 --run 1 --variant high_occupancy --lean --online-log
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
V2_DIR = os.path.join(ROOT, "real_case_common")
sys.path.insert(0, V2_DIR)
from v2_common import apply_agents, build_sampler, fix_sys_path  # noqa: E402

fix_sys_path()

from main import load_config_from_path            # noqa: E402  (main.py ROOT)
from run_experiment import run_single_experiment  # noqa: E402

from overrides import install_v2_bookings, normalize_time_weight  # noqa: E402
from booking_report import tulis_laporan  # noqa: E402

AGENT_SLUGS = {
    "DLST": "dlst",
    "DLSTMBOM": "dlstmbom",
    "DLSTMBOMNewMechanism": "dlstmbomnew",
}


def cli() -> None:
    ap = argparse.ArgumentParser(description="Nego network-scaling real-case v2 (1 exp/invokasi).")
    ap.add_argument("--agent-fleet", required=True, choices=sorted(AGENT_SLUGS))
    ap.add_argument("--mode", default="PADDED", choices=("PADDED", "ENCODER"))
    ap.add_argument("--run", type=int, default=1, choices=(1, 2, 3, 4, 5))
    ap.add_argument("--hidden", type=int, required=True,
                    help="hidden_dim fleet (harus sudah di-generate gen_configs.py)")
    ap.add_argument("--layers", type=int, required=True,
                    help="hidden_layers fleet")
    ap.add_argument("--variant", default="medium_occupancy",
                    choices=("medium_occupancy", "high_occupancy"),
                    help="Keluarga skenario, sama dengan v_2: medium_occupancy "
                         "(permintaan 185 booking) atau high_occupancy "
                         "(permintaan 366 booking).")
    ap.add_argument("--empiris", action="store_true",
                    help="Pakai sampler EMPIRIS lama untuk durasi & jam mulai "
                         "booking (undian berbobot histogram, dipotong 2 jam). "
                         "Bawaan tanpa flag ini: sampling dari distribusi "
                         "teoretis hasil fit.")
    ap.add_argument("--lean", action="store_true",
                    help="Matikan trace & online-log (aggregate_metrics tetap ditulis).")
    ap.add_argument("--online-log", action="store_true",
                    help="Nyalakan log online training (deret loss per-update) MESKI "
                         "--lean; trace tetap mengikuti --lean (hemat disk).")
    ap.add_argument("--opp-hidden", type=int, default=None,
                    help="hidden_dim sisi SPKLU (lawan); config harus di-generate "
                         "dgn gen_configs.py --opp-hidden yang sama.")
    ap.add_argument("--opp-layers", type=int, default=None)
    args = ap.parse_args()
    if (args.opp_hidden is None) != (args.opp_layers is None):
        raise SystemExit("--opp-hidden dan --opp-layers harus dipakai bersamaan")

    tag = f"h{args.hidden}x{args.layers}"
    if args.opp_hidden is not None:
        tag += f"_opp{args.opp_hidden}x{args.opp_layers}"
    vpref = f"{args.variant}_"
    cfg_path = os.path.join(HERE, "configs", f"run{args.run}_{vpref}{tag}.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"Config {cfg_path} belum ada — jalankan gen_configs.py dulu.")
    cfg = load_config_from_path(cfg_path)
    cfg = normalize_time_weight(cfg)
    cfg = apply_agents(cfg, args.agent_fleet, args.mode)

    fh = cfg.agent.fleet.hyperparameters
    sh = cfg.agent.spklu.hyperparameters
    assert fh.hidden_dim == args.hidden and fh.hidden_layers == args.layers, \
        f"override fleet tidak terpasang: {fh.hidden_dim}x{fh.hidden_layers}"
    if args.opp_hidden is not None:
        assert sh.hidden_dim == args.opp_hidden and sh.hidden_layers == args.opp_layers, \
            f"override spklu tidak terpasang: {sh.hidden_dim}x{sh.hidden_layers}"
    diminta = {"medium_occupancy": 185, "high_occupancy": 366}[args.variant]
    assert cfg.app_user_booking.initial_booking_count == diminta, \
        (f"permintaan booking tidak sesuai skenario {args.variant}: "
         f"{cfg.app_user_booking.initial_booking_count} != {diminta}")
    opp_lbl = "default" if args.opp_hidden is None else "DISKALAKAN"
    print(f"[net-scaling] fleet net {fh.hidden_dim}x{fh.hidden_layers} | "
          f"spklu net {sh.hidden_dim}x{sh.hidden_layers} ({opp_lbl}) | "
          f"varian={args.variant} n_days={cfg.time.n_days} "
          f"N={cfg.app_user_booking.initial_booking_count}")

    sampler = build_sampler(empiris=args.empiris)
    install_v2_bookings(sampler)

    name = f"run{args.run}_{args.mode.lower()}_{AGENT_SLUGS[args.agent_fleet]}"
    exp_root = os.path.join(HERE, "data", f"{vpref}{tag}", "negotiation", "real_case_v2")
    os.makedirs(exp_root, exist_ok=True)

    enable_online = (not args.lean) or args.online_log
    if args.online_log:
        # Log online RAMPING: matikan snapshot memori per-update (insiden disk
        # 2026-07-19: snapshot membuat log run 30-hari fase-sulit membengkak
        # ~GB). Deret loss (aggregate_loss, epoch_losses, result) tetap utuh.
        import simulation.training.orchestrator as _orch
        _orch._memory_snapshot = lambda trainer: None
        print("[net-scaling] online-log RAMPING (tanpa snapshot memori)")
    print(f"[net-scaling] exp={name} agen={args.agent_fleet}/{args.mode} vs RLNegotiator "
          f"seed={cfg.seed} (fixed_mechanism, fleet {tag}, varian {args.variant}, "
          f"trace={not args.lean}, online_log={enable_online}) -> {exp_root}")
    run_single_experiment(
        name,
        cfg,
        seed=cfg.seed,
        render_runs=False,
        enable_trace=not args.lean,
        enable_online_log=enable_online,
        exp_root=exp_root,
        config_override_label=os.path.relpath(cfg_path, ROOT),
    )
    tulis_laporan(os.path.join(exp_root, name), sampler,
                  n_days=cfg.time.n_days, label=args.variant)


if __name__ == "__main__":
    cli()
