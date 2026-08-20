"""Bulk experiment runner: the full 9-pair × 2-mode matrix for ONE config.

Where ``run_experiment.py`` runs a **single** agent-pairing simulation,
``run_experiment_bulk_per_config.py`` runs the **whole 18-experiment matrix**
(the 9 fleet/spklu agent pairings, each in both PADDED and ENCODER mode) for a
**single, fixed config**, and bundles them under one folder:

    data/experiment_bulk/<bulk_name>/
      exp01_padded_01_RL_vs_Rule/        # each is a full experiment folder
      exp02_padded_02_RL_vs_RL/          #   (logs/ registry/ history/ metrices/)
      ...
      exp18_encoder_09_DLST_vs_RL/
      bulk_meta.json                     # which config/seed/modes this bulk used
      bulk_results.csv                   # headline metrics for all 18 (comparable)
      bulk_results.json

The point: **one bulk = one config**. To compare configs, run several bulks with
different ``--config`` (and different ``--name``); every bulk uses the *same*
sub-experiment names, so cross-config comparison is a simple lookup by name.

**By default the heavy ``runs/`` artefacts (per-negotiation JSON + all PNG
diagrams) are NOT written** — they are redundant with the trace / training logs
and metrics, and rendering them (matplotlib) is the main compute cost. All
metrics, including the deal-only utilities in ``bulk_results.csv``, are computed
from the in-memory collector, so analysis stays exact. Pass ``--with-runs`` to
also render the ``runs/`` diagrams (same output as a single ``run_experiment.py``).

The 9 agent pairings (fleet vs spklu) are exactly the ones studied in the thesis
experiment (see ``laporan_eksperiment.md``):

    01 RL vs Rule            04 MBOM-NZS vs RL        07 DLST-MBOM-NZS vs RL
    02 RL vs RL              05 MBOM-New vs RL        08 DLST-MBOM-New vs RL
    03 MBOM vs RL            06 DLST-MBOM vs RL        09 DLST vs RL

Usage::

    # Default config, both modes (18 runs) -> data/experiment_bulk/baseline_1d/
    python run_experiment_bulk_per_config.py --name baseline_1d

    # A longer config (proper H1 test), fixed seed
    python run_experiment_bulk_per_config.py --name horizon_30d \\
        --config config_30d.json --seed 42

    # Only a subset of pairings, only ENCODER, keep going on errors
    python run_experiment_bulk_per_config.py --name quick --only 6 7 8 9 \\
        --modes ENCODER --continue-on-error

Each sub-experiment reuses :func:`run_experiment.run_single_experiment`, so its
folder layout, logs, metrics and diagrams are identical to a single
``run_experiment.py`` run — only the parent directory differs.

--------------------------------------------------------------------------
DIREKTORI KORPUS HASIL EKSPERIMEN YANG SUDAH DIJALANKAN (project clean_code)
--------------------------------------------------------------------------

Ini peta korpus **yang benar-benar ada di disk**, dicacah ulang pada 18 Agustus
2026, bukan salinan rencana. Skrip ini adalah mesin di balik korpus negosiasi
v_1, sebab ``v_1/main_bulk_v1.py`` memasang tambalan dunia lama lalu memanggil
:func:`main` di berkas ini. Korpus v_2 memakai titik masuknya sendiri.

**HARD CONSTRAINT. Data hasil di direktori-direktori berikut TIDAK BOLEH
disentuh.** Tidak dihapus, tidak dipindah, tidak ditimpa, tidak dinormalisasi.
Skrip analisis dan skrip penulis laporan hanya boleh MEMBACA.

1. Korpus v_1, skenario non-real-case, 13 dunia
------------------------------------------------

Titik masuk ``v_1/main_bulk_v1.py`` (negosiasi), ``v_1/main_nonneg_v1.py``
(non-negosiasi), ``v_1/main_greedy_v1.py`` (Greedy). Config di ``v_1/configs/``,
13 berkas, satu per dunia. Total 6,02 GB, 78 sel, seluruhnya lengkap.

    v_1/data/experiment_bulk/<dunia>/       # negosiasi, 13 dunia x 3 agen = 39
        exp06_padded_06_DLSTMBOM_vs_RL/
        exp08_padded_08_DLSTMBOMnew_vs_RL/  # DLST-MBOM Nego
        exp09_padded_09_DLST_vs_RL/
        bulk_results_exp06.csv  bulk_results_exp08.csv  bulk_results_exp09.csv
    v_1/data/as_is/<dunia>/                 # tanpa pemesanan, 13
    v_1/data/booking/<dunia>/               # pemesanan statis, 13
    v_1/data/greedy/<dunia>/                # Greedy lawan Greedy, 13

Ketiga belas ``<dunia>`` itu, dipakai persis sebagai nama folder maupun nama
config, adalah::

    ref_n10_upd1_s42            10 hari, dunia REFERENSI
    control_default_s42         default tanpa-belajar, 1 hari, upd5
    hz_n1_upd1_s42              horizon 1 hari
    hz_n5_upd1_s42              horizon 5 hari
    hz_n30_upd1_s42             horizon 30 hari
    uf_n10_upd2_s42             pembaruan tiap 2 episode
    uf_n10_upd5_s42             pembaruan tiap 5 episode
    sc_scarce_n10_upd1_s42      kelangkaan tinggi 6:2
    sc_abundant_n10_upd1_s42    sumber daya longgar 2:6
    soc_low_n10_upd1_s42        SoC rendah
    soc_high_n10_upd1_s42       SoC tinggi
    cc_alternating_n10_upd1_s42 lawan bergiliran
    cc_fixedopp_n10_upd1_s42    lawan beku

2. Korpus v_2, skenario real-case, 4 keluarga
----------------------------------------------

Titik masuk ``v_2/main_nego.py``, ``v_2/main_nonneg_v2.py``,
``v_2/main_greedy.py``. Config dibangkitkan ``v_2/gen_configs.py`` ke
``v_2/configs/``. Total 22,53 GB.

    v_2/data_medium_occupancy/                  # permintaan N = 185
    v_2/data_high_occupancy/                    # permintaan N = 366
    v_2/data_mahalkompen_medium_occupancy/      # 185, harga lantai dilarang saat digeser
    v_2/data_mahalkompen_high_occupancy/        # 366, harga lantai dilarang saat digeser

Isi tiap keluarga sama polanya::

    <keluarga>/negotiation/real_case_v2/run{n}_padded_{slug}/
        slug = dlst | dlstmbom | dlstmbomnew | directed | greedy
    <keluarga>/non_negotiation/real_case_v2/{as_is,booking}/run{n}/

Perhatikan bedanya penamaan. Folder negosiasi memakai ``run{n}_padded_<slug>``
sedangkan non-negosiasi memakai ``run{n}`` saja, dan **nomornya tidak diberi nol
di depan**, jadi ``run1`` bukan ``run01``.

3. Cacah ulangan, 30 dan BUKAN 40
----------------------------------

Kampanye 14 sampai 17 Agustus 2026 menghasilkan 40 ulangan per sel. Sesudah itu
**ulangan 31 sampai 40 dipindahkan keluar dari project ini oleh peneliti**, ke
``thesis_final/hehe_clean_code/`` dan ``thesis_final/hehe/``, beserta config-nya.
Keadaan sekarang, seluruh 28 sel v_2 berisi **run 1 sampai 30, kontigu, tanpa
lubang**, dan ``v_2/configs/`` berisi 120 config ditambah 2 berkas meta.

    per keluarga : 5 x 30 negosiasi + 2 x 30 non-negosiasi  = 210 run
    empat keluarga                                          = 840 run
    ditambah v_1                                            =  78 run
    TOTAL korpus clean_code                                 = 918 run

⚠️ ``laporan/_tooling/data_babv2.json`` masih merekam ``n = 40`` per sel, sebab
dihasilkan sebelum pemindahan. Angka di berkas itu **basi**. Jalankan ulang
``laporan/_tooling/data_babv2.py`` sebelum memakainya, dan turunkan ``N_ULANG``
dari 40 menjadi 30 lebih dulu, kalau tidak skrip akan berhenti dengan galat.

4. Isi tiap folder run
-----------------------

    experiment_meta.json / run_meta.json   parameter dan seed run
    metrices/aggregate_metrics.json        SUMBER ANGKA LAPORAN
    metrices/booking_generation.json       okupansi hasil, khusus v_2
    metrices/distribusi_durasi.png         diagram sampling, khusus v_2
    metrices/distribusi_jam.png
    logs/simulation/                       trace simulasi
    logs/online_training/*.jsonl           log pelatihan per episode
    history/  registry/  runs/

Sebuah run dianggap **lengkap** bila ``metrices/aggregate_metrics.json`` ada.
Untuk v_2 ditambah syarat ``metrices/booking_generation.json`` ada.

5. Pengendali kampanye
-----------------------

    _antrean/antrean_v1.sh  antrean_v2.sh  antrean_semua.sh
    _antrean/status.py  periksa.py  progres.sh  hentikan.sh
    _antrean/logs/

``periksa.py`` memeriksa kelengkapan dan kerusakan seluruh korpus, dan itulah
alat yang dipakai untuk menyatakan korpus utuh sesudah kampanye dihentikan lalu
dilanjutkan.

6. Yang TIDAK punya korpus
---------------------------

``network_scaling/`` hanya berisi ``gen_configs.py``, ``main_nego_ns.py``, dan
README. ``configs/`` maupun folder datanya kosong, jadi eksperimen network
scaling **belum pernah dijalankan di project ini**. Hasil network scaling yang
dibahas laporan berasal dari project lama ``nyoba_general``.

Aturan penulisan laporan beserta peta korpus versi naratifnya ada di
``laporan/aturan_penulisan_laporan.md``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from main import load_config_from_path
from run_experiment import run_single_experiment


BULK_ROOT = os.path.join("data", "experiment_bulk")

# The 9 fleet/spklu agent pairings (base_index, name_suffix, fleet_type, spklu_type).
# Fleet = first agent, spklu = second — matching the thesis experiment matrix.
BASE_PAIRS: List[Tuple[int, str, str, str]] = [
    (1, "01_RL_vs_Rule",        "RLNegotiator",         "Rule"),
    (2, "02_RL_vs_RL",          "RLNegotiator",         "RLNegotiator"),
    (3, "03_MBOM_vs_RL",        "MBOM",                 "RLNegotiator"),
    (4, "04_MBOMnzs_vs_RL",     "MBOMNonZeroSum",       "RLNegotiator"),
    (5, "05_MBOMnew_vs_RL",     "MBOMNewMechanism",     "RLNegotiator"),
    (6, "06_DLSTMBOM_vs_RL",    "DLSTMBOM",             "RLNegotiator"),
    (7, "07_DLSTMBOMnzs_vs_RL", "DLSTMBOMNonZeroSum",   "RLNegotiator"),
    (8, "08_DLSTMBOMnew_vs_RL", "DLSTMBOMNewMechanism", "RLNegotiator"),
    (9, "09_DLST_vs_RL",        "DLST",                 "RLNegotiator"),
]

MODES: Tuple[str, ...] = ("PADDED", "ENCODER")


def _apply_agents(cfg: Any, fleet_type: str, spklu_type: str, mode: str) -> Any:
    """Return a copy of ``cfg`` with fleet/spklu agent_type + mode overridden.

    Only ``agent_type`` and ``mode`` are changed; every other side field
    (``trainer_type``, hyperparameters, reward modes, critic enrichment) is
    inherited from the base config via :func:`dataclasses.replace` (so a
    config-driven ``trainer_type`` such as ``PG`` is respected). The original
    ``cfg`` is left untouched (G14).
    """
    fleet = dataclasses.replace(cfg.agent.fleet, agent_type=fleet_type, mode=mode)
    spklu = dataclasses.replace(cfg.agent.spklu, agent_type=spklu_type, mode=mode)
    agent = dataclasses.replace(cfg.agent, fleet=fleet, spklu=spklu)
    return dataclasses.replace(cfg, agent=agent)


def _make_bulk_dir(name: str, root: str = None) -> str:
    """Create ``<root>/<name>/`` (collision-suffixed) + return it. Default root = BULK_ROOT."""
    base = os.path.join(root or BULK_ROOT, name)
    bulk_dir = base
    suffix = 2
    while os.path.exists(bulk_dir):
        bulk_dir = f"{base}_{suffix}"
        suffix += 1
    os.makedirs(bulk_dir, exist_ok=True)
    return bulk_dir


def _plan(only: Optional[List[int]], modes: List[str]) -> List[Tuple[int, str, str, str, str]]:
    """Expand the matrix into ordered ``(num, mode, name, fleet_type, spklu_type)`` rows.

    ``num`` numbers 1..9 for the first mode and 10..18 for the second, matching the
    naming in the original thesis experiment.
    """
    pairs = BASE_PAIRS if not only else [p for p in BASE_PAIRS if p[0] in set(only)]
    rows: List[Tuple[int, str, str, str, str]] = []
    for mode_idx, mode in enumerate(modes):
        tag = "padded" if mode.upper() == "PADDED" else "encoder"
        offset = mode_idx * len(BASE_PAIRS)
        for base_idx, suffix, fleet, spklu in pairs:
            num = base_idx + offset
            name = f"exp{num:02d}_{tag}_{suffix}"
            rows.append((num, mode.upper(), name, fleet, spklu))
    return rows


def main() -> None:
    """Parse args, run the 9×modes matrix for one config, write a bulk summary."""

    parser = argparse.ArgumentParser(
        description="Run the full 9-pair × mode experiment matrix for ONE config."
    )
    parser.add_argument("--name", required=True,
                        help="Bulk name -> data/experiment_bulk/<name>/.")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config override applied to ALL sub-experiments "
                             "(deep-merged on the default). Omit for the default scenario.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (same for all 18).")
    parser.add_argument("--modes", nargs="+", default=list(MODES),
                        metavar="MODE", help="Modes to run (default: PADDED ENCODER).")
    parser.add_argument("--only", nargs="+", type=int, default=None, metavar="N",
                        help="Only these base pairings (1..9); default all 9.")
    parser.add_argument("--load-fleet", type=str, default=None, metavar="VERSION",
                        help="Warm-start every fleet from the GLOBAL registry "
                             "(must be compatible with each fleet agent_type).")
    parser.add_argument("--load-spklu", type=str, default=None, metavar="VERSION",
                        help="Warm-start every spklu from the GLOBAL registry.")
    parser.add_argument("--save-version", type=str, default=None, metavar="VERSION",
                        help="Version label for saved agents (default: per-run UTC ts).")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable the simulation trace log for all runs.")
    parser.add_argument("--no-training-log", action="store_true",
                        help="Disable the online training log for all runs.")
    parser.add_argument("--with-runs", action="store_true",
                        help="Also write the per-negotiation JSON + PNG diagrams "
                             "under each sub-experiment's runs/ (and copy them into "
                             "metrices/). DEFAULT: OFF — those are skipped in bulk "
                             "mode to save compute; all metrics (incl. deal-only "
                             "utilities) are still computed from the collector.")
    parser.add_argument("--out-root", type=str, default=None,
                        help="Root folder output (default: data/experiment_bulk).")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep going if a sub-experiment raises (record the error).")
    args = parser.parse_args()

    modes = [m.upper() for m in args.modes]
    for m in modes:
        if m not in ("PADDED", "ENCODER"):
            raise SystemExit(f"[bulk] unknown mode {m!r} (use PADDED / ENCODER).")

    bulk_dir = _make_bulk_dir(args.name, args.out_root)
    plan = _plan(args.only, modes)
    print(f"[bulk] {os.path.basename(bulk_dir)} -> {bulk_dir}")
    print(f"[bulk] config={args.config or 'DEFAULT'} seed={args.seed} "
          f"modes={modes} runs={'ON' if args.with_runs else 'OFF (default)'} "
          f"-> {len(plan)} experiments")

    # Load the base config once; each sub-experiment gets its own agent-override copy.
    if args.config is not None:
        print(f"[bulk] Loading base config override: {args.config}")
    base_cfg = load_config_from_path(args.config)

    results: List[Dict[str, Any]] = []
    t_start = time.time()
    for i, (num, mode, name, fleet, spklu) in enumerate(plan, start=1):
        print("\n" + "#" * 70)
        print(f"[bulk] ({i}/{len(plan)}) exp{num:02d} {mode}  {fleet} vs {spklu}")
        print("#" * 70)
        cfg = _apply_agents(base_cfg, fleet, spklu, mode)
        row: Dict[str, Any] = {"num": num, "mode": mode, "name": name,
                               "fleet": fleet, "spklu": spklu}
        t0 = time.time()
        try:
            summary = run_single_experiment(
                name,
                cfg,
                seed=args.seed,
                load_fleet=args.load_fleet,
                load_spklu=args.load_spklu,
                save_version=args.save_version,
                enable_trace=not args.no_trace,
                enable_online_log=not args.no_training_log,
                render_runs=args.with_runs,
                exp_root=bulk_dir,
                config_override_label=args.config,
            )
            agg = summary["aggregate"]
            row.update({
                "sim_id": summary["sim_id"],
                "deal_rate": agg.get("avg_deal_rate"),
                "u_fleet_all": agg.get("avg_utility_fleet"),
                "u_spklu_all": agg.get("avg_utility_spklu"),
                # Deal-only utilities come from the collector (in aggregate), so they
                # are exact whether or not runs/ diagrams were written.
                "u_fleet_deal": agg.get("avg_utility_fleet_deal"),
                "u_spklu_deal": agg.get("avg_utility_spklu_deal"),
                "welfare_deal": agg.get("social_welfare_deal"),
                "n_deal": agg.get("n_deal"),
                "n_no_deal": agg.get("n_no_deal"),
                "avg_episode_length": agg.get("avg_episode_length"),
                "fleet_inf_prop_ms": agg.get("fleet_avg_inference_ms_propose"),
                "fleet_inf_resp_ms": agg.get("fleet_avg_inference_ms_respond"),
                "fleet_inf_avg_ms": agg.get("fleet_avg_inference_ms_overall"),
                "negotiations": summary["negotiations"],
                "episodes": summary["episodes"],
                "error": None,
                "duration_s": round(time.time() - t0, 1),
            })
        except Exception as exc:  # noqa: BLE001
            row.update({"error": f"{type(exc).__name__}: {exc}",
                        "duration_s": round(time.time() - t0, 1)})
            print(f"[bulk] ERROR in {name}: {row['error']}")
            traceback.print_exc()
            if not args.continue_on_error:
                _write_summary(bulk_dir, args, modes, plan, results + [row])
                raise SystemExit(f"[bulk] aborted at {name} "
                                 f"(use --continue-on-error to skip failures).")
        results.append(row)

    _write_summary(bulk_dir, args, modes, plan, results)

    total = round(time.time() - t_start, 1)
    ok = sum(1 for r in results if not r.get("error"))
    print("\n" + "=" * 70)
    print(f"[bulk] DONE {ok}/{len(results)} OK in {total}s -> {bulk_dir}")
    _print_table(results)
    print(f"[bulk] Summary: {os.path.join(bulk_dir, 'bulk_results.csv')}")
    print("=" * 70)


def _write_summary(bulk_dir: str, args: Any, modes: List[str],
                   plan: List[Tuple[int, str, str, str, str]],
                   results: List[Dict[str, Any]]) -> None:
    """Write bulk_meta.json + bulk_results.json + bulk_results.csv."""
    meta = {
        "bulk_name": os.path.basename(bulk_dir),
        "config_override": args.config,
        "seed": args.seed,
        "modes": modes,
        "only": args.only,
        "render_runs": args.with_runs,
        "n_experiments": len(plan),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(bulk_dir, "bulk_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(os.path.join(bulk_dir, "bulk_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    cols = ["num", "mode", "fleet", "spklu", "deal_rate", "u_fleet_all", "u_spklu_all",
            "u_fleet_deal", "u_spklu_deal", "welfare_deal", "avg_episode_length",
            "fleet_inf_prop_ms", "fleet_inf_resp_ms", "fleet_inf_avg_ms",
            "n_deal", "n_no_deal", "negotiations", "episodes", "sim_id", "duration_s",
            "error"]
    with open(os.path.join(bulk_dir, "bulk_results.csv"), "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.get("num", 0)):
            writer.writerow(r)


def _print_table(results: List[Dict[str, Any]]) -> None:
    """Print a compact comparison table of the bulk results."""
    def f(x: Any, w: int = 8, p: int = 3) -> str:
        if x is None:
            return " " * (w - 1) + "-"
        try:
            return f"{float(x):{w}.{p}f}"
        except (TypeError, ValueError):
            return str(x).rjust(w)

    hdr = (f"{'#':>2} {'mode':7} {'fleet':22} {'deal':>6} {'uF_deal':>8} "
           f"{'uS_deal':>8} {'welf':>7} {'epLen':>6} {'nNeg':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: x.get("num", 0)):
        if r.get("error"):
            print(f"{r['num']:>2} {r['mode']:7} {r['fleet']:22} ERROR: {r['error']}")
            continue
        print(f"{r['num']:>2} {r['mode']:7} {r['fleet']:22} "
              f"{f(r.get('deal_rate'), 6, 3)} {f(r.get('u_fleet_deal'))} "
              f"{f(r.get('u_spklu_deal'))} {f(r.get('welfare_deal'), 7, 3)} "
              f"{f(r.get('avg_episode_length'), 6, 2)} {r.get('negotiations', '-'):>4}")


if __name__ == "__main__":
    main()
