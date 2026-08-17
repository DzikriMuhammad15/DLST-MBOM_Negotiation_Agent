# -*- coding: utf-8 -*-
"""Plot distribusi variabel data asli YANG DIPAKAI SIMULASI.

Hanya variabel yang benar-benar masuk ke simulasi yang diplot di sini. Variabel
lain yang ada di data tapi tidak dipakai (energi per sesi, SoC, hari, bulan,
konektor, pengguna) ada di ``filtering/eda/``.

Variabel yang dipakai simulasi
------------------------------
=== disampel acak per booking ===
1. Jam mulai booking   <- histogram jam transaksi nyata
2. Durasi booking      <- histogram durasi sesi nyata (dibulatkan ke kisi
                          setengah jam lalu dipotong pada 2 jam)
3. Stasiun (lalu port) <- bobot volume transaksi 13 stasiun dunia simulasi,
                          dihaluskan Laplace alpha = 1

=== dipakai sebagai angka tetap, bukan disampel ===
4. Harga per kWh       <- median seluruh transaksi (tarif PLN, praktis flat)
5. Bobot waktu order   <- indeks kemacetan TomTom Bandung per jam

Masukan  : ``filtering/output/bandung_trx.csv``, ``filtering/output/bandung_spklu.json``,
           ``real_data/time_weight_24.json``
Keluaran : ``plots/*.png``

    python reference_data_simulation/visualization/plot_variabel_simulasi.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REF_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REF_ROOT, "filtering", "output")
PLOT_DIR = os.path.join(HERE, "plots")

BIRU, ORANYE, HIJAU, ABU, MERAH = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#e34948"
TINTA, TINTA2, REDUP, GARIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10, "font.family": "sans-serif",
    "axes.edgecolor": GARIS, "axes.labelcolor": TINTA2, "text.color": TINTA,
    "xtick.color": REDUP, "ytick.color": REDUP, "axes.titlecolor": TINTA,
    "axes.grid": True, "grid.color": GARIS, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# Aturan konversi durasi milik engine (lihat booking_sampler.py).
GRID_MENIT = 30
GRID_MIN, GRID_MAKS = 1, 4
LAPLACE_ALPHA = 1.0


def rapikan(ax, judul, xlabel="", ylabel=""):
    ax.set_title(judul, fontsize=11, fontweight="bold", loc="left", pad=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    for sisi in ("left", "bottom"):
        ax.spines[sisi].set_color(GARIS)
    ax.tick_params(length=0)
    return ax


def simpan(fig, nama):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, nama), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> plots/{nama}")


def ke_grid(menit):
    """menit -> kisi setengah jam terdekat, dipotong pada [1, 4] grid."""
    g = np.maximum(GRID_MIN, (np.asarray(menit) / GRID_MENIT + 0.5).astype(int))
    return np.minimum(GRID_MAKS, g)


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "bandung_trx.csv"))
    for kolom in ("hour", "hour_frac", "durasi_min", "price_per_kwh"):
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    with open(os.path.join(DATA_DIR, "bandung_spklu.json"), "r", encoding="utf-8") as fh:
        spklu = json.load(fh)
    for e in spklu:
        e["ports"] = sum(int(cb["connectors"]) for cb in e["chargeboxes"])
    pln = sorted([e for e in spklu if "PLN" in str(e["nama"]).upper() and e["ports"] > 0],
                 key=lambda e: (e["ports"], e["nama"]))
    terpilih13 = pln[:13]

    # ── 1. Jam mulai booking ────────────────────────────────────────────────
    h = df["hour"].value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "Variabel 1 — Jam mulai booking (jam bulat, versi yang dipakai sampler)",
            "jam", "jumlah transaksi")
    ax.bar(h.index, h.values, color=BIRU, width=0.82)
    simpan(fig, "01_jam_mulai_bulat.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "Variabel 1 — Jam mulai booking (resolusi menit)",
            "jam (0-24)", "jumlah transaksi")
    ax.hist(df["hour_frac"].dropna(), bins=96, color=BIRU)
    simpan(fig, "02_jam_mulai_menit.png")

    # ── 2. Durasi booking ───────────────────────────────────────────────────
    durasi = df["durasi_min"].dropna()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "Variabel 2 — Durasi booking (menit, nilai mentah)",
            "menit", "jumlah transaksi")
    ax.hist(durasi.clip(upper=300), bins=60, color=BIRU)
    ax.axvline(105, color=MERAH, lw=1.5, ls=":")
    ax.text(109, ax.get_ylim()[1] * 0.75,
            "mulai 105 menit\nsemua dipotong\nmenjadi 2 jam", color=MERAH, fontsize=8)
    simpan(fig, "03_durasi_mentah.png")

    grid = ke_grid(durasi.values)
    cacah = pd.Series(grid).value_counts().reindex([1, 2, 3, 4], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    rapikan(ax, "Variabel 2 — Durasi booking setelah aturan engine",
            "durasi (jam)", "jumlah transaksi")
    batang = ax.bar(range(4), cacah.values, color=[BIRU, BIRU, BIRU, ORANYE], width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["0,5", "1,0", "1,5", "2,0"])
    for b, v in zip(batang, cacah.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 150, f"{v:,}\n{v/len(grid)*100:.1f}%",
                ha="center", fontsize=8, color=TINTA2)
    ax.text(3, cacah.values.max() * 0.55,
            "batang oranye = tumpukan\nsemua sesi >= 105 menit", color=ORANYE, fontsize=8,
            ha="center")
    simpan(fig, "04_durasi_pasca_engine.png")

    # ── 3. Stasiun ──────────────────────────────────────────────────────────
    volume = {e["id"]: int((df["id_spklu"].astype(str) == e["id"]).sum())
              for e in terpilih13}
    penyebut = sum(v + LAPLACE_ALPHA for v in volume.values())
    bobot = {sid: (v + LAPLACE_ALPHA) / penyebut for sid, v in volume.items()}
    urut = sorted(terpilih13, key=lambda e: volume[e["id"]], reverse=True)
    label = [str(e["nama"]).replace("SPKLU PLN ", "")[:28] for e in urut]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    rapikan(ax, "Variabel 3 — Volume transaksi 13 stasiun dunia simulasi",
            "jumlah sesi 2024", "")
    ax.barh(range(len(urut)), [volume[e["id"]] for e in urut], color=ORANYE)
    ax.set_yticks(range(len(urut)))
    ax.set_yticklabels(label, fontsize=8)
    ax.invert_yaxis()
    simpan(fig, "05_volume_stasiun.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    rapikan(ax, "Variabel 3 — Peluang pemilihan stasiun setelah penghalusan Laplace",
            "peluang", "")
    ax.barh(range(len(urut)), [bobot[e["id"]] for e in urut], color=BIRU)
    ax.set_yticks(range(len(urut)))
    ax.set_yticklabels(label, fontsize=8)
    ax.invert_yaxis()
    nol = [e for e in urut if volume[e["id"]] == 0]
    if nol:
        ax.text(max(bobot.values()) * 0.45, len(urut) - 1.5,
                f"{len(nol)} stasiun bervolume nol tetap\nmendapat peluang "
                f"{LAPLACE_ALPHA/penyebut:.5f}", color=TINTA2, fontsize=8)
    simpan(fig, "06_bobot_stasiun_laplace.png")

    # ── 4. Harga per kWh ────────────────────────────────────────────────────
    harga = df["price_per_kwh"].dropna()
    harga = harga[(harga > 500) & (harga < 6000)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "Variabel 4 — Harga per kWh (simulasi memakai mediannya sebagai konstanta)",
            "Rp/kWh", "jumlah transaksi")
    ax.hist(harga, bins=60, color=BIRU)
    ax.axvline(harga.median(), color=ORANYE, lw=2, ls="--")
    ax.text(harga.median() + 60, ax.get_ylim()[1] * 0.82,
            f"median {harga.median():.1f} Rp/kWh\n(dipakai simulasi)",
            color=ORANYE, fontsize=9)
    simpan(fig, "07_harga_per_kwh.png")

    # ── 5. Bobot waktu order ────────────────────────────────────────────────
    tw_path = os.path.join(REF_ROOT, "real_data", "time_weight_24.json")
    with open(tw_path, "r", encoding="utf-8") as fh:
        tw = json.load(fh)["time_weight"]
    nilai = [float(tw[str(j)]) for j in range(24)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "Variabel 5 — Bobot waktu order penumpang (TomTom Bandung)",
            "jam", "bobot (puncak = 1)")
    ax.bar(range(24), nilai, color=HIJAU, width=0.82)
    simpan(fig, "08_bobot_waktu_order.png")

    # ── ringkasan angka ─────────────────────────────────────────────────────
    ringkas = {
        "jam_mulai": {
            "sumber": "kolom hour / hour_frac, seluruh 22.514 transaksi Bandung",
            "nilai_berbeda_jam_bulat": int(df["hour"].nunique()),
            "nilai_berbeda_jam_menit": int(df["hour_frac"].nunique()),
            "jam_puncak": int(h.idxmax()),
            "jam_lembah": int(h.idxmin()),
        },
        "durasi": {
            "sumber": "kolom durasi_min, seluruh 22.514 transaksi Bandung",
            "median_menit": float(durasi.median()),
            "mean_menit": round(float(durasi.mean()), 4),
            "maks_menit": float(durasi.max()),
            "kemiringan": round(float(durasi.skew()), 4),
            "proporsi_pasca_engine": {
                f"{g/2:.1f} jam": round(float((grid == g).mean()), 6) for g in (1, 2, 3, 4)
            },
            "mean_grid_pasca_engine": round(float(grid.mean()), 6),
            "mean_jam_pasca_engine": round(float(grid.mean()) * 0.5, 6),
        },
        "stasiun": {
            "sumber": "volume transaksi 2024 pada 13 stasiun dunia simulasi",
            "total_sesi_13_stasiun": int(sum(volume.values())),
            "stasiun_tanpa_transaksi": int(sum(1 for v in volume.values() if v == 0)),
            "laplace_alpha": LAPLACE_ALPHA,
            "bobot": {e["nama"]: round(bobot[e["id"]], 6) for e in urut},
        },
        "harga": {
            "sumber": "kolom price_per_kwh (RPKWH / KWH)",
            "median": round(float(harga.median()), 4),
            "mean": round(float(harga.mean()), 4),
            "simpangan_baku": round(float(harga.std()), 4),
            "catatan": "praktis flat (tarif PLN); simulasi hanya memakai median",
        },
        "bobot_waktu_order": {
            "sumber": "real_data/time_weight_24.json (TomTom Traffic Index Bandung)",
            "jam_puncak": int(np.argmax(nilai)),
            "jam_berbobot_nol": [j for j in range(24) if nilai[j] == 0.0],
        },
    }
    with open(os.path.join(HERE, "ringkasan_variabel_simulasi.json"), "w",
              encoding="utf-8") as fh:
        json.dump(ringkas, fh, indent=1, ensure_ascii=False)

    print("\n[visualisasi] 8 grafik + ringkasan_variabel_simulasi.json")
    print(f"  durasi pasca-engine: "
          f"{ringkas['durasi']['proporsi_pasca_engine']}")
    print(f"  E[durasi] pasca-engine = {ringkas['durasi']['mean_jam_pasca_engine']} jam")


if __name__ == "__main__":
    main()
