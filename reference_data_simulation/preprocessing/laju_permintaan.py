# -*- coding: utf-8 -*-
"""Laju permintaan harian SPKLU Bandung 2024 — dasar angka N pada skenario v_2.

Masukan  : ``../filtering/output/bandung_trx.csv`` dan ``bandung_spklu.json``
Keluaran : ``output/laju_permintaan.json`` + ``plots/07_laju_harian.png``

Skenario real-case pada project ``clean_code`` menetapkan **permintaan booking**,
bukan target okupansi. Dua angkanya diturunkan di sini:

* ``medium_occupancy`` = rata-rata jumlah transaksi harian x 3 hari
* ``high_occupancy``  = kuantil 95 jumlah transaksi harian x 3 hari

Keduanya statistik dari sebaran yang sama, yaitu 366 nilai jumlah transaksi per
hari sepanjang 2024, sehingga penjelasannya konsisten. Yang satu beban rata-rata,
yang satu beban puncak.

Berkas ini juga menghitung **faktor pemampatan**, yaitu perbandingan jumlah port
nyata yang melayani Bandung terhadap 14 port dunia simulasi. Volume seluruh
Bandung dimampatkan ke 14 port itu, dan pemampatan tersebut harus disebut apa
adanya, bukan disajikan seolah diambil mentah dari data.

    python reference_data_simulation/preprocessing/laju_permintaan.py
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
FILTER_OUT = os.path.join(HERE, "..", "filtering", "output")
OUT_DIR = os.path.join(HERE, "output")
PLOT_DIR = os.path.join(HERE, "plots")

PORT_SIMULASI = 14      # dunia real-case v2: 13 SPKLU, 14 port
N_HARI_SIMULASI = 3     # horizon satu run

BIRU, ORANYE, MERAH = "#2a78d6", "#eb6834", "#e34948"
TINTA, TINTA2, REDUP, GARIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10, "font.family": "sans-serif",
    "axes.edgecolor": GARIS, "axes.labelcolor": TINTA2, "text.color": TINTA,
    "xtick.color": REDUP, "ytick.color": REDUP, "axes.titlecolor": TINTA,
    "axes.grid": True, "grid.color": GARIS, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(FILTER_OUT, "bandung_trx.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df["durasi_min"] = pd.to_numeric(df["durasi_min"], errors="coerce")

    with open(os.path.join(FILTER_OUT, "bandung_spklu.json"), "r", encoding="utf-8") as fh:
        spklu = json.load(fh)
    # Satu chargebox dihitung satu port. Kolom ``connectors`` TIDAK dijumlahkan
    # karena pada 28 stasiun nilainya seragam di seluruh chargebox dan lebih
    # dari satu, yaitu tanda jumlah konektor TOTAL stasiun tersalin ke setiap
    # baris chargebox. Menjumlahkannya melipatgandakan cacah port, yaitu 236
    # lawan 60 pada stasiun yang bertransaksi. Cara satu port per chargebox
    # inilah yang konsisten dengan angka pada laporan tesis.
    port_per_spklu = {
        str(e["id"]): len(e["chargeboxes"])
        for e in spklu
    }
    aktif = set(df["id_spklu"].astype(str))
    port_nyata = sum(v for k, v in port_per_spklu.items() if k in aktif)
    kompresi = port_nyata / PORT_SIMULASI

    harian = df.groupby("date").agg(trx=("durasi_min", "size"),
                                    menit=("durasi_min", "sum"))
    harian["port_jam"] = harian["menit"] / 60.0
    harian["okupansi_nyata"] = harian["port_jam"] / (port_nyata * 24) * 100
    harian["okupansi_dimampatkan"] = harian["okupansi_nyata"] * kompresi

    rerata = float(harian["trx"].mean())
    k95 = float(harian["trx"].quantile(0.95))
    n_medium = int(round(rerata * N_HARI_SIMULASI))
    n_high = int(round(k95 * N_HARI_SIMULASI))

    bulanan = df.groupby(df["date"].dt.to_period("M")).agg(
        trx=("id_spklu", "size"), hari=("date", "nunique"),
        stasiun=("id_spklu", "nunique"))
    bulanan["per_hari"] = bulanan["trx"] / bulanan["hari"]

    laporan = {
        "sumber": "filtering/output/bandung_trx.csv",
        "cakupan": {
            "tanggal_awal": str(df["date"].min().date()),
            "tanggal_akhir": str(df["date"].max().date()),
            "hari_kalender": int((df["date"].max() - df["date"].min()).days + 1),
            "hari_ada_transaksi": int(df["date"].nunique()),
            "total_transaksi": int(len(df)),
            "stasiun_bertransaksi": int(df["id_spklu"].nunique()),
            "stasiun_aktif_master": len(spklu),
        },
        "port": {
            "port_pada_stasiun_bertransaksi": int(port_nyata),
            "port_dunia_simulasi": PORT_SIMULASI,
            "faktor_pemampatan": round(kompresi, 4),
            "keterangan": "Volume seluruh Bandung dimampatkan ke 14 port dunia "
                          "simulasi. Pemampatan ini disengaja supaya ada rebutan "
                          "slot yang dapat dinegosiasikan.",
        },
        "transaksi_per_hari": {
            "rerata": round(rerata, 4),
            "median": round(float(harian["trx"].median()), 4),
            "minimum": int(harian["trx"].min()),
            "maksimum": int(harian["trx"].max()),
            **{f"kuantil_{int(q*100)}": round(float(harian["trx"].quantile(q)), 4)
               for q in (0.05, 0.25, 0.75, 0.90, 0.95, 0.99)},
        },
        "beban_port_jam_per_hari": {
            "rerata": round(float(harian["port_jam"].mean()), 4),
            "kuantil_95": round(float(harian["port_jam"].quantile(0.95)), 4),
            "okupansi_nyata_rerata_persen": round(float(harian["okupansi_nyata"].mean()), 4),
            "okupansi_dimampatkan_rerata_persen":
                round(float(harian["okupansi_dimampatkan"].mean()), 4),
            "okupansi_dimampatkan_kuantil_95_persen":
                round(float(harian["okupansi_dimampatkan"].quantile(0.95)), 4),
        },
        "pertumbuhan_bulanan_per_hari": {
            str(p): round(float(v), 3) for p, v in bulanan["per_hari"].items()
        },
        "stasiun_aktif_per_bulan": {
            str(p): int(v) for p, v in bulanan["stasiun"].items()
        },
        "N_SKENARIO": {
            "n_hari_simulasi": N_HARI_SIMULASI,
            "medium_occupancy": {
                "N": n_medium,
                "dasar": "rata-rata transaksi harian",
                "laju_harian": round(rerata, 4),
            },
            "high_occupancy": {
                "N": n_high,
                "dasar": "kuantil 95 transaksi harian",
                "laju_harian": round(k95, 4),
            },
        },
    }
    with open(os.path.join(OUT_DIR, "laju_permintaan.json"), "w", encoding="utf-8") as fh:
        json.dump(laporan, fh, indent=1, ensure_ascii=False)

    # ── Gambar: laju harian sepanjang tahun ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(harian.index, harian["trx"], color=BIRU, lw=0.9, alpha=0.55,
            label="transaksi per hari")
    ax.plot(harian.index, harian["trx"].rolling(30, center=True).mean(),
            color=ORANYE, lw=2.0, label="rata-rata bergerak 30 hari")
    # Tanpa anotasi teks di dalam gambar, dan tanpa angka pada legend. Nilai
    # 61,51 serta 122 sudah dinyatakan pada prosa IV.2.3.6, sehingga legend
    # cukup menamai garisnya saja.
    ax.axhline(rerata, color=TINTA2, lw=1.4, ls="--",
               label="rata-rata harian")
    ax.axhline(k95, color=MERAH, lw=1.4, ls="--",
               label="kuantil 95 harian")
    ax.set_title("Jumlah transaksi harian seluruh SPKLU Bandung 2024",
                 fontsize=11, fontweight="bold", loc="left", pad=9)
    ax.set_xlabel("tanggal")
    ax.set_ylabel("transaksi per hari")
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "07_laju_harian.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)

    print("LAJU PERMINTAAN")
    print(f"  cakupan            : {laporan['cakupan']['tanggal_awal']} sampai "
          f"{laporan['cakupan']['tanggal_akhir']}, "
          f"{laporan['cakupan']['hari_kalender']} hari")
    print(f"  transaksi per hari : rata-rata {rerata:.2f}, kuantil 95 {k95:.0f}")
    print(f"  pemampatan         : {port_nyata} port nyata / {PORT_SIMULASI} port "
          f"simulasi = {kompresi:.3f}x")
    print(f"  medium_occupancy   : N = {n_medium}")
    print(f"  high_occupancy     : N = {n_high}")
    print(f"  pertumbuhan 2024   : {bulanan['per_hari'].iloc[0]:.1f} -> "
          f"{bulanan['per_hari'].iloc[-1]:.1f} per hari")


if __name__ == "__main__":
    main()
