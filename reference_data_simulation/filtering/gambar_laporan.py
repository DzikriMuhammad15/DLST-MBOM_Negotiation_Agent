# -*- coding: utf-8 -*-
"""Gambar yang disisipkan ke laporan tesis, terpisah dari gambar EDA internal.

Gambar EDA di ``eda/`` bernomor dan berjudul gaya kerja, misalnya
``01 — Transaksi per jam``. Yang masuk laporan perlu judul yang sama dengan
caption-nya dan harus menyajikan sebarannya seakurat mungkin. Karena itu
keduanya dipisah, dan berkas ini yang dipakai dokumen.

Masukan  : ``output/bandung_trx.csv``
Keluaran : ``gambar_laporan/*.png``

    python reference_data_simulation/filtering/gambar_laporan.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step3_eda import ABU, BIRU, ORANYE, rapikan  # gaya identik dengan gambar EDA

#: Benih dequantisasi, sama dengan ``preprocessing/preprocess.py`` supaya gambar
#: laporan menampilkan tepat data yang dipakai pencocokan distribusi.
BENIH_DEQUANTISASI = 20260811

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
GAMBAR_DIR = os.path.join(HERE, "gambar_laporan")

#: Lebar bin waktu dalam menit. Lima belas menit cukup halus untuk memperlihatkan
#: struktur harian tanpa menjadi berisik pada 22.514 transaksi.
LEBAR_BIN_MENIT = 15


def simpan(fig, nama: str) -> None:
    os.makedirs(GAMBAR_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(GAMBAR_DIR, nama), dpi=115, bbox_inches="tight")
    plt.close(fig)


def gambar_waktu(df: pd.DataFrame) -> None:
    """Sebaran waktu transaksi sepanjang hari, memakai resolusi sebenarnya.

    Versi sebelumnya membin per jam bulat sehingga hanya menyisakan 24 batang
    dan menyembunyikan menit serta detik yang sebenarnya tercatat. Data mencatat
    waktu sampai detik lewat kolom ``hour_frac``, yaitu
    ``hour + minute/60 + second/3600``, jadi sebarannya digambar apa adanya
    dengan bin selebar seperempat jam.
    """
    h = df["hour_frac"].dropna()
    tepi = np.arange(0, 24 + LEBAR_BIN_MENIT / 60.0, LEBAR_BIN_MENIT / 60.0)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "Distribusi Waktu Transaksi Charging",
            "waktu dalam sehari (jam)", "jumlah sesi")
    ax.hist(h, bins=tepi, color=BIRU)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{j:02d}.00" for j in range(0, 25, 3)])
    simpan(fig, "waktu_transaksi.png")


def gambar_durasi(df: pd.DataFrame) -> None:
    """Sebaran lama sesi pengisian, dijepit tampilannya pada 300 menit.

    Garis batas atas engine dan keterangannya sengaja tidak digambar, sebab
    aturan penulisan melarang teks penjelas di dalam gambar.
    """
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rapikan(ax, "Distribusi Durasi Sesi Charging", "menit", "jumlah sesi")
    ax.hist(df["durasi_min"].dropna().clip(upper=300), bins=60, color=BIRU)
    simpan(fig, "durasi_sesi.png")


def _durasi_dequantisasi(d: np.ndarray) -> np.ndarray:
    """Dequantisasi durasi memakai benih yang sama dengan tahap prapemrosesan."""
    rng = np.random.default_rng(BENIH_DEQUANTISASI)
    return d.astype(float) + rng.random(len(d))


def gambar_frekuensi_durasi(df: pd.DataFrame) -> None:
    """Cacah transaksi pada setiap nilai durasi bulat, SELURUH sebaran.

    Versi sebelumnya hanya menampilkan nol sampai dua puluh menit dengan satu
    batang merah dan keterangan angka di dalam gambar. Keduanya dibuang, sebab
    aturan penulisan melarang teks penjelas di dalam gambar dan pemotongan pada
    dua puluh menit menyembunyikan sebagian besar sebarannya.
    """
    d = df["durasi_min"].dropna().to_numpy(dtype=int)
    maks = int(d.max())
    k = np.arange(0, maks + 1)
    cacah = np.bincount(d, minlength=maks + 1)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "Frekuensi Setiap Durasi Pengecasan",
            "durasi tercatat (menit)", "cacah transaksi")
    ax.bar(k, cacah, color=BIRU, width=1.0)
    ax.set_xlim(-5, maks + 5)
    simpan(fig, "frekuensi_durasi.png")


def gambar_dequantisasi(df: pd.DataFrame) -> None:
    """Sebaran durasi sebelum dan sesudah dequantisasi, SELURUH sebaran.

    Bin selebar satu menit, yaitu selang pencatatan aslinya, supaya efek
    dequantisasi terlihat sebagai perataan gerigi dan bukan sekadar dua kurva
    yang saling menimpa.
    """
    d = df["durasi_min"].dropna().to_numpy(dtype=float)
    dq = _durasi_dequantisasi(d)
    tepi = np.arange(0, d.max() + 2, 1.0)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "Durasi Sebelum dan Sesudah Dequantisasi",
            "durasi (menit)", "cacah transaksi")
    ax.hist(d, bins=tepi, color=ABU, label="sebelum dequantisasi")
    ax.hist(dq, bins=tepi, histtype="step", lw=1.4, color=ORANYE,
            label="sesudah dequantisasi")
    ax.set_xlim(0, d.max() + 1)
    ax.legend(frameon=False)
    simpan(fig, "durasi_dequantisasi.png")


def main() -> None:
    df = pd.read_csv(os.path.join(OUT_DIR, "bandung_trx.csv"))
    gambar_waktu(df)
    gambar_durasi(df)
    gambar_frekuensi_durasi(df)
    gambar_dequantisasi(df)
    print(f"durasi maksimum    : {df['durasi_min'].max():.0f} menit")
    print(f"transaksi          : {len(df):,}".replace(",", "."))
    print(f"nilai waktu unik   : {df['hour_frac'].nunique():,}".replace(",", "."))
    print(f"lebar bin waktu    : {LEBAR_BIN_MENIT} menit "
          f"({int(24 * 60 / LEBAR_BIN_MENIT)} bin)")
    print(f"keluaran           : {GAMBAR_DIR}")


if __name__ == "__main__":
    main()
