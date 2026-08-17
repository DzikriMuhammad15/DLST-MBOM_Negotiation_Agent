# -*- coding: utf-8 -*-
"""Gambar pembanding data terhadap hasil fit distribusi.

Satu gambar berisi satu panel. Fungsi kepadatan peluang dan fungsi distribusi
kumulatif dipisah menjadi dua gambar tersendiri. Satu-satunya keterangan di
dalam gambar adalah jarak Kolmogorov-Smirnov.

Masukan  : ``results/parameter__<target>.json`` dan ``results/metrik__<target>.csv``
Keluaran : ``plots/<target>/pdf__<keluarga>.png``
           ``plots/<target>/cdf__<keluarga>.png``

    python reference_data_simulation/fitting/plot_fits.py
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import (KELUARGA_MENURUT_NAMA, PLOT_DIR, RESULT_DIR, Target,
                    ad_angka, muat_target)

BIRU, ORANYE = "#2a78d6", "#eb6834"
TINTA, TINTA2, REDUP, GARIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10, "font.family": "sans-serif",
    "axes.edgecolor": GARIS, "axes.labelcolor": TINTA2, "text.color": TINTA,
    "xtick.color": REDUP, "ytick.color": REDUP, "axes.titlecolor": TINTA,
    "axes.grid": True, "grid.color": GARIS, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# Judul menyebut variabelnya supaya gambar dari dua target tidak berjudul sama.
# Ini syarat caption pada laporan tesis, sebab Daftar Gambar memakai judul yang
# sama persis dengan judul di dalam gambar, sehingga judul kembar tidak terbaca.
JUDUL_VARIABEL = {
    "durasi_mentah": "Charging Session Duration",
    "jam_mentah_menit": "Session Start Hour",
}


def judul_pdf(t: Target) -> str:
    return f"Probability Distribution Function of {JUDUL_VARIABEL[t.kunci]}"


def judul_cdf(t: Target) -> str:
    return f"Cumulative Distribution Function of {JUDUL_VARIABEL[t.kunci]}"


def _rapikan(ax, judul: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(judul, fontsize=11, fontweight="bold", loc="left", pad=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    for sisi in ("left", "bottom"):
        ax.spines[sisi].set_color(GARIS)
    ax.tick_params(length=0)


def _simpan(fig, folder: str, nama: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(folder, nama), dpi=120, bbox_inches="tight")
    plt.close(fig)


def gambar_pdf(t: Target, nama_keluarga: str, param: List[float], ks: float,
               folder: str) -> None:
    dist = getattr(stats, nama_keluarga)
    label = KELUARGA_MENURUT_NAMA[nama_keluarga].label
    batas = t.batas_gambar
    tepi = np.arange(0.0, batas + t.lebar_bin, t.lebar_bin)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(t.data, bins=tepi, density=True, color=BIRU, alpha=0.55, label="data")
    xs = np.linspace(1e-9, batas, 1200)
    ax.plot(xs, dist.pdf(xs, *param), color=ORANYE, lw=2.0,
            label=f"hasil fit distribusi {label}")
    _rapikan(ax, judul_pdf(t), f"{t.judul.lower()} ({t.satuan})", "kepadatan peluang")
    ax.set_xlim(0, batas)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.985, 0.72, f"jarak KS = {ad_angka(ks, 4)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=TINTA2)
    _simpan(fig, folder, f"pdf__{nama_keluarga}.png")


def gambar_cdf(t: Target, nama_keluarga: str, param: List[float], ks: float,
               folder: str) -> None:
    dist = getattr(stats, nama_keluarga)
    label = KELUARGA_MENURUT_NAMA[nama_keluarga].label
    batas = t.batas_gambar

    x_urut = np.sort(t.data)
    ecdf = np.arange(1, len(x_urut) + 1, dtype=float) / len(x_urut)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x_urut, ecdf, drawstyle="steps-post", color=BIRU, lw=2.0, label="data")
    xs = np.linspace(0.0, batas, 1200)
    ax.plot(xs, dist.cdf(xs, *param), color=ORANYE, lw=2.0,
            label=f"hasil fit distribusi {label}")
    _rapikan(ax, judul_cdf(t), f"{t.judul.lower()} ({t.satuan})", "peluang kumulatif")
    ax.set_xlim(0, batas)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.985, 0.52, f"jarak KS = {ad_angka(ks, 4)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=TINTA2)
    _simpan(fig, folder, f"cdf__{nama_keluarga}.png")


def main() -> None:
    n_gambar = 0
    for t in muat_target():
        folder = os.path.join(PLOT_DIR, t.kunci)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(RESULT_DIR, f"parameter__{t.kunci}.json"),
                  "r", encoding="utf-8") as fh:
            hasil: List[Dict] = json.load(fh)
        for b in hasil:
            if b["status"] != "berhasil":
                print(f"  lewati {b['keluarga']} ({b['status']})")
                continue
            gambar_pdf(t, b["keluarga"], b["parameter"], b["ks"], folder)
            gambar_cdf(t, b["keluarga"], b["parameter"], b["ks"], folder)
            n_gambar += 2
        print(f"  {t.kunci}: gambar ditulis ke {os.path.relpath(folder, PLOT_DIR)}")
    print(f"SELESAI, {n_gambar} gambar")


if __name__ == "__main__":
    main()
