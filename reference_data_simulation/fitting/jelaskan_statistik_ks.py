# -*- coding: utf-8 -*-
"""Gambar penjelas: fungsi distribusi kumulatif empiris dan statistik KS.

Menghasilkan dua gambar:

``plots/belajar/tangga_empiris.png``
    Tiga panel bertahap yang menerangkan apa itu "tangga data", yaitu fungsi
    distribusi kumulatif empiris, dan mengapa bentuknya menyerupai anak tangga.

``plots/belajar/statistik_ks.png``
    Arti rumus statistik Kolmogorov-Smirnov, disertai tabel perhitungan langkah
    demi langkah untuk contoh kecil berisi delapan pengamatan sehingga rumusnya
    dapat diperiksa dengan hitungan tangan.

    python reference_data_simulation/fitting/jelaskan_statistik_ks.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import BELAJAR_DIR, RESULT_DIR, muat_data

BIRU, ORANYE, HIJAU, ABU, MERAH = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#e34948"
TINTA, TINTA2, REDUP, GARIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 9.5, "font.family": "sans-serif",
    "axes.edgecolor": GARIS, "axes.labelcolor": TINTA2, "text.color": TINTA,
    "xtick.color": REDUP, "ytick.color": REDUP, "axes.titlecolor": TINTA,
    "axes.grid": True, "grid.color": GARIS, "grid.linewidth": 0.7,
    "axes.axisbelow": True,
})

# Delapan durasi nyata (menit) sebagai contoh kecil yang bisa dihitung tangan.
CONTOH = np.array([4.0, 12.0, 25.0, 35.0, 50.0, 86.0, 143.0, 151.0])


def rapikan(ax, judul, xlabel="", ylabel=""):
    ax.set_title(judul, fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    for sisi in ("left", "bottom"):
        ax.spines[sisi].set_color(GARIS)
    ax.tick_params(length=0)
    return ax


def tangga(ax, xs, warna, label):
    """Gambar fungsi distribusi kumulatif empiris sebagai tangga."""
    n = len(xs)
    ax.plot([xs[0] - (xs[-1] - xs[0]) * 0.08, xs[0]], [0, 0], color=warna, lw=1.8)
    for i, x in enumerate(xs):
        bawah, atas = i / n, (i + 1) / n
        ax.vlines(x, bawah, atas, color=warna, lw=1.8)
        kanan = xs[i + 1] if i + 1 < n else x + (xs[-1] - xs[0]) * 0.08
        ax.plot([x, kanan], [atas, atas], color=warna, lw=1.8,
                label=label if i == 0 else None)
        ax.plot([x], [atas], "o", ms=3.5, color=warna)


def hitung(xs, cdf):
    n = len(xs)
    F = cdf(xs)
    i = np.arange(1, n + 1)
    naik = i / n - F              # tangga di ATAS kurva
    turun = F - (i - 1) / n       # kurva di ATAS tangga
    return F, naik, turun, int(np.argmax(naik)), int(np.argmax(turun))


def gambar_tangga(durasi: np.ndarray) -> str:
    """Tiga panel bertahap: dari titik data menjadi tangga, lalu ke kasus nyata."""
    kecil = np.array([10.0, 20.0, 35.0, 60.0, 120.0])
    n = len(kecil)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 4.1))

    # Panel 1 — data mentah sebagai titik pada garis bilangan
    rapikan(a1, "Langkah 1: lima durasi, ditaruh pada garis", "durasi (menit)", "")
    a1.plot(kecil, np.zeros(n), "o", ms=9, color=BIRU, zorder=3)
    for x in kecil:
        a1.annotate(f"{x:.0f}", xy=(x, 0), xytext=(0, 12),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=TINTA2)
    a1.set_ylim(-1, 1)
    a1.set_yticks([])
    a1.set_xlim(-8, 138)
    a1.grid(axis="y", visible=False)

    # Panel 2 — tangga dibangun, satu anak tangga per data
    rapikan(a2, "Langkah 2: tiap data menaikkan kurva 1/5 = 0,2",
            "durasi (menit)", "proporsi sesi yang <= x")
    tangga(a2, kecil, BIRU, "tangga data")
    for i, x in enumerate(kecil):
        a2.annotate(f"+0,2", xy=(x, (i + 0.5) / n), xytext=(6, -4),
                    textcoords="offset points", fontsize=8, color=HIJAU)
    a2.axhline(0, color=GARIS, lw=0.8)
    a2.axhline(1, color=GARIS, lw=0.8)
    a2.text(-6, 1.045, "1,0 = semua sesi sudah tercakup", fontsize=8, color=TINTA2)
    a2.text(-6, -0.085, "0,0 = belum ada sesi sependek itu", fontsize=8, color=TINTA2)
    a2.set_ylim(-0.13, 1.13)
    a2.set_xlim(-8, 138)
    a2.legend(frameon=False, fontsize=8.5, loc="center right")

    # Panel 3 — kasus nyata: anak tangganya jadi terlalu rapat untuk terlihat
    rapikan(a3, f"Langkah 3: kasus nyata, n = {len(durasi):,}".replace(",", "."),
            "durasi (menit)", "proporsi sesi yang <= x")
    xr = np.sort(durasi)
    N = len(xr)
    a3.plot(xr, np.arange(1, N + 1) / N, color=BIRU, lw=1.8)
    a3.set_xlim(0, float(np.percentile(durasi, 99.5)))
    a3.set_ylim(-0.04, 1.06)
    a3.text(0.36, 0.28,
            "bentuknya tetap tangga,\ntetapi tiap anak tangga\nhanya setinggi "
            f"1/{N:,} ".replace(",", ".") + "sehingga\nterlihat mulus",
            transform=a3.transAxes, fontsize=8.5, color=TINTA2)

    fig.tight_layout(rect=[0, 0.05, 1, 0.87])
    fig.text(0.005, 0.985,
             "Tangga data: menjawab pertanyaan “berapa proporsi sesi yang "
             "durasinya kurang dari atau sama dengan x?”",
             fontsize=12, fontweight="bold", color=TINTA, ha="left", va="top")
    fig.text(0.005, 0.930,
             "Istilah resminya fungsi distribusi kumulatif empiris. Disebut tangga "
             "karena naik mendadak tepat di tiap pengamatan, lalu datar di antaranya.",
             fontsize=9, color=TINTA2, ha="left", va="top")
    fig.text(0.005, 0.012,
             "Kurva inilah yang dibandingkan dengan kurva distribusi hasil "
             "fit distribusi. Jarak tegak terjauh antara keduanya = statistik "
             "Kolmogorov-Smirnov.",
             fontsize=8.5, color=TINTA2, ha="left", va="bottom")

    keluar = os.path.join(BELAJAR_DIR, "tangga_empiris.png")
    fig.savefig(keluar, dpi=115)
    plt.close(fig)
    return keluar


def main() -> None:
    os.makedirs(BELAJAR_DIR, exist_ok=True)

    # Model pembanding: keluarga juara hasil fit pada durasi.
    with open(os.path.join(RESULT_DIR, "parameter__durasi_mentah.json"),
              encoding="utf-8") as fh:
        hasil = json.load(fh)
    juara = next(b for b in hasil if b["peringkat"] == 1)
    beku = getattr(stats, juara["keluarga"])(*juara["parameter"])
    print(f"Model pembanding: {juara['label']} hasil fit pada seluruh data durasi")

    # ── gambar penjelas tangga empiris ──────────────────────────────────────
    durasi_penuh = muat_data()["durasi_min"].dropna().to_numpy(float)
    print(f"-> {os.path.relpath(gambar_tangga(durasi_penuh))}")
    print()

    # ── contoh kecil ────────────────────────────────────────────────────────
    xs = np.sort(CONTOH)
    n = len(xs)
    F, naik, turun, i_naik, i_turun = hitung(xs, beku.cdf)

    print("Contoh kecil, n = 8 (delapan durasi nyata, dalam menit)")
    print()
    print(f"{'i':>2s} {'x_i':>7s} {'(i-1)/n':>9s} {'i/n':>7s} {'F(x_i)':>9s} "
          f"{'i/n - F':>10s} {'F - (i-1)/n':>12s}")
    print("-" * 62)
    for k in range(n):
        tanda = ""
        if k == i_naik:
            tanda += "  <- D+"
        if k == i_turun:
            tanda += "  <- D-"
        print(f"{k+1:>2d} {xs[k]:>7.1f} {k/n:>9.4f} {(k+1)/n:>7.4f} {F[k]:>9.4f} "
              f"{naik[k]:>10.4f} {turun[k]:>12.4f}{tanda}")
    print("-" * 62)
    print(f"D+ = maks kolom ke-6 = {naik.max():.4f}   (pada x = {xs[i_naik]:.1f})")
    print(f"D- = maks kolom ke-7 = {turun.max():.4f}   (pada x = {xs[i_turun]:.1f})")
    print(f"D  = maks(D+, D-)    = {max(naik.max(), turun.max()):.4f}")
    print()

    # ── gambar ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    rapikan(ax1, f"Contoh kecil, n = {n}", "durasi (menit)", "peluang kumulatif")
    grid = np.linspace(0, xs[-1] * 1.12, 500)
    ax1.plot(grid, beku.cdf(grid), color=ORANYE, lw=2,
             label="kurva model  F(x)", zorder=3)
    tangga(ax1, xs, BIRU, "tangga data")
    # penanda D+ dan D-
    ax1.vlines(xs[i_naik], F[i_naik], (i_naik + 1) / n, color=HIJAU, lw=3, zorder=5)
    ax1.annotate(f"D+ = {naik.max():.4f}",
                 xy=(xs[i_naik], 0.5 * (F[i_naik] + (i_naik + 1) / n)),
                 xytext=(xs[i_naik] + 14, (i_naik + 1) / n + 0.02),
                 color=HIJAU, fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color=HIJAU, lw=1.2))
    ax1.vlines(xs[i_turun], i_turun / n, F[i_turun], color=MERAH, lw=3, zorder=5)
    ax1.annotate(f"D- = {turun.max():.4f}",
                 xy=(xs[i_turun], 0.5 * (i_turun / n + F[i_turun])),
                 xytext=(xs[i_turun] + 16, i_turun / n - 0.13),
                 color=MERAH, fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color=MERAH, lw=1.2))
    ax1.set_ylim(-0.04, 1.10)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

    # ── kasus nyata ─────────────────────────────────────────────────────────
    durasi = durasi_penuh
    xr = np.sort(durasi)
    N = len(xr)
    Fr = beku.cdf(xr)
    ir = np.arange(1, N + 1)
    naik_r = ir / N - Fr
    turun_r = Fr - (ir - 1) / N
    j = int(np.argmax(np.maximum(naik_r, turun_r)))

    rapikan(ax2, f"Kasus nyata, n = {N:,}".replace(",", "."),
            "durasi (menit)", "peluang kumulatif")
    batas = float(np.percentile(durasi, 99.5))
    ax2.plot(xr, ir / N, color=BIRU, lw=1.8, label="tangga data", zorder=3)
    gr = np.linspace(0, batas, 600)
    ax2.plot(gr, beku.cdf(gr), color=ORANYE, lw=1.8, ls="--",
             label="kurva model  F(x)", zorder=4)
    ax2.vlines(xr[j], min(ir[j] / N, Fr[j]), max(ir[j] / N, Fr[j]),
               color=HIJAU, lw=3, zorder=5)
    ax2.annotate(f"D = {max(naik_r.max(), turun_r.max()):.4f}\n"
                 f"pada x = {xr[j]:.0f} menit",
                 xy=(xr[j], 0.5 * (ir[j] / N + Fr[j])),
                 xytext=(xr[j] + batas * 0.12, ir[j] / N - 0.22),
                 color=HIJAU, fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color=HIJAU, lw=1.2))
    ax2.set_xlim(0, batas)
    ax2.set_ylim(-0.04, 1.06)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout(rect=[0, 0.055, 1, 0.88])
    fig.text(0.006, 0.985,
             "Statistik Kolmogorov-Smirnov: jarak tegak TERBESAR antara tangga data "
             "dan kurva model",
             fontsize=12, fontweight="bold", color=TINTA, ha="left", va="top")
    fig.text(0.006, 0.930,
             "Tangga data melompat 1/n di tiap pengamatan, sehingga di titik itu ia "
             "punya dua nilai: bawah (i-1)/n dan atas i/n. Karena itu jaraknya "
             "diperiksa dua arah.",
             fontsize=9, color=TINTA2, ha="left", va="top")
    fig.text(0.006, 0.012,
             "D+ mengukur saat tangga berada DI ATAS kurva; D- mengukur saat kurva "
             "berada DI ATAS tangga. Statistik akhirnya D = maks(D+, D-).",
             fontsize=8.5, color=TINTA2, ha="left", va="bottom")

    keluar = os.path.join(BELAJAR_DIR, "statistik_ks.png")
    fig.savefig(keluar, dpi=115)
    plt.close(fig)
    print(f"-> {os.path.relpath(keluar)}")
    print(f"Kasus nyata: D = {max(naik_r.max(), turun_r.max()):.6f} "
          f"pada durasi {xr[j]:.0f} menit")


if __name__ == "__main__":
    main()
