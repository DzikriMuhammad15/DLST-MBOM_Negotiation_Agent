# -*- coding: utf-8 -*-
"""Gambar penjelas: apa itu parameter lokasi, dan mengapa ia dikunci nol.

Menghasilkan ``plots/belajar/parameter_lokasi.png``, gambar dua panel yang
menerangkan apa yang terjadi bila parameter lokasi DIBIARKAN BEBAS diduga, dan
karena itu menjelaskan mengapa pada ``fit_distributions.py`` lokasi dikunci di
nol untuk kesepuluh keluarga.

Panel kanan memakai contoh nyata, yaitu pendugaan Gaussian invers pada durasi
dengan lokasi dibiarkan bebas. Lokasinya terduga di −8,71 menit, sehingga model
itu memberi peluang 0,417 persen bagi durasi bernilai NEGATIF. Durasi negatif
tidak punya arti fisik dan tidak dapat disampel simulasi, dan itulah alasan
lokasi dikunci.

    python reference_data_simulation/fitting/jelaskan_parameter_lokasi.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import BELAJAR_DIR, muat_data

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

# Parameter Gaussian invers hasil pencocokan nyata pada durasi (lokasi_bebas),
# yaitu kecocokan peringkat satu untuk variabel itu. Dipakai sebagai contoh
# karena lokasinya diduga bernilai negatif.
INVGAUSS = dict(mu=0.967383, loc=-8.707809, scale=87.208185)


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


def main() -> None:
    os.makedirs(BELAJAR_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # ── Panel kiri: lokasi hanya MENGGESER bentuk yang sama ─────────────────
    rapikan(ax1, "Lokasi hanya menggeser bentuk yang sama",
            "nilai x", "kepadatan")
    x = np.linspace(-40, 120, 900)
    for loc, warna, gaya in ((0, BIRU, "-"), (30, HIJAU, "-"), (-8.71, ORANYE, "-")):
        ax1.plot(x, stats.gamma.pdf(x, a=2.0, loc=loc, scale=14),
                 color=warna, lw=2, ls=gaya, label=f"loc = {loc}")
        ax1.axvline(loc, color=warna, lw=1, ls=":", alpha=0.7)
    ax1.axvspan(-40, 0, color=MERAH, alpha=0.07)
    ax1.axvline(0, color=TINTA2, lw=1.1)
    ax1.text(-38, ax1.get_ylim()[1] * 0.93, "daerah mustahil\nuntuk durasi",
             color=MERAH, fontsize=8.5, va="top")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_xlim(-40, 120)

    # ── Panel kanan: kasus nyata pada data durasi ───────────────────────────
    df = muat_data()
    durasi = df["durasi_min"].dropna().to_numpy(dtype=float)
    rapikan(ax2, "Kasus nyata: kecocokan Gaussian invers pada durasi",
            "durasi (menit)", "kepadatan")
    pot = durasi[durasi <= 300]
    ax2.hist(pot, bins=70, density=True, color=ABU, label="data asli", zorder=2)

    xx = np.linspace(-40, 300, 900)
    beku = stats.invgauss(INVGAUSS["mu"], loc=INVGAUSS["loc"], scale=INVGAUSS["scale"])
    ax2.plot(xx, beku.pdf(xx), color=ORANYE, lw=2, zorder=3,
             label=f"lokasi bebas, loc = {INVGAUSS['loc']:.2f}")
    ax2.axvspan(-40, 0, color=MERAH, alpha=0.10, zorder=1)
    ax2.axvline(0, color=TINTA2, lw=1.1, zorder=4)
    ax2.annotate(f"model menaruh {beku.cdf(0.0):.2%} peluang\ndi durasi NEGATIF",
                 xy=(-4.5, beku.pdf(-4.5)), xytext=(38, 0.0125),
                 fontsize=8.5, color=MERAH,
                 arrowprops=dict(arrowstyle="->", color=MERAH, lw=1.2))
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_xlim(-40, 300)

    fig.tight_layout(rect=[0, 0.055, 1, 0.90])
    fig.text(0.006, 0.985,
             "Parameter lokasi: di mana bentuk distribusi diletakkan pada garis bilangan",
             fontsize=12, fontweight="bold", color=TINTA, ha="left", va="top")
    fig.text(0.006, 0.935,
             "Varian lokasi_bebas membiarkan titik awal ini diduga dari data. "
             "Varian lokasi_nol menguncinya tepat di angka nol.",
             fontsize=9, color=TINTA2, ha="left", va="top")
    fig.text(0.006, 0.012,
             "Bentuk kurva ketiganya di panel kiri PERSIS SAMA — yang berubah hanya "
             "posisinya. Itulah yang dikerjakan parameter lokasi.",
             fontsize=8.5, color=TINTA2, ha="left", va="bottom")

    keluar = os.path.join(BELAJAR_DIR, "parameter_lokasi.png")
    fig.savefig(keluar, dpi=115)
    plt.close(fig)

    # angka pendukung untuk README
    print(f"-> {os.path.relpath(keluar)}")
    print(f"Peluang durasi NEGATIF menurut kecocokan Gaussian invers lokasi bebas: "
          f"{beku.cdf(0.0):.4%}")
    print(f"Titik awal daerah dukungannya: {INVGAUSS['loc']:.4f} menit")


if __name__ == "__main__":
    main()
