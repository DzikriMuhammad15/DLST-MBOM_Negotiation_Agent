# -*- coding: utf-8 -*-
"""Perkakas bersama pencocokan distribusi.

Berisi definisi target, daftar sepuluh keluarga distribusi kandidat, pendugaan
parameter dengan kemungkinan maksimum, dan perhitungan metrik kecocokan.

Titik awal setiap keluarga **dikunci di nol** (``floc=0``). Parameter lokasi
tidak diduga, sehingga tidak ikut dihitung sebagai parameter bebas pada AIC dan
BIC.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PREP_DIR = os.path.join(HERE, "..", "preprocessing", "output")
RESULT_DIR = os.path.join(HERE, "results")
PLOT_DIR = os.path.join(HERE, "plots")
BELAJAR_DIR = os.path.join(PLOT_DIR, "belajar")


# ═════════════════════════════════════════════════════════════════════════════
# 1. TARGET
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Target:
    """Satu variabel yang dicocokkan."""
    kunci: str          # nama berkas dan folder
    judul: str          # judul manusiawi
    satuan: str
    data: np.ndarray
    lebar_bin: float    # lebar bin histogram pembanding
    batas_gambar: float  # batas kanan sumbu mendatar pada gambar
    keterangan: str = ""

    @property
    def n(self) -> int:
        return int(len(self.data))


def muat_target() -> List[Target]:
    """Baca kedua variabel hasil preprocessing."""
    durasi = pd.read_csv(os.path.join(PREP_DIR, "durasi_mentah.csv"))["durasi_mentah"]
    jam = pd.read_csv(os.path.join(PREP_DIR, "jam_mentah_menit.csv"))["jam_mentah_menit"]
    d = durasi.to_numpy(dtype=float)
    h = jam.to_numpy(dtype=float)
    return [
        Target(
            kunci="durasi_mentah",
            judul="Durasi sesi",
            satuan="menit",
            data=d,
            lebar_bin=5.0,
            batas_gambar=float(np.quantile(d, 0.995)),
            keterangan="Lama satu sesi pengisian, dipakai simulasi sebagai lama "
                       "AppUserBooking.",
        ),
        Target(
            kunci="jam_mentah_menit",
            judul="Jam mulai sesi",
            satuan="jam",
            data=h,
            lebar_bin=0.25,
            batas_gambar=24.0,
            keterangan="Waktu mulai satu sesi pengisian dalam sehari, resolusi "
                       "menit dan detik. Dipakai simulasi sebagai jam mulai "
                       "AppUserBooking.",
        ),
    ]


def muat_data() -> pd.DataFrame:
    """Sampel analisis hasil preprocessing sebagai DataFrame.

    Dipakai oleh skrip penjelas ``jelaskan_*.py``. Kolom ``durasi_min`` di sini
    adalah durasi SETELAH dequantisasi dan penyaringan kesahihan fisik, yaitu
    nilai yang benar-benar dicocokkan, bukan nilai bulat pada data hasil filter.
    """
    df = pd.read_csv(os.path.join(PREP_DIR, "sampel_analisis.csv"))
    df["durasi_min"] = pd.to_numeric(df["durasi_dq"], errors="coerce")
    df["hour_frac"] = pd.to_numeric(df["jam_dq"], errors="coerce")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. SEPULUH KELUARGA KANDIDAT
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Keluarga:
    nama: str            # nama scipy
    label: str           # nama Indonesia untuk laporan dan legenda
    alasan: str          # mengapa keluarga ini diuji
    catatan: str = ""    # peringatan khusus


KELUARGA: List[Keluarga] = [
    Keluarga("expon", "Eksponensial",
             "Laju kejadian konstan dan sifat tanpa-ingatan. Acuan dasar untuk "
             "waktu antar-kejadian dan waktu sampai kejadian pertama."),
    Keluarga("gamma", "Gamma",
             "Fleksibel mengatur kemiringan. Eksponensial dan chi-kuadrat adalah "
             "bentuk khususnya, sehingga sekaligus menjadi pembanding bertingkat."),
    Keluarga("weibull_min", "Weibull",
             "Baku dalam analisis keandalan. Laju kegagalan bisa menurun, tetap, "
             "atau menaik menurut parameter bentuknya."),
    Keluarga("lognorm", "Lognormal",
             "Muncul ketika besaran terbentuk dari perkalian banyak faktor acak "
             "bebas. Lazim untuk lama pengerjaan dan waktu pemulihan."),
    Keluarga("invgauss", "Inverse Gaussian (Wald)",
             "Waktu yang dibutuhkan gerak Brown mencapai suatu ambang. Tafsir "
             "fisiknya cocok untuk pengisian sampai suatu tingkat daya tertentu."),
    Keluarga("fisk", "Log-Logistik (Fisk)",
             "Bentuknya mirip lognormal dengan ekor lebih tebal. Lazim pada "
             "analisis ketahanan hidup dan lama sesi pengguna."),
    Keluarga("rayleigh", "Rayleigh",
             "Bentuk khusus Weibull dengan parameter bentuk dua. Menguji apakah "
             "satu parameter skala saja sudah memadai."),
    Keluarga("chi2", "Chi-Kuadrat",
             "Jumlah kuadrat peubah normal baku. Diuji sebagai bentuk khusus "
             "Gamma yang derajat kebebasannya punya tafsir tersendiri."),
    Keluarga("pareto", "Pareto",
             "Ekor sangat tebal, landasan aturan 80/20. Menguji apakah ekor "
             "kanan data memang selebar itu.",
             catatan="Pada scipy, Pareto dengan lokasi dikunci nol bersupport "
                     "x >= skala, sehingga titik awal efektifnya berada di skala, "
                     "bukan tepat di nol. Penduga kemungkinan maksimum menaruh "
                     "skala pada nilai terkecil sampel."),
    Keluarga("halfnorm", "Half-Normal",
             "Nilai mutlak sebaran normal berpusat nol. Modusnya selalu tepat di "
             "nol, sehingga menguji apakah nilai terkecil memang yang paling "
             "sering muncul."),
]

NAMA_KELUARGA = [k.nama for k in KELUARGA]
KELUARGA_MENURUT_NAMA = {k.nama: k for k in KELUARGA}


# ═════════════════════════════════════════════════════════════════════════════
# 3. PENDUGAAN PARAMETER
# ═════════════════════════════════════════════════════════════════════════════
def nama_parameter(nama: str) -> List[str]:
    """Nama parameter bentuk keluarga, ditambah skala."""
    dist = getattr(stats, nama)
    bentuk = [] if dist.shapes is None else [s.strip() for s in dist.shapes.split(",")]
    return bentuk + ["skala"]


def duga_parameter(nama: str, x: np.ndarray) -> Tuple[Tuple[float, ...], int]:
    """Duga parameter dengan kemungkinan maksimum, lokasi dikunci nol.

    Returns:
        Pasangan ``(parameter_lengkap, jumlah_parameter_bebas)``. Parameter
        lengkap mengikuti urutan scipy ``(*bentuk, lokasi, skala)`` dengan
        lokasi selalu nol. Jumlah parameter bebas TIDAK menghitung lokasi.
    """
    dist = getattr(stats, nama)
    param = dist.fit(x, floc=0.0)
    n_bentuk = 0 if dist.shapes is None else len(dist.shapes.split(","))
    bebas = n_bentuk + 1          # bentuk + skala; lokasi dikunci, tidak dihitung
    return tuple(float(p) for p in param), bebas


# ═════════════════════════════════════════════════════════════════════════════
# 4. METRIK KECOCOKAN
# ═════════════════════════════════════════════════════════════════════════════
def _cdf_empiris(x_urut: np.ndarray) -> np.ndarray:
    return np.arange(1, len(x_urut) + 1, dtype=float) / len(x_urut)


def metrik(nama: str, param: Sequence[float], x: np.ndarray, k_bebas: int,
           tepi_bin: np.ndarray) -> Dict[str, float]:
    """Hitung seluruh metrik kecocokan satu keluarga.

    Metrik yang dihitung:

    * ``log_kemungkinan``, ``aic``, ``bic`` — berbasis fungsi kemungkinan;
      hanya parameter bebas yang dihukum, sehingga lokasi yang dikunci tidak
      merugikan keluarga manapun.
    * ``ks`` — jarak Kolmogorov-Smirnov, yaitu jarak tegak terbesar antara
      fungsi distribusi kumulatif empiris dan teoretis.
    * ``cvm``, ``ad`` — Cramer-von Mises dan Anderson-Darling, yang menimbang
      seluruh rentang dan ekor secara lebih berat daripada KS.
    * ``wasserstein`` — jarak pemindahan massa, bersatuan sama dengan datanya.
    * ``tvd`` — jarak variasi total atas bin histogram, yaitu bagian massa
      peluang yang salah tempat.
    """
    dist = getattr(stats, nama)
    n = len(x)

    log_kepadatan = dist.logpdf(x, *param)
    berhingga = np.isfinite(log_kepadatan)
    titik_diabaikan = int(n - berhingga.sum())
    # Satu saja titik data yang kepadatannya nol membuat fungsi kemungkinan
    # bernilai minus tak hingga. Itu dilaporkan apa adanya, bukan ditambal.
    loglik = float(np.sum(log_kepadatan)) if titik_diabaikan == 0 else float("-inf")

    aic = 2.0 * k_bebas - 2.0 * loglik
    bic = k_bebas * math.log(n) - 2.0 * loglik

    x_urut = np.sort(x)
    cdf_teori = dist.cdf(x_urut, *param)
    ecdf_atas = _cdf_empiris(x_urut)
    ecdf_bawah = ecdf_atas - 1.0 / n
    d_plus = float(np.max(ecdf_atas - cdf_teori))
    d_minus = float(np.max(cdf_teori - ecdf_bawah))
    ks = max(d_plus, d_minus)

    i = np.arange(1, n + 1, dtype=float)
    cvm = float(1.0 / (12.0 * n) + np.sum((cdf_teori - (2 * i - 1) / (2 * n)) ** 2))
    aman = np.clip(cdf_teori, 1e-15, 1 - 1e-15)
    ad = float(-n - np.mean((2 * i - 1) * (np.log(aman) + np.log(1 - aman[::-1]))))

    acuan = dist.rvs(*param, size=200000, random_state=20260811)
    wass = float(stats.wasserstein_distance(x, acuan))

    cacah, _ = np.histogram(x, bins=tepi_bin)
    p_data = cacah / n
    p_model = np.diff(dist.cdf(tepi_bin, *param))
    tvd = float(0.5 * np.sum(np.abs(p_data - p_model))
                + 0.5 * abs(1.0 - float(p_model.sum())))

    return {
        "log_kemungkinan": loglik,
        "k_bebas": int(k_bebas),
        "aic": aic,
        "bic": bic,
        "ks": ks,
        "ks_d_plus": d_plus,
        "ks_d_minus": d_minus,
        "cvm": cvm,
        "ad": ad,
        "wasserstein": wass,
        "tvd": tvd,
        "titik_kepadatan_diabaikan": titik_diabaikan,
        "massa_di_luar_bin": float(max(0.0, 1.0 - float(p_model.sum()))),
    }


def tepi_bin(t: Target) -> np.ndarray:
    """Tepi bin histogram pembanding, dari nol sampai melewati nilai terbesar."""
    atas = math.ceil(float(t.data.max()) / t.lebar_bin) * t.lebar_bin
    return np.arange(0.0, atas + t.lebar_bin, t.lebar_bin)


# ═════════════════════════════════════════════════════════════════════════════
# 5. FORMAT ANGKA (gaya Indonesia)
# ═════════════════════════════════════════════════════════════════════════════
def rb(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def ad_angka(x: float, desimal: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    teks = f"{x:,.{desimal}f}"
    utuh, _, pecahan = teks.partition(".")
    utuh = utuh.replace(",", ".")
    return f"{utuh},{pecahan}" if pecahan else utuh
