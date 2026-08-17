# -*- coding: utf-8 -*-
"""Sampler AppUserBooking berbasis distribusi teoretis hasil fit.

Menggantikan **hanya dua** cara sampling milik
:class:`booking_sampler.EmpiricalBookingSampler`, yaitu **lama booking** dan
**jam mulai booking**. Cara sampling stasiun dan port, harga dasar, serta
seluruh perilaku lain diwarisi apa adanya dari kelas empiris.

Sumber angka
------------
Parameter distribusi TIDAK di-hardcode. Semuanya dibaca dari berkas hasil
analisis di ``reference_data_simulation/`` (prinsip grounding desain v2 §2.0):

- ``reference_data_simulation/fitting/best_parameters.json`` — keluarga juara
  beserta parameternya untuk ``durasi_mentah`` dan ``jam_mentah_menit``.
- ``reference_data_simulation/preprocessing/output/preprocessing_report.json``
  — nilai durasi terbesar pada sampel analisis, dipakai sebagai batas jepitan.

Aturan sampling
---------------
**Lama booking.** Satu nilai ditarik dari distribusi juara ``durasi_mentah``
dengan alih-ragam kebalikan, dijepit pada durasi terbesar yang ada pada data
hasil preprocessing, lalu **dibulatkan ke ATAS** ke kelipatan 30 menit karena
satu grid engine berdurasi 30 menit. Contoh: 1 menit menjadi 1 grid, 31 menit
menjadi 2 grid.

**Jam mulai booking.** Hari ditarik seragam dari ``{0..n_days-1}``. Jam dalam
hari ditarik dari distribusi juara ``jam_mentah_menit``, dijepit pada 24, lalu
**dibulatkan ke kelipatan 30 menit TERDEKAT**. Contoh: pukul 16.24 menjadi
16.30, pukul 17.43 menjadi 17.30. Hasil pembulatan dijepit lagi pada grid 47
(pukul 23.30) karena satu hari hanya memuat grid 0 sampai 47.

Alih-ragam kebalikan
--------------------
Penarikan memakai fungsi kuantil bentuk tertutup dan ``random.Random`` milik
engine, bukan penarik acak scipy. Dua alasan: hasilnya tetap dapat diulang
persis dari seed dunia yang sama, dan simulasi tidak jadi bergantung pada
scipy saat berjalan.
"""
from __future__ import annotations

import json
import math
import os
import random
from statistics import NormalDist
from typing import Callable, Dict, Sequence, Tuple

from booking_sampler import (GRID_MINUTES, GRIDS_PER_DAY, MIN_DURATION_GRIDS,
                             EmpiricalBookingSampler)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REF_DIR = os.path.join(_ROOT, "reference_data_simulation")
PARAM_PATH = os.path.join(REF_DIR, "fitting", "best_parameters.json")
PREP_PATH = os.path.join(REF_DIR, "preprocessing", "output",
                         "preprocessing_report.json")

MAX_START_GRID_IN_DAY = GRIDS_PER_DAY - 1   # satu hari = grid 0..47
MAX_HOUR = 24.0

_NORMAL = NormalDist()


# ═════════════════════════════════════════════════════════════════════════════
# Fungsi kuantil bentuk tertutup, lokasi selalu nol
# ═════════════════════════════════════════════════════════════════════════════
def _q_expon(u: float, skala: float) -> float:
    return -skala * math.log1p(-u)


def _q_weibull_min(u: float, c: float, skala: float) -> float:
    return skala * (-math.log1p(-u)) ** (1.0 / c)


def _q_rayleigh(u: float, skala: float) -> float:
    return skala * math.sqrt(-2.0 * math.log1p(-u))


def _q_fisk(u: float, c: float, skala: float) -> float:
    return skala * (u / (1.0 - u)) ** (1.0 / c)


def _q_lognorm(u: float, s: float, skala: float) -> float:
    return skala * math.exp(s * _NORMAL.inv_cdf(u))


def _q_pareto(u: float, b: float, skala: float) -> float:
    return skala * (1.0 - u) ** (-1.0 / b)


def _q_halfnorm(u: float, skala: float) -> float:
    return skala * _NORMAL.inv_cdf(0.5 * (1.0 + u))


# Keluarga yang punya fungsi kuantil bentuk tertutup tanpa scipy. Keluarga lain
# (gamma, chi2, invgauss) sengaja tidak didukung: fungsi kuantilnya perlu
# pembalikan numerik, dan sampai saat ini tidak satupun menjadi juara.
KUANTIL: Dict[str, Callable[..., float]] = {
    "expon": _q_expon,
    "weibull_min": _q_weibull_min,
    "rayleigh": _q_rayleigh,
    "fisk": _q_fisk,
    "lognorm": _q_lognorm,
    "pareto": _q_pareto,
    "halfnorm": _q_halfnorm,
}


def _fungsi_kuantil(keluarga: str, parameter: Sequence[float]) -> Callable[[float], float]:
    """Kembalikan ``u -> nilai`` untuk satu keluarga beserta parameternya.

    ``parameter`` mengikuti urutan scipy ``(*bentuk, lokasi, skala)``. Lokasi
    wajib nol; itulah syarat yang dipakai saat pencocokan.
    """
    if keluarga not in KUANTIL:
        raise ValueError(
            f"Keluarga '{keluarga}' tidak punya fungsi kuantil bentuk tertutup di "
            f"modul ini. Yang didukung: {sorted(KUANTIL)}.")
    param = [float(p) for p in parameter]
    lokasi, skala = param[-2], param[-1]
    if abs(lokasi) > 1e-12:
        raise ValueError(f"Lokasi keluarga '{keluarga}' harus nol, terbaca {lokasi}.")
    bentuk = param[:-2]
    fn = KUANTIL[keluarga]
    return lambda u: fn(u, *bentuk, skala)


def _cdf_dari_kuantil(keluarga: str, parameter: Sequence[float]) -> Callable[[float], float]:
    """Fungsi distribusi kumulatif bentuk tertutup untuk keluarga yang didukung."""
    param = [float(p) for p in parameter]
    skala = param[-1]
    bentuk = param[:-2]
    if keluarga == "expon":
        return lambda x: 1.0 - math.exp(-x / skala)
    if keluarga == "weibull_min":
        c = bentuk[0]
        return lambda x: 1.0 - math.exp(-((x / skala) ** c))
    if keluarga == "rayleigh":
        return lambda x: 1.0 - math.exp(-0.5 * (x / skala) ** 2)
    if keluarga == "fisk":
        c = bentuk[0]
        return lambda x: 1.0 / (1.0 + (x / skala) ** (-c)) if x > 0 else 0.0
    if keluarga == "lognorm":
        s = bentuk[0]
        return lambda x: _NORMAL.cdf(math.log(x / skala) / s) if x > 0 else 0.0
    if keluarga == "pareto":
        b = bentuk[0]
        return lambda x: 1.0 - (skala / x) ** b if x > skala else 0.0
    if keluarga == "halfnorm":
        return lambda x: 2.0 * _NORMAL.cdf(x / skala) - 1.0 if x > 0 else 0.0
    raise ValueError(f"Keluarga '{keluarga}' tidak didukung.")


# ═════════════════════════════════════════════════════════════════════════════
def muat_hasil_fit(param_path: str = PARAM_PATH,
                   prep_path: str = PREP_PATH) -> Dict[str, object]:
    """Baca keluarga juara kedua variabel beserta batas jepitan durasi."""
    with open(param_path, "r", encoding="utf-8") as fh:
        best = json.load(fh)
    with open(prep_path, "r", encoding="utf-8") as fh:
        prep = json.load(fh)

    def _juara(kunci: str) -> Tuple[str, Sequence[float]]:
        blok = best[kunci]
        baris = next(b for b in blok["keluarga"] if b["peringkat"] == 1)
        return baris["keluarga"], baris["parameter"]

    kel_durasi, par_durasi = _juara("durasi_mentah")
    kel_jam, par_jam = _juara("jam_mentah_menit")
    return {
        "durasi_keluarga": kel_durasi,
        "durasi_parameter": list(par_durasi),
        "jam_keluarga": kel_jam,
        "jam_parameter": list(par_jam),
        "durasi_maks_menit": float(
            prep["sampel_analisis"]["durasi_mentah"]["maksimum"]),
        "n_sampel_analisis": int(prep["sampel_analisis"]["n"]),
    }


class FittedBookingSampler(EmpiricalBookingSampler):
    """Sampler booking dengan durasi dan jam mulai dari distribusi hasil fit.

    Stasiun, port, harga dasar, dan seluruh perilaku lain identik dengan
    :class:`booking_sampler.EmpiricalBookingSampler`.
    """

    def __init__(self, *args, param_path: str = PARAM_PATH,
                 prep_path: str = PREP_PATH, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        fit = muat_hasil_fit(param_path, prep_path)
        self.fit_meta = fit
        self.durasi_keluarga = str(fit["durasi_keluarga"])
        self.durasi_parameter = list(fit["durasi_parameter"])
        self.jam_keluarga = str(fit["jam_keluarga"])
        self.jam_parameter = list(fit["jam_parameter"])
        self.durasi_maks_menit = float(fit["durasi_maks_menit"])
        self._q_durasi = _fungsi_kuantil(self.durasi_keluarga, self.durasi_parameter)
        self._q_jam = _fungsi_kuantil(self.jam_keluarga, self.jam_parameter)
        self._cdf_durasi = _cdf_dari_kuantil(self.durasi_keluarga, self.durasi_parameter)
        self.max_duration_grids = int(
            math.ceil(self.durasi_maks_menit / GRID_MINUTES))

    # ------------------------------------------------------------------
    # Sampling yang diganti
    # ------------------------------------------------------------------
    def sample_duration_grids(self, rng: random.Random) -> int:
        """Durasi: tarik dari distribusi juara, jepit, lalu bulatkan ke ATAS."""
        u = rng.random()
        menit = self._q_durasi(u)
        menit = min(menit, self.durasi_maks_menit)
        grid = int(math.ceil(menit / GRID_MINUTES))
        return max(MIN_DURATION_GRIDS, min(self.max_duration_grids, grid))

    def sample_start_grid(self, rng: random.Random, n_days: int) -> int:
        """Jam mulai: hari seragam, jam dari distribusi juara, bulat ke TERDEKAT."""
        day = rng.randrange(max(1, int(n_days)))
        u = rng.random()
        jam = min(self._q_jam(u), MAX_HOUR)
        # Pembulatan setengah-ke-atas, konvensi yang sama dengan engine.
        grid_in_day = int(jam * (60.0 / GRID_MINUTES) + 0.5)
        grid_in_day = max(0, min(MAX_START_GRID_IN_DAY, grid_in_day))
        return day * GRIDS_PER_DAY + grid_in_day

    # ------------------------------------------------------------------
    # Ekspektasi analitik (deterministik)
    # ------------------------------------------------------------------
    def expected_capped_duration_grids(self) -> float:
        """E[grid durasi] eksak dari fungsi distribusi kumulatif.

        Grid bernilai ``k`` tepat ketika durasi jatuh pada ``((k-1)*30, k*30]``,
        sehingga ekspektasinya dapat dijumlah persis tanpa simulasi.
        """
        k_maks = self.max_duration_grids
        acc = 0.0
        sebelum = 0.0
        for k in range(1, k_maks + 1):
            batas = min(k * GRID_MINUTES, self.durasi_maks_menit)
            kini = self._cdf_durasi(batas) if k < k_maks else 1.0
            acc += k * (kini - sebelum)
            sebelum = kini
        return acc

    # ------------------------------------------------------------------
    def provenance(self) -> Dict[str, object]:
        """Ringkasan angka utk audit; menimpa versi empiris pada dua variabel."""
        dasar = super().provenance()
        dasar.update({
            "sampler": "distribusi hasil fit",
            "durasi_keluarga": self.durasi_keluarga,
            "durasi_parameter": self.durasi_parameter,
            "durasi_jepit_menit": self.durasi_maks_menit,
            "durasi_grid_maksimum": self.max_duration_grids,
            "duration_grid_rule":
                "grid = ceil(min(D, durasi_jepit) / 30), D ~ distribusi juara",
            "jam_keluarga": self.jam_keluarga,
            "jam_parameter": self.jam_parameter,
            "jam_grid_rule":
                "grid = min(47, int(min(H, 24) * 2 + 0.5)), H ~ distribusi juara",
            "n_sampel_analisis": self.fit_meta["n_sampel_analisis"],
            "expected_capped_duration_grids":
                round(self.expected_capped_duration_grids(), 6),
            "expected_capped_duration_hours":
                round(self.expected_capped_duration_hours(), 6),
        })
        return dasar
