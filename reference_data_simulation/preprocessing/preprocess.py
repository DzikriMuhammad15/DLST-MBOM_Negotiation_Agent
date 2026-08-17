# -*- coding: utf-8 -*-
"""PREPROCESSING — menyiapkan sampel analisis sebelum pencocokan distribusi.

Masukan  : ``../filtering/output/bandung_trx.csv``   (22.514 transaksi Bandung)
Keluaran : ``output/sampel_analisis.csv``            (sampel siap dicocokkan)
           ``output/durasi_mentah.csv``              (satu kolom, menit)
           ``output/jam_mentah_menit.csv``           (satu kolom, jam 0..24)
           ``output/preprocessing_report.json``      (angka tiap langkah)
           ``plots/*.png``                           (diagnostik tiap langkah)

Dua variabel yang disiapkan adalah dua variabel yang benar-benar disampel oleh
engine simulasi untuk membentuk AppUserBooking, yaitu lama sesi dan jam mulai
sesi. Alasan tiap langkah ditulis lengkap di ``documentation.md``.

    python reference_data_simulation/preprocessing/preprocess.py
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUMBER = os.path.join(HERE, "..", "filtering", "output", "bandung_trx.csv")
OUT_DIR = os.path.join(HERE, "output")
PLOT_DIR = os.path.join(HERE, "plots")

# Benih tetap supaya dequantisasi dapat diulang persis.
BENIH = 20260811

# Ambang kesahihan fisik: daya terhitung dibanding daya terpasang stasiun.
# Nilainya diturunkan dari data (lihat documentation.md langkah 4).
AMBANG_RASIO_DAYA = 1.25

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


def rb(n) -> str:
    """Pemisah ribuan gaya Indonesia: 22514 menjadi '22.514'."""
    return f"{int(n):,}".replace(",", ".")


def ad(x: float, desimal: int = 3) -> str:
    """Angka desimal gaya Indonesia: 1.25 menjadi '1,250'."""
    teks = f"{x:,.{desimal}f}"
    utuh, _, pecahan = teks.partition(".")
    utuh = utuh.replace(",", ".")
    return f"{utuh},{pecahan}" if pecahan else utuh


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
    fig.savefig(os.path.join(PLOT_DIR, nama), dpi=115, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Langkah-langkah
# ═════════════════════════════════════════════════════════════════════════════
def langkah_1_audit(df: pd.DataFrame) -> Dict:
    """Audit integritas. Tidak membuang satu baris pun; hanya melaporkan.

    Yang diperiksa: baris kembar, kembar pada kunci identitas transaksi, dan
    nilai hilang pada kedua variabel target.
    """
    kunci = ["id_spklu", "chargebox_id", "date", "hour", "minute", "second", "id_user"]
    lap = {
        "baris_masuk": int(len(df)),
        "baris_kembar_penuh": int(df.duplicated().sum()),
        "baris_kembar_kunci_transaksi": int(df.duplicated(subset=kunci).sum()),
        "durasi_hilang": int(df["durasi_min"].isna().sum()),
        "jam_hilang": int(df["hour_frac"].isna().sum()),
        "baris_dibuang": 0,
    }
    return lap


def langkah_2_domain(df: pd.DataFrame) -> Dict:
    """Pemeriksaan domain. Tidak membuang baris; hanya melaporkan.

    Kesepuluh keluarga distribusi kandidat bersupport tak-negatif dengan titik
    awal dikunci di nol, sehingga satu saja nilai negatif membuat fungsi
    kemungkinan bernilai minus tak hingga.
    """
    d = df["durasi_min"]
    h = df["hour_frac"]
    return {
        "durasi_negatif": int((d < 0).sum()),
        "durasi_nol": int((d == 0).sum()),
        "durasi_bilangan_bulat_persen": round(float((d % 1 == 0).mean()) * 100, 4),
        "jam_di_luar_0_24": int(((h < 0) | (h >= 24)).sum()),
        "jam_nol": int((h == 0).sum()),
        "baris_dibuang": 0,
    }


def langkah_3_dequantisasi(df: pd.DataFrame, rng: np.random.Generator) -> Dict:
    """Dequantisasi: kembalikan resolusi yang hilang karena pembulatan.

    ``durasi_min`` tercatat sebagai bilangan bulat menit, sehingga nilai ``k``
    sebenarnya berarti selang ``[k, k+1)``. Nilai pengganti diambil seragam di
    dalam selang itu. ``hour_frac`` tercatat sampai detik, sehingga nilai
    penggantinya diambil seragam di dalam detik yang tercatat.

    Menambahkan kolom ``durasi_dq`` dan ``jam_dq``; tidak membuang baris.
    """
    n = len(df)
    df["durasi_dq"] = df["durasi_min"].to_numpy(dtype=float) + rng.random(n)
    df["jam_dq"] = df["hour_frac"].to_numpy(dtype=float) + rng.random(n) / 3600.0

    d0 = df["durasi_min"]
    return {
        "benih": BENIH,
        "durasi_selang_pencatatan_menit": 1.0,
        "jam_selang_pencatatan_detik": 1.0,
        "cacah_durasi_0": int((d0 == 0).sum()),
        "cacah_durasi_1": int((d0 == 1).sum()),
        "rasio_cacah_0_terhadap_1": round(float((d0 == 0).sum() / max(1, (d0 == 1).sum())), 4),
        "durasi_nol_setelah_dequantisasi": int((df["durasi_dq"] == 0).sum()),
        "durasi_min_setelah": round(float(df["durasi_dq"].min()), 6),
        "nilai_kembar_jam_sebelum": int(len(df) - df["hour_frac"].nunique()),
        "nilai_kembar_jam_sesudah": int(len(df) - df["jam_dq"].nunique()),
        "baris_dibuang": 0,
    }


def langkah_4_kesahihan_fisik(df: pd.DataFrame) -> tuple:
    """Buang transaksi yang melanggar kekekalan energi.

    Daya terhitung ``kwh / durasi_jam`` tidak boleh melampaui daya terpasang
    stasiun. Ambangnya ``AMBANG_RASIO_DAYA`` kali daya terpasang.
    """
    tier = df["kw_tier"].str.replace(" kW", "", regex=False).astype(float)
    daya = df["kwh"] / (df["durasi_dq"] / 60.0)
    rasio = daya / tier
    df["daya_terhitung_kw"] = daya
    df["rasio_daya"] = rasio

    langgar = rasio > AMBANG_RASIO_DAYA
    lap = {
        "ambang_rasio": AMBANG_RASIO_DAYA,
        "rasio_kuantil_50": round(float(rasio.quantile(0.50)), 4),
        "rasio_kuantil_90": round(float(rasio.quantile(0.90)), 4),
        "rasio_kuantil_99": round(float(rasio.quantile(0.99)), 4),
        "rasio_maksimum": round(float(rasio.max()), 4),
        "melanggar_rasio_1x": int((rasio > 1.0).sum()),
        "baris_dibuang": int(langgar.sum()),
        "persen_dibuang": round(float(langgar.mean()) * 100, 4),
        "dibuang_durasi_tercatat_0": int((langgar & (df["durasi_min"] == 0)).sum()),
        "dibuang_durasi_tercatat_le5": int((langgar & (df["durasi_min"] <= 5)).sum()),
    }
    return df.loc[~langgar].copy(), lap


def langkah_5_pencilan(df: pd.DataFrame) -> Dict:
    """Periksa pencilan ekor kanan durasi, lalu putuskan TIDAK membuangnya.

    Pagar Tukey mengandaikan sebaran kira-kira setangkup. Pada sebaran yang
    kemiringannya besar, pagar itu menandai ekor karena bentuk sebarannya,
    bukan karena datanya salah. Pemeriksaan fisik menunjukkan sesi panjang tetap
    menyalurkan energi nyata pada daya jauh di bawah daya terpasang, sehingga
    sesi itu sah.
    """
    d = df["durasi_dq"]
    q1, q3 = float(d.quantile(0.25)), float(d.quantile(0.75))
    iqr = q3 - q1
    lap = {
        "kemiringan": round(float(d.skew()), 4),
        "q1": round(q1, 3), "q3": round(q3, 3), "iqr": round(iqr, 3),
        "pagar_tukey": round(q3 + 1.5 * iqr, 3),
        "pagar_ekstrem": round(q3 + 3.0 * iqr, 3),
        "di_atas_pagar_tukey": int((d > q3 + 1.5 * iqr).sum()),
        "di_atas_pagar_ekstrem": int((d > q3 + 3.0 * iqr).sum()),
        "baris_dibuang": 0,
        "keputusan": "tidak dibuang",
    }
    for amb in (120, 240, 480, 720):
        s = df[df["durasi_dq"] > amb]
        lap[f"di_atas_{amb}_menit"] = {
            "n": int(len(s)),
            "median_kwh": round(float(s["kwh"].median()), 3) if len(s) else None,
            "median_daya_kw": round(float(s["daya_terhitung_kw"].median()), 3) if len(s) else None,
            "median_rasio_daya": round(float(s["rasio_daya"].median()), 4) if len(s) else None,
        }
    return lap


# ═════════════════════════════════════════════════════════════════════════════
# Diagnostik
# ═════════════════════════════════════════════════════════════════════════════
def gambar(df_awal: pd.DataFrame, df_akhir: pd.DataFrame) -> List[str]:
    nama: List[str] = []

    # 1. Bukti pembulatan durasi ke menit bulat.
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    d = df_awal["durasi_min"]
    k = np.arange(0, 21)
    ax.bar(k, [int((d == v).sum()) for v in k], color=BIRU, width=0.75)
    ax.bar([0], [int((d == 0).sum())], color=MERAH, width=0.75)
    rapikan(ax, "Cacah durasi tercatat pada nilai bulat 0 sampai 20 menit",
            "durasi tercatat (menit)", "cacah transaksi")
    ax.text(0.4, int((d == 0).sum()) * 0.97,
            f"nilai 0 = {rb((d == 0).sum())} transaksi\n"
            f"nilai 1 = {rb((d == 1).sum())} transaksi",
            fontsize=9, color=TINTA2, va="top")
    simpan(fig, "01_durasi_pembulatan_menit.png")
    nama.append("01_durasi_pembulatan_menit.png")

    # 2. Sebelum dan sesudah dequantisasi pada ujung kiri.
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    tepi = np.arange(0, 21, 0.5)
    ax.hist(df_awal["durasi_min"], bins=tepi, color=ABU, label="sebelum dequantisasi")
    ax.hist(df_awal["durasi_dq"], bins=tepi, histtype="step", lw=1.8,
            color=ORANYE, label="sesudah dequantisasi")
    rapikan(ax, "Ujung kiri durasi sebelum dan sesudah dequantisasi",
            "durasi (menit)", "cacah transaksi")
    ax.legend(frameon=False)
    simpan(fig, "02_durasi_dequantisasi.png")
    nama.append("02_durasi_dequantisasi.png")

    # 3. Diagnostik kesahihan fisik.
    #
    # Garis ambang beserta keterangan "ambang 1,25" dan cacah transaksi yang
    # dibuang sengaja tidak digambar. Gambar ini dipakai laporan, dan aturan
    # penulisannya melarang teks penjelas di dalam gambar. Angka ambang serta
    # jumlah baris yang tersaring sudah dinyatakan pada prosa dan pada
    # ``preprocessing_report.json``, jadi tidak ada informasi yang hilang.
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    r = df_awal["rasio_daya"].clip(upper=3.0)
    ax.hist(r, bins=np.arange(0, 3.05, 0.05), color=BIRU)
    rapikan(ax, "Daya terhitung dibanding daya terpasang stasiun",
            "daya terhitung / daya terpasang", "cacah transaksi")
    simpan(fig, "03_kesahihan_fisik.png")
    nama.append("03_kesahihan_fisik.png")

    # 4. Sesi panjang tetap menyalurkan energi nyata.
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    s = df_awal[df_awal["durasi_dq"] > 60]
    ax.scatter(s["durasi_dq"], s["daya_terhitung_kw"], s=3, alpha=0.18, color=BIRU,
               edgecolors="none")
    rapikan(ax, "Daya terhitung terhadap durasi pada sesi di atas satu jam",
            "durasi (menit)", "daya terhitung (kW)")
    ax.set_yscale("log")
    simpan(fig, "04_sesi_panjang_daya.png")
    nama.append("04_sesi_panjang_daya.png")

    # 5. Durasi akhir.
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(df_akhir["durasi_dq"], bins=np.arange(0, 605, 5), color=BIRU)
    rapikan(ax, "Durasi sesi pada sampel analisis akhir",
            "durasi (menit)", "cacah transaksi")
    ax.text(0.98, 0.92,
            f"n = {rb(len(df_akhir))}\n"
            f"median = {ad(float(df_akhir['durasi_dq'].median()), 1)} menit\n"
            f"maksimum = {ad(float(df_akhir['durasi_dq'].max()), 1)} menit",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=TINTA2)
    simpan(fig, "05_durasi_akhir.png")
    nama.append("05_durasi_akhir.png")

    # 6. Jam akhir.
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(df_akhir["jam_dq"], bins=np.arange(0, 24.25, 0.25), color=HIJAU)
    rapikan(ax, "Jam mulai sesi pada sampel analisis akhir",
            "jam dalam sehari", "cacah transaksi")
    ax.set_xticks(range(0, 25, 3))
    simpan(fig, "06_jam_akhir.png")
    nama.append("06_jam_akhir.png")

    return nama


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    rng = np.random.default_rng(BENIH)

    df = pd.read_csv(SUMBER)
    for kolom in ("hour", "minute", "second", "hour_frac", "kwh",
                  "durasi_min", "first_soc", "last_soc", "price_per_kwh"):
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    laporan: Dict[str, object] = {"sumber": os.path.relpath(SUMBER, HERE)}
    laporan["langkah_1_audit_integritas"] = langkah_1_audit(df)
    laporan["langkah_2_pemeriksaan_domain"] = langkah_2_domain(df)
    laporan["langkah_3_dequantisasi"] = langkah_3_dequantisasi(df, rng)
    df_awal = df.copy()
    df, lap4 = langkah_4_kesahihan_fisik(df)
    # Kolom diagnostik juga dihitung untuk salinan sebelum penyaringan (utk gambar).
    tier_awal = df_awal["kw_tier"].str.replace(" kW", "", regex=False).astype(float)
    df_awal["daya_terhitung_kw"] = df_awal["kwh"] / (df_awal["durasi_dq"] / 60.0)
    df_awal["rasio_daya"] = df_awal["daya_terhitung_kw"] / tier_awal
    laporan["langkah_4_kesahihan_fisik"] = lap4
    laporan["langkah_5_pemeriksaan_pencilan"] = langkah_5_pencilan(df)

    durasi = df["durasi_dq"].to_numpy(dtype=float)
    jam = df["jam_dq"].to_numpy(dtype=float)
    laporan["sampel_analisis"] = {
        "n": int(len(df)),
        "n_dibuang_total": int(len(df_awal) - len(df)),
        "persen_dipertahankan": round(len(df) / len(df_awal) * 100, 4),
        "durasi_mentah": {
            "satuan": "menit",
            "n": int(len(durasi)),
            "minimum": round(float(durasi.min()), 6),
            "maksimum": round(float(durasi.max()), 6),
            "median": round(float(np.median(durasi)), 4),
            "rerata": round(float(durasi.mean()), 4),
            "simpangan_baku": round(float(durasi.std(ddof=1)), 4),
            "kemiringan": round(float(pd.Series(durasi).skew()), 4),
        },
        "jam_mentah_menit": {
            "satuan": "jam dalam sehari (0 sampai 24), resolusi menit dan detik",
            "n": int(len(jam)),
            "minimum": round(float(jam.min()), 6),
            "maksimum": round(float(jam.max()), 6),
            "median": round(float(np.median(jam)), 4),
            "rerata": round(float(jam.mean()), 4),
            "simpangan_baku": round(float(jam.std(ddof=1)), 4),
            "kemiringan": round(float(pd.Series(jam).skew()), 4),
        },
    }
    laporan["gambar"] = gambar(df_awal, df)

    # ── Keluaran ────────────────────────────────────────────────────────────
    df[["id_spklu", "chargebox_id", "date", "hour", "minute", "second",
        "durasi_min", "durasi_dq", "hour_frac", "jam_dq", "kwh",
        "daya_terhitung_kw", "rasio_daya"]].to_csv(
        os.path.join(OUT_DIR, "sampel_analisis.csv"), index=False)
    pd.DataFrame({"durasi_mentah": durasi}).to_csv(
        os.path.join(OUT_DIR, "durasi_mentah.csv"), index=False)
    pd.DataFrame({"jam_mentah_menit": jam}).to_csv(
        os.path.join(OUT_DIR, "jam_mentah_menit.csv"), index=False)
    with open(os.path.join(OUT_DIR, "preprocessing_report.json"), "w", encoding="utf-8") as fh:
        json.dump(laporan, fh, indent=1, ensure_ascii=False)

    # ── Ringkasan layar ─────────────────────────────────────────────────────
    print("PREPROCESSING SELESAI")
    print(f"  baris masuk           : {rb(len(df_awal))}")
    print(f"  dibuang langkah 4     : {rb(lap4['baris_dibuang'])} "
          f"({ad(lap4['persen_dibuang'], 3)} persen)")
    print(f"  sampel analisis       : {rb(len(df))}")
    print(f"  durasi  min/median/maks : {ad(float(durasi.min()), 3)} / "
          f"{ad(float(np.median(durasi)), 2)} / {ad(float(durasi.max()), 2)} menit")
    print(f"  jam     min/median/maks : {ad(float(jam.min()), 4)} / "
          f"{ad(float(np.median(jam)), 3)} / {ad(float(jam.max()), 3)} jam")
    print(f"  keluaran              : {os.path.relpath(OUT_DIR, HERE)}")


if __name__ == "__main__":
    main()
