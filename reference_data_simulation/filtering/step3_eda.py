# -*- coding: utf-8 -*-
"""LANGKAH 3 — EDA (analisis data eksploratif) atas data hasil penyaringan.

Masukan  : ``output/bandung_trx.csv`` + ``output/bandung_spklu.json``
Keluaran : ``eda/*.png`` + ``eda/EDA.md``

Cakupan: SELURUH kolom data hasil filter, termasuk kolom yang tidak dipakai
simulasi (hari, bulan, konektor, pengguna). Plot khusus variabel yang dipakai
simulasi ada terpisah di ``reference_data_simulation/visualization/``.

    python reference_data_simulation/filtering/step3_eda.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
EDA_DIR = os.path.join(HERE, "eda")

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
    fig.savefig(os.path.join(EDA_DIR, nama), dpi=115, bbox_inches="tight")
    plt.close(fig)


def rb(n) -> str:
    """Pemisah ribuan gaya Indonesia: 22514 menjadi '22.514'."""
    return f"{int(n):,}".replace(",", ".")


def main() -> None:
    os.makedirs(EDA_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(OUT_DIR, "bandung_trx.csv"))
    for kolom in ("hour", "hour_frac", "minute", "second", "dow", "month", "kwh",
                  "rpkwh", "rppakai", "price_per_kwh", "durasi_min",
                  "first_soc", "last_soc"):
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    with open(os.path.join(OUT_DIR, "bandung_spklu.json"), "r", encoding="utf-8") as fh:
        spklu = json.load(fh)
    for e in spklu:
        e["ports"] = sum(int(cb["connectors"]) for cb in e["chargeboxes"])
    pln = sorted([e for e in spklu if "PLN" in str(e["nama"]).upper() and e["ports"] > 0],
                 key=lambda e: (e["ports"], e["nama"]))
    terpilih13 = pln[:13]
    nama13 = {e["nama"] for e in terpilih13}
    id13 = {e["id"] for e in terpilih13}

    gambar = []
    n = len(df)

    # 1 — jam bulat
    h = df["hour"].value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "01 — Transaksi per jam (jam bulat, seperti ekstraksi lama)",
            "jam", "jumlah sesi")
    ax.bar(h.index, h.values, color=BIRU, width=0.82)
    simpan(fig, "01_jam_bulat.png")
    gambar.append(("01_jam_bulat.png",
                   f"Puncak {int(h.idxmax())}:00, lembah {int(h.idxmin())}:00. "
                   "Hanya 24 nilai berbeda."))

    # 2 — jam pecahan
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rapikan(ax, "02 — Transaksi per jam (jam pecahan, menit ikut dihitung)",
            "jam (0-24)", "jumlah sesi")
    ax.hist(df["hour_frac"].dropna(), bins=96, color=BIRU)
    simpan(fig, "02_jam_pecahan.png")
    gambar.append(("02_jam_pecahan.png",
                   f"{rb(df['hour_frac'].nunique())} nilai berbeda (bandingkan 24 pada "
                   "gambar 01). Bentuk harian sama, resolusi jauh lebih halus."))

    # 3 — hari dalam minggu
    dow = df["dow"].value_counts().reindex(range(7), fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    rapikan(ax, "03 — Hari dalam minggu", "", "jumlah sesi")
    ax.bar(range(7), dow.values, color=BIRU, width=0.72)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"])
    simpan(fig, "03_hari.png")
    gambar.append(("03_hari.png",
                   "Sebaran hari cukup rata. Kolom ini TIDAK dipakai simulasi — "
                   "sampler mengulang bentuk harian yang sama untuk setiap hari."))

    # 4 — tren bulanan
    mo = df["month"].value_counts().reindex(range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rapikan(ax, "04 — Tren bulanan 2024", "bulan", "jumlah sesi")
    ax.plot(mo.index, mo.values, color=BIRU, lw=2, marker="o", ms=5)
    simpan(fig, "04_bulan.png")
    gambar.append(("04_bulan.png", "Tren adopsi sepanjang 2024."))

    # 5 — energi
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rapikan(ax, "05 — Energi per sesi", "kWh", "jumlah sesi")
    ax.hist(df["kwh"].dropna().clip(upper=100), bins=40, color=BIRU)
    ax.axvline(df["kwh"].mean(), color=ORANYE, lw=2, ls="--")
    simpan(fig, "05_energi.png")
    gambar.append(("05_energi.png",
                   f"Mean {df['kwh'].mean():.1f} kWh, median {df['kwh'].median():.1f} kWh."))

    # 6 — durasi
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rapikan(ax, "06 — Durasi sesi", "menit", "jumlah sesi")
    # Garis batas atas engine dan keterangannya SENGAJA tidak digambar. Gambar
    # ini dipakai di laporan sebagai sebaran data apa adanya, dan aturan
    # penulisan melarang teks penjelas di dalam gambar.
    ax.hist(df["durasi_min"].dropna().clip(upper=300), bins=60, color=BIRU)
    simpan(fig, "06_durasi.png")
    gambar.append(("06_durasi.png",
                   f"Miring kanan: median {df['durasi_min'].median():.0f} mnt, "
                   f"mean {df['durasi_min'].mean():.1f} mnt, maks {df['durasi_min'].max():.0f} mnt. "
                   "Engine memotong di 2 jam."))

    # 7 — SoC datang & selesai
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rapikan(ax, "07 — SoC saat datang dan saat selesai", "SoC (%)", "jumlah sesi")
    ax.hist(df["first_soc"].dropna().clip(0, 100), bins=40, color=BIRU,
            alpha=0.75, label="datang")
    ax.hist(df["last_soc"].dropna().clip(0, 100), bins=40, color=HIJAU,
            alpha=0.6, label="selesai")
    ax.legend(frameon=False)
    simpan(fig, "07_soc.png")
    gambar.append(("07_soc.png",
                   f"SoC datang mean {df['first_soc'].mean():.0f}%, "
                   f"SoC selesai mean {df['last_soc'].mean():.0f}%."))

    # 8 — harga
    harga = df["price_per_kwh"].dropna()
    harga = harga[(harga > 500) & (harga < 6000)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rapikan(ax, "08 — Harga per kWh", "Rp/kWh", "jumlah sesi")
    ax.hist(harga, bins=60, color=BIRU)
    ax.axvline(harga.median(), color=ORANYE, lw=2, ls="--")
    ax.text(harga.median() + 60, ax.get_ylim()[1] * 0.85,
            f"median {harga.median():.1f}", color=ORANYE, fontsize=9)
    simpan(fig, "08_harga.png")
    gambar.append(("08_harga.png",
                   f"Nyaris flat, median {harga.median():.1f} Rp/kWh (tarif PLN). "
                   "Simulasi hanya memakai mediannya sebagai konstanta."))

    # 9 — volume per stasiun (20 teratas)
    vol = df.groupby("nama_spklu").size().sort_values(ascending=False).head(20)
    warna = [ORANYE if nm in nama13 else ABU for nm in vol.index]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    rapikan(ax, "09 — Volume transaksi per SPKLU (20 teratas)", "jumlah sesi 2024", "")
    ax.barh(range(len(vol)), vol.values, color=warna)
    ax.set_yticks(range(len(vol)))
    ax.set_yticklabels([str(nm)[:36] for nm in vol.index], fontsize=8)
    ax.invert_yaxis()
    simpan(fig, "09_volume_stasiun.png")
    gambar.append(("09_volume_stasiun.png",
                   "Sangat timpang. Oranye = termasuk 13 stasiun dunia simulasi "
                   "(semuanya stasiun sepi)."))

    # 10 — tier daya
    kt = df["kw_tier"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    rapikan(ax, "10 — Tier daya charger", "", "jumlah sesi")
    ax.bar(range(len(kt)), kt.values, color=BIRU)
    ax.set_xticks(range(len(kt)))
    ax.set_xticklabels(kt.index, rotation=40, ha="right", fontsize=8)
    simpan(fig, "10_tier_daya.png")
    gambar.append(("10_tier_daya.png", "Tier daya yang paling sering dipakai."))

    # 11 — konektor
    ct = df["connector"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    rapikan(ax, "11 — Jenis konektor", "", "jumlah sesi")
    ax.bar(range(len(ct)), ct.values, color=BIRU)
    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(ct.index, rotation=30, ha="right", fontsize=8)
    simpan(fig, "11_konektor.png")
    gambar.append(("11_konektor.png", "Jenis konektor. Tidak dipakai simulasi."))

    # 12 — sesi per pengguna
    uc = df.groupby("id_user").size()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rapikan(ax, "12 — Sesi per pengguna", "sesi per pengguna", "jumlah pengguna")
    ax.hist(uc.clip(upper=50), bins=50, color=BIRU)
    ax.set_yscale("log")
    simpan(fig, "12_pengguna.png")
    gambar.append(("12_pengguna.png",
                   f"{rb(uc.size)} pengguna unik; mayoritas jarang, sedikit sangat sering."))

    # 13 — sebaran spasial
    fig, ax = plt.subplots(figsize=(7, 6))
    rapikan(ax, "13 — Sebaran spasial 77 SPKLU Bandung", "longitude", "latitude")
    ax.grid(False)
    for e in spklu:
        pilih = e["id"] in id13
        ax.scatter(e["lon"], e["lat"], s=14 + e["ports"] * 6,
                   c=(ORANYE if pilih else ABU),
                   edgecolors=TINTA2 if pilih else "none", linewidths=0.6,
                   alpha=0.95 if pilih else 0.6, zorder=3 if pilih else 2)
    simpan(fig, "13_peta.png")
    gambar.append(("13_peta.png",
                   "Oranye = 13 stasiun dunia simulasi (dipilih karena port paling sedikit)."))

    # 14 — port per stasiun
    pp = [e["ports"] for e in spklu if e["ports"] > 0]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rapikan(ax, "14 — Port per stasiun (77 SPKLU aktif)", "port per stasiun", "jumlah SPKLU")
    ax.hist(pp, bins=range(0, max(pp) + 3, 2), color=BIRU)
    simpan(fig, "14_port_per_stasiun.png")
    gambar.append(("14_port_per_stasiun.png",
                   "Mayoritas stasiun kecil; sedikit hub besar."))

    # 15 — volume 13 stasiun terpilih
    c13 = (df[df["nama_spklu"].isin(nama13)].groupby("nama_spklu").size()
           .reindex(sorted(nama13), fill_value=0).sort_values())
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rapikan(ax, "15 — Volume 2024 pada 13 stasiun dunia simulasi", "jumlah sesi 2024", "")
    ax.barh(range(len(c13)), c13.values, color=ORANYE)
    ax.set_yticks(range(len(c13)))
    ax.set_yticklabels([str(nm).replace("SPKLU PLN ", "")[:32] for nm in c13.index],
                       fontsize=8)
    ax.invert_yaxis()
    simpan(fig, "15_volume_13_terpilih.png")
    total13 = int(c13.sum())
    gambar.append(("15_volume_13_terpilih.png",
                   f"Total {rb(total13)} sesi sepanjang 2024. "
                   f"{int((c13 == 0).sum())} stasiun tanpa transaksi sama sekali — "
                   "inilah alasan bobot stasiun dihaluskan Laplace."))

    # ── laporan ──────────────────────────────────────────────────────────────
    baris = [
        "# EDA — data transaksi SPKLU Bandung 2024 (hasil penyaringan)",
        "",
        f"> Sumber: `real_data/DATA TRX SPKLU 2024.xlsx` disaring ke **{rb(n)} transaksi** "
        f"pada {df['nama_spklu'].nunique()} SPKLU wilayah Bandung "
        "(lihat `step1_filter_spklu.py` dan `step2_filter_transaksi.py`).",
        "",
        "## Angka kunci",
        "",
        f"| Besaran | Nilai |",
        f"|---|---|",
        f"| Transaksi | {rb(n)} |",
        f"| SPKLU bertransaksi | {df['nama_spklu'].nunique()} dari {len(spklu)} aktif |",
        f"| Pengguna unik | {rb(uc.size)} |",
        f"| Energi per sesi | mean {df['kwh'].mean():.2f} kWh, median {df['kwh'].median():.2f} kWh |",
        f"| Durasi sesi | median {df['durasi_min'].median():.0f} mnt, mean {df['durasi_min'].mean():.2f} mnt, maks {df['durasi_min'].max():.0f} mnt |",
        f"| Kemiringan durasi | {df['durasi_min'].skew():.3f} (miring kanan) |",
        f"| SoC datang | mean {df['first_soc'].mean():.1f}% |",
        f"| SoC selesai | mean {df['last_soc'].mean():.1f}% |",
        f"| Harga per kWh | median {harga.median():.1f} Rp/kWh |",
        f"| Jam puncak / lembah | {int(h.idxmax())}:00 / {int(h.idxmin())}:00 |",
        f"| Nilai berbeda kolom `hour` | 24 |",
        f"| Nilai berbeda kolom `hour_frac` | {rb(df['hour_frac'].nunique())} |",
        f"| Nilai berbeda kolom `durasi_min` | {rb(df['durasi_min'].nunique())} |",
        "",
        "## Catatan penting untuk pemodelan distribusi",
        "",
        "1. **Kolom `hour` terlalu kasar.** Ekstraksi lama membuang menit sehingga "
        "22.514 transaksi hanya menempati 24 nilai. Kolom `hour_frac` yang baru "
        "mengembalikan resolusi menit dan membuat uji Kolmogorov-Smirnov sah dipakai.",
        "2. **Durasi terpotong engine.** Simulasi membulatkan durasi ke kisi setengah "
        "jam lalu memotongnya di 2 jam, sehingga seluruh ekor panjang menumpuk pada "
        "satu nilai. Karena itu proses fitting dijalankan pada dua basis terpisah "
        "(nilai mentah dan nilai pasca-transformasi) — lihat folder `fitting/`.",
        "3. **Volume stasiun sangat timpang** dan beberapa stasiun dunia simulasi "
        "tidak punya transaksi sama sekali, sehingga bobot pemilihan stasiun "
        "dihaluskan Laplace.",
        "",
        "## Grafik",
        "",
    ]
    for nama_file, keterangan in gambar:
        baris += [f"### {nama_file}", "", keterangan, "", f"![{nama_file}]({nama_file})", ""]

    with open(os.path.join(EDA_DIR, "EDA.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(baris))

    print(f"[step3] {len(gambar)} grafik + EDA.md -> filtering/eda/")


if __name__ == "__main__":
    main()
