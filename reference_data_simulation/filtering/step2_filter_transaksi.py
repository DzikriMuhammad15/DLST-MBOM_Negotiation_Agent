# -*- coding: utf-8 -*-
"""LANGKAH 2 — Filter transaksi charging nasional 2024 menjadi transaksi Bandung.

Masukan  : ``real_data/DATA TRX SPKLU 2024.xlsx`` (sheet ``TRX 2024``, ±402 ribu
           transaksi se-Indonesia, ±48 MB) + ``output/bandung_spklu.json``
           (keluaran langkah 1).
Keluaran : ``output/bandung_trx.csv`` — transaksi yang ``ID_SPKLU``-nya termasuk
           77 stasiun Bandung, ditambah ``output/bandung_trx_summary.json``.

ATURAN FILTER (identik dengan yang dipakai simulasi):

    ID_SPKLU termasuk daftar 77 stasiun Bandung  ->  22.514 transaksi

BEDA DENGAN BERKAS TURUNAN LAMA — kolom waktu
    Ekstraksi lama (``simulation_data/_derived/_extract_bandung_rows.py``) hanya
    menyimpan JAM BULAT: ``hour = int(waktu.split(":")[0])``, sehingga menit dan
    detik dibuang dan 22.514 transaksi hanya menempati 24 nilai berbeda.
    Skrip ini menyimpan tambahan tiga kolom tanpa mengubah kolom lama:

        minute     menit transaksi (0-59)
        second     detik transaksi (0-59)
        hour_frac  jam sebagai pecahan: hour + minute/60 + second/3600

    ``hour_frac`` diperlukan agar variabel jam-mulai dapat diuji dengan
    Kolmogorov-Smirnov secara sah. Pada jam bulat, fungsi distribusi kumulatif
    empiris berbentuk tangga dengan 24 anak tangga besar sehingga jaraknya
    terhadap kurva kontinu manapun otomatis besar — statistik KS menjadi bias
    dan peringkat antar distribusi tidak informatif.

    Kolom ``hour`` yang lama TETAP DIPERTAHANKAN apa adanya, sehingga kedua
    versi dapat dibandingkan berdampingan.

Skrip memverifikasi seluruh kolom bersama terhadap
``simulation_data/_derived/bandung_trx.csv`` (berkas yang benar-benar dipakai
simulasi) — berkas tersebut hanya dibaca, tidak pernah ditulis.

    python reference_data_simulation/filtering/step2_filter_transaksi.py
"""
from __future__ import annotations

import csv
import datetime
import json
import os
from collections import Counter

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
REF_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REF_ROOT, "real_data", "DATA TRX SPKLU 2024.xlsx")
SHEET = "TRX 2024"
OUT_DIR = os.path.join(HERE, "output")

REF_CSV = os.path.join(os.path.dirname(REF_ROOT), "simulation_data", "_derived",
                       "bandung_trx.csv")

COLUMNS = ["id_spklu", "nama_spklu", "chargebox_id", "kw_tier", "date",
           "hour", "minute", "second", "hour_frac", "dow", "month",
           "kwh", "rpkwh", "rppakai", "price_per_kwh", "durasi_min",
           "first_soc", "last_soc", "connector", "id_user"]
# Kolom yang juga ada pada berkas turunan lama (dipakai untuk verifikasi).
SHARED = [c for c in COLUMNS if c not in ("minute", "second", "hour_frac")]


def _num(value):
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return ""


def _parse_waktu(stamp):
    """'01/01/2024 13:47:22,2960' -> (tanggal, jam, menit, detik, dow, bulan)."""
    try:
        tanggal, waktu = str(stamp).split(" ", 1)
        dd, mm, yy = tanggal.split("/")
        jam_str = waktu.split(",")[0]
        bagian = jam_str.split(":")
        hh = int(bagian[0])
        mi = int(bagian[1]) if len(bagian) > 1 else 0
        ss = int(float(bagian[2])) if len(bagian) > 2 else 0
        dow = datetime.date(int(yy), int(mm), int(dd)).weekday()
        return f"{yy}-{mm}-{dd}", hh, mi, ss, dow, int(mm)
    except (ValueError, IndexError, AttributeError):
        return "", "", "", "", "", ""


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "bandung_spklu.json"), "r", encoding="utf-8") as fh:
        spklu = json.load(fh)
    bandung_ids = {e["id"] for e in spklu}
    nama = {e["id"]: e["nama"] for e in spklu}
    print(f"[step2] daftar acuan: {len(bandung_ids)} stasiun Bandung")

    wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
    ws = wb[SHEET]
    it = ws.iter_rows(values_only=True)
    header = list(next(it))
    idx = {h: i for i, h in enumerate(header) if h is not None}

    def get(row, key):
        i = idx.get(key)
        return row[i] if i is not None and i < len(row) else None

    out_path = os.path.join(OUT_DIR, "bandung_trx.csv")
    n_total = n_bandung = 0
    hour_hist, dur_hist = Counter(), Counter()
    kwh_sum, kwh_n = 0.0, 0
    fs_sum, fs_n = 0.0, 0

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in it:
            if get(row, "ID_SPKLU") is None:
                continue
            n_total += 1
            sid = str(get(row, "ID_SPKLU")).split(".")[0]
            if sid not in bandung_ids:
                continue
            n_bandung += 1

            tanggal, hh, mi, ss, dow, bulan = _parse_waktu(get(row, "TANGGAL"))
            hour_frac = (round(hh + mi / 60.0 + ss / 3600.0, 6)
                         if hh != "" else "")
            kwh = _num(get(row, "KWH"))
            rpkwh = _num(get(row, "RPKWH"))
            harga = (round(rpkwh / kwh, 2)
                     if isinstance(kwh, float) and kwh > 0 and isinstance(rpkwh, float)
                     else "")
            durasi = _num(get(row, "DURASI"))
            first_soc = _num(get(row, "FIRST_SOC"))
            last_soc = _num(get(row, "LAST_SOC"))

            writer.writerow([
                sid, nama.get(sid, ""), get(row, "CHARGEBOX_ID"),
                str(get(row, "DAYA_CHARGEBOX")).strip(),
                tanggal, hh, mi, ss, hour_frac, dow, bulan,
                kwh, rpkwh, _num(get(row, "RPPAKAI")), harga, durasi,
                first_soc, last_soc, get(row, "TYPE_CONNECTOR"),
                str(get(row, "ID_USER")).split(".")[0],
            ])

            if hh != "":
                hour_hist[hh] += 1
            if isinstance(durasi, float) and durasi >= 0:
                dur_hist[min(int(durasi // 15) * 15, 240)] += 1
            if isinstance(kwh, float) and kwh > 0:
                kwh_sum += kwh
                kwh_n += 1
            if isinstance(first_soc, float) and 0 <= first_soc <= 100:
                fs_sum += first_soc
                fs_n += 1
    wb.close()

    summary = {
        "rows_bandung": n_bandung,
        "hour_hist": {str(h): hour_hist.get(h, 0) for h in range(24)},
        "durasi_hist_bin15": {str(e): dur_hist[e] for e in sorted(dur_hist)},
        "kwh_mean": round(kwh_sum / max(kwh_n, 1), 2),
        "firstsoc_mean_pct": round(fs_sum / max(fs_n, 1), 1),
    }
    with open(os.path.join(OUT_DIR, "bandung_trx_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    print(f"[step2] transaksi nasional dibaca : {n_total}")
    print(f"[step2] transaksi Bandung lolos   : {n_bandung}  -> output/bandung_trx.csv")
    print(f"[step2] kwh_mean={summary['kwh_mean']}  "
          f"firstsoc_mean={summary['firstsoc_mean_pct']}%")

    verify(out_path)


def verify(out_path: str) -> None:
    """Bandingkan kolom bersama terhadap berkas turunan yang dipakai simulasi."""
    # Lokasi dicatat relatif terhadap akar project, bukan mutlak, supaya laporan
    # ini identik di project manapun berkas ini disalin.
    akar_project = os.path.dirname(REF_ROOT)
    report = {"referensi": os.path.relpath(REF_CSV, akar_project).replace("\\", "/"),
              "referensi_tersedia": os.path.exists(REF_CSV)}
    if not report["referensi_tersedia"]:
        print("[step2] berkas referensi tidak ditemukan — verifikasi dilewati")
    else:
        with open(REF_CSV, "r", encoding="utf-8") as fh:
            ref = list(csv.DictReader(fh))
        with open(out_path, "r", encoding="utf-8") as fh:
            new = list(csv.DictReader(fh))
        beda = Counter()
        n_cmp = min(len(ref), len(new))
        for a, b in zip(ref, new):
            for col in SHARED:
                if col in a and str(a[col]) != str(b[col]):
                    beda[col] += 1
        report.update({
            "n_referensi": len(ref),
            "n_hasil": len(new),
            "jumlah_baris_sama": len(ref) == len(new),
            "baris_dibandingkan": n_cmp,
            "kolom_dibandingkan": SHARED,
            "sel_berbeda_per_kolom": dict(beda),
            "IDENTIK": len(ref) == len(new) and not beda,
        })
        tanda = "IDENTIK" if report["IDENTIK"] else "BERBEDA"
        print(f"[step2] verifikasi kolom bersama vs berkas simulasi: {tanda} "
              f"({len(new)} vs {len(ref)} baris)")
        if beda:
            print(f"[step2]   kolom berbeda: {dict(beda)}")

    # Verifikasi ringkasan histogram terhadap berkas yang dibaca sampler.
    ref_sum = os.path.join(os.path.dirname(REF_ROOT), "simulation_data", "_derived",
                           "bandung_only_trx_summary.json")
    if os.path.exists(ref_sum):
        with open(ref_sum, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        with open(os.path.join(OUT_DIR, "bandung_trx_summary.json"), "r",
                  encoding="utf-8") as fh:
            n = json.load(fh)
        sama = (r["rows_bandung"] == n["rows_bandung"]
                and r["hour_hist"] == n["hour_hist"]
                and r["durasi_hist_bin15"] == n["durasi_hist_bin15"])
        report["ringkasan_sampler_identik"] = sama
        print(f"[step2] verifikasi histogram yang dibaca sampler: "
              f"{'IDENTIK' if sama else 'BERBEDA'}")

    with open(os.path.join(OUT_DIR, "filtering_transaksi_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
