# -*- coding: utf-8 -*-
"""Pencocokan sepuluh keluarga distribusi ke dua variabel simulasi.

Masukan  : ``../preprocessing/output/{durasi_mentah,jam_mentah_menit}.csv``
Keluaran : ``results/metrik__<target>.csv``
           ``results/parameter__<target>.json``
           ``results/RINGKASAN.md``
           ``best_parameters.json`` dan ``best_parameters.md``

Lokasi setiap keluarga dikunci di nol dan tidak diduga, sehingga seluruh
distribusi bermula dari titik nol dan tidak ada satupun keluarga yang menang
karena menggeser titik awalnya.

    python reference_data_simulation/fitting/fit_distributions.py
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np

from common import (KELUARGA, KELUARGA_MENURUT_NAMA, PLOT_DIR, RESULT_DIR,
                    HERE, Target, ad_angka, duga_parameter, metrik,
                    muat_target, nama_parameter, rb, tepi_bin)

# Metrik yang dipakai untuk memeringkat. AIC dipilih karena berbasis fungsi
# kemungkinan, sehingga membandingkan keluarga dengan jumlah parameter berbeda
# secara setara, dan karena seluruh keluarga diduga dengan cara yang sama.
METRIK_PERINGKAT = "aic"


def cocokkan(t: Target) -> List[Dict]:
    """Duga parameter dan hitung metrik untuk kesepuluh keluarga."""
    tepi = tepi_bin(t)
    baris: List[Dict] = []
    for k in KELUARGA:
        mulai = time.time()
        catatan = ""
        try:
            param, bebas = duga_parameter(k.nama, t.data)
            m = metrik(k.nama, param, t.data, bebas, tepi)
            status = "berhasil" if np.isfinite(m["aic"]) else "gagal: kemungkinan nol"
        except Exception as exc:                                    # noqa: BLE001
            param, bebas = (), 0
            m = {}
            status = f"gagal: {type(exc).__name__}"
            catatan = str(exc)[:200]
        baris.append({
            "keluarga": k.nama,
            "label": k.label,
            "status": status,
            "n": t.n,
            "parameter": list(param),
            "nama_parameter": nama_parameter(k.nama),
            "detik": round(time.time() - mulai, 2),
            "catatan": catatan or k.catatan,
            **m,
        })
        tanda = "ok " if status == "berhasil" else "GAGAL"
        aic = m.get("aic")
        print(f"  [{tanda}] {k.nama:<12} "
              f"AIC={ad_angka(aic, 1) if aic is not None else '-':>16}  "
              f"KS={ad_angka(m.get('ks'), 4) if m else '-':>8}  "
              f"({round(time.time() - mulai, 1)} detik)")
    return baris


def peringkatkan(baris: List[Dict]) -> List[Dict]:
    """Urutkan menaik menurut AIC; yang gagal ditaruh paling belakang."""
    berhasil = [b for b in baris if b["status"] == "berhasil"]
    gagal = [b for b in baris if b["status"] != "berhasil"]
    berhasil.sort(key=lambda b: b[METRIK_PERINGKAT])
    for i, b in enumerate(berhasil, start=1):
        b["peringkat"] = i
    for b in gagal:
        b["peringkat"] = None
    return berhasil + gagal


# Lama eksekusi sengaja TIDAK ditulis ke berkas hasil. Angkanya bergantung mesin,
# sehingga akan membuat berkas hasil berbeda antar project padahal seluruh angka
# statistiknya identik. Lama eksekusi tetap dicetak ke layar saat dijalankan.
KOLOM_CSV = ["peringkat", "keluarga", "label", "status", "n", "k_bebas",
             "log_kemungkinan", "aic", "bic", "ks", "ks_d_plus", "ks_d_minus",
             "cvm", "ad", "wasserstein", "tvd", "massa_di_luar_bin",
             "titik_kepadatan_diabaikan", "catatan"]


def tulis_csv(baris: List[Dict], path: str) -> None:
    import csv
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KOLOM_CSV, extrasaction="ignore")
        w.writeheader()
        for b in baris:
            w.writerow(b)


def tabel_markdown(baris: List[Dict]) -> str:
    kepala = ("| Peringkat | Distribusi | Parameter bebas | Log-kemungkinan | AIC "
              "| Jarak KS | Cramer-von Mises | Anderson-Darling | Wasserstein "
              "| Massa salah tempat |\n")
    kepala += "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    isi = ""
    for b in baris:
        if b["status"] != "berhasil":
            isi += (f"| - | {b['label']} | - | - | - | - | - | - | - | "
                    f"{b['status']} |\n")
            continue
        isi += (f"| {b['peringkat']} | {b['label']} | {b['k_bebas']} "
                f"| {ad_angka(b['log_kemungkinan'], 1)} "
                f"| {ad_angka(b['aic'], 1)} "
                f"| {ad_angka(b['ks'], 4)} "
                f"| {ad_angka(b['cvm'], 2)} "
                f"| {ad_angka(b['ad'], 2)} "
                f"| {ad_angka(b['wasserstein'], 3)} "
                f"| {ad_angka(b['tvd'] * 100, 2)} persen |\n")
    return kepala + isi


def tabel_parameter(baris: List[Dict]) -> str:
    kepala = "| Distribusi | Parameter terbaik keluarga itu | Lokasi |\n|---|---|---|\n"
    isi = ""
    for b in baris:
        if b["status"] != "berhasil":
            isi += f"| {b['label']} | - | - |\n"
            continue
        nama_p = b["nama_parameter"]
        nilai = list(b["parameter"])
        # scipy mengembalikan (*bentuk, lokasi, skala); lokasi dipisah.
        lokasi = nilai[-2]
        pasangan = list(zip(nama_p[:-1], nilai[:-2])) + [(nama_p[-1], nilai[-1])]
        teks = "; ".join(f"{n} = {ad_angka(v, 5)}" for n, v in pasangan)
        isi += f"| {b['label']} | {teks} | {ad_angka(lokasi, 1)} (dikunci) |\n"
    return kepala + isi


def main() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    target = muat_target()
    semua: Dict[str, List[Dict]] = {}
    ringkas = ["# Hasil fit distribusi\n",
               "Sepuluh keluarga distribusi bersupport tak-negatif dicocokkan ke "
               "dua variabel yang disampel simulasi. Titik awal seluruh keluarga "
               "dikunci di nol.\n"]

    for t in target:
        print(f"\n=== {t.kunci} (n = {rb(t.n)}) ===")
        baris = peringkatkan(cocokkan(t))
        semua[t.kunci] = baris

        tulis_csv(baris, os.path.join(RESULT_DIR, f"metrik__{t.kunci}.csv"))
        with open(os.path.join(RESULT_DIR, f"parameter__{t.kunci}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump([{
                "peringkat": b["peringkat"], "keluarga": b["keluarga"],
                "label": b["label"], "status": b["status"],
                "nama_parameter": b["nama_parameter"],
                "parameter": b["parameter"],
                "aic": b.get("aic"), "ks": b.get("ks"),
            } for b in baris], fh, indent=1, ensure_ascii=False)

        ringkas.append(f"\n## {t.judul} (`{t.kunci}`)\n")
        ringkas.append(f"Satuan {t.satuan}. Banyak pengamatan {rb(t.n)}. "
                       f"{t.keterangan}\n")
        ringkas.append("\n### Peringkat menurut AIC\n\n" + tabel_markdown(baris))
        ringkas.append("\n### Parameter hasil pendugaan\n\n" + tabel_parameter(baris))

    with open(os.path.join(RESULT_DIR, "RINGKASAN.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(ringkas))

    # ── Parameter terbaik per keluarga per variabel ──────────────────────────
    terbaik: Dict[str, object] = {
        "keterangan": "Parameter terbaik hasil pendugaan kemungkinan maksimum "
                      "untuk setiap keluarga distribusi, lokasi dikunci di nol.",
        "metrik_peringkat": METRIK_PERINGKAT,
    }
    for kunci, baris in semua.items():
        terbaik[kunci] = {
            "n": baris[0]["n"],
            "juara": next((b["keluarga"] for b in baris if b["peringkat"] == 1), None),
            "keluarga": [{
                "peringkat": b["peringkat"],
                "keluarga": b["keluarga"],
                "label": b["label"],
                "status": b["status"],
                "nama_parameter": b["nama_parameter"],
                "parameter": b["parameter"],
                "lokasi_dikunci": 0.0,
                "aic": b.get("aic"), "bic": b.get("bic"), "ks": b.get("ks"),
                "cvm": b.get("cvm"), "ad": b.get("ad"),
                "wasserstein": b.get("wasserstein"), "tvd": b.get("tvd"),
            } for b in baris],
        }
    with open(os.path.join(HERE, "best_parameters.json"), "w", encoding="utf-8") as fh:
        json.dump(terbaik, fh, indent=1, ensure_ascii=False)

    md = ["# Parameter terbaik tiap keluarga distribusi\n",
          "\nParameter di bawah adalah hasil pendugaan kemungkinan maksimum untuk "
          "setiap keluarga, dengan parameter lokasi dikunci di nol dan karena itu "
          "tidak diduga. Urutan peringkat memakai AIC.\n"]
    for t in target:
        baris = semua[t.kunci]
        md.append(f"\n## {t.judul} (`{t.kunci}`, satuan {t.satuan})\n\n")
        md.append(tabel_parameter(baris))
        md.append("\n### Peringkat dan kecocokan\n\n")
        md.append(tabel_markdown(baris))
    with open(os.path.join(HERE, "best_parameters.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(md))

    print("\nSELESAI")
    for t in target:
        juara = next(b for b in semua[t.kunci] if b["peringkat"] == 1)
        print(f"  {t.kunci:<18} juara = {juara['keluarga']} "
              f"(AIC {ad_angka(juara['aic'], 1)}, KS {ad_angka(juara['ks'], 4)})")


if __name__ == "__main__":
    main()
