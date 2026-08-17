# -*- coding: utf-8 -*-
"""Memeriksa keutuhan korpus setelah antrean dihentikan.

Tiga hal yang dipastikan:

1. Setiap run yang dianggap selesai penandanya benar-benar JSON utuh dan berisi
   blok metrik yang diharapkan, bukan berkas terpotong.
2. Tidak ada folder run yang setengah jadi, yaitu folder yang sudah terbentuk
   tetapi tidak punya penanda. Folder seperti itu akan dihapus dan diulang saat
   continue, jadi keberadaannya bukan kerusakan, tetapi tetap perlu dilaporkan.
3. Tidak ada penanda tiruan sisa pengujian.

Pakai dari root project::

    .venv/Scripts/python.exe _antrean/periksa.py
"""
import glob
import json
import os

PENANDA = os.path.join("metrices", "aggregate_metrics.json")

# Folder yang boleh berisi run, beserta kedalaman folder run-nya.
AKAR_RUN = [
    "v_1/data/experiment_bulk/*/exp*_padded_*",
    "v_1/data/greedy/*",
    "v_1/data/as_is/*",
    "v_1/data/booking/*",
    "v_2/data_*/negotiation/real_case_v2/run*_padded_*",
    "v_2/data_*/non_negotiation/real_case_v2/*/run*",
]


def main() -> None:
    utuh, rusak, parsial, tiruan = [], [], [], []

    for pola in AKAR_RUN:
        for folder in sorted(glob.glob(pola)):
            if not os.path.isdir(folder):
                continue
            m = os.path.join(folder, PENANDA)
            if not os.path.isfile(m):
                parsial.append(folder)
                continue
            try:
                isi = json.load(open(m, encoding="utf-8"))
            except Exception as e:
                rusak.append((folder, f"{type(e).__name__}: {e}"))
                continue
            if isi.get("stub"):
                tiruan.append(folder)
            elif "summary" not in isi:
                rusak.append((folder, "tanpa blok 'summary'"))
            else:
                utuh.append(folder)

    print(f"run utuh dan lengkap        : {len(utuh)}")
    print(f"penanda rusak / tak lengkap : {len(rusak)}")
    print(f"folder parsial tanpa penanda: {len(parsial)}")
    print(f"penanda tiruan sisa uji     : {len(tiruan)}")

    for f, sebab in rusak:
        print(f"  RUSAK   {f}  ({sebab})")
    for f in parsial:
        print(f"  PARSIAL {f}  -> akan dihapus dan diulang saat continue")
    for f in tiruan:
        print(f"  TIRUAN  {f}  -> WAJIB dihapus")

    # Folder singgah v_1 diperiksa terpisah, sebab isinya memang belum di tujuan.
    singgah = glob.glob("v_1/data/_singgah/*/*")
    if singgah:
        print("\nfolder singgah:")
        for f in singgah:
            lengkap = os.path.isfile(os.path.join(f, PENANDA))
            status = ("LENGKAP, akan diselamatkan tanpa dijalankan ulang"
                      if lengkap else "belum selesai, akan diulang dari awal")
            print(f"  {f}  -> {status}")

    aman = not rusak and not tiruan
    print("\n" + ("KORPUS AMAN, tidak ada yang rusak."
                  if aman else "ADA MASALAH, lihat daftar di atas."))


if __name__ == "__main__":
    main()
