# -*- coding: utf-8 -*-
"""Tabel status kampanye, skenario mana yang sudah dan yang belum.

Dihitung langsung dari penanda ``metrices/aggregate_metrics.json`` di disk, jadi
selalu mencerminkan keadaan sebenarnya. Aman dijalankan saat antrean berjalan.

Pakai dari root project::

    .venv/Scripts/python.exe _antrean/status.py
"""
import glob
import os

# ── v_1, non-real-case. Satu ulangan per sel, tiga pasangan agen mode PADDED. ──
DUNIA = [
    ("ref_n10_upd1_s42", "10 hari (REF)"),
    ("control_default_s42", "Default tanpa-belajar"),
    ("hz_n1_upd1_s42", "1 hari"),
    ("hz_n5_upd1_s42", "5 hari"),
    ("hz_n30_upd1_s42", "30 hari"),
    ("uf_n10_upd2_s42", "Tiap 2 episode"),
    ("uf_n10_upd5_s42", "Tiap 5 episode"),
    ("sc_scarce_n10_upd1_s42", "Kelangkaan tinggi"),
    ("sc_abundant_n10_upd1_s42", "Sumber daya longgar"),
    ("soc_low_n10_upd1_s42", "SoC rendah"),
    ("soc_high_n10_upd1_s42", "SoC tinggi"),
    ("cc_alternating_n10_upd1_s42", "Lawan bergiliran"),
    ("cc_fixedopp_n10_upd1_s42", "Lawan beku"),
]
SEL_V1 = [
    ("exp06", "DLST-MBOM"),
    ("exp08", "DLST-MBOM Nego"),
    ("exp09", "DLST murni"),
    ("greedy", "Greedy"),
    ("booking", "Pemesanan statis"),
    ("as_is", "Tanpa pemesanan"),
]

# ── v_2, real-case. Empat puluh ulangan per sel. ──────────────────────────────
SKENARIO = [
    ("medium_occupancy", "Permintaan 185"),
    ("high_occupancy", "Permintaan 366"),
    ("mahalkompen_medium_occupancy", "Mahalkompen 185"),
    ("mahalkompen_high_occupancy", "Mahalkompen 366"),
]
SEL_V2 = [
    ("dlst", "DLST"),
    ("dlstmbom", "DLST-MBOM"),
    ("dlstmbomnew", "DLST-MBOM Nego"),
    ("directed", "DLSTMBOMDirected"),
    ("greedy", "Greedy"),
    ("booking", "Pemesanan statis"),
    ("as_is", "Tanpa pemesanan"),
]
ULANGAN_V2 = 40


def cacah(pola: str) -> int:
    return sum(1 for m in glob.glob(pola) if os.path.isfile(m))


def sel_v1_selesai(dunia: str, sel: str) -> int:
    if sel.startswith("exp"):
        pola = (f"v_1/data/experiment_bulk/{dunia}/{sel}_padded_*/"
                "metrices/aggregate_metrics.json")
    else:
        pola = f"v_1/data/{sel}/{dunia}/metrices/aggregate_metrics.json"
    return min(1, cacah(pola))


def sel_v2_selesai(skenario: str, sel: str) -> int:
    if sel in ("booking", "as_is"):
        pola = (f"v_2/data_{skenario}/non_negotiation/real_case_v2/{sel}/run*/"
                "metrices/aggregate_metrics.json")
    else:
        pola = (f"v_2/data_{skenario}/negotiation/real_case_v2/run*_padded_{sel}/"
                "metrices/aggregate_metrics.json")
    return cacah(pola)


def garis(lebar):
    return "+" + "+".join("-" * w for w in lebar) + "+"


def baris(sel_teks, lebar, rata="<"):
    isi = "|".join(f"{t:{rata}{w}}" for t, w in zip(sel_teks, lebar))
    return "|" + isi + "|"


def tabel_v1() -> int:
    lebar = [23] + [17] * len(SEL_V1)
    print("v_1 NON-REAL-CASE, target 1 ulangan tiap sel")
    print(garis(lebar))
    print(baris([" Dunia"] + [f" {n}" for _, n in SEL_V1], lebar))
    print(garis(lebar))
    total = 0
    for kunci, label in DUNIA:
        nilai = [sel_v1_selesai(kunci, s) for s, _ in SEL_V1]
        total += sum(nilai)
        teks = [f" {label}"] + [("   selesai" if v else "   BELUM") for v in nilai]
        print(baris(teks, lebar))
    print(garis(lebar))
    print(f"  subtotal v_1: {total} dari {len(DUNIA) * len(SEL_V1)} run\n")
    return total


def tabel_v2() -> int:
    lebar = [20] + [18] * len(SEL_V2)
    print(f"v_2 REAL-CASE, target {ULANGAN_V2} ulangan tiap sel")
    print(garis(lebar))
    print(baris([" Skenario"] + [f" {n}" for _, n in SEL_V2], lebar))
    print(garis(lebar))
    total = 0
    for kunci, label in SKENARIO:
        nilai = [sel_v2_selesai(kunci, s) for s, _ in SEL_V2]
        total += sum(nilai)
        teks = [f" {label}"] + [f"   {v} / {ULANGAN_V2}" for v in nilai]
        print(baris(teks, lebar))
    print(garis(lebar))
    n_sel = len(SKENARIO) * len(SEL_V2) * ULANGAN_V2
    print(f"  subtotal v_2: {total} dari {n_sel} run\n")
    return total


def main() -> None:
    a = tabel_v1()
    b = tabel_v2()
    print(f"TOTAL {a + b} dari 1198 run")


if __name__ == "__main__":
    main()
