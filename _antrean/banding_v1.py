# -*- coding: utf-8 -*-
"""Membandingkan hasil v_1 yang baru dengan angka Bab V yang berlaku sekarang.

Dipakai untuk memantau sel mana yang kolaps. Jalankan dari root project::

    .venv/Scripts/python.exe _antrean/banding_v1.py

Angka pembanding diambil dari Tabel V.4 (deal-rate) dan Tabel V.5 (order
terlayani) pada ``laporan/thesis_ujung_parah.docx``, yaitu korpus LAMA yang
belum memakai perbaikan fallback menghadap depan. Selisih besar bukan otomatis
bug, sebab kolaps memang gejala yang diteliti, tetapi wajib disadari karena
v_1 hanya punya SATU seed per dunia sehingga satu run menentukan satu baris.
"""
import glob
import json
import os

# (dunia, exp) -> (deal-rate lama, order terlayani lama)
LAMA = {
    ("hz_n1_upd1_s42", "exp06"): (1.000, 87),
    ("hz_n1_upd1_s42", "exp08"): (1.000, 90),
    ("hz_n1_upd1_s42", "exp09"): (0.900, 83),
    ("hz_n5_upd1_s42", "exp06"): (1.000, 453),
    ("hz_n5_upd1_s42", "exp08"): (1.000, 443),
    ("hz_n5_upd1_s42", "exp09"): (0.976, 442),
    ("ref_n10_upd1_s42", "exp06"): (0.988, 898),
    ("ref_n10_upd1_s42", "exp08"): (1.000, 901),
    ("ref_n10_upd1_s42", "exp09"): (0.988, 897),
    ("hz_n30_upd1_s42", "exp06"): (1.000, 2700),
    ("hz_n30_upd1_s42", "exp08"): (1.000, 2704),
    ("hz_n30_upd1_s42", "exp09"): (0.996, 2680),
    ("control_default_s42", "exp06"): (0.200, 66),
    ("control_default_s42", "exp08"): (0.692, 84),
    ("control_default_s42", "exp09"): (0.643, 79),
    ("uf_n10_upd2_s42", "exp06"): (0.953, 896),
    ("uf_n10_upd2_s42", "exp08"): (1.000, 912),
    ("uf_n10_upd2_s42", "exp09"): (0.976, 891),
    ("uf_n10_upd5_s42", "exp06"): (1.000, 915),
    ("uf_n10_upd5_s42", "exp08"): (1.000, 905),
    ("uf_n10_upd5_s42", "exp09"): (0.941, 886),
    ("sc_scarce_n10_upd1_s42", "exp06"): (0.988, 1853),
    ("sc_scarce_n10_upd1_s42", "exp08"): (1.000, 1866),
    ("sc_scarce_n10_upd1_s42", "exp09"): (0.994, 1861),
    ("sc_abundant_n10_upd1_s42", "exp06"): (1.000, 615),
    ("sc_abundant_n10_upd1_s42", "exp08"): (1.000, 616),
    ("sc_abundant_n10_upd1_s42", "exp09"): (0.981, 594),
    ("soc_low_n10_upd1_s42", "exp06"): (1.000, 894),
    ("soc_low_n10_upd1_s42", "exp08"): (1.000, 901),
    ("soc_low_n10_upd1_s42", "exp09"): (0.976, 896),
    ("soc_high_n10_upd1_s42", "exp06"): (1.000, 904),
    ("soc_high_n10_upd1_s42", "exp08"): (1.000, 897),
    ("soc_high_n10_upd1_s42", "exp09"): (0.988, 908),
    ("cc_alternating_n10_upd1_s42", "exp06"): (1.000, 904),
    ("cc_alternating_n10_upd1_s42", "exp08"): (1.000, 903),
    ("cc_alternating_n10_upd1_s42", "exp09"): (0.976, 891),
    ("cc_fixedopp_n10_upd1_s42", "exp06"): (0.976, 879),
    ("cc_fixedopp_n10_upd1_s42", "exp08"): (1.000, 901),
    ("cc_fixedopp_n10_upd1_s42", "exp09"): (0.097, 364),
}
AGEN = {"exp06": "DLST-MBOM", "exp08": "DLST-MBOM Nego", "exp09": "DLST murni"}
AMBANG = 0.30  # selisih deal-rate yang dianggap perlu diperhatikan

# Pendapatan BERSIH fleet dalam juta rupiah, yaitu pendapatan penumpang dikurangi
# biaya pengisian. Itu definisi kolom "Revenue fleet" pada laporan, BUKAN
# pendapatan kotor. Angka lama hanya tersedia untuk dunia referensi (Tabel V.2).
LAMA_REVENUE = {
    ("ref_n10_upd1_s42", "exp06"): 16.04,
    ("ref_n10_upd1_s42", "exp08"): 15.96,
    ("ref_n10_upd1_s42", "exp09"): 15.23,
}


def main() -> None:
    pola = "v_1/data/experiment_bulk/*/exp*_padded_*/metrices/aggregate_metrics.json"
    fs = sorted(glob.glob(pola))
    if not fs:
        raise SystemExit("belum ada run v_1 yang selesai")

    print(f"{'dunia':<28}{'agen':<16}{'deal':>7}{'lama':>7}"
          f"{'order':>8}{'lama':>8}{'laba':>8}{'lama':>7}  catatan")
    n_kolaps = 0
    for f in fs:
        bagian = f.replace(os.sep, "/").split("/")
        dunia, exp = bagian[3], bagian[4].split("_")[0]
        s = json.load(open(f, encoding="utf-8"))["summary"]
        deal = s["negotiation_system"]["deal_success_rate"]
        fb = s["fleet_business"]
        order = fb["orders_served"]
        laba = (fb["passenger_revenue"] - fb["charging_cost"]) / 1e6
        deal_lama, order_lama = LAMA.get((dunia, exp), (None, None))
        laba_lama = LAMA_REVENUE.get((dunia, exp))

        catatan = ""
        if deal_lama is not None and deal_lama - deal > AMBANG:
            catatan = "KOLAPS"
            n_kolaps += 1
        sdl = f"{deal_lama:.3f}" if deal_lama is not None else "-"
        sol = str(order_lama) if order_lama is not None else "-"
        sll = f"{laba_lama:.2f}" if laba_lama is not None else "-"
        print(f"{dunia:<28}{AGEN.get(exp, exp):<16}{deal:>7.3f}{sdl:>7}"
              f"{order:>8}{sol:>8}{laba:>8.2f}{sll:>7}  {catatan}")

    print(f"\n{len(fs)} sel selesai, {n_kolaps} di antaranya kolaps "
          f"(deal-rate turun lebih dari {AMBANG:.2f} dari korpus lama).")


if __name__ == "__main__":
    main()
