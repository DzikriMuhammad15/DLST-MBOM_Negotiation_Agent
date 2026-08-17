# -*- coding: utf-8 -*-
"""LANGKAH 1 — Filter master SPKLU nasional menjadi daftar SPKLU wilayah Bandung.

Masukan  : ``real_data/MASTER SPKLU PER 14 MAR 2025_.xlsx`` (sheet
           ``SPKLU AKTIF PER MESIN (2)``, 3.586 baris mesin / 2.431 stasiun
           se-Indonesia).
Keluaran : ``output/bandung_spklu.json`` — 77 stasiun aktif wilayah Bandung,
           lengkap dengan koordinat dan daftar chargebox (daya + jumlah
           konektor).

ATURAN FILTER (direkonstruksi dan diverifikasi identik dengan berkas
``simulation_data/_derived/bandung_spklu.json`` yang dipakai simulasi):

    IS_ACTIVE_SPKLU == "Y"
    DAN  -7,05 <= LATITUDE  <= -6,80
    DAN 107,50 <= LONGITUDE <= 107,78

Kotak koordinat dipakai (bukan kolom ``KOTA KABUPATEN``) karena 41 dari 77
stasiun adalah SPKLU mitra non-PLN (Hyundai, Toyota, Wuling, Voltron, Starvo)
yang kolom kota/UID-nya kosong pada master, sehingga penyaringan berbasis nama
wilayah akan membuang stasiun-stasiun itu.

CATATAN JUJUR — jumlah konektor
    Kolom yang tersalin ke medan ``connectors`` adalah ``JML_CONNECTOR_SPKLU``
    (total konektor SATU STASIUN), bukan ``JML_CONNECTOR_CHARGER`` (konektor
    milik chargebox itu sendiri). Untuk stasiun ber-chargebox banyak, angka
    total stasiun tersalin berulang ke setiap chargebox, sehingga
    ``sum(cb["connectors"])`` melebih-hitung jumlah port. Perilaku ini
    DIPERTAHANKAN apa adanya supaya berkas keluaran identik dengan yang dipakai
    simulasi; besarnya penyimpangan dilaporkan skrip ini (lihat
    ``output/filtering_spklu_report.json``). 13 stasiun dunia simulasi tidak
    terdampak karena masing-masing hanya punya satu chargebox.

    python reference_data_simulation/filtering/step1_filter_spklu.py
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
REF_ROOT = os.path.dirname(HERE)                      # reference_data_simulation/
SRC = os.path.join(REF_ROOT, "real_data", "MASTER SPKLU PER 14 MAR 2025_.xlsx")
SHEET = "SPKLU AKTIF PER MESIN (2)"
OUT_DIR = os.path.join(HERE, "output")

# Kotak koordinat wilayah Bandung raya (lihat docstring).
LAT_MIN, LAT_MAX = -7.05, -6.80
LON_MIN, LON_MAX = 107.50, 107.78

# Berkas pembanding: turunan yang BENAR-BENAR dipakai simulasi. Dipakai hanya
# untuk verifikasi (tidak pernah ditulis ulang).
REF_JSON = os.path.join(os.path.dirname(REF_ROOT), "simulation_data", "_derived",
                        "bandung_spklu.json")


def _num(value):
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
    ws = wb[SHEET]
    it = ws.iter_rows(values_only=True)
    header = list(next(it))
    idx = {h: i for i, h in enumerate(header) if h is not None}

    def get(row, key):
        i = idx.get(key)
        return row[i] if i is not None and i < len(row) else None

    stations: Dict[str, Dict] = {}
    n_rows = n_station_all = 0
    seen_all = set()
    for row in it:
        if get(row, "ID_SPKLU") is None:
            continue
        n_rows += 1
        sid = str(get(row, "ID_SPKLU")).split(".")[0]
        seen_all.add(sid)

        if str(get(row, "IS_ACTIVE_SPKLU")).strip().upper() != "Y":
            continue
        lat, lon = _num(get(row, "LATITUDE")), _num(get(row, "LONGITUDE"))
        if lat is None or lon is None:
            continue
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue

        entry = stations.setdefault(sid, {
            "id": sid,
            "nama": get(row, "NAMA_SPKLU"),
            "lat": lat,
            "lon": lon,
            "chargeboxes": [],
        })
        entry["chargeboxes"].append({
            "kw": str(get(row, "DAYA_CHARGEBOX")).strip(),
            # lihat CATATAN JUJUR pada docstring modul
            "connectors": int(_num(get(row, "JML_CONNECTOR_SPKLU")) or 0),
            "id": get(row, "CHARGEBOX_ID"),
            # kolom pembanding, TIDAK dipakai simulasi — hanya untuk laporan
            "_connectors_charger": int(_num(get(row, "JML_CONNECTOR_CHARGER")) or 0),
        })
    wb.close()
    n_station_all = len(seen_all)

    # Urutan keluaran mengikuti urutan kemunculan pertama di master.
    out: List[Dict] = []
    for sid, entry in stations.items():
        out.append({
            "id": entry["id"],
            "nama": entry["nama"],
            "lat": entry["lat"],
            "lon": entry["lon"],
            "chargeboxes": [{"kw": cb["kw"], "connectors": cb["connectors"],
                             "id": cb["id"]} for cb in entry["chargeboxes"]],
        })

    out_path = os.path.join(OUT_DIR, "bandung_spklu.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    # ── verifikasi terhadap berkas yang dipakai simulasi ────────────────────
    verification = {"referensi_tersedia": os.path.exists(REF_JSON)}
    if verification["referensi_tersedia"]:
        with open(REF_JSON, "r", encoding="utf-8") as fh:
            ref = json.load(fh)
        ref_by_id = {e["id"]: e for e in ref}
        ids_new, ids_ref = {e["id"] for e in out}, set(ref_by_id)
        beda_isi = []
        for e in out:
            r = ref_by_id.get(e["id"])
            if r is None:
                continue
            if (round(e["lat"], 6) != round(r["lat"], 6)
                    or round(e["lon"], 6) != round(r["lon"], 6)
                    or e["nama"] != r["nama"]
                    or [(c["kw"], c["connectors"]) for c in e["chargeboxes"]]
                    != [(c["kw"], c["connectors"]) for c in r["chargeboxes"]]):
                beda_isi.append(e["id"])
        verification.update({
            "n_referensi": len(ref),
            "n_hasil": len(out),
            "id_sama_persis": ids_new == ids_ref,
            "hanya_di_hasil": sorted(ids_new - ids_ref),
            "hanya_di_referensi": sorted(ids_ref - ids_new),
            "stasiun_beda_isi": beda_isi,
            "IDENTIK": ids_new == ids_ref and not beda_isi,
        })

    # ── laporan penyimpangan hitungan konektor ──────────────────────────────
    over = []
    for sid, entry in stations.items():
        s_spklu = sum(cb["connectors"] for cb in entry["chargeboxes"])
        s_charger = sum(cb["_connectors_charger"] for cb in entry["chargeboxes"])
        if s_spklu != s_charger:
            over.append({"id": sid, "nama": entry["nama"],
                         "n_chargebox": len(entry["chargeboxes"]),
                         "port_via_JML_CONNECTOR_SPKLU": s_spklu,
                         "port_via_JML_CONNECTOR_CHARGER": s_charger})

    report = {
        "sumber": os.path.relpath(SRC, REF_ROOT).replace("\\", "/"),
        "sheet": SHEET,
        "aturan_filter": {
            "IS_ACTIVE_SPKLU": "Y",
            "lat": [LAT_MIN, LAT_MAX],
            "lon": [LON_MIN, LON_MAX],
        },
        "baris_mesin_dibaca": n_rows,
        "stasiun_nasional": n_station_all,
        "stasiun_bandung_lolos": len(out),
        "verifikasi_vs_berkas_simulasi": verification,
        "catatan_konektor": {
            "kolom_dipakai": "JML_CONNECTOR_SPKLU (total per stasiun) — "
                             "dipertahankan agar identik dengan berkas simulasi",
            "kolom_alternatif": "JML_CONNECTOR_CHARGER (per chargebox)",
            "stasiun_yang_hitungan_portnya_berbeda": len(over),
            "rincian": over,
        },
    }
    with open(os.path.join(OUT_DIR, "filtering_spklu_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)

    print(f"[step1] baris mesin dibaca : {n_rows}")
    print(f"[step1] stasiun nasional   : {n_station_all}")
    print(f"[step1] stasiun Bandung    : {len(out)}  -> output/bandung_spklu.json")
    if verification["referensi_tersedia"]:
        tanda = "IDENTIK" if verification["IDENTIK"] else "BERBEDA"
        print(f"[step1] verifikasi vs berkas simulasi: {tanda} "
              f"({verification['n_hasil']} vs {verification['n_referensi']} stasiun)")
    print(f"[step1] stasiun dgn hitungan port berbeda antar kolom: {len(over)} "
          f"(lihat output/filtering_spklu_report.json)")


if __name__ == "__main__":
    main()
