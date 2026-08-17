# -*- coding: utf-8 -*-
"""
Ekstraksi data kemacetan per-jam Bandung dari halaman TomTom Traffic Index
(bukti mentah: tomtom_traffic_index_city_bandung_raw_2026-07-07.html, diunduh
2026-07-07 dari https://www.tomtom.com/traffic-index/city/bandung/).

Halaman meng-embed JSON (Astro island props, HTML-escaped) berisi blok
`traffic.metro.all.weekHour`: 168 titik (7 hari x 24 jam) dengan
  c  = congestion level (%, ekstra waktu tempuh vs free-flow)
  v  = kecepatan rata-rata aktual (km/jam)
  fv = kecepatan free-flow (km/jam)
Basis data: Traffic Index tahun 2025 (blok monthly = 2025-01..2025-12).

Output:
  1) tomtom_bandung_weekhour.json  — 168 titik mentah + metadata (audit)
  2) time_weight_24.json           — 24 bobot jam final utk config
     `marking_area_order.time_weight` (rata-rata c per jam lintas 7 hari,
     dinormalisasi puncak = 1.0, dibulatkan 4 desimal)

Jalankan:  python extract_tomtom_weekhour.py
"""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "tomtom_traffic_index_city_bandung_raw_2026-07-07.html"

DAYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]


def main() -> None:
    raw = RAW.read_text(encoding="utf-8", errors="replace")
    text = html.unescape(raw)

    # Ambil hanya segmen weekHour (1 kemunculan; scope = traffic.metro.all)
    i = text.find('"weekHour"')
    if i < 0:
        raise SystemExit("blok weekHour tidak ditemukan — struktur halaman berubah?")
    seg = text[i : i + 40000]

    # Entri berbentuk {"time":[0,"FRIDAY-17"],"c":[0,112.2],"v":[0,15.9],"fv":[0,33.8]}
    pat = re.compile(
        r'\{"time":\[0,"([A-Z]+)-(\d+)"\],"c":\[0,([\d.]+)\],'
        r'"v":\[0,([\d.]+)\],"fv":\[0,([\d.]+)\]\}'
    )
    points = {}
    for m in pat.finditer(seg):
        day, hour, c, v, fv = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        if day in DAYS and 0 <= hour <= 23:
            points[(day, hour)] = {"c": c, "v": v, "fv": fv}

    if len(points) != 7 * 24:
        raise SystemExit(f"diharapkan 168 titik, dapat {len(points)} — cek parser")

    # Konteks tambahan: jam puncak am/pm resmi TomTom (blok "am"/"pm")
    rush = {}
    for label in ("am", "pm"):
        mm = re.search(
            r'"%s":\[0,\{"time":\[0,"(\d+)"\],"c":\[0,([\d.]+)\]' % label, seg[:4000]
        ) or re.search(
            r'"%s":\[0,\{"time":\[0,"(\d+)"\],"c":\[0,([\d.]+)\]' % label, text[i - 3000 : i]
        )
        if mm:
            rush[label] = {"hour": int(mm.group(1)), "c": float(mm.group(2))}

    raw_out = {
        "source": "TomTom Traffic Index — city page Bandung (metro area, agregat tahun-data 2025)",
        "url": "https://www.tomtom.com/traffic-index/city/bandung/",
        "downloaded": "2026-07-07",
        "raw_file": RAW.name,
        "fields": {"c": "congestion level % (ekstra waktu tempuh)", "v": "kecepatan aktual km/jam", "fv": "kecepatan free-flow km/jam"},
        "rush_hours_tomtom": rush,
        "week_hour": [
            {"day": d, "hour": h, **points[(d, h)]} for d in DAYS for h in range(24)
        ],
    }
    (HERE / "tomtom_bandung_weekhour.json").write_text(
        json.dumps(raw_out, indent=2), encoding="utf-8"
    )

    # ---- Penurunan 24 bobot ----
    # bobot_jam = mean(c[hari, jam] utk 7 hari) / max_jam(mean) ; puncak = 1.0
    mean_c = {h: sum(points[(d, h)]["c"] for d in DAYS) / len(DAYS) for h in range(24)}
    peak_hour = max(mean_c, key=mean_c.get)
    peak_val = mean_c[peak_hour]
    weights = {str(h): round(mean_c[h] / peak_val, 4) for h in range(24)}

    weight_out = {
        "description": (
            "Bobot waktu order per jam utk config marking_area_order.time_weight "
            "(eksperiment_real_case_v2). bobot_jam = rata-rata congestion level "
            "lintas 7 hari / nilai jam puncak (puncak=1.0)."
        ),
        "derivation": "mean_c(jam) = mean(c[hari,jam], 7 hari); bobot = mean_c/max(mean_c)",
        "source": "tomtom_bandung_weekhour.json (TomTom Traffic Index Bandung, data 2025)",
        "peak_hour": peak_hour,
        "peak_mean_congestion_pct": round(peak_val, 2),
        "mean_congestion_pct_per_hour": {str(h): round(mean_c[h], 2) for h in range(24)},
        "time_weight": weights,
    }
    (HERE / "time_weight_24.json").write_text(
        json.dumps(weight_out, indent=2), encoding="utf-8"
    )

    print(f"OK: 168 titik; puncak jam {peak_hour}:00 (mean c={peak_val:.1f}%)")
    for h in range(24):
        print(f"  {h:02d}:00  mean_c={mean_c[h]:6.2f}%  bobot={weights[str(h)]:.4f}")


if __name__ == "__main__":
    main()
