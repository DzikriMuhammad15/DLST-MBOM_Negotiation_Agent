# DOKUMENTASI SAMPLING & PROVENANCE — Eksperimen Real-Case v2 (Bandung)

> **Status: HIDUP — diperbarui setiap angka/keputusan final berubah.**
> Dokumen REPORT-READY (desain v2 §10, kerangka A–J): bab "Data & Metodologi
> Simulasi Real-Case v2" laporan/tesis dapat ditulis **hanya dari dokumen ini**
> — setiap angka dunia tertelusur ke sumber (data vs asumsi) DAN ke
> perhitungannya (rumus + aritmetika).
> Terakhir diperbarui: **2026-07-07** (pasca-generate config + uji wiring).

---

## A. Ringkasan & tujuan

Simulasi **real-case v2** adalah satu skenario dunia Kota Bandung yang
di-*ground* penuh ke data transaksi charging nyata 2024 + master SPKLU PLN,
dijalankan **engine simulasi asli** (bukan sintetis), untuk menguji **H1**:
integrasi MBOM (model-based opponent modelling) ke framework DLST meningkatkan
kinerja agen fleet dalam negosiasi bilateral multi-isu alokasi charging.

- **Dunia fisik nyata**: 13 SPKLU PLN (14 port, lat/lon master PLN), 18 marking
  area nyata, tarif PLN nyata 2.466,8 Rp/kWh flat.
- **Proses permintaan nyata**: kapan/berapa-lama/di-mana AppUserBooking
  disampel dari distribusi 22.514 transaksi charging Bandung 2024; bobot waktu
  order penumpang dari TomTom Traffic Index Bandung (data 2025).
- **Intensitas skenario** (bukan data): 50 taksi listrik, target okupansi
  booking 22,5% → N = 229.
- **Kontensi 50 taksi / 14 port = 3,57 taksi/port** — sengaja berada di rezim
  ≈ `sc_scarce` non-RC (3,0), kondisi di mana mekanisme negosiasi & H1 paling
  teruji (temuan eksperimen non-RC: nilai opponent modelling menguat saat
  kepadatan interaksi tinggi). Pembanding lintas-dunia: non-RC ref 0,75;
  sc_scarce 3,0; **v2 3,57**; rc lama (dibekukan) 12,5; Bandung nyata penuh
  1,24 (semua 77 SPKLU) / 2,56 (PLN saja).
- Eksperimen: negosiasi (DLST, DLSTMBOM, DLSTMBOMNewMechanism vs SPKLU-RL;
  PADDED; ≥2 run random-init), non-negosiasi (booking & as_is, 3 run), dan
  baseline metaheuristic greedy (folder `metaheuristic/`).

## B. Sumber data mentah

| File (`simulation_data/`) | Isi | Periode | Ukuran | Pemakaian |
|---|---|---|---|---|
| `DATA TRX SPKLU 2024.xlsx` | ±402 ribu transaksi charging SPKLU nasional (per-transaksi: stasiun, chargebox, kW-tier, tanggal/jam, kWh, Rupiah, durasi, SoC awal/akhir, konektor, user) | 2024 penuh | ±48 MB | difilter ke Bandung → semua distribusi booking & harga |
| `MASTER SPKLU PER 14 MAR 2025_.xlsx` | Master lokasi SPKLU (nama, koordinat, chargebox/kW, konektor, status) | per 14 Mar 2025 | — | identitas + lat/lon + port/kW 13 SPKLU dunia |
| `Master SPKLU 20250311 (…).xlsx` | Master pembanding (aktif & non-aktif) | per 11 Mar 2025 | — | triangulasi ekstraksi |

**Filter Bandung**: transaksi dipetakan ke daftar **77 SPKLU aktif wilayah
Bandung** pada master (ID stasiun) → **22.514 transaksi** di 34 SPKLU yang
tercatat bertransaksi. Subset PLN pada master Bandung = **25 stasiun**.

**File turunan (`simulation_data/_derived/`)** — dibuat skrip tercatat
(`_extract_trx.py`, `_extract_bandung_rows.py`, `_eda.py`; EDA 15 plot:
`simulation_data/eda/EDA.md`), sesi 2026-07-06/07:

| File | Isi |
|---|---|
| `bandung_spklu.json` | 77 SPKLU aktif Bandung: id, nama, lat/lon, chargebox (kW) + jumlah konektor |
| `bandung_trx.csv` | 22.514 baris transaksi Bandung (kolom: id_spklu, nama, chargebox, kw_tier, date, hour, dow, month, kwh, rpkwh, rppakai, price_per_kwh, durasi_min, first_soc, last_soc, connector, id_user) |
| `bandung_only_trx_summary.json` | `hour_hist` (24 jam), `durasi_hist_bin15` (bin 15 mnt), kwh_mean, firstsoc_mean |
| `simulation_data/sumber_data_traffic_kendaraanUmum/` | Sumber bobot waktu order (TomTom) + skrip + bukti mentah — lihat §E.7 & README di folder itu |

## C. Statistik deskriptif data nyata (rujukan: `simulation_data/eda/EDA.md`)

- **Harga**: median `price_per_kwh` = **2.466,8 Rp/kWh**, **flat lintas tier kW**
  (bukti: plot `08_price.png`; nilai per-baris konsisten ±sen) — tarif PLN.
- **Energi/sesi**: mean **23,4 kWh**, median 21,2 kWh (`04_kwh.png`).
- **Durasi sesi**: median **45 mnt**, mean 75,6 mnt, **skewness +3,82**
  (miring kanan), p10 = 13 mnt, p90 = 172 mnt (`05_durasi.png`).
- **SoC datang**: mean **28%** (= ambang orang mulai mengecas, BUKAN SoC
  awal-hari); SoC selesai mean 56% (`06/07_first/last_soc.png`).
- **Kurva jam transaksi**: puncak **13:00**, lembah **03:00–04:00**
  (`01_hourly.png`; histogram penuh di §J.7).
- **Volume 13 stasiun terpilih**: **2.142 sesi/2024** (§J.8) — sangat timpang;
  3 stasiun ≈ 0 transaksi (Icon Plus 0, Rancaekek 0, Teuan 1) → perlu
  smoothing Laplace (`09_station_volume.png`, `15_chosen13.png`).
- 7.025 pengguna unik; mayoritas jarang, sedikit sering (`12_users.png`).

## D. Spesifikasi dunia final (angka EKSAK yang masuk config)

### D.1 SPKLU — 13 stasiun PLN ber-port-tersedikit (data master, apa adanya)

Seleksi deterministik: subset PLN (nama mengandung "PLN") diurut jumlah port
menaik (tie-break: nama), ambil 13 pertama → 12 stasiun × 1 port + 1 stasiun
× 2 port = **14 port**. `base_price_per_kwh` = **2.466,8** untuk SEMUA port.

| # | station_id (config) | Nama master | lat | lon | Port × kW | Vol. 2024 |
|---|---|---|---|---|---|---|
| 1 | spklu_hotel_novena | SPKLU PLN HOTEL NOVENA | -6.84075 | 107.59739 | 1×22 | 9 |
| 2 | spklu_icon_plus_bandung | SPKLU PLN ICON PLUS BANDUNG | -6.90816 | 107.63112 | 1×22 | 0 |
| 3 | spklu_teuan_cafe_resto | SPKLU PLN TEUAN CAFE & RESTO | -6.81467 | 107.62495 | 1×22 | 1 |
| 4 | spklu_uid_jawa_barat | SPKLU PLN UID JAWA BARAT | -6.92073 | 107.60844 | 1×22 | 480 |
| 5 | spklu_ulp_banjaran | SPKLU PLN ULP BANJARAN | -7.03542 | 107.59332 | 1×22 | 132 |
| 6 | spklu_ulp_cijawura | SPKLU PLN ULP CIJAWURA | -6.95401 | 107.64058 | 1×22 | 163 |
| 7 | spklu_ulp_cimahi_selatan | SPKLU PLN ULP CIMAHI SELATAN | -6.89858 | 107.55414 | 1×22 | 187 |
| 8 | spklu_ulp_kopo | SPKLU PLN ULP KOPO | -6.93387 | 107.57143 | 1×22 | 226 |
| 9 | spklu_ulp_lembang | SPKLU PLN ULP LEMBANG | -6.81833 | 107.62200 | 1×22 | 282 |
| 10 | spklu_ulp_prima_cibabat | SPKLU PLN ULP PRIMA CIBABAT | -6.87574 | 107.54486 | 1×22 | 248 |
| 11 | spklu_ulp_rancaekek | SPKLU PLN ULP RANCAEKEK | -6.95895 | 107.76474 | 1×22 | 0 |
| 12 | spklu_ulp_ujungberung | SPKLU PLN ULP UJUNGBERUNG | -6.91308 | 107.69491 | 1×22 | 76 |
| 13 | spklu_geowisata_inn | SPKLU PLN GEOWISATA INN | -6.91753 | 107.57861 | **2×40** | 338 |

Catatan jujur: beberapa stasiun berada di wilayah metropolitan/kabupaten
(Lembang, Banjaran, Rancaekek, Cimahi Selatan, Prima Cibabat) — master PLN
mengelompokkannya sebagai wilayah Bandung; dipertahankan sesuai keputusan
desain (daftar 13 nama ini tercantum eksplisit di desain v2 §2.1).

### D.2 Marking area — 18 lokasi nyata (data peneliti; desain v2 §7)

d'Botanica Pasteur (-6.8812, 107.5800); Ciwalk (-6.8938, 107.6052); PVJ
(-6.8895, 107.5957); BIP (-6.9112, 107.6097); Summarecon (-6.9600, 107.7170);
TSM (-6.9276, 107.6364); Stasiun Bandung (-6.9175, 107.6030); Tegalluar Whoosh
(-6.9850, 107.7400); Terminal Leuwipanjang (-6.9333, 107.5742); **Kertajati
Airport (-6.5569, 108.2314 — ±60 km, sengaja dipertahankan)**; Cititrans
Dipatiukur (-6.8840, 107.6186); Shuttle Pasteur (-6.8855, 107.5779); Majesty
Apartement (-6.8858, 107.5790); Kota Baru Parahyangan (-6.8655, 107.4750);
W Super Club (-6.9180, 107.6150); MOD Pool and Club (-6.9205, 107.6162);
Lembang Park Zoo (-6.8247, 107.6133); Rest Area Alun-Alun Lembang (-6.8180,
107.6169). — 4 dari 22 lokasi asli DIHAPUS (keputusan peneliti 2026-07-07):
Padalarang Whoosh, Dusun Bambu, The Lodge Maribaya, Soreang.
Config hanya memakai center lat/lon (order_min=0, order_max=50 global; radius
1 km tidak di-enforce engine — limitasi lama).

### D.3 Parameter dunia & mekanisme (nilai config eksak)

| Parameter | Nilai | Status |
|---|---|---|
| `time.n_days` | 3 | keputusan (ikut rc lama) |
| Taksi | 50 unit BYD e6: baterai 61 kWh, 0,2 kWh/km, 30 km/jam | spec data peneliti |
| Posisi taksi | ~ U(bbox kota: lat −6.98..−6.84, lon 107.55..107.72), per run | asumsi |
| SoC awal taksi | ~ U(0,35, 0,90), per run | asumsi (lihat §E.5) |
| `port_station` | 13 SPKLU/14 port (D.1), base 2466,8 semua port | **data** |
| `negotiation.price_levels` | [1973,44, 2466,8, 3453,52, 4000,0] | turunan base (J.4) |
| `negotiation.max_action_library` / `K_max` | **30 / 30** (kontrak wajib sama; RALAT 2026-07-08, semula 50) | keputusan peneliti — gerbang runtime |
| `negotiation.max_steps` | **15** (RALAT 2026-07-08, semula 100) | keputusan peneliti — gerbang runtime |
| `app_user_booking.initial_booking_count` | **229** (target) | turunan okupansi (J.2) |
| `fallback.allowed_slot_shift_range` | (1, 4) grid | default engine |
| `fallback.compensation_per_kwh_range` | **(986,72, 986,72)** — fix | turunan 40%×base (J.5) |
| `marking_area_order.time_weight` | 24 bobot TomTom (J.7) | **data (T1)** |
| `fare` | PURE, 3.000 Rp/km | asumsi (ikut lama) |
| `encoder_dims` | N_max=50, P_max=14, A_max=18, K_max=50 | struktur dunia |
| `training.train_every_n_episodes` | 1; SIMULTANEOUS | keputusan (ikut REF) |
| `hyperparameters.mbom_M` / `rollout_max_steps` | **2 / None (faithful = ikut max_steps 15)** (RALAT 2026-07-08; semula 3/20) | keputusan peneliti — gerbang runtime |
| `seed` run1 / run2 / run3 | 42 / 7 / 123 | keputusan (tradisi multi-seed) |
| Battery threshold | default engine (B_charge 0,30; B_area 0,50; SoC_target 1,0) | default |
| `dead_battery_mode` | LET_DIE (default; ikut semua eksperimen lama) | default |

## E. Metodologi SAMPLING per kuantitas

Semua sampler **membaca file `_derived` saat runtime** (tidak ada angka
distribusi yang disalin ke kode — `booking_sampler.py`); RNG = `random.Random`
dunia engine yang di-seed `seed` run (§H) → hasil reproducible bit-identik.

### E.1 Jam mulai AppUserBooking — SAMPLING empiris
- Sumber: `hour_hist` SELURUH Bandung (22.514 trx; §J.7) — bukan 13 stasiun,
  karena bentuk-harian kedatangan = pola perilaku kota (sampel besar), sedangkan
  13 stasiun terpilih terlalu jarang (2.142 sesi, 3 di antaranya ~0).
- Algoritma per booking: (1) hari ~ U{0,1,2} (bentuk harian diulang tiap
  hari); (2) jam ~ kategorikal `hour_hist` (dadu berbobot via CDF); (3) menit
  ~ U dalam jam → dibulatkan ke grid ½-jam (paruh 0/1 uniform) →
  `slot_start = hari×48 + jam×2 + paruh`; (4) bila span melewati akhir horizon
  (144 grid), resample (≤100×; menjaga bentuk distribusi jam).

### E.2 Durasi AppUserBooking — SAMPLING empiris + cap engine
- Sumber: `durasi_hist_bin15` SELURUH Bandung (bin 15-menit; alasan sama E.1).
- Algoritma: (1) bin ~ kategorikal histogram; (2) menit ~ U dalam bin;
  (3) konversi grid ½-jam TERDEKAT: `grid = int(menit/30 + 0,5)`, min 1;
  (4) **cap [1, 4] grid = 0,5–2 jam** (batas struktural engine
  `_MIN/_MAX_BOOKING_DURATION`). Bin 15-menit sejajar ambang pembulatan
  (45/75/105 mnt) → satu bin memetakan ke tepat satu nilai grid (§J.1).

### E.3 Port AppUserBooking — SAMPLING berbobot volume nyata + Laplace
- Sumber: volume transaksi 2024 per 13 stasiun terpilih (`bandung_trx.csv`,
  match `id_spklu`; §D.1 kolom terakhir).
- Algoritma: (1) stasiun ~ bobot `(volume_s + α) / Σ(volume + α)`, **α = 1**
  (Laplace — menyelamatkan 3 stasiun bervolume ~0 agar tetap mungkin muncul);
  (2) port ~ U{port stasiun itu}. Bobot eksak: §J.8.
- Ini SATU-SATUNYA komponen sampling yang station-specific.

### E.4 Penempatan & fallback — semantik ENGINE dipertahankan identik
- Bila span (port, slot, durasi) terisi → booking **di-skip diam-diam**
  (N = target, bukan jaminan; terpasang aktual dicatat di log & §J.3).
- Fallback tunggal per booking via fungsi engine asli
  `world_assembly._choose_fallback_destination`: coba geser waktu ±1..4 grid
  di port sama (urutan magnitude acak), lalu port lain slot sama; destinasi
  harus kosong; tanpa fallback valid → skip.
- Kompensasi displacement = rumus engine energy-based:
  `komp = rate × power_kw × duration_grids × 0,5` dengan rate ~
  U(986,72, 986,72) = **986,72 tetap** (§J.5).
- Alokasi port dikunci sejak awal (`add_allocation(..., "booking")`).

### E.5 Posisi & SoC taksi — SAMPLING asumsi (bukan data)
- Posisi ~ U(bbox kota; §D.3), SoC ~ U(0,35, 0,90); disampel ULANG per run
  (run1/2/3 = "3× random init") dgn `random.Random(10000 + seed_run)` →
  run1: 10042, run2: 10007, run3: 10123; hasil dibekukan ke config
  (`initial_taxi_state` eksplisit, committed).
- Mengapa bukan dari data: `FIRST_SOC` data (mean 28%) = SoC saat orang
  MEMUTUSKAN mengecas (ambang), bukan SoC awal-hari armada; tidak ada data
  posisi taksi. U(0,35–0,90) memberi campuran urgensi wajar (≈30% armada mulai
  di bawah B_area 0,50 → jadi kandidat charge lebih awal).

### E.6 Order penumpang per jam — mekanisme engine + bobot data
- Engine (`order_generator`): tiap jam, per area, jumlah order =
  `order_min + share·(order_max − order_min)` di-bias `clip(bobot_jam, 0, 1)`;
  v2: order_min=0, order_max=50 (global, desain §7).
- `time_weight` 24 jam ← TomTom (E.7). Destinasi order: mekanisme engine
  default (soft-reachable; tanpa order_reference_point — tidak diubah).

### E.7 Bobot waktu order (`time_weight`) — turunan data traffic (T1)
- Rantai fallback keputusan peneliti: (1) pola per-jam taksi Bandung → tidak
  tersedia terbuka; (2) kepadatan penumpang angkutan umum Bandung per jam →
  tidak tersedia terbuka; (3) **traffic umum Bandung: TomTom Traffic Index
  Bandung (dipakai)** — penelusuran ketiganya terdokumentasi di
  `simulation_data/sumber_data_traffic_kendaraanUmum/README.md` (+ bukti HTML
  mentah, skrip ekstraksi, JSON hasil).
- Sumber: blok `weekHour` halaman kota TomTom (metro Bandung, tahun-data 2025):
  congestion level per (hari × jam), 168 titik.
- Penurunan: `mean_c(jam)` = rata-rata 7 hari → normalisasi puncak
  `bobot = mean_c / max(mean_c)`; puncak 17:00 = 1,0. Tabel penuh: §J.7b.

## F. Parameter FIXED/ASUMSI (bukan data) + alasan

| Parameter | Nilai | Alasan |
|---|---|---|
| N booking | target okupansi 22,5% → 229 | data 2024 terlalu tipis (murni data: 2.142×3/365 ≈ 18 booking → okupansi ~2%, tak terasa sbg kontensi; DITOLAK peneliti). 22,5% = intensitas skenario "app-user terasa tapi taksi tetap kontensi utama" |
| SoC awal U(0,35–0,90) | asumsi | §E.5 (FIRST_SOC ≠ SoC awal-hari) |
| Posisi taksi U(bbox) | asumsi | tidak ada data posisi armada; uniform = netral |
| Fare 3.000/km PURE | asumsi | di luar cakupan data charging; konstan lintas rezim → netral utk perbandingan |
| price_levels [0,8; 1; 1,4]×base + 4000 | keputusan | base = jangkar tarif nyata; 0,8×base = diskon slot fleksibel; 1,4×base = premi yang MENUTUP kompensasi 40% (break-even SPKLU saat menggeser, §J.6); 4000 = ceiling premi kelangkaan |
| Kompensasi fix 40%×base | keputusan final peneliti | mekanisme engine lama (energy-based) tanpa ubah kode — range degenerate |
| Fallback p = 1,0 | keputusan | semua booking punya tepat 1 fallback (engine as-is); varian p<1 = cadangan sensitivitas |
| n_days=3, upd=1 | keputusan | konsistensi dgn desain rc lama & REF non-RC |
| **K=30, max_steps=15, mbom_M=2, rollout faithful** | **keputusan peneliti 2026-07-08 (RALAT gerbang runtime)** | smoke DLSTMBOM pada K=50/steps=100/M=3 terbukti ≫6 jam/exp (54% dari 1 hari memakan 7,3 jam-CPU; 1 negosiasi maraton >2 jam). Estimasi biaya: (30/50)×(2/3)×(15/100) ≈ ×0,06 → ~2,5–5 jam/exp MBOM. Konsekuensi: menu bundle terpangkas ke 30 teratas (sort desc utility fleet); kedalaman imajinasi M=2 (level 0+1) — analisis H1 v2 diframe "hybrid M=2" (korpus nonRC memakai M=3); rollout mengikuti horizon nyata 15 (cap>horizon tidak bermakna). SEMUA run v2 (non-neg & greedy) diulang dgn config ini — hasil non-neg terbukti bit-identik dgn pra-ralat (menu ≤30 tak pernah mengikat; arsip: `data_pra_ralat_K50/`) |
| Seeds 42/7/123 | keputusan | tradisi multi-seed proyek (REF/ms) |

## G. Asumsi & LIMITASI (jujur)

1. **N terpasang < N target (semantik skip engine).** Sampling empiris
   menumpuk di jam siang & stasiun populer → tabrakan slot lebih sering
   daripada penempatan uniform. Uji wiring run1: terpasang **170/229**
   (okupansi terpasang ~15,2% dari target 22,5%; §J.3). Bias tambahan:
   booking durasi panjang lebih sering tertolak (grid-4 turun dari ekspektasi
   ~19% ke ~14%). Konsekuensi dipertahankan (desain memilih semantik engine);
   angka terpasang per run dilaporkan.
2. **Bobot waktu 02:00–04:00 = 0** (congestion TomTom 0 dini hari) → tidak ada
   order penumpang dini hari; kenyataannya kecil tapi ≠ 0. Konservatif; searah
   lembah transaksi charging nyata (03–04).
3. **Congestion ≠ permintaan order taksi** — proxy intensitas aktivitas
   (rantai fallback tahap-3; tahap 1–2 tak tersedia terbuka; didokumentasikan).
4. **Radius area 1 km tidak di-enforce engine** (area = titik spawn); field
   capacity/order_rate per-area data peneliti tidak dipakai (schema global).
5. **Kertajati (±60 km) & Tegalluar dipertahankan** (hub penting, keputusan
   peneliti): trip ke sana menguras ±12 kWh ≈ 20% SoC sekali jalan — taksi
   bisa "tersedot" keluar kota (unproductive_km besar). Fitur realisme, bukan
   bug; efeknya dianalisis, tidak "diperbaiki" diam-diam.
6. **Beberapa SPKLU di luar kota administratif** (Lembang, Banjaran, Rancaekek,
   Cimahi) — konsisten master PLN wilayah Bandung & bbox order tetap kota.
7. **PPJ + biaya admin (~+6% + ~1.500 flat) TIDAK dimodelkan** — identik utk
   semua rezim (booking/as_is/nego) → saling meniadakan dalam perbandingan.
8. **Data EV 2024 masih tipis** di 13 stasiun kecil (2.142 sesi; 3 stasiun ~0)
   → volume stasiun di-smoothing Laplace; N memakai target okupansi (F).
9. **~30% taksi mulai dgn SoC < B_area (0,50)** → warning advisory engine
   (by design: campuran urgensi).
10. **Ufun REF_\*** memakai default engine (REF_cost 100k, REF_comp 50k, dll.)
    — skala biaya v2 (komp tipikal ~21,7 ribu; biaya charge slot 22 kW 1 jam
    ~54 ribu) masih dalam rentang wajar normalisasi; konstanta tidak diubah
    (kode lama tak disentuh); efek normalisasi identik lintas rezim.
11. **1–3 run per konfigurasi** — klaim statistik per-kondisi terbatas (pola
    proyek: konsistensi arah, bukan p-value tunggal).

## H. Reproduksibilitas

1. **Generator config**: `python eksperiment_real_case_v2/build_world_config.py`
   → `configs/run{1,2,3}.json` (committed) + `configs/v2_world_meta.json`
   (audit: seleksi stasiun, bobot, rumus N, provenance sampler). Deterministik
   (seed taksi 10000+seed_run; seleksi stasiun & bobot tanpa RNG).
2. **Bobot waktu**: `python simulation_data/sumber_data_traffic_kendaraanUmum/extract_tomtom_weekhour.py`
   (dari bukti HTML mentah yang tersimpan) → `time_weight_24.json`.
3. **Eksperimen negosiasi** (1 exp = 1 invokasi):
   `python eksperiment_real_case_v2/main.py --agent-fleet {DLST|DLSTMBOM|DLSTMBOMNewMechanism} --run {1|2|3} [--mode PADDED]`
   → `data/negotiation/run{n}_{mode}_{agen}/`. Seed = `seed` config run.
4. **Non-negosiasi**: `python eksperiment_real_case_v2/run_nonneg.py --mech {booking|as_is} --run {n}`
   → `data/{mech}/run{n}/`.
5. **Injeksi v2** (aditif, kode lama utuh): `overrides.normalize_time_weight`
   (key JSON string → int; tanpa ini engine diam-diam memakai bobot 1,0) +
   `overrides.install_v2_bookings` (monkeypatch
   `world_assembly._build_initial_bookings`; berlaku utk pipeline nego DAN
   non-neg karena keduanya lewat `build_world_from_config`).
6. **Audit angka**: tiap angka dunia di config dapat dicek ke
   `configs/v2_world_meta.json` → file `_derived` → xlsx mentah; booking
   terpasang per run tercetak di log (`[v2-bookings] target=… terpasang=…`).
7. Smoke test: tambah `--smoke` (n_days=1, N diskala 1/3, output `_smoke/` —
   bukan data penelitian).

## I. Justifikasi ke-real-an (ringkas untuk sidang)

**NYATA (data, bukan asumsi):** dunia fisik — identitas/lokasi/daya 13 SPKLU
PLN + 14 port (master PLN Mar-2025); tarif 2.466,8 Rp/kWh (median 22.514
transaksi 2024, flat); 18 marking area (lokasi nyata, data peneliti); proses
permintaan charging — jam-mulai & durasi booking (distribusi empiris 22.514
trx), pilihan stasiun (volume nyata + Laplace); intensitas order penumpang
per jam (TomTom Traffic Index Bandung 2025).

**KEPUTUSAN skenario (diakui, beralasan):** jumlah taksi (50) & bookingnya
(target okupansi 22,5% → 229), SoC/posisi awal, fare, parameter mekanisme
negosiasi (price_levels/kompensasi/fallback), parameter training.

Klaim yang defendable: *"dunia dan proses permintaan di-ground ke data nyata
Bandung dan dapat diaudit angka-per-angka; intensitas skenario adalah
keputusan eksperimen yang didokumentasikan — bukan klaim data."*

## J. PERHITUNGAN TURUNAN (rumus + angka + aritmetika)

### J.1 Durasi efektif booking (E[durasi tersampel ter-cap])
Pemetaan bin → grid (aturan `grid = min(4, max(1, int(menit/30 + 0,5)))`):

| Grid (jam) | Bin menit | Jumlah trx |
|---|---|---|
| 1 (0,5 j) | [0,15)+[15,30)+[30,45) | 2.520 + 3.797 + 4.824 = **11.141** |
| 2 (1,0 j) | [45,60)+[60,75) | 3.047 + 1.954 = **5.001** |
| 3 (1,5 j) | [75,90)+[90,105) | 1.319 + 773 = **2.092** |
| 4 (2,0 j, cap) | ≥105 | 537+530+348+323+468+263+213+201+136+1.261 = **4.280** |

`E[grid] = (11.141·1 + 5.001·2 + 2.092·3 + 4.280·4) / 22.514 = 44.539 / 22.514
= 1,97828 grid` → **E[durasi] = 1,97828 × 0,5 = 0,98914 jam**.
(Mean mentah tanpa cap = 1,26 jam; cap memangkas ekor >2 jam. Desain
mengestimasi ≈0,99 — cocok.)

### J.2 Kapasitas & N booking
- **Kapasitas port-jam** = 14 port × 3 hari × 24 jam = **1.008 port-jam**.
- **N** = okupansi × kapasitas ÷ E[durasi] = 0,225 × 1.008 ÷ 0,98914
  = 226,8 ÷ 0,98914 = 229,29 → **round = 229**.
  (Angka desain "≈228" = estimasi pembulatan E=0,99; generator menghitung
  eksak sesuai mandat desain → 229. Pembanding data-murni: 2.142 × 3/365 ≈ 18
  → okupansi ~1,8% — DITOLAK sbg skenario.)

### J.3 N terpasang aktual (semantik skip engine — dicatat per run)
Run1 (seed 42): target 229 → **terpasang 170** (59 skip span-terisi, 0 skip
tanpa-fallback) → okupansi terpasang = Σ(durasi grid)/(14×144) = 307/2.016 =
**15,2%**. Run2 (seed 7): **176/229** (53 skip). Run3 (seed 123): **166/229**
(63 skip). Deterministik per seed (booking identik utk semua mekanisme/agen dalam
run yang sama).

### J.4 Harga
- **Base** = median `price_per_kwh` 22.514 trx = **2.466,8 Rp/kWh** (flat).
- **price_levels** = [0,8×2466,8; 2466,8; 1,4×2466,8; 4000]
  = [**1.973,44**; **2.466,8**; **3.453,52**; **4.000**] Rp/kWh.

### J.5 Kompensasi displacement
- Rate = 40% × base = 0,4 × 2.466,8 = **986,72 Rp/kWh** → config
  `compensation_per_kwh_range = (986,72, 986,72)` (Uniform degenerate =
  deterministik; TANPA ubah kode).
- Total per geser = 986,72 × energi_kWh; energi = power_kw × grid × 0,5.
- Contoh: port 22 kW, booking 1 jam (2 grid) → energi 22 kWh → komp =
  986,72 × 22 = **Rp 21.707,84**. Port 40 kW (Geowisata), 2 jam → energi 80
  kWh → komp = Rp 78.937,60.

### J.6 Konsistensi ceiling ↔ kompensasi
Premi level-3 = (3.453,52 − 2.466,8) × E = 986,72 × E = **persis** rate
kompensasi × E → bila energi taksi ≈ energi booking yang digeser, pendapatan
premi menutup kompensasi (SPKLU break-even saat menggeser); level 4.000
(premi 1.533,2/kWh > 986,72) = margin kelangkaan.

### J.7 Bobot waktu order (T1 — TomTom Bandung 2025)
(a) Histogram jam transaksi charging (utk sampling jam booking, E.1; Σ=22.514):
0: 388 · 1: 219 · 2: 150 · 3: 112 · 4: 137 · 5: 308 · 6: 625 · 7: 1.068 ·
8: 1.167 · 9: 1.282 · 10: 1.373 · 11: 1.343 · 12: 1.332 · 13: 1.491 ·
14: 1.471 · 15: 1.465 · 16: 1.390 · 17: 1.359 · 18: 1.234 · 19: 1.171 ·
20: 1.047 · 21: 1.057 · 22: 799 · 23: 526.

(b) `time_weight` order penumpang = mean congestion TomTom ÷ puncak
(rumus & mean per jam: README sumber; puncak 17:00, mean_c 95,17%):

| Jam | Bobot | Jam | Bobot | Jam | Bobot | Jam | Bobot |
|---|---|---|---|---|---|---|---|
| 00 | 0,0242 | 06 | 0,3379 | 12 | 0,7535 | 18 | 0,7932 |
| 01 | 0,0029 | 07 | 0,4484 | 13 | 0,8930 | 19 | 0,6600 |
| 02 | 0,0000 | 08 | 0,4332 | 14 | 0,9403 | 20 | 0,5410 |
| 03 | 0,0000 | 09 | 0,5018 | 15 | 0,9138 | 21 | 0,4400 |
| 04 | 0,0000 | 10 | 0,6034 | 16 | 0,9685 | 22 | 0,2672 |
| 05 | 0,0296 | 11 | 0,7097 | 17 | **1,0000** | 23 | 0,1094 |

### J.8 Bobot lokasi port (Laplace α = 1)
`bobot_s = (volume_s + 1) / (2.142 + 13) = (volume_s + 1) / 2.155`:

| Stasiun | Vol. | Bobot | Stasiun | Vol. | Bobot |
|---|---|---|---|---|---|
| UID Jawa Barat | 480 | 0,2232 | ULP Banjaran | 132 | 0,0617 |
| Geowisata Inn | 338 | 0,1573 | ULP Ujungberung | 76 | 0,0357 |
| ULP Lembang | 282 | 0,1313 | Hotel Novena | 9 | 0,0046 |
| ULP Prima Cibabat | 248 | 0,1155 | Teuan Cafe & Resto | 1 | 0,0009 |
| ULP Kopo | 226 | 0,1053 | Icon Plus Bandung | 0 | 0,0005 |
| ULP Cimahi Selatan | 187 | 0,0872 | ULP Rancaekek | 0 | 0,0005 |
| ULP Cijawura | 163 | 0,0761 | **Σ** | **2.142** | **1,0000** |

Port dalam stasiun dibagi rata (Geowisata: tiap port = 0,1573/2 = 0,0786).

### J.9 Kontensi
= 50 taksi ÷ 14 port = **3,571 taksi/port** (≈ `sc_scarce` non-RC = 6/2 = 3,0;
rezim kelangkaan tempat H1 paling relevan — §A).

---

## Changelog
- 2026-07-07 — dokumen dibuat (pasca T1 + generate config + uji wiring):
  seluruh kerangka A–J terisi; N final 229 (rekomputasi eksak; desain ≈228);
  temuan wiring: terpasang run1 = 170/229 (okupansi 15,2%) + bias durasi
  panjang — dicatat G.1/J.3.
- 2026-07-08 — N terpasang run2 (176) & run3 (166) diisi (J.3); **RALAT GERBANG
  RUNTIME (keputusan peneliti)**: K 50→30, max_steps 100→15, mbom_M 3→2,
  rollout_max_steps 20→None (faithful) — D.3 & F diperbarui; configs
  di-regenerate; semua run v2 diulang (non-neg bit-identik dgn pra-ralat;
  arsip `data_pra_ralat_K50/`). Engine non-neg dipakai versi pasca-fix T4.
