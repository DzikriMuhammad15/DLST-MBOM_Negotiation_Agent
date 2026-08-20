# Konteks Project `clean_code`

Dokumen ini adalah rujukan pengetahuan lengkap untuk sesi Claude Code yang dibuka
di direktori `clean_code/`. Isinya apa project ini, bagaimana kodenya bekerja,
apa yang sudah diputuskan, dan jebakan apa saja yang sudah pernah ditemukan.
Baca ini lebih dulu sebelum mengubah apa pun.

Terakhir diperbarui 2026-08-12.

---

## 1. Ini project apa

Kode simulasi untuk tesis magister tentang **negosiasi otomatis untuk alokasi
slot pengisian daya kendaraan listrik**. Sebuah armada taksi listrik dan
pengelola SPKLU bernegosiasi memperebutkan slot pengisian yang terbatas, dengan
agen berbasis *deep reinforcement learning*. Judul kerja tesisnya menyebut
*Model-Based Opponent Modelling dalam Adaptive Strategy Templates Menggunakan
Deep Learnable Strategy Templates*.

Project ini menghasilkan **data mentah eksperimen**. Analisis, agregasi, dan
penulisan laporan dilakukan di tempat lain.

### Hubungannya dengan `nyoba_general`

`clean_code` awalnya salinan bersih dari `nyoba_general` yang dimaksudkan
menghasilkan angka identik. **Kesetaraan itu sudah tidak berlaku dan bukan lagi
sasaran.** Dua perubahan sengaja membuatnya berbeda, yaitu sampler booking baru
dan perbaikan bug fallback. Peneliti sudah menyatakan fokus kerjanya sekarang
**hanya di `clean_code`**. Jangan menjadikan `nyoba_general` patokan, dan jangan
menjanjikan reproduksi korpus tesis lama dari sini.

---

## 2. Aturan kerja yang tidak boleh dilanggar

**Data hasil simulasi tidak boleh diubah, dihapus, atau ditimpa.** Ini hard
constraint dari peneliti, berlaku di project mana pun. Data hasil adalah patokan
kebenaran, dan itulah alasan ia dijadikan patokan. Saat ini folder data memang
sedang kosong karena sengaja dikosongkan sebelum regenerasi, tetapi begitu ada
isinya aturan ini berlaku penuh.

**Jangan menjalankan korpus penuh tanpa diperintahkan.** Lihat bagian 12.

**Ukur, jangan berargumen.** Peneliti berulang kali meminta bukti angka, dan
beberapa kesimpulan yang tampak masuk akal ternyata terbalik setelah diukur.
Kalau ada dua tafsir yang menghasilkan pekerjaan berbeda, tanyakan, jangan
berasumsi.

**Sebut keterbatasan terang-terangan.** Jangan menyajikan pilihan pemodelan
seolah diambil mentah dari data.

### Gaya penulisan yang diminta

- Bahasa Indonesia.
- **Tanpa em-dash.** Hindari titik koma dan titik dua di tengah kalimat.
- Satu gambar satu panel. PDF dan CDF menjadi gambar terpisah.
- Judul gambar hasil fit persis `Probability Distribution Function` dan
  `Cumulative Distribution Function`.
- Tulis "hasil fit distribusi ...", bukan "hasil pencocokan".
- Jangan menaruh keterangan berlebih di dalam gambar. Cukup jarak KS dan legend.

---

## 3. Peta direktori

```
clean_code/
├── negotiation/              # KODE INTI. Agen, model, trainer, reward shaper, env NegMAS
├── simulation/               # KODE INTI. Perakitan dunia, engine, action library, metrik
├── real_case_common/         # Kode bersama dunia real-case Bandung (dipakai v_2 & network_scaling)
├── metaheuristic/            # Baseline Greedy dua sisi
├── non_negotiation_mechanism/# Baseline As-Is dan Booking statis
├── dashboard/                # HANYA config_io.py (deep-merge + validasi config). Bukan server web
├── v_1/                      # Config 13 dunia sintetis OFAT (korpus non-real-case)
├── v_2/                      # Entry point + 160 config real-case (korpus utama)
├── network_scaling/          # Entry point + generator config eksperimen ukuran jaringan
├── reference_data_simulation/# Rantai analisis data nyata. BUKAN bagian pipeline simulasi
├── simulation_data/          # Subset data mentah Bandung yang dibaca saat simulasi berjalan
├── main.py                   # Satu simulasi, artefak tersebar di data/
├── run_experiment.py         # Satu eksperimen rapi, semua artefak terkumpul
├── run_experiment_bulk_per_config.py  # Matriks pasangan agen x mode untuk satu dunia
└── simulation_variable_description.md # Asal-usul SETIAP variabel dunia simulasi
```

**Berkas paling berguna untuk memahami dunia simulasi adalah
`simulation_variable_description.md` di root.** Isinya menjawab, untuk setiap
variabel, apakah ia tetap, disampel berakar data, disampel tanpa akar data, atau
turunan. Rujuk berkas itu sebelum menjawab pertanyaan tentang asal sebuah angka.

---

## 4. Lingkungan

- Python **3.12.7**, sudah terpasang di `.venv/Scripts/python.exe`.
- Dependency utama `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`,
  `negmas==0.15.7` (versi terkunci, jangan dinaikkan), `networkx`, `osmnx`.
- **Seluruh perintah dijalankan dari root `clean_code/`**, bukan dari dalam
  subfolder, karena `negotiation` dan `simulation` diimpor sebagai paket
  top-level relatif terhadap direktori kerja.
- Windows. Tersedia PowerShell dan Bash.
- `fastapi`/`uvicorn` ikut terpasang tetapi tidak dipakai. Server web dashboard
  sengaja tidak disalin ke sini.

---

## 5. Cara kerja simulasi

### 5.1 Dunia dan waktu

Dunia berisi taksi, port pengisian di beberapa SPKLU, marking area tempat order
penumpang muncul, dan satu stasiun darurat. Waktu berjalan dengan centang 5
menit. **Satu grid slot pengisian panjangnya 30 menit**, jadi ada 48 grid per
hari. Konstanta `GRID_MINUTES` ada di kode, bukan di config.

Negosiasi dipicu setiap `interval_min`, yaitu 30 menit.

### 5.2 Satu negosiasi memutuskan satu Bundle

Satu negosiasi tidak memperebutkan satu slot untuk satu taksi, melainkan
memutuskan **satu Bundle sekaligus** untuk semua taksi kandidat. Bundle adalah
tuple `Assignment`, satu per taksi kandidat, di mana tiap assignment berupa
`(port, slot, durasi, harga)` atau sentinel UNASSIGNED.

`simulation/bridge/action_library.py` membangun `action_library`, yaitu daftar
Bundle yang boleh ditawarkan. Outcome negosiasi adalah indeks langsung ke daftar
itu, tanpa offset. Poin pentingnya:

- **Full assignment.** Setiap bundle menugaskan SEMUA kandidat yang secara fisik
  dapat ditempatkan. UNASSIGNED hanya muncul untuk kandidat yang memang tidak
  bisa ditempatkan sama sekali. Bundle tidak pernah *memilih* membiarkan taksi
  yang sebenarnya bisa ditugaskan menjadi menganggur. Ini "fixed mechanism"
  2026-07-08 yang menggantikan produk kartesian lama.
- Keragaman menu datang dari **rotasi preferensi port** yang deterministik,
  disilangkan dengan satu tingkat harga seragam per bundle.
- Menu dipotong ke `max_action_library` bundle dengan utilitas fleet tertinggi.
  Pada config v_2 nilainya **30**.
- Waktu tiba dan SoC tiba **eksak**, karena laju dan konsumsi konstan.

### 5.3 Kapan sebuah slot boleh direbut taksi

- Span port yang kosong selalu boleh diambil.
- Span yang berisi **tepat satu** AppUserBooking yang **masih punya rencana
  fallback** boleh diambil, dengan cara menggeser booking itu ke tujuan
  fallback-nya.
- Span yang berisi booking tanpa fallback tersisa, slot yang sudah dimenangkan
  fleet, alokasi darurat, atau lebih dari satu booking, tidak pernah boleh
  diambil.
- Stasiun darurat tidak pernah masuk action library.

### 5.4 Protokol negosiasi

Mekanisme SAO dari NegMAS. Fleet menawar lebih dulu. Propose dan respond
dipisahkan penuh, jadi indeks propose adalah indeks murni ke action library.
`negotiation/environments/negmas_env.py` membungkus utilitas dunia menjadi
`BundleUFun`. `max_steps` pada config v_2 adalah **15**.

`utility_min` dan `utility_max` hanya dipakai untuk **penskalaan reward**, bukan
normalisasi intrinsik. Sejak korpus fixed_mechanism, batasnya di-stamp
per-negosiasi sehingga reward deal berada pada 0 sampai 1 dan kegagalan bernilai
−1.

### 5.5 Utilitas

Sisi fleet menimbang pendapatan order yang bisa dilayani, biaya energi, biaya
pengisian, dan kerugian siklus menunggu. Sisi SPKLU menimbang pendapatan
penjualan energi. Kalibrasi terhadap tarif 6.300 rupiah per km menghasilkan
`lost/cycle` 0,447733 dan `g_max` 21,4912 dengan `tau` 48. Rumusnya tercatat di
`v_2/configs/calibration_meta.json`.

---

## 6. Agen negosiasi

`negotiation/agents/agent_factory.py` adalah pintu masuk tunggal pembangunan
agen. Nama tipe kanoniknya:

| Tipe | Sifat |
|---|---|
| `RLNegotiator` | Aksi diskret, dipakai sebagai **lawan** di seluruh eksperimen v_2 |
| `MBOM`, `MBOMNonZeroSum`, `MBOMNewMechanism` | Diskret, dengan pemodelan lawan berbasis rollout |
| `DLST` | DDPG, aktor mengeluarkan parameter *strategy template* |
| `DLSTMBOM` | DLST digabung pemodelan lawan MBOM |
| `DLSTMBOMNonZeroSum`, `DLSTMBOMNewMechanism` | Varian mekanisme fan-out |
| `DLSTMBOMDirected` | Varian anti-kolaps, seluruh saklarnya otomatis menyala |
| `RandomIOP`, `Rule`, `GreedyMH` | Baseline |

### DLST

Tiga sub-keputusan, masing-masing punya aktor-kritik DDPG, trainer, memori, dan
reward shaper **sendiri**, tanpa state bersama.

- `threshold_trainer` memprediksi ambang utilitas dinamis.
- `propose_trainer` menghasilkan strategi penawaran.
- `respond_trainer` menghasilkan strategi penerimaan.

Model lawan frekuensi dipakai bersama oleh strategi tawar Pareto dan reward
shaper respond.

### DLSTMBOMDirected

Agen anti-kolaps. Saat `agent_type` bernilai `"DLSTMBOMDirected"`,
`simulation/run_pipeline.py` otomatis menyalakan seluruh lapisnya, yaitu `l1`,
`persistent_freq`, `stuck_rescue`, `handover`, `shaping`, dan `hygiene`. Tidak
perlu wiring tambahan di entry point. Spesifikasi arsitekturnya ada di
`negotiation/agents/DLSTMBOMDirected_SPEC.md`. Variabel lingkungan `DIRECTED_HP`
berisi JSON dapat menimpa field `DirectedConfig` untuk keperluan tuning, dan
tanpa variabel itu perilakunya tidak berubah sama sekali.

### Mode PADDED dan ENCODER

Hanya mengubah front-end jaringan. Seluruh korpus yang dipakai memakai
**PADDED**.

### Kebijakan konkurensi pelatihan

`SIMULTANEOUS` melatih kedua sisi, `ALTERNATING` membekukan satu sisi
bergantian, `FIXED_OPPONENT` hanya melatih satu sisi.

---

## 7. Tiga keluarga eksperimen

| Keluarga | Folder | Dunia | Booking |
|---|---|---|---|
| Non-real-case | `v_1/` | sintetis kecil, 3 taksi, 2 SPKLU, 4 port, 3 marking area | 2 booking, seluruhnya acak seragam |
| Real-case v2 | `v_2/` | Bandung nyata, 50 taksi, 13 SPKLU, 14 port, 18 marking area | 185 atau 366 permintaan, berakar data |
| Network scaling | `network_scaling/` | sama persis dengan v_2 | sama persis dengan v_2 |

**Network scaling bukan dunia baru.** Generatornya menyalin config v_2 lalu hanya
mengganti `hidden_dim` dan `hidden_layers` sisi fleet. Sisi SPKLU sengaja tidak
disentuh supaya lawan dan lingkungan tetap identik.

Pada dunia non-real-case, **tidak satu pun variabel berakar data SPKLU nyata**.
Dunia itu memang sintetis untuk menguji perilaku mekanisme. Ia juga memakai tarif
bawaan engine 3.000 rupiah per km, bukan 6.300, karena config `v_1` tidak
menuliskan blok `fare` sama sekali.

---

## 8. Skenario v_2

Sejak 2026-08-12, istilah **okupansi 22,5 persen dan 50 persen sudah tidak
dipakai lagi**. Yang dikendalikan sekarang adalah **permintaan booking**, dan
okupansi menjadi keluaran yang diukur.

| Skenario | Permintaan | `prune_floor_price_on_shift` | Okupansi yang muncul |
|---|---:|---|---:|
| `medium_occupancy` | 185 | tidak diset | sekitar 26 persen |
| `high_occupancy` | 366 | tidak diset | sekitar 47 persen |
| `mahalkompen_medium_occupancy` | 185 | `true` | sekitar 26 persen |
| `mahalkompen_high_occupancy` | 366 | `true` | sekitar 47 persen |

### Asal angka 185 dan 366

Keduanya statistik dari sebaran yang sama, yaitu **jumlah transaksi per hari
seluruh SPKLU Bandung sepanjang 2024**, 366 nilai harian dari 22.514 transaksi,
dikalikan horizon tiga hari.

- **185** adalah rata-rata harian 61,51 transaksi dikali 3.
- **366** adalah kuantil 95 harian 122 transaksi dikali 3.

Keduanya dapat dihitung ulang kapan saja, jangan dihitung manual:

```bash
python reference_data_simulation/preprocessing/laju_permintaan.py
```

Skrip itu menulis `laju_permintaan.json` dan gambar laju hariannya, dan
terverifikasi menghasilkan tepat 185 dan 366.

### Mengapa permintaan, bukan okupansi

Dalam pemodelan sistem berkapasitas, yang ditetapkan adalah proses permintaan
sedangkan tingkat keterpakaian adalah keluaran. Memaksa okupansi mencapai angka
tertentu berarti membengkokkan permintaan demi mengejar keluaran. Kolom okupansi
di tabel di atas karena itu **perkiraan hasil, bukan sasaran**.

### Pemampatan yang harus selalu disebut

Volume seluruh Bandung dimampatkan ke 14 port dunia simulasi, yaitu **4,3 kali
kepadatan nyata**, sebab 34 SPKLU yang bertransaksi punya **60 port** nyata.
Pemampatan ini disengaja supaya ada rebutan slot yang dapat dinegosiasikan, dan
wajib disebut apa adanya.

⚠️ **Angka 236 port dan pemampatan 16,857 kali adalah SALAH**, jangan dipakai
lagi di mana pun. Keduanya lahir dari menjumlahkan kolom `connectors` tiap
chargebox, padahal pada **28 stasiun** kolom itu memuat jumlah konektor TOTAL
stasiun yang tersalin ke setiap baris. Cara yang benar satu port per chargebox.

| | 77 SPKLU | 34 SPKLU bertransaksi |
|---|---:|---:|
| jumlahkan kolom `connectors` (salah) | 323 | 236 |
| satu port per chargebox (benar) | 117 | **60** |

`laju_permintaan.py` sudah diperbaiki. **N = 185 dan 366 TIDAK terpengaruh**,
sebab keduanya diturunkan dari cacah transaksi harian, bukan dari jumlah port.

### Config dan seed

- Satu generator tunggal `v_2/gen_configs.py` menghasilkan **160 config**, yaitu
  40 run untuk masing-masing dari 4 skenario. Empat generator lama sudah dihapus.
- **`seed(run n) = 7 + 104729 * n`**, dan **sama lintas skenario**, sehingga
  keempat skenario berpasangan pada dunia yang seed-nya identik.
- `taxi_spec.initial_taxi_state` diturunkan deterministik dari seed lewat
  `sample_taxis`, lalu dibekukan ke dalam config.
- Seluruh field dunia lain diambil apa adanya dari
  `v_2/configs/_template_dunia.json`.
- Yang berbeda antar skenario hanya dua saklar, yaitu
  `app_user_booking.initial_booking_count` dan
  `negotiation.prune_floor_price_on_shift`.

Parameter penting di config v_2, yaitu `n_days` 3, `max_action_library` 30,
`max_steps` 15, `price_levels` 1973,44 / 2466,8 / 3453,52 / 4000, `fare` 6.300
mode PURE, `g_max` 21,4912, `tau` 48, `mbom_M` 2.

⚠️ **`v_2/configs/calibration_meta.json` sebagian sudah basi.** Blok
`varian_okup50`, `varian_30hari`, dan `seed_per_run` menggambarkan rancangan lama
yang sudah dibuang. Yang masih berlaku dari berkas itu hanya blok
`fixed_mechanism` dan rumus kalibrasinya.

---

## 9. Sampler booking

### Dua cara, satu kebijakan penempatan

| Cara | Argumen | Lama booking | Jam mulai booking |
|---|---|---|---|
| **Distribusi hasil fit** (bawaan) | tanpa argumen | Log-Logistik (Fisk), dijepit 1.502,2205 menit, dibulatkan **ke atas** ke kelipatan 30 menit | Weibull, dijepit 24 jam, dibulatkan ke kelipatan 30 menit **terdekat**, dijepit pada grid 47 |
| **Empiris lama** | `--empiris` | undian berbobot histogram, dibulatkan ke grid terdekat, dipotong 4 grid | jam bulat berbobot histogram, paruh jam seragam |

Parameter juara, dengan titik lokasi dikunci di nol:

| Variabel | Distribusi | Parameter | AIC | Jarak KS |
|---|---|---|---:|---:|
| Lama sesi | **Fisk** | c = 1,74291; skala = 47,06317 | 237.252,5 | 0,0359 |
| Jam mulai | **Weibull** | c = 2,76424; skala = 15,57593 | 141.388,3 | 0,0425 |

Parameter **tidak ditulis di kode**. `real_case_common/fitted_booking_sampler.py`
membacanya saat berjalan dari
`reference_data_simulation/fitting/best_parameters.json` dan
`reference_data_simulation/preprocessing/output/preprocessing_report.json`.

Penarikan memakai **fungsi kuantil bentuk tertutup** dan `random.Random` milik
engine, bukan penarik acak scipy, supaya hasilnya dapat diulang persis dari seed
dunia dan simulasi tidak bergantung scipy saat berjalan.

Contoh pembulatan jam, pukul 16.24 menjadi 16.30 dan pukul 17.43 menjadi 17.30.
Contoh pembulatan durasi, 1 menit menjadi 1 grid dan 31 menit menjadi 2 grid.

Batas jepit 1.502,2205 menit adalah durasi terbesar pada sampel analisis hasil
preprocessing, bukan angka pilihan.

`--empiris` **hanya mengganti sumber distribusinya**. Kebijakan penempatan sama
pada kedua mode, sehingga keduanya dapat dibandingkan secara adil. Jalur empiris
sudah diverifikasi identik bit demi bit dengan perilaku sebelum sampler baru
ditambahkan.

### Kebijakan penempatan booking

Permintaan diundi **sekali**, lalu yang dicari hanyalah tempatnya. **Durasi tidak
pernah diundi ulang.** Urutannya:

1. Coba port dan jam persis seperti yang diundi.
2. Masih pada jam yang sama, coba port lain menurut urutan acak berbobot volume
   stasiun, memakai algoritma Efraimidis-Spirakis.
3. Geser jam, besaran kecil dulu, sampai 4 grid ke depan atau ke belakang. Pada
   tiap jam baru seluruh port dicoba lagi.
4. Gagal semua berarti permintaan itu **ditolak**, tanpa undian kedua.

Cara lama mengundi ulang port, durasi, dan jam sekaligus, sehingga booking
panjang yang gagal digantikan booking pendek dan menghilang diam-diam dari dunia
simulasi. Terukur pada permintaan 366, penolakan turun dari 20,5 menjadi 7,4
persen dan penyimpangan sebaran durasi turun dari 12,35 menjadi 3,54 persen.

`max_tries` sudah **dihapus** dari `install_v2_bookings`.

Melebarkan `allowed_slot_shift_range` dari 1 sampai 4 menjadi 1 sampai 8 hanya
menurunkan penolakan dari 7,4 menjadi sekitar 5,4 persen, jadi **diputuskan tetap
1 sampai 4**.

### Sumber booking lain

Stasiun ditarik berbobot `(volume + 1) / Σ(volume + 1)` dari volume transaksi
2024. Penghalusan Laplace dipakai karena hanya 34 dari 77 stasiun yang pernah
bertransaksi, sehingga tanpa penambahan satu, stasiun bervolume nol akan
berpeluang nol selamanya. Port ditarik seragam di dalam stasiun. Hari mulai
seragam. Tarif kompensasi per kWh sebenarnya **angka tetap 986,72**, ditulis
sebagai `rng.uniform` dengan selang degenerate hanya karena mengikuti bentuk
rumus engine.

---

## 10. Fallback booking selalu menghadap depan

Setiap booking punya **tepat satu** rencana pindah, dipilih sekali saat dunia
dirakit oleh `_choose_fallback_destination` di
`simulation/world/world_assembly.py`.

1. Coba port yang sama, mulai **pada atau sesudah span asli booking berakhir**,
   jarak 0 sampai 3 grid, urutan diacak.
2. Bila gagal, coba port lain pada rentang slot yang sama, jarak terdekat dulu.
3. Bila keduanya gagal, booking itu dilewati sama sekali.

**Mengapa selalu di depan.** Rencana fallback dipilih saat dunia dirakit, ketika
belum ada satu grid pun yang berlalu, sehingga rencana itu sendiri tidak mengenal
konsep masa lalu. Tetapi ia baru dipakai jauh kemudian saat negosiasi berjalan.
Taksi hanya pernah meminta slot pada atau sesudah grid berjalan, dan ia hanya
bertabrakan dengan booking bila slot itu berada di dalam span booking. Karena
tujuan fallback ditaruh pada atau sesudah span itu berakhir, tujuannya pasti
berada di depan slot yang diminta, sehingga pasti di depan jam simulasi juga.

**Sebelum diperbaiki**, arah pergeseran diacak maju atau mundur, dan **45,7
persen** rencana fallback menunjuk ke slot yang lebih awal daripada booking itu
sendiri. Rencana semacam itu sudah terlewati saat dibutuhkan, sehingga
booking-nya berubah menjadi kendala keras dan slotnya tidak dapat direbut taksi
mana pun. Sekitar empat dari sepuluh kesempatan menggeser booking hilang karena
itu, bukan karena mekanisme negosiasinya lemah melainkan karena dunianya dirakit
menghadap arah yang salah.

**Setelah diperbaiki**, 100 persen rencana fallback menghadap ke depan, dan
jumlah kesempatan pergeseran yang benar-benar dapat dipakai naik dari 144 menjadi
195 pada sapuan 48 grid dengan tiga seed.

Sebagai lapis kedua, `simulation/bridge/action_library.py` memuat
`_earliest_shiftable_grid` dan menolak setiap pergeseran yang grid tujuannya
lebih awal dari `ceil(sim_time_min / 30)`. Setelah perbaikan di atas penjaga itu
tidak pernah lagi menolak apa pun, dan sengaja dipertahankan sebagai pengaman.

Perbaikan yang sama sudah **ikut diterapkan di `nyoba_general`** atas permintaan
peneliti, dan itulah satu-satunya perubahan kode yang pernah masuk ke sana.

---

## 11. Menjalankan eksperimen

### v_2, korpus utama

```bash
python v_2/gen_configs.py                 # sekali saja, 160 config
python v_2/gen_configs.py --skenario high_occupancy

python v_2/main_nego.py --agent-fleet DLST --run 1 --skenario medium_occupancy --lean
python v_2/main_greedy.py --run 1 --skenario high_occupancy --lean
python v_2/main_nonneg_v2.py --mech booking --run 1 --skenario medium_occupancy --lean
python v_2/main_nonneg_v2.py --mech as_is   --run 1 --skenario high_occupancy --lean
```

- `--agent-fleet` menerima `DLST`, `DLSTMBOM`, `DLSTMBOMNewMechanism`,
  `DLSTMBOMDirected`. Lawannya selalu `RLNegotiator`.
- `--run` menerima 1 sampai 40.
- `--skenario` menerima keempat nama pada bagian 8. Argumen lama `--okup50` dan
  `--mahalkompen` **sudah dihapus** dan ditolak argparse.
- `--lean` mematikan trace dan log detail, `aggregate_metrics` tetap ditulis.
- `--empiris` memakai sampler lama.
- `--smoke` menjalankan versi cepat dengan `n_days` 1 dan N diskala sepertiga,
  keluarannya ke `v_2/_smoke/`.
- `--online-log` menyalakan log training meski `--lean`, dan `--online-log-full`
  ikut menulis snapshot memori sehingga sangat boros disk.

Keluaran negosiasi ke
`v_2/data_<skenario>/negotiation/real_case_v2/run{n}_padded_<agen>/`, dan
non-negosiasi ke `v_2/data_<skenario>/non_negotiation/real_case_v2/<mech>/run{n}/`.

### v_1, dunia sintetis, tidak diubah sama sekali

```bash
python run_experiment_bulk_per_config.py --name <NAMA> \
    --config v_1/configs/<NAMA>.json --only 6 8 9 --modes PADDED \
    --out-root v_1/data/experiment_bulk

python metaheuristic/run_greedy.py --config v_1/configs/<NAMA>.json \
    --name <NAMA> --exp-root v_1/data/greedy

python non_negotiation_mechanism/as_is_mechanism/main_as_is.py \
    --name <NAMA> --config v_1/configs/<NAMA>.json --seed 42 --out-root v_1/data/as_is
python non_negotiation_mechanism/booking_mechanism/main_booking.py \
    --name <NAMA> --config v_1/configs/<NAMA>.json --seed 42 --out-root v_1/data/booking
```

🔴 **v_1 WAJIB dijalankan lewat titik masuk di `v_1/`, bukan skrip root.**

| Jenis | Titik masuk |
|---|---|
| Negosiasi | `v_1/main_bulk_v1.py` |
| Greedy lawan greedy | `v_1/main_greedy_v1.py` |
| Tanpa pemesanan dan pemesanan statis | `v_1/main_nonneg_v1.py` |

Ketiganya memanggil `overrides.pasang_dunia_lama()` lebih dulu, dan **tidak ada
saklar untuk mematikannya**. Itu disengaja supaya tidak mungkin ada run v_1 yang
memakai dunia yang salah.

**Sebabnya.** Kode simulasi sekarang menghasilkan dunia yang bergeser dari korpus
non-real-case lama. Perbedaannya sudah ditemukan dan hanya satu, yaitu pemilih
tujuan fallback versi lama **menghabiskan tepat tiga undian rng lebih banyak**.
Karena rangkaian bilangan acak berurutan, tiga undian yang hilang menggeser
seluruh undian sesudahnya, sehingga **order penumpang yang terbangkit berbeda**.
Dunia punya dua pemesanan, jadi fungsinya dipanggil dua kali.

**Cara angka tiga itu ditemukan.** Pada mekanisme non-negosiasi tidak ada
pemesanan yang benar-benar digeser, sehingga satu-satunya jalan perubahan itu
berpengaruh adalah lewat jumlah undian. Ruang pencariannya jadi satu dimensi,
lalu disisir 0 sampai 16 dan dibandingkan dengan Tabel V.1 laporan. Pada tiga
undian, pemesanan statis mereproduksi Tabel V.1 **persis lima kolom** untuk
`control_default`, `hz_n1`, `hz_n5`, dan `ref_n10`.

**Batasnya, sebutkan apa adanya.** `hz_n30` mendekat tetapi belum persis.
Mekanisme tanpa pemesanan tidak akan cocok, sebab perbaikan bug jarak 7 Juli
membuatnya berperilaku seperti pemesanan statis, dan itu **tidak** dikembalikan
karena akan memasang bug lagi. Dunia pulih juga **tidak** menjamin negosiasi
ikut pulih, sebab kode agen berubah dan versi Juli-nya sudah tidak ada.

Pilihan tujuan fallback sendiri **terbukti tidak berpengaruh** di dunia v_1. Dua
aturan tujuan berlawanan memberi hasil identik, sebab dunianya hanya punya dua
pemesanan sehingga praktis tak pernah bertabrakan dengan permintaan taksi.

**13 nama dunia** ada di `v_1/configs/`, seluruhnya berseed 42, sehingga
`--seed 42` berlaku untuk semuanya. Isinya dipangkas pada 2026-08-12 supaya
persis memuat apa yang muncul di laporan. Tiga config lama dihapus karena tidak
pernah dirujuk, yaitu `ex_entropy05_n10_upd1_s42`, `ms_ref_n10_upd1_s7`, dan
`ms_ref_n10_upd1_s123`.

`run_experiment_bulk_per_config.py` sanggup menjalankan matriks penuh sembilan
pasangan agen dikali dua mode, tetapi **korpus ini hanya memakai tiga pasangan
pada mode PADDED**, yaitu `exp06` (DLST-MBOM), `exp08` (DLST-MBOM Nego), dan
`exp09` (DLST murni). Karena itu `--only 6 8 9 --modes PADDED` wajib disertakan.

⚠️ **Mode tidak boleh dijalankan sebagai faktor eksperimen.** Laporan sengaja
tidak menyebut mode sama sekali supaya tidak memancing pertanyaan tentang varian
yang tidak diuji. Yang dipakai PADDED, dan ENCODER tidak dijalankan.

⚠️ Tabel IV.17 pada Bab IV mendaftar **enam** kombinasi agen sedangkan Bab V
melaporkan tiga. Peneliti memutuskan menjalankan yang tiga saja. Ketidakcocokan
antar-bab itu diketahui dan menunggu keputusan terpisah.

### network_scaling

⚠️ **`network_scaling/configs/` sekarang kosong.** Generator harus dijalankan
lebih dulu, dan config v_2 harus sudah ada karena ia sumbernya.

```bash
python network_scaling/gen_configs.py --hidden 512 --layers 3
python network_scaling/gen_configs.py --hidden 512 --layers 3 --variant high_occupancy

python network_scaling/main_nego_ns.py --agent-fleet DLST --hidden 512 --layers 3 \
    --run 1 --lean
```

Hanya **dua** keluarga skenario di sini, yaitu `medium_occupancy` dan
`high_occupancy`, tanpa mahalkompen, sebab yang diuji ukuran jaringan bukan
aturan kompensasi. Hanya **5 ulangan** per kombinasi, bukan 40, karena satu
kombinasi sudah mahal. `--agent-fleet` mendukung `DLST`, `DLSTMBOM`, dan
`DLSTMBOMNewMechanism`. `DLSTMBOMDirected` bukan bagian eksperimen ini.

Korpus ini **belum dipakai di laporan tesis** dan masih dalam pertimbangan.

### Uji coba bebas di luar korpus

```bash
python main.py --seed 42
python run_experiment.py --name coba_dulu
```

---

## 12. Status sekarang dan langkah berikutnya

### Yang sudah selesai dan terverifikasi

- Sampler hasil fit menjadi bawaan, `--empiris` mengembalikan histogram.
- Kebijakan penempatan P3 terpasang, `max_tries` dihapus.
- 160 config v_2 sudah ada dari satu generator.
- `booking_report.py` menulis metrik dan dua gambar per run.
- network_scaling menyesuaikan dua keluarga skenario.
- `v_1` kodenya tidak diubah sama sekali. Yang berubah hanya daftar config, dari
  16 dunia menjadi 13 dunia, mengikuti isi laporan.
- Seluruh berkas `.py` lolos kompilasi. Keempat entry point sudah diuji jalan
  ujung ke ujung.

### Data hasil sedang kosong

Seluruh folder data di `clean_code` sudah dihapus atas perintah peneliti, yaitu
`v_2/data_*`, `v_1/data`, `network_scaling/data`, `metaheuristic/data`, dan
`non_negotiation_mechanism/*/data`. **Jangan menjanjikan pemulihan data lama.**

### ✅ KAMPANYE SELESAI 2026-08-17 08:59, korpus penuh 1.198/1.198

Regenerasi korpus **TUNTAS**, yaitu v_1 78/78 dan v_2 1.120/1.120, dengan
**nol run rusak, nol folder parsial, nol gagal, dan nol baris DISK TIPIS**
sepanjang seluruh kampanye. Disk sisa 84 GB.

### 🔴 KEPUTUSAN PENAMAAN SKENARIO v_2, dari user 2026-08-17

Nama skenario di laporan memakai **"medium occupancy"** dan **"high occupancy"**
beserta varian **mahalkompen**, bukan penamaan berbasis permintaan. Saya sempat
menyarankan sebaliknya karena okupansi adalah hasil dan bukan kendali, tetapi
**user menegaskan pilihannya. Ikuti, jangan diungkit lagi.**

Belum diterapkan ke berkas mana pun. Yang akan tersentuh, Bab IV Tabel IV.37
beserta prosa IV.2.4.3, dan Bab V.2 pada empat heading dan dua belas caption.

Okupansi sebenarnya, medium 26,3 persen dan high 47,8 persen, sedangkan N tetap
persis 185 dan 366.

### ⚠️ Bab IV menulis 30 ulangan, korpusnya 40

Tabel IV.37 dan prosa IV.2.4.3 menyatakan 30 ulangan per keluarga, padahal
korpus yang sudah jadi berisi 40. Salah satu harus mengalah, dan ini **belum
diputuskan user**.

### Hasil v_2, ringkasan yang perlu diingat

Angka lengkap ada di memori `hasil-v2-analisis-lengkap`, perkakasnya
`laporan/_tooling/data_babv2.py`. Intinya:

- **Median negosiasi mengungguli pemesanan statis di seluruh 16 sel**, tetapi
  rata-ratanya kalah pada kedua keluarga medium karena ekor bawah.
- **Kolaps itu sifat agen, bukan sifat dunia.** Irisan kegagalan antar-agen
  selalu sekitar atau di bawah harapan acak.
- **Yang merusak bisnis adalah lamanya gagal, bukan gagalnya.**
- `avg_utility_fleet_deal` **bersyarat pada terjadinya deal**, jangan dibaca
  sendirian.
- **Penyaringan run terjelek tidak boleh menjadi hasil Bab V**, hanya
  diagnostik, sebab pemilihannya bersandar pada hasilnya sendiri.

Kalau nanti perlu menjalankan lagi, perkakasnya tetap yang di bawah ini, dan
aturan memeriksa proses hidup sebelum melepas ulang tetap berlaku.

```bash
.venv/Scripts/python.exe _antrean/status.py      # tabel sudah/belum per agen
.venv/Scripts/python.exe _antrean/periksa.py     # keutuhan, wajib sesudah stop
.venv/Scripts/python.exe _antrean/banding_v1.py  # hasil v_1 lawan Bab V lama
bash _antrean/hentikan.sh                        # berhenti dengan aman
bash _antrean/antrean_semua.sh                   # lanjutkan
```

**Sebelum melepas ulang, WAJIB cek dulu apakah masih hidup**, yaitu proses
`python.exe` yang CommandLine-nya memuat `clean_code`. Melepas dua kali membuat
dua antrean berebut folder yang sama.

**v_1 SUDAH TUNTAS 78/78** pada dunia korpus lama. Ringkasan hasilnya:

| Agen | Sel kolaps dari 13 |
|---|---:|
| DLST-MBOM Nego | **0**, seluruhnya deal-rate 1,000 |
| DLST-MBOM | 4 |
| DLST murni | 7 |

Temuan menarik, `cc_fixedopp` DLST murni justru **membaik jauh**, dari 0,097 di
korpus lama menjadi 0,975.

### Langkah berikutnya yang ditunggu, cakupannya SUDAH DIPUTUSKAN

Peneliti akan memerintahkan **menjalankan ulang korpus penuh**. Cakupannya sudah
ditetapkan pada 2026-08-12 dan tidak perlu ditanyakan lagi.

**Jangan mulai menjalankan sebelum ada aba-aba eksplisit.**

| Bagian | Rincian | Run |
|---|---|---:|
| v_2 negosiasi | 4 skenario x 4 agen x 40 ulangan | 640 |
| v_2 greedy | 4 skenario x 40 ulangan | 160 |
| v_2 non-negosiasi | 4 skenario x 2 mekanisme x 40 ulangan | 320 |
| v_1 negosiasi | 13 dunia x 3 pasangan, PADDED | 39 |
| v_1 greedy lawan greedy | 13 dunia | 13 |
| v_1 tanpa pemesanan | 13 dunia | 13 |
| v_1 pemesanan statis | 13 dunia | 13 |
| **TOTAL** | | **1.198** |

Waktu terukur dari korpus lama. v_2 sekitar **35 jam**, didominasi 640 run
negosiasi. v_1 justru murah, sebab dunianya cuma tiga taksi dan dua SPKLU,
sehingga 39 sub-eksperimennya selesai sekitar **25 menit**.

**Tingkat log yang diminta peneliti mengikuti korpus lama**, yaitu run negosiasi
memakai `--lean --online-log-full`, sedangkan greedy dan non-negosiasi memakai
`--lean` saja. Konsekuensinya boros, sebab satu run negosiasi terukur 18 sampai
122 MB, jadi v_2 saja perlu **40 sampai 80 GB**.

⚠️ **Halangan disk.** Per 2026-08-12 drive C tinggal **2,7 GB kosong** dari
466 GB. Peneliti akan memindahkan `nyoba_general` yang memakan 110,7 GB ke
perangkat lain lebih dulu. Kampanye **tidak boleh dimulai sebelum itu selesai**.
Riwayat `nyoba_general` mencatat pembersihan disk otomatis pernah menghapus run
31 sampai 40 ketika disk mencapai 94 persen.

---

## 13. Keluaran dan metrik per run

### Struktur folder satu eksperimen

```
<exp_root>/<name>/
  experiment_meta.json          # marker lengkap, DITULIS TERAKHIR
  logs/simulation/<sim_id>.jsonl
  logs/online_training/<sim_id>.jsonl
  runs/<sim_id>/                # JSON per hari dan per negosiasi, PNG
  registry/<side>/<version>/
  history/<sim_id>.pkl
  metrices/
    aggregate_metrics.json
    booking_generation.json     # hanya v_2 dan network_scaling
    distribusi_durasi.png       # hanya v_2 dan network_scaling
    distribusi_jam.png          # hanya v_2 dan network_scaling
```

**Folder tanpa `experiment_meta.json` berarti parsial atau korup**, dan harus
dihapus sebelum melanjutkan.

### `aggregate_metrics.json`

Tiga blok, yaitu `summary`, `numeric_flat`, dan `aggregate`. `summary` sendiri
punya tiga kelompok, yaitu `spklu_business`, `fleet_business`, dan
`negotiation_system`. Kunci penting di `aggregate` antara lain
`avg_utility_fleet`, `avg_utility_spklu`, `avg_deal_rate`,
`avg_partial_deal_rate`, `avg_cycles_waited`, `avg_episode_length`,
`negotiations_total`, serta versi deal-only `avg_utility_fleet_deal`,
`avg_utility_spklu_deal`, `social_welfare_deal`, `n_deal`, dan `n_no_deal`.

Untuk v_2 dan network_scaling, berkas ini juga disisipi `booking_generation` dan
`pergeseran_fallback`.

### `booking_generation.json`

Nasib setiap permintaan booking saat dunia dirakit.

| Kunci | Arti |
|---|---|
| `permintaan` | N yang diminta config |
| `terpasang` | yang berhasil masuk dunia |
| `tanpa_menggeser` | berhasil pada port dan jam persis seperti diundi |
| `menggeser_pindah_port` | jam sama, port lain |
| `menggeser_geser_jam` | jamnya digeser |
| `ditolak_kapasitas_penuh` | seluruh port penuh pada seluruh jam yang dicoba |
| `ditolak_di_luar_horizon` | durasinya tidak muat sebelum simulasi berakhir |
| `ditolak_tanpa_fallback` | tertempatkan tetapi tidak ada tujuan fallback yang muat |
| `okupansi_persen` | keterisian port yang dihasilkan |
| `massa_salah_tempat_*` | jarak sebaran terpasang terhadap sebaran hasil fit |
| `ambang_derau_*` | jarak yang muncul dari derau pencuplikan saja pada n yang sama |

⚠️ **Dua kunci terakhir wajib dibaca berpasangan.** Satu run hanya memuat
beberapa ratus booking sedangkan sebarannya punya puluhan kelas, sehingga
sebagian jarak yang terukur murni kebetulan. Massa salah tempat baru bermakna
bila jelas melampaui ambang derau. Contoh nyata dari run yang sudah diukur,
massa salah tempat durasi 6,23 persen dengan ambang derau 7,11 persen, yang
berarti tidak ada distorsi sistematis yang terdeteksi.

Penyebab penolakan terbesar sekarang adalah `ditolak_tanpa_fallback`, karena
jendela fallback menyempit setelah perbaikan menghadap-depan.

### `pergeseran_fallback`

Berapa kali booking benar-benar dipindahkan ke rencana fallback-nya karena
slotnya diambil taksi **saat negosiasi berjalan**. Ini berbeda dari pergeseran
saat dunia dirakit. Nilainya diambil dari
`summary.spklu_business.shifted_bookings_count` yang sudah ada di engine sejak
awal, jadi berlaku juga untuk v_1 tanpa perubahan kode.

---

## 14. `reference_data_simulation/`

Folder swasembada berisi seluruh rantai dari data mentah sampai model
distribusi. **Bukan bagian pipeline simulasi**, dan tidak dijalankan saat
eksperimen berjalan. Isinya identik dengan salinannya di `nyoba_general`.

| Subfolder | Isi |
|---|---|
| `real_data/` | salinan mentah seluruh data referensi, apa adanya |
| `filtering/` | penyaringan Bandung, data hasil, dan 15 grafik eksploratif |
| `visualization/` | 8 grafik sebaran variabel yang dipakai simulasi |
| `preprocessing/` | lima langkah penyiapan sampel analisis, 22.514 menjadi 22.479 baris |
| `fitting/` | pencocokan 10 keluarga distribusi ke 2 variabel, tabel metrik, dan gambar |

Rantai penyaringannya **mereproduksi persis** data yang mendasari seluruh
eksperimen, yaitu 77 stasiun identik dan 22.514 transaksi identik sel demi sel.
Kesamaan itu diperiksa otomatis setiap kali skripnya dijalankan.

### Preprocessing

Lima langkah, yaitu audit, pemeriksaan domain, **dequantisasi**, penyaringan
kesahihan fisik, dan pemeriksaan pencilan. Hasilnya 22.514 menjadi 22.479 baris.

- **Dequantisasi** dipakai karena durasi tercatat sebagai menit bulat. Nilai `k`
  diganti `k + U(0,1)`. Bukti bahwa konvensinya pemotongan bukan pembulatan,
  cacah nilai 0 adalah 283 sedangkan cacah nilai 1 hanya 141, yaitu rasio 2,007
  kali, padahal pembulatan setengah ke atas akan memprediksi sekitar 0,5 kali.
- **Penyaringan kesahihan fisik** membuang baris yang daya tersiratnya melampaui
  1,25 kali daya terpasang stasiun. Ambang itu diturunkan dari data, yaitu
  persentil 99 rasionya bernilai 1,0045. Hanya 35 baris yang terbuang.
- **Ekor kanan sengaja tidak dibuang.** Pagar Tukey tidak sah pada data
  sekondong ini dengan kemencengan +3,82. Sesi panjang di atas 12 jam sudah
  diperiksa dan memang nyata, yaitu kendaraan yang tetap tercolok setelah selesai
  mengisi.

### Fitting

Sepuluh keluarga bersupport tak negatif, yaitu `expon`, `gamma`, `weibull_min`,
`lognorm`, `invgauss`, `fisk`, `rayleigh`, `chi2`, `pareto`, dan `halfnorm`.
Titik lokasi **dikunci di nol** dengan `floc=0`, sehingga lokasi bukan parameter
bebas dan tidak dihitung dalam AIC maupun BIC. Hasil juaranya ada di bagian 9.

### Temuan sampingan yang diketahui

Master SPKLU `bandung_spklu.json` punya galat hitungan port, yaitu kolom jumlah
konektor total stasiun tersalin ke setiap chargebox, mengenai 30 dari 77 stasiun.
Dampaknya satu dari 13 stasiun dunia simulasi berubah, dan totalnya menjadi 13
port bila kolom yang benar dipakai. **Galat ini sengaja tidak diperbaiki** supaya
dunia simulasi tetap konsisten dengan korpus, dan dicatat sebagai keterbatasan di
`filtering/documentation.md`.

---

## 15. Jebakan teknis yang sudah pernah ditemukan

1. **`build_world_from_config(cfg)` tanpa argumen `rng` memakai seed acak sistem
   operasi.** Setiap pengujian perbandingan wajib memberi
   `rng=random.Random(cfg.seed)`. Ini pernah membuat sebuah uji melaporkan dunia
   berbeda padahal kodenya benar.
2. **Greedy `deal_success_rate` bernilai 0,0 itu bawaan, bukan regresi.** Korpus
   greedy yang tersimpan di `nyoba_general` juga menunjukkan angka yang sama
   dengan `charging_cost` 0,0. Jangan mengejarnya sebagai bug. Kolapsnya
   emergent, yaitu irisan daerah penerimaan kedua sisi memang sering kosong.
3. **`v2_world_meta.json` ada di `real_case_common/configs/`**, bukan
   `v_2/configs/`. Aman saat menghapus config v_2.
4. **Grep rekursif di seluruh folder project bisa timeout** karena `.venv`.
   Pakai tool Grep dengan `path` yang sempit.
5. **Menulis berkas Python lewat heredoc bash rawan menyisipkan null byte** dan
   gagal escaping. Tulis lewat tool Write, atau lewat skrip di scratchpad.
6. **Angka TVD per run harus dibaca berpasangan dengan `ambang_derau_*`.**
   Lihat bagian 13.
7. **Nilai okupansi bergeser saat sampler berganti.** Dengan N ditahan tetap,
   sampler hasil fit menghasilkan okupansi lebih tinggi daripada sampler empiris,
   sebab sampler empiris memotong durasi pada 4 grid sedangkan sampler hasil fit
   membiarkannya sampai 51 grid.

---

## 16. Peta dokumentasi

| Berkas | Isi |
|---|---|
| `README.md` | ikhtisar project, environment, cara pakai ringkas |
| `simulation_variable_description.md` | **asal-usul setiap variabel dunia simulasi**, untuk ketiga keluarga skenario |
| `v_1/README.md` | 13 dunia sintetis, padanannya di laporan, dan tiga korpus non-real-case |
| `v_2/README.md` | empat skenario, dua sampler, kebijakan penempatan, arti tiap metrik |
| `network_scaling/README.md` | dua keluarga skenario dan opsi ukuran jaringan |
| `real_case_common/README.md` | peran tiap modul kode bersama |
| `real_case_common/DOKUMENTASI_SAMPLING_DAN_PROVENANCE.md` | provenance angka dunia real-case |
| `metaheuristic/README.md` | aturan agen Greedy dan guard kejujurannya |
| `non_negotiation_mechanism/SCHEMA.md` | skema keluaran run non-negosiasi |
| `negotiation/agents/DLSTMBOMDirected_SPEC.md` | arsitektur agen anti-kolaps |
| `reference_data_simulation/README.md` | ikhtisar rantai analisis data |
| `reference_data_simulation/filtering/documentation.md` | langkah penyaringan dan galat port yang diketahui |
| `reference_data_simulation/preprocessing/documentation.md` | lima langkah preprocessing dan penurunan angka 185 dan 366 |
| `reference_data_simulation/fitting/documentation.md` | metodologi pencocokan, `floc=0`, tabel metrik |
| `reference_data_simulation/fitting/best_parameters.md` | parameter juara dalam bentuk terbaca |
| `v_2/configs/calibration_meta.json` | kalibrasi tarif dan `g_max`. ⚠️ sebagian blok sudah basi |

---

## 17. Rujukan kode cepat

| Berkas | Peran |
|---|---|
| `simulation/config/default.py` | seluruh nilai bawaan engine, dasar dunia non-real-case |
| `simulation/config/config.py` | pohon `SimulationConfig`, sumber kebenaran seluruh parameter |
| `simulation/world/world_assembly.py` | perakitan dunia, booking bawaan, pemilihan tujuan fallback |
| `simulation/world/order_generator.py` | sampling order penumpang saat berjalan |
| `simulation/bridge/action_library.py` | pembentukan Bundle, penjaga pergeseran ke masa lalu |
| `simulation/bridge/negotiation_runner.py` | menjalankan satu negosiasi bilateral |
| `simulation/run_pipeline.py` | perakitan dunia dan agen, penyalaan saklar Directed |
| `simulation/training/orchestrator.py` | kebijakan konkurensi pelatihan |
| `simulation/metrics/experiment_metrics.py` | pembentukan `aggregate_metrics.json` |
| `negotiation/agents/agent_factory.py` | pintu masuk tunggal pembangunan agen |
| `negotiation/environments/negmas_env.py` | pembungkusan mekanisme SAO NegMAS |
| `real_case_common/build_world_config.py` | pembuatan config dunia v2, pemilihan 13 SPKLU, sampling taksi |
| `real_case_common/booking_sampler.py` | sampler empiris, dipakai lewat `--empiris` |
| `real_case_common/fitted_booking_sampler.py` | sampler distribusi hasil fit, dipakai secara bawaan |
| `real_case_common/overrides.py` | pemasangan sampler v2 dan kebijakan penempatan |
| `real_case_common/booking_report.py` | penulisan metrik dan gambar per run |
| `real_case_common/v2_common.py` | `build_sampler`, tempat pilihan bawaan ditentukan |
| `v_2/gen_configs.py` | generator tunggal 160 config |
| `network_scaling/gen_configs.py` | penyalinan config v2 dengan penggantian ukuran jaringan |
| `reference_data_simulation/preprocessing/laju_permintaan.py` | penurunan angka 185 dan 366 |

---

## 18. Laporan tesis `laporan/thesis_ujung_parah.docx`

Berkas laporan tesis kini **berada di project ini**, di `laporan/`. Salinan di
`nyoba_general` sengaja dibiarkan utuh sebagai arsip dan **tidak boleh diubah**.

> 📕 **BACA DULU `laporan/aturan_penulisan_laporan.md`** sebelum menyentuh laporan.
> Berkas itu aturan lengkapnya, 24 bagian, mencakup aturan proses, gaya bahasa,
> tanda baca, istilah italic, penomoran, konvensi tabel dan gambar, API perkakas,
> jebakan, **peta seluruh direktori korpus data hasil**, keadaan dokumen terukur,
> dan daftar periksa akhir. Bagian 18 ini hanya ringkasannya.
>
> ⚠️ **Korpus v_2 kini 30 ulangan, bukan 40.** Run 31 sampai 40 sudah dipindahkan
> keluar project. Seluruh angka v_2 yang beredar dihitung dari 40 ulangan dan
> wajib dihitung ulang. Lihat bagian 20.4 berkas itu.

### Aturan proses, paling mengikat

1. **Hanya ubah bagian yang diperintahkan.** Satu-satunya pengecualian yang
   berlaku tanpa diminta ulang adalah pembetulan nomor gambar, tabel, dan rumus
   agar tetap konsisten.
2. **Setiap penambahan atau penghapusan gambar, tabel, atau rumus wajib diikuti
   penelusuran ulang seluruh dokumen** untuk membetulkan mention di prosa.
   Pembetulan itu hanya boleh mengubah angkanya, tidak boleh menyentuh kata lain.
3. **Kalau ada keputusan konten yang ambigu, tanya peneliti dulu.** Jangan
   berasumsi. Kalau menemukan masalah di luar cakupan yang diminta, laporkan dan
   minta persetujuan, jangan langsung diperbaiki.
4. **Cadangkan berkas penuh sebelum penyuntingan besar**, dengan pola nama
   `thesis_ujung_parah.backup_before_<nama_perubahan>.docx`.
5. **Investigasi data dulu sebelum menulis.** Setiap angka di laporan harus dapat
   ditelusuri ke berkas sumber yang benar-benar ada. Jangan mengarang angka.
6. Untuk rombakan besar, **susun outline dan minta persetujuan lebih dulu.**

### Gaya bahasa dan tanda baca

- Hanya **titik dan koma** sebagai tanda baca pemisah dalam prosa. Tidak boleh
  titik koma, titik dua, maupun dash pemisah klausa. Hyphen di dalam istilah
  majemuk tetap boleh, contohnya DLST-MBOM dan real-case. Tanda kurung boleh.
- Kalimat panjang mengalir yang disambung koma dan kata sambung, bukan daftar
  poin, kecuali bagian yang aslinya memang berupa daftar.
- Rujukan ke gambar dan tabel **wajib bernomor eksplisit**, contohnya
  Gambar IV.40. Dilarang memakai frasa seperti gambar berikut atau tabel di atas.
- Jangan menyatakan klaim kuantitatif yang belum diverifikasi.

### Istilah teknis

Istilah ML dan RL yang di komunitas riset lazim dipakai apa adanya ditulis
**Inggris-italic**, contohnya *state*, *action*, *policy*, *reward*, *return*,
*value function*, *head*, *layer*, *gate*, *frozen*, *real-case*, *deal-rate*,
*action library*, *propose*, *respond*. Kata kerja umum dan kata benda cognate
lazim **tetap Indonesia**, contohnya hitung, latih, perbarui, estimasi,
distribusi, jaringan, bobot, ambang, utilitas, episode. Nama agen tidak
di-italic. Setelah mengganti istilah secara menyeluruh, **wajib periksa gloss
dobel** seperti *policy* (*policy*).

### Penomoran, caption, dan gambar

- Caption memakai field `STYLEREF 1 \s` ditambah `SEQ Gambar/Tabel/Rumus \*
  ARABIC \s 1`, berstyle `Caption`. **Teks hasil field yang ter-cache sering
  basi**, jangan pernah dipercaya sebagai nomor yang benar.
- Nomor yang benar dihitung dengan **ground-truth simulation**, yaitu menelusuri
  paragraf berstyle `Caption` per bab lalu menghitung posisi urut per jenis.
  Rumus dikenali dari adanya elemen `<m:oMath>`. Cara ini sudah diverifikasi
  cocok dengan penomoran Word sungguhan.
- **Jangan memakai heuristik nomor tetangga terdekat.** Wajib mencocokkan isi
  caption dengan konteks paragraf yang merujuknya.
- **Satu gambar satu panel.** Caption Word harus **sama persis** dengan judul di
  dalam gambar. Di dalam gambar tidak boleh ada teks penjelasan tambahan.
- Angka format Indonesia, koma desimal dan titik ribuan.
- Tabel bergaya `TableGrid`, dipusatkan. Tabel padat perlu font 8 sampai 9 pt.

### Jebakan yang sudah terbukti mahal

1. **Frasa dan mention kerap terpecah antar-run** akibat penanda ejaan Word.
   Penggantian per-run akan gagal diam-diam. Pernah terjadi, 48 dari 153 mention
   terlewat. **Wajib mengganti pada tingkat paragraf** lalu menuliskan kembali ke
   run yang bersangkutan. Helper `ganti_frasa` menangani ini.
2. **Daftar Rumus memakai entri manual** berupa hyperlink ke bookmark
   `_TocRumusNNN` beserta field PAGEREF, sehingga tidak ikut terbarui sendiri
   ketika rumus ditambah atau dihapus. Saat refresh di Word, Daftar Rumus **wajib
   memakai "Update page numbers only"**, sebab "Update entire table" akan
   menimpa seluruh entri manual dan merusaknya. Daftar Gambar dan Daftar Tabel
   tidak punya batasan ini.
3. **Label entri Daftar Rumus tidak boleh dibaca dari gabungan seluruh run**,
   sebab pemisah ke nomor halaman berupa elemen tab dan bukan karakter teks,
   sehingga nomor rumus dan nomor halaman akan menempel. Baca run teks pertama.
4. **Kondisi lama yang sudah ada sejak sebelum pekerjaan ini**, yaitu 73 dari 121
   caption rumus tidak punya bookmark `_TocRumus`, serta Gambar IV.8, Rumus
   IV.54, dan Rumus IV.67 tidak pernah dirujuk prosa mana pun. Jangan dikira
   regresi baru.
5. Verifikasi rupa nyata dokumen hanya dapat dilakukan dengan **merender lewat
   MS Word COM** lalu membaca hasilnya sebagai gambar. `win32com` dan `pymupdf`
   hanya tersedia di Python sistem, bukan di `.venv` project.

### Keadaan Bab IV per 2026-08-12

Bab IV sudah diperbarui menyeluruh dan berisi **50 gambar, 39 tabel, dan 89
rumus**. Yang berubah, yaitu subbab baru IV.2.3 tentang pemodelan data referensi
dan sampling pemesanan, subbab baru IV.2.2.4 dan IV.2.4.2 tentang nilai variabel
dunia kedua skenario, penggantian skenario okupansi menjadi skenario permintaan
185 dan 366, penghapusan seluruh pembahasan kontensi, serta penyelarasan angka
port ke 60 port nyata dengan pemampatan 4,3 kali. **IV.1 dan IV.3 tidak
disentuh.**

⚠️ **Bab V masih memakai istilah lama**, yaitu Skenario Real-Case Okupansi 22,5
Persen dan 50 Persen, dan angkanya berasal dari korpus lama yang belum
diregenerasi. Ketidaksinkronan ini disengaja dan diketahui peneliti.

Satu gambar di IV.2.3.5 berupa **penanda**, sumbernya
`laporan/diagram_alur_sampling_pemesanan.drawio`, dan peneliti mengekspornya
sendiri lalu menempelkannya di Word.

### Perkakas perakitnya tersimpan, jangan menyunting docx manual

Seluruh perubahan Bab IV dihasilkan skrip di **`laporan/_tooling/`**, bukan
suntingan tangan. Baca `laporan/_tooling/README.md` sebelum menyentuh dokumen
lagi. Urutan menjalankannya `build.py` lalu `renumber.py` lalu `daftar.py`, dan
`build.py` selalu memulai dari cadangan sehingga aman diulang. Cadangannya
`laporan/thesis_ujung_parah.backup_before_bab4_data_sampling.docx`.

### Konvensi istilah italic, terukur dari dokumen sendiri

Rasio italic lawan tegak pada Bab IV dokumen asli, dipakai sebagai dasar
keputusan dan sudah diterapkan pada seluruh konten baru.

| Istilah | Prosa | Tabel | Caption | Heading | Keputusan |
|---|---|---|---|---|---|
| real-case | 22 : 5 | 3 : 0 | 8 : 0 | 4 : 0 | **italic** |
| port | 23 : 13 | 17 : 0 | 2 : 0 | 2 : 0 | **italic** |
| bundle | 17 : 9 | 9 : 0 | 2 : 0 | 1 : 0 | **italic** |
| slot | 8 : 6 | 8 : 0 | — | 1 : 0 | **italic** |
| fallback | 5 : 0 | 1 : 0 | — | 1 : 0 | **italic** |
| action library | 4 : 2 | 3 : 0 | — | 1 : 0 | **italic** |
| grid | 3 : 11 | 7 : 0 | — | 1 : 0 | tegak di prosa, italic di tabel dan heading |
| fleet | 6 : 15 | — | — | — | **tegak** |

Di dalam tabel, caption, dan heading konvensinya **mutlak italic tanpa satu pun
pengecualian**. `dxh.tandai_istilah` menerapkannya otomatis.

### Bukti verifikasi yang sudah dilakukan

Diff seluruh dokumen terhadap cadangan menunjukkan hanya **tiga blok perubahan
di luar Bab IV**, seluruhnya disengaja, yaitu entri Daftar Rumus, satu kata di
Bab III, dan satu frasa di Bab V. Dokumen berhasil dirender Word menjadi 199
halaman tanpa peringatan perbaikan berkas. Mention 203 buah seluruhnya sah dan
isinya dicocokkan dengan caption. Paragraf merah tetap 13, tidak bertambah.
Entri Daftar Rumus yang bookmark-nya menggantung tetap 39, sama seperti sebelum
pekerjaan ini.

### Yang masih menjadi tugas peneliti

1. Mengekspor `laporan/diagram_alur_sampling_pemesanan.drawio` lalu
   menempelkannya menggantikan penanda di IV.2.3.5.
2. Me-refresh field di Word. **Daftar Rumus wajib "Update page numbers only"**,
   sedangkan Daftar Gambar dan Daftar Tabel boleh "Update entire table".
3. Memutuskan apakah rujukan "Tabel IV.31, Tabel IV.32, dan Tabel IV.34" perlu
   dibuat berurutan. Angkanya sudah benar, hanya tabel tingkat harga menyisip di
   antaranya sebagai IV.33.

### Catatan integritas berkas sumber

Salinan di `nyoba_general/laporan/` kehilangan **satu paragraf kosong** dan
bertanggal dua menit sesudah penyalinan. Isinya secara substansi utuh, Bab IV di
sana masih 39 gambar, 30 tabel, dan 80 rumus, serta kontensi masih ada. Tidak
ada skrip yang menulis ke path itu. Dugaan terkuat berkasnya sempat terbuka di
Word. Disebutkan supaya tidak diklaim lebih dari yang dapat dibuktikan.
