# Asal-usul variabel dunia simulasi

Dokumen ini menjawab satu pertanyaan untuk **setiap** variabel yang membentuk
dunia simulasi: *angka itu datang dari mana?* Jawabannya selalu salah satu dari
empat kemungkinan berikut.

| Golongan | Arti |
|---|---|
| **Tetap** | ditulis langsung di config atau di kode sebagai satu angka |
| **Disampel, berakar data** | ditarik acak dari sebaran yang diduga dari data transaksi SPKLU nyata |
| **Disampel, tanpa akar data** | ditarik acak dari asumsi seragam atau dari logika algoritmik, bukan dari pengamatan |
| **Turunan** | dihitung dengan rumus dari besaran lain, tanpa keacakan |

Seluruh isi dokumen ini digrounding langsung ke kode di project ini. Analisis
data yang mendasari golongan kedua ada di `reference_data_simulation/`.

> **Dokumen ini menggambarkan project `clean_code`.** Project `nyoba_general`
> punya berkas dengan nama sama yang menggambarkan keadaan di sana, dan pada satu
> hal keduanya **berbeda**: di sini lama dan jam mulai booking bawaan ditarik dari
> distribusi teoretis hasil fit, sedangkan di `nyoba_general` keduanya tetap
> ditarik dari histogram data.

---

## Tiga keluarga skenario

| Keluarga | Folder | Dunia | Booking |
|---|---|---|---|
| **Non-*real-case*** | `v_1/` | sintetis kecil bawaan engine, 3 taksi, 2 SPKLU, 4 port, 3 marking area | 2 booking, seluruhnya acak seragam |
| **Real-case v2 fixed** | `v_2/` | Bandung nyata, 50 taksi, 13 SPKLU, 14 port, 18 marking area | 185 atau 366 permintaan booking, berakar data transaksi |
| **Network scaling** | `network_scaling/` | **sama persis dengan real-case v2**, hanya lebar dan kedalaman jaringan agen fleet yang divariasikan | sama persis dengan real-case v2 |

Kode bersama ketiganya ada di `real_case_common/` untuk dua keluarga terakhir,
dan di `simulation/` untuk ketiganya.

**Network scaling bukan dunia baru.** `network_scaling/gen_configs.py` menyalin
config `v_2/configs/run{n}_<skenario>.json` lalu hanya
mengganti `agent.fleet.hyperparameters.hidden_dim` dan `hidden_layers`. Sisi
SPKLU sengaja tidak disentuh supaya lingkungan dan lawan tetap identik.
Konsekuensinya, **setiap baris tentang real-case v2 pada dokumen ini berlaku
apa adanya untuk network scaling**, dan hanya bagian arsitektur jaringan yang
berbeda.

---

## A. Variabel dunia fisik

### A.1 Waktu

| Variabel | Non-RC | Real-case v2 dan network scaling | Golongan |
|---|---|---|---|
| Panjang simulasi `n_days` | 1, 5, 10, atau 30 hari menurut config | 3 hari | Tetap |
| Resolusi centang `tick_resolution_min` | 5 menit | 5 menit | Tetap |
| Selang negosiasi `interval_min` | 30 menit | 30 menit | Tetap |
| Panjang satu grid | 30 menit, `GRID_MINUTES` di kode | sama | Tetap, konstanta engine |
| Grid per hari | 48 | 48 | Turunan, 24 jam dibagi 30 menit |

### A.2 Stasiun pengisian dan port

| Variabel | Non-RC | Real-case v2 dan network scaling | Golongan |
|---|---|---|---|
| Jumlah stasiun | 2 | 13 | Tetap |
| Jumlah port | 4 | 14 | Tetap |
| Koordinat stasiun | karangan, sekitar pusat Bandung | koordinat nyata dari master SPKLU PLN | Tetap, berakar data untuk v2 |
| Pemilihan 13 stasiun | tidak berlaku | SPKLU PLN dengan port paling sedikit, pemutus seri nama, tanpa keacakan | Turunan dari data |
| Daya port `power_kw` | 50 kW | daya nyata per chargebox dari master | Tetap |
| Harga dasar per kWh | 2.500 rupiah, karangan | **2.466,8 rupiah**, median seluruh 22.514 transaksi Bandung | Tetap, berakar data untuk v2 |

Sumber pemilihan stasiun dan harga: `real_case_common/build_world_config.py`,
`EmpiricalBookingSampler.real_base_price()`.

**Catatan keterbatasan.** Jumlah port 14 itu ikut membawa satu galat pada master
SPKLU, yaitu kolom jumlah konektor total stasiun yang tersalin ke setiap
chargebox. Bila kolom yang benar dipakai, stasiun ke-13 berubah dan totalnya
menjadi 13 port. Rinciannya di `reference_data_simulation/filtering/documentation.md`.

### A.3 Marking area dan stasiun darurat

| Variabel | Non-RC | Real-case v2 dan network scaling | Golongan |
|---|---|---|---|
| Jumlah marking area | 3 | 18 | Tetap |
| Koordinat marking area | karangan | lokasi nyata pilihan peneliti | Tetap |
| Lokasi stasiun darurat | karangan | pilihan peneliti | Tetap |
| Harga darurat | 10.000 rupiah per kWh | pilihan peneliti | Tetap |

### A.4 Armada taksi

| Variabel | Non-RC | Real-case v2 dan network scaling | Golongan |
|---|---|---|---|
| Jumlah taksi | 3 | 50 | Tetap |
| Kapasitas baterai | 50 kWh | 50 kWh | Tetap |
| Konsumsi | 0,15 kWh per km | 0,15 kWh per km | Tetap |
| Laju | 30 km per jam | 30 km per jam | Tetap |
| Posisi awal | tiga koordinat tertulis di config | lintang seragam pada −6,98 sampai −6,84; bujur seragam pada 107,55 sampai 107,72 | Non-RC tetap; v2 **disampel tanpa akar data** |
| SoC awal | tiga nilai tertulis, 0,20 / 0,45 / 0,80 | seragam pada 0,35 sampai 0,90 | Non-RC tetap; v2 **disampel tanpa akar data** |

**Mengapa SoC awal taksi tidak diambil dari data.** Kolom `first_soc` pada data
transaksi adalah tingkat baterai saat pengguna **memutuskan mengisi**, yaitu
sebuah ambang keputusan dengan rerata 28 persen, bukan tingkat baterai armada di
awal hari. Selain itu 36 persen barisnya bernilai nol yang lebih mungkin berarti
tidak tercatat. Sebaran seragam 0,35 sampai 0,90 adalah asumsi peneliti yang
disebut terang-terangan sebagai asumsi.

Posisi dan SoC awal disampel dengan `random.Random(10000 + seed_run)` di
`real_case_common/build_world_config.py` fungsi `sample_taxis`, lalu hasilnya
**dibekukan ke dalam config** sebagai `taxi_spec.initial_taxi_state`. Jadi saat
simulasi berjalan angkanya sudah tetap; keacakan hanya terjadi sekali saat
config dibuat.

### A.5 Ambang baterai dan tarif perjalanan

| Variabel | Non-RC | Real-case v2 dan network scaling | Golongan |
|---|---|---|---|
| `B_charge` | 0,30 | 0,30 | Tetap |
| `B_area` | 0,50 | 0,50 | Tetap |
| `critical_soc` | 0,10 | 0,10 | Tetap |
| `SoC_target` | 1,00 | 1,00 | Tetap |
| **Tarif perjalanan per km** | **3.000 rupiah**, bawaan engine, tidak diset config | **6.300 rupiah** | Tetap |
| Mode tarif | `PURE` | `PURE` | Tetap |

**Asal angka 6.300.** Ralat 2026-07-08, yaitu tarif riil taksi listrik Bluebird
BYD, menggantikan 3.000 yang dinilai tidak mencerminkan tarif nyata. Perubahan
itu ikut mengubah kalibrasi fungsi utilitas, yaitu `lost/cycle` menjadi 0,4477
dan `g_max` menjadi 21,4912. Tercatat di `v_2/README.md` dan
`v_2/configs/calibration_meta.json`.

**Non-RC memakai 3.000** karena config `v_1/configs/*.json` sama sekali tidak
menuliskan blok `fare`, sehingga nilai bawaan `simulation/config/default.py`
yang berlaku. Ini bukan kelalaian: korpus non-RC memang dunia sintetis yang
tidak dimaksudkan mewakili tarif nyata.

### A.6 Tingkat harga negosiasi

| Variabel | Non-RC | Real-case v2 dan network scaling |
|---|---|---|
| `price_levels` | 2.000 / 3.000 / 4.000 rupiah, tetap | 1.973,44 / 2.466,8 / 3.453,52 / 4.000 rupiah |

Untuk real-case, tiga tingkat pertama adalah **turunan** dari harga dasar
2.466,8, yaitu 0,8 kali, 1,0 kali, dan 1,4 kali, sedangkan 4.000 adalah batas
atas pilihan peneliti.

---

## B. AppUserBooking

Inilah bagian yang paling berakar pada data, dan satu-satunya bagian yang
berbeda antara `clean_code` dan `nyoba_general`.

### B.1 Non-*real-case*: seluruhnya acak seragam

Config `v_1/configs/*.json` **tidak memasang override apa pun**, sehingga yang
berjalan adalah fungsi bawaan engine
`simulation/world/world_assembly.py` `_build_initial_bookings`:

| Variabel | Cara | Golongan |
|---|---|---|
| Jumlah booking | 2, dari `initial_booking_count` | Tetap |
| Port | `rng.choice(all_ports)`, seragam | Disampel, tanpa akar data |
| Lama booking | `rng.randint(1, 4)` grid, seragam | Disampel, tanpa akar data |
| Grid mulai | `rng.randint(0, total_grids - durasi)`, seragam | Disampel, tanpa akar data |
| Tarif kompensasi per kWh | `rng.uniform(0, 1000)` | Disampel, tanpa akar data |

**Tidak satu pun berakar pada data transaksi nyata.** Korpus non-RC memang dunia
sintetis untuk menguji perilaku mekanisme, bukan untuk mewakili Bandung.

### B.2 Real-case v2 dan network scaling: berakar data

Di sini `real_case_common/overrides.py` `install_v2_bookings` menambal
`_build_initial_bookings` dengan versi v2. Semantik lain dipertahankan identik
dengan engine, yaitu booking dilewati diam-diam bila span asalnya sudah terisi
atau tidak ada tujuan fallback yang muat.

| Variabel | Cara | Golongan |
|---|---|---|
| Jumlah permintaan booking | 185 skenario medium, 366 skenario high | Tetap, dari data, lihat B.4 |
| Stasiun lalu port | stasiun ditarik berbobot `(volume + 1) / Σ(volume + 1)` dari volume transaksi 2024 pada 13 stasiun; port ditarik seragam di dalam stasiun | **Disampel, berakar data** |
| **Lama booking** | lihat B.3 | **Disampel, berakar data** |
| **Jam mulai booking** | lihat B.3 | **Disampel, berakar data** |
| Hari mulai | `rng.randrange(n_days)`, seragam | Disampel, tanpa akar data |
| Tujuan fallback | logika engine `_choose_fallback_destination` | Disampel, tanpa akar data |
| Tarif kompensasi per kWh | `rng.uniform(986,72 , 986,72)` | Tetap, meskipun ditulis sebagai tarikan acak |

**Penghalusan Laplace pada bobot stasiun** dipakai karena hanya 34 dari 77
stasiun yang pernah bertransaksi sepanjang 2024. Tanpa penambahan satu pada
setiap pembilang, stasiun bervolume nol akan berpeluang nol selamanya.

**Tarif kompensasi itu sebenarnya angka tetap.** Selangnya degenerate, yaitu
batas bawah sama dengan batas atas, sehingga `rng.uniform` selalu mengembalikan
986,72. Ditulis sebagai tarikan acak hanya karena mengikuti bentuk rumus engine.
Kompensasi akhir satu booking adalah **turunan**:
`986,72 × power_kw × durasi_grid × 0,5`.

### B.3 Lama dan jam mulai booking: dua cara sampling

Kedua variabel inilah yang dianalisis penuh di `reference_data_simulation/`.
Tersedia dua cara, dan **cara bawaan di project ini adalah yang pertama**.

#### Cara bawaan: distribusi teoretis hasil fit

Kode: `real_case_common/fitted_booking_sampler.py`.

Parameter **tidak ditulis di kode**. Semuanya dibaca saat berjalan dari
`reference_data_simulation/fitting/best_parameters.json` dan
`reference_data_simulation/preprocessing/output/preprocessing_report.json`.

| Variabel | Distribusi juara | Parameter | Alih-ragam |
|---|---|---|---|
| Lama booking | **Log-Logistik (Fisk)** | c = 1,74291; skala = 47,06317; lokasi 0 | jepit pada 1.502,2205 menit, lalu **bulat ke ATAS** ke kelipatan 30 menit |
| Jam mulai booking | **Weibull** | c = 2,76424; skala = 15,57593; lokasi 0 | jepit pada 24 jam, lalu **bulat ke kelipatan 30 menit TERDEKAT**, lalu jepit pada grid 47 |

Contoh aturan pembulatan jam: pukul 16.24 menjadi 16.30, pukul 17.43 menjadi
17.30. Contoh aturan pembulatan durasi: 1 menit menjadi 1 grid, 31 menit menjadi
2 grid, 121 menit menjadi 5 grid.

Penarikan memakai **fungsi kuantil bentuk tertutup** dan `random.Random` milik
engine, bukan penarik acak scipy. Dua alasannya: hasilnya tetap dapat diulang
persis dari seed dunia yang sama, dan simulasi tidak menjadi bergantung pada
scipy saat berjalan.

Batas jepit 1.502,2205 menit adalah **durasi terbesar pada sampel analisis
hasil preprocessing**, bukan angka pilihan. Grid maksimum yang mungkin karena
itu 51, jauh di atas batas 4 grid yang berlaku pada cara empiris.

#### Cara lama: penarikan ulang empiris, lewat argumen `--empiris`

Kode: `real_case_common/booking_sampler.py`.

| Variabel | Cara |
|---|---|
| Lama booking | bin 15 menit ditarik berbobot dari `durasi_hist_bin15`; menit ditarik seragam di dalam bin; dibulatkan ke kisi setengah jam **terdekat**; **dipotong pada [1, 4] grid**, yaitu 0,5 sampai 2 jam |
| Jam mulai booking | jam bulat ditarik berbobot dari `hour_hist`; paruh jam ditarik seragam |

Tidak ada parameter yang diduga dan tidak ada asumsi bentuk. Cara ini
**dipertahankan utuh dan terbukti identik bit demi bit** dengan perilaku sebelum
sampler baru ditambahkan.

Menjalankannya:

```bash
python v_2/main_nego.py --agent-fleet dlst --run 1 --empiris
python v_2/main_greedy.py --run 1 --empiris
python v_2/main_nonneg_v2.py --mech booking --run 1 --empiris
python network_scaling/main_nego_ns.py --agent-fleet dlst --run 1 --hidden 512 --layers 3 --empiris
```

Tanpa `--empiris`, keempatnya memakai distribusi hasil fit.

#### Kebijakan penempatan booking

Permintaan diundi **sekali**, lalu yang dicari hanyalah tempatnya. Durasi tidak
pernah diundi ulang. Urutannya: coba port dan jam persis seperti diundi; lalu
port lain pada jam yang sama menurut urutan acak berbobot volume stasiun; lalu
geser jam sampai 4 grid ke depan atau ke belakang, mencoba seluruh port pada tiap
jam baru. Gagal semua berarti permintaan itu ditolak, tanpa undian kedua.

Cara lama mengundi ulang port, durasi, dan jam sekaligus, sehingga booking
panjang yang gagal digantikan booking pendek dan menghilang diam-diam. Terukur
pada permintaan 366: penolakan turun dari 20,5 menjadi 7,4 persen, dan
penyimpangan sebaran durasi turun dari 12,35 menjadi 3,54 persen.

Kebijakan ini **sama untuk kedua sampler**; `--empiris` hanya mengganti sumber
distribusinya.

### B.4 Jumlah permintaan booking

**Tetap, diturunkan dari data.** Dua nilai, keduanya statistik dari sebaran
jumlah transaksi per hari seluruh SPKLU Bandung sepanjang 2024, yaitu 366 nilai
harian dari 22.514 transaksi, dikali horizon tiga hari:

| Skenario | Permintaan | Dasar | Okupansi yang muncul |
|---|---:|---|---:|
| `medium_occupancy` | **185** | rata-rata harian 61,51 transaksi | sekitar 26 persen |
| `high_occupancy` | **366** | kuantil 95 harian 122 transaksi | sekitar 47 persen |

**Yang dikendalikan adalah permintaan, bukan okupansi.** Dalam pemodelan sistem
berkapasitas, yang ditetapkan adalah proses permintaan sedangkan tingkat
keterpakaian adalah keluaran. Memaksa okupansi mencapai suatu angka berarti
membengkokkan permintaan demi mengejar keluaran. Kolom terakhir karena itu
perkiraan hasil, bukan sasaran; angka sebenarnya dicatat per run pada
`metrices/booking_generation.json`.

Permintaan yang tidak tertampung dicatat sebagai **permintaan tertolak**, bukan
dibuang diam-diam. Pada permintaan 185 penolakannya sekitar 4,7 persen, dan pada
366 sekitar 7,4 persen.

**Volume seluruh Bandung dimampatkan ke 14 port** dunia simulasi, yaitu sekitar
16,9 kali kepadatan nyata, sebab Bandung nyata melayani 61,51 transaksi per hari
dengan 236 port. Pemampatan itu disengaja supaya ada rebutan slot yang dapat
dinegosiasikan, dan harus disebut apa adanya. Okupansi nyata seluruh Bandung
sepanjang 2024 hanya 1,37 persen.

### B.5 Fallback booking

Setiap booking punya **tepat satu** rencana pindah. Tujuannya dipilih sekali saat
dunia dirakit oleh `simulation/world/world_assembly.py`
`_choose_fallback_destination`:

1. Coba port yang sama, mulai **pada atau sesudah span asli booking berakhir**.
   Jaraknya diambil dari `allowed_slot_shift_range`, yaitu 0 sampai 3 grid
   sesudah span itu berakhir, dengan urutan diacak.
2. Bila gagal, coba port lain pada rentang slot yang sama, jarak terdekat dulu.
3. Bila keduanya gagal, booking itu dilewati sama sekali.

**Golongan: disampel tanpa akar data.** Ini logika algoritmik engine, bukan
pengamatan perilaku pengguna nyata.

**Mengapa tujuannya selalu di depan.** Rencana fallback dipilih saat dunia
dirakit, ketika belum ada satu grid pun yang berlalu, sehingga rencana itu
sendiri tidak mengenal konsep masa lalu. Tetapi ia baru **dipakai** jauh
kemudian, saat negosiasi berjalan. Taksi hanya pernah meminta slot pada atau
sesudah grid berjalan, dan ia hanya bertabrakan dengan booking ini bila slot itu
berada di dalam span booking. Karena tujuan fallback ditaruh pada atau sesudah
span itu berakhir, tujuannya pasti berada di depan slot yang diminta, sehingga
pasti di depan jam simulasi juga.

Dulu arah pergeseran diacak maju atau mundur, dan **45,7 persen** rencana
fallback menunjuk ke slot yang lebih awal daripada booking itu sendiri. Rencana
semacam itu sudah terlewati pada saat dibutuhkan, sehingga booking-nya berubah
menjadi kendala keras dan slotnya tidak dapat direbut taksi mana pun. Sekitar
empat dari sepuluh kesempatan menggeser booking hilang karena itu, bukan karena
mekanisme negosiasinya lemah melainkan karena dunianya dirakit menghadap arah
yang salah.

Setelah diperbaiki, **100 persen** rencana fallback menghadap ke depan, dan
jumlah kesempatan pergeseran yang benar-benar dapat dipakai naik dari 144
menjadi 195 pada sapuan 48 grid dengan tiga seed.

Sebagai lapis kedua, `simulation/bridge/action_library.py` memuat
`_earliest_shiftable_grid` dan tetap menolak setiap pergeseran yang grid
tujuannya lebih awal dari `ceil(sim_time_min / 30)`. Setelah perbaikan di atas,
penjaga itu tidak pernah lagi menolak apa pun, dan memang sengaja dipertahankan
sebagai pengaman.

---

## C. Order penumpang

Berbeda dari booking yang hanya disampel sekali di awal, order penumpang
disampel **berulang setiap selang negosiasi** sepanjang simulasi. Sumber:
`simulation/world/order_generator.py`. Berlaku sama untuk ketiga keluarga
skenario kecuali nilai parameternya.

| Variabel | Cara | Non-RC | Real-case v2 dan NS | Golongan |
|---|---|---|---|---|
| Jumlah order per area per selang | `order_min + round(u_condong × (order_max − order_min))`, dengan `u_condong` tarikan seragam yang dicondongkan ke atas sebanding bobot jam | `order_min` 1, `order_max` 5 | `order_min` 0, `order_max` 50 | Disampel, sebagian berakar data lewat bobot jam |
| Bobot jam `time_weight` | tabel 24 angka tetap | bawaan engine | **TomTom Traffic Index Bandung** | Tetap, berakar data untuk v2 |
| Titik jemput | seragam dalam cakram berjari-jari 0,5 km dari pusat area, sudut seragam dan jarak sebanding akar tarikan seragam | sama | sama | Disampel, tanpa akar data |
| Jarak tujuan | seragam pada 0 sampai jari-jari aman, dengan jari-jari aman dihitung dari sisa energi taksi | sama | sama | Disampel, tanpa akar data |
| Titik tujuan | seragam dalam cakram jari-jari di atas, lalu dijepit ke cakram acuan bila `order_reference_point` diset | acuan **diset** | acuan **tidak diset**, penjepitan tidak aktif | Disampel, tanpa akar data |

Jari-jari 0,5 km adalah konstanta engine `_PICKUP_RADIUS_KM` di
`simulation/world/order_generator.py` baris 53.

**Bobot jam adalah satu-satunya bagian order yang berakar data**, dan itu pun
hanya sebagai pengali kecondongan, bukan sebagai sebaran yang langsung disampel.

---

## D. Parameter mekanisme, bukan dunia fisik

Angka-angka berikut menentukan perilaku agen dan negosiasi, bukan dunia fisik.
Seluruhnya **tetap**, ditulis di config.

| Variabel | Non-RC | Real-case v2 | Network scaling |
|---|---|---|---|
| `max_action_library` | 10 atau 16 | menurut config skenario | sama dengan v2 |
| `max_steps` | 20 | menurut config skenario | sama dengan v2 |
| `prune_floor_price_on_shift` | mati | mati pada base, **hidup pada varian mahalkompen** | sama dengan v2 |
| `ufun.shape.g_max` | 1,0 | **21,4912**, hasil kalibrasi terhadap tarif 6.300 | sama dengan v2 |
| `ufun.shape.tau` | 5,0 | **48,0** | sama dengan v2 |
| `hidden_dim` dan `hidden_layers` sisi fleet | bawaan | bawaan | **inilah satu-satunya yang divariasikan** |
| Seed per run | tertulis di nama berkas config | tertulis di config, ditetapkan sebelum eksperimen | mewarisi dari v2 |

---

## E. Ringkasan menurut golongan

| Golongan | Non-RC | Real-case v2 dan network scaling |
|---|---|---|
| **Disampel, berakar data** | tidak ada | lama booking, jam mulai booking, stasiun booking |
| **Disampel, tanpa akar data** | port booking, lama booking, grid mulai booking, kompensasi, seluruh order penumpang | posisi awal taksi, SoC awal taksi, hari mulai booking, tujuan fallback, seluruh order penumpang kecuali bobot jam |
| **Tetap, berakar data** | tidak ada | harga dasar per kWh, koordinat dan daya stasiun, bobot jam order |
| **Tetap, keputusan peneliti** | seluruh dunia fisik, tarif 3.000 per km | jumlah taksi, jumlah marking area, tarif 6.300 per km, n_days, seluruh parameter mekanisme |
| **Turunan** | tidak ada | jumlah booking N, `price_levels`, kompensasi per booking, pemilihan 13 stasiun |

**Kesimpulan pokoknya.** Pada korpus non-*real-case*, tidak satu pun variabel
berakar pada data SPKLU nyata; dunianya sengaja sintetis. Pada korpus real-case
v2 dan network scaling, yang berakar data adalah **tiga variabel booking**
ditambah **harga, koordinat dan daya stasiun, serta bobot jam order**. Sisanya
tetap merupakan asumsi peneliti atau logika algoritmik, dan itu disebut
terang-terangan supaya tidak ditafsirkan lebih jauh daripada semestinya.

---

## F. Rujukan kode

| Berkas | Peran |
|---|---|
| `simulation/config/default.py` | seluruh nilai bawaan engine, dasar dunia non-RC |
| `simulation/world/world_assembly.py` | perakitan dunia, booking bawaan, pemilihan tujuan fallback |
| `simulation/world/order_generator.py` | sampling order penumpang saat berjalan |
| `simulation/bridge/action_library.py` | pembentukan opsi, penjaga pergeseran booking ke masa lalu |
| `real_case_common/build_world_config.py` | pembuatan config dunia v2, pemilihan 13 SPKLU, sampling posisi dan SoC taksi |
| `real_case_common/booking_sampler.py` | sampler empiris, dipakai lewat `--empiris` |
| `real_case_common/fitted_booking_sampler.py` | sampler distribusi hasil fit, dipakai secara bawaan |
| `real_case_common/overrides.py` | pemasangan sampler v2 di atas engine |
| `real_case_common/v2_common.py` | `build_sampler`, tempat pilihan bawaan dan `--empiris` ditentukan |
| `network_scaling/gen_configs.py` | penyalinan config v2 dengan penggantian ukuran jaringan fleet |
| `reference_data_simulation/` | seluruh rantai analisis data yang mendasari distribusi hasil fit |
