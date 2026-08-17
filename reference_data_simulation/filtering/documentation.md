# `filtering/` — dokumentasi penyaringan data asli menjadi data Bandung

Tahap pertama dari tiga tahap `reference_data_simulation/`. Masukannya data
mentah nasional di `../real_data/`; keluarannya data Bandung yang **terbukti
identik** dengan data yang benar-benar dipakai simulasi.

`README.md` di folder ini adalah ringkasan singkat. Berkas ini adalah
dokumentasi lengkapnya, yaitu apa yang dikerjakan, mengapa dikerjakan begitu,
dan bagaimana kebenarannya dibuktikan.

```bash
python reference_data_simulation/filtering/step1_filter_spklu.py      # ~10 detik
python reference_data_simulation/filtering/step2_filter_transaksi.py  # ~2 menit
python reference_data_simulation/filtering/step3_eda.py               # ~20 detik
```

Ketiganya berurutan karena langkah berikutnya memakai keluaran sebelumnya.

## Kedudukan tahap ini

Berkas turunan yang dipakai simulasi, yaitu `simulation_data/_derived/`, sudah
ada sejak lama tetapi **aturan pembuatannya tidak pernah didokumentasikan**.
Tanpa aturan itu, tidak ada yang dapat memastikan dari mana 77 stasiun dan
22.514 transaksi itu berasal. Tahap ini merekayasa balik aturannya, menuliskannya
sebagai kode yang dapat dijalankan ulang, lalu **membuktikan** bahwa keluarannya
sama persis dengan berkas turunan tersebut.

Berkas di `simulation_data/_derived/` hanya **dibaca** untuk pembuktian itu dan
tidak pernah ditulis.

## Langkah 1 — menyaring stasiun

`step1_filter_spklu.py` membaca master SPKLU nasional
(`MASTER SPKLU PER 14 MAR 2025_.xlsx`, lembar `SPKLU AKTIF PER MESIN (2)`,
3.586 baris mesin yang menjadi 2.431 stasiun) dan menyaring stasiun Bandung.

**Aturan hasil rekayasa balik:**

```
IS_ACTIVE_SPKLU = "Y"
DAN   -7,05 <= LATITUDE  <= -6,80
DAN  107,50 <= LONGITUDE <= 107,78
```

Hasilnya **77 stasiun**.

**Mengapa kotak koordinat, bukan nama kota.** Empat puluh satu dari 77 stasiun
itu adalah SPKLU mitra non-PLN, yaitu Hyundai, Toyota, Wuling, Voltron, dan
Starvo, yang kolom kota dan kolom unit induknya kosong pada master. Menyaring
berdasarkan nama wilayah akan membuang seluruh stasiun mitra itu dan hanya
menyisakan 43 stasiun, sehingga tidak mungkin menghasilkan 77 stasiun yang ada
pada berkas turunan. Kotak koordinat adalah satu-satunya aturan yang cocok.

**Keluaran:** `output/bandung_spklu.json`, `output/filtering_spklu_report.json`.

### Temuan: dua kolom jumlah konektor memberi angka berbeda

Master menyediakan dua kolom yang keduanya dapat dibaca sebagai jumlah port:

| Kolom | Arti | Sifat |
|---|---|---|
| `JML_CONNECTOR_SPKLU` | total konektor satu stasiun | nilainya sama untuk semua baris chargebox milik stasiun itu |
| `JML_CONNECTOR_CHARGER` | konektor milik chargebox pada baris itu | berbeda antar baris |

Berkas turunan yang dipakai simulasi menyalin `JML_CONNECTOR_SPKLU` ke setiap
chargebox lalu menjumlahkannya. Karena angka total stasiun tersalin berulang,
penjumlahan itu melebih-hitung port. Contoh paling jelas: SPKLU PLN ULP Bandung
Utara punya 4 chargebox dan angka total stasiunnya 9, sehingga penjumlahan
menghasilkan 36 port padahal jumlah sebenarnya 8. Dari 77 stasiun, **30 stasiun**
terdampak.

**Dampaknya ke dunia simulasi.** Dunia simulasi memakai 13 SPKLU PLN dengan port
paling sedikit. Bila kolom yang dipakai diganti, susunan 13 stasiun itu berubah
pada **tepat satu stasiun**:

| | Kolom yang dipakai (`JML_CONNECTOR_SPKLU`) | Kolom alternatif (`JML_CONNECTOR_CHARGER`) |
|---|---|---|
| Stasiun ke-13 | SPKLU PLN Geowisata Inn, 2 port | SPKLU PLN ULP Bandung Barat, 1 port |
| Total port | **14** | 13 |

**Keputusan: perilaku lama dipertahankan.** Skrip ini sengaja menyalin kolom
yang sama seperti berkas turunan, supaya keluarannya identik, sambil melaporkan
besar penyimpangannya di `output/filtering_spklu_report.json`. Alasannya:
memperbaikinya akan membuat seluruh data hasil eksperimen yang sudah ada tidak
lagi dapat direproduksi. Temuan ini dicatat sebagai keterbatasan, bukan
diperbaiki diam-diam.

## Langkah 2 — menyaring transaksi

`step2_filter_transaksi.py` membaca 402.119 transaksi nasional secara mengalir,
sebab berkasnya 48 MB dan tidak perlu dimuat sekaligus ke memori, lalu menyimpan
transaksi yang `ID_SPKLU`-nya termasuk 77 stasiun Bandung.

Hasilnya **22.514 transaksi**.

### Tambahan: tiga kolom waktu baru

Ekstraksi lama membuang menit dan detik:

```python
hh = int(waktu.split(":")[0])     # "13:47:22" menjadi 13
```

Akibatnya 22.514 transaksi hanya menempati **24 nilai berbeda**. Untuk keperluan
pemodelan distribusi, skrip ini menambahkan tiga kolom **tanpa mengubah kolom
lama**:

| Kolom baru | Isi |
|---|---|
| `minute` | menit transaksi, 0 sampai 59 |
| `second` | detik transaksi, 0 sampai 59 |
| `hour_frac` | jam sebagai pecahan, `hour + minute/60 + second/3600` |

`hour_frac` punya **19.169 nilai berbeda**.

**Mengapa ini perlu.** Pada jam bulat, fungsi distribusi kumulatif empiris
berbentuk tangga dengan 24 anak tangga besar, sehingga jaraknya terhadap kurva
kontinu manapun otomatis besar. Statistik Kolmogorov-Smirnov, Cramer-von Mises,
dan Anderson-Darling menjadi bias dan peringkat antar distribusi tidak lagi
informatif. Dengan resolusi menit dan detik, kurva kumulatifnya mulus dan
ketiganya sah dipakai. Argumen yang sama dipakai lagi pada `../preprocessing/`
langkah 3 untuk variabel durasi.

Kolom `hour` yang lama tetap ada sehingga kedua versi dapat dibandingkan
berdampingan.

**Keluaran:** `output/bandung_trx.csv` (20 kolom), `output/bandung_trx_summary.json`,
`output/filtering_transaksi_report.json`.

### Hasil verifikasi

Setiap kali dijalankan, skrip membandingkan keluarannya dengan berkas turunan
yang benar-benar dipakai simulasi. Hasilnya tercatat pada berkas laporan:

| Yang diperiksa | Hasil |
|---|---|
| Daftar 77 stasiun beserta koordinat dan chargebox | identik, `IDENTIK: true` |
| Jumlah baris transaksi | 22.514 lawan 22.514 |
| Seluruh 17 kolom bersama, sel demi sel, 22.514 baris | nol perbedaan |
| Histogram jam dan histogram durasi yang dibaca sampler | identik |

Tujuh belas kolom yang dibandingkan: `id_spklu`, `nama_spklu`, `chargebox_id`,
`kw_tier`, `date`, `hour`, `dow`, `month`, `kwh`, `rpkwh`, `rppakai`,
`price_per_kwh`, `durasi_min`, `first_soc`, `last_soc`, `connector`, `id_user`.
Tiga kolom sisanya, yaitu `minute`, `second`, dan `hour_frac`, memang tambahan
baru sehingga tidak ada padanannya untuk dibandingkan.

**Arti hasil ini.** Aturan penyaringan yang direkayasa balik terbukti benar.
Seluruh analisis pada `../preprocessing/` dan `../fitting/` karena itu berpijak
pada data yang sama persis dengan yang dipakai simulasi, bukan pada data yang
mirip.

## Langkah 3 — analisis data eksploratif

`step3_eda.py` menghasilkan 15 grafik dan satu laporan di `eda/`, mencakup
seluruh kolom termasuk yang tidak dipakai simulasi, yaitu hari, bulan, jenis
konektor, dan pengguna. Grafik khusus variabel yang dipakai simulasi ada
terpisah di `../visualization/`.

Angka pokok hasil penyaringan:

| Besaran | Nilai |
|---|---|
| Transaksi | 22.514 |
| Stasiun yang bertransaksi | 34 dari 77 aktif |
| Pengguna unik | 7.025 |
| Energi per sesi | rata-rata 23,43 kWh |
| Durasi sesi | median 45 menit, rata-rata 75,66 menit, maksimum 1.502 menit |
| Kemiringan durasi | +3,82, miring kanan |
| SoC saat datang | rata-rata 28,3 persen |
| Harga per kWh | median 2.466,8 rupiah |
| Jam puncak dan lembah | 13:00 dan 03:00 |

### Temuan lain yang perlu diketahui

**SoC saat datang bernilai nol pada 36 persen baris.** Kolom `first_soc`
seharusnya berisi tingkat baterai saat pengguna mulai mengisi. Nilai nol pada
lebih dari sepertiga baris hampir pasti berarti data tidak terbaca dari
kendaraan, bukan baterai benar-benar kosong. Kolom ini **tidak dipakai** simulasi
maupun tahap fitting, sehingga temuan ini hanya dicatat.

**Hanya 34 dari 77 stasiun yang pernah bertransaksi.** Empat puluh tiga stasiun
lain aktif menurut master tetapi tidak muncul sama sekali pada data transaksi
2024. Inilah sebabnya sampler stasiun memakai penghalusan Laplace, sebab tanpa
itu stasiun bervolume nol akan berpeluang nol untuk selamanya.

## Isi folder

```
filtering/
├── step1_filter_spklu.py          menyaring stasiun
├── step2_filter_transaksi.py      menyaring transaksi + kolom waktu baru
├── step3_eda.py                   analisis data eksploratif
├── README.md                      ringkasan singkat
├── documentation.md               berkas ini
├── output/
│   ├── bandung_spklu.json                77 stasiun
│   ├── bandung_trx.csv                   22.514 transaksi, 20 kolom
│   ├── bandung_trx_summary.json          histogram ringkas
│   ├── filtering_spklu_report.json       laporan + hasil verifikasi
│   └── filtering_transaksi_report.json   laporan + hasil verifikasi
└── eda/
    ├── EDA.md                     laporan terbaca
    └── 01..15_*.png               15 grafik
```

## Tahap berikutnya

`../preprocessing/` membaca `output/bandung_trx.csv` dan menyiapkan sampel
analisis, lalu `../fitting/` mencocokkan sepuluh keluarga distribusi ke dua
variabel yang disampel simulasi.
