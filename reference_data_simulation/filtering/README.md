# `filtering/` — penyaringan data asli menjadi data Bandung

Data mentah di `../real_data/` mencakup seluruh Indonesia. Simulasi hanya
memakai wilayah Bandung. Folder ini berisi kode penyaringannya, data hasil
saringan, dan analisis data eksploratif atas hasil tersebut.

Penyaringan di sini **menghasilkan data yang identik** dengan yang dipakai
simulasi. Kesamaan itu diperiksa otomatis setiap kali skrip dijalankan, dan
hasil pemeriksaannya ditulis ke berkas laporan.

## Cara menjalankan

Ketiga langkah dijalankan berurutan karena langkah berikutnya memakai keluaran
langkah sebelumnya.

```bash
python reference_data_simulation/filtering/step1_filter_spklu.py      # ~10 detik
python reference_data_simulation/filtering/step2_filter_transaksi.py  # ~2 menit
python reference_data_simulation/filtering/step3_eda.py               # ~20 detik
```

## Langkah 1 — menyaring stasiun

`step1_filter_spklu.py` membaca master SPKLU nasional dan menyaring stasiun
wilayah Bandung.

**Aturannya:**

```
IS_ACTIVE_SPKLU = "Y"
DAN   -7,05 <= LATITUDE  <= -6,80
DAN  107,50 <= LONGITUDE <= 107,78
```

Hasilnya **77 stasiun** dari 2.431 stasiun nasional.

Penyaringan memakai kotak koordinat, bukan kolom nama kota. Alasannya: 41 dari
77 stasiun tersebut adalah SPKLU mitra non-PLN (Hyundai, Toyota, Wuling,
Voltron, Starvo) yang kolom kota dan kolom unit induknya kosong pada master.
Menyaring berdasarkan nama wilayah akan membuang seluruh stasiun mitra itu dan
hanya menyisakan 43 stasiun.

**Keluaran:** `output/bandung_spklu.json`, `output/filtering_spklu_report.json`.

### Temuan: dua kolom jumlah konektor memberi angka berbeda

Master menyediakan dua kolom yang keduanya bisa dibaca sebagai jumlah port:

| Kolom | Arti | Sifat |
|---|---|---|
| `JML_CONNECTOR_SPKLU` | total konektor satu stasiun | nilainya sama untuk semua baris chargebox milik stasiun itu |
| `JML_CONNECTOR_CHARGER` | konektor milik chargebox pada baris itu | berbeda antar baris |

Berkas turunan yang dipakai simulasi menyalin **`JML_CONNECTOR_SPKLU`** ke
setiap chargebox, lalu menjumlahkannya. Karena angka total stasiun tersalin
berulang, penjumlahan itu melebih-hitung port. Contoh paling jelas: SPKLU PLN
ULP Bandung Utara punya 4 chargebox; angka total stasiunnya 9, sehingga
penjumlahan menghasilkan 36 port padahal jumlah sebenarnya 8. Dari 77 stasiun,
**30 stasiun** terdampak. Rinciannya ada di `output/filtering_spklu_report.json`.

Skrip ini **mempertahankan perilaku tersebut** supaya keluarannya identik dengan
yang dipakai simulasi, dan sekaligus melaporkan besar penyimpangannya.

**Seberapa besar dampaknya ke dunia simulasi?** Dunia simulasi memakai 13 SPKLU
PLN dengan port paling sedikit. Kalau kolom yang dipakai diganti, susunan 13
stasiun itu berubah pada **tepat satu stasiun**:

| | Kolom yang dipakai (`JML_CONNECTOR_SPKLU`) | Kolom alternatif (`JML_CONNECTOR_CHARGER`) |
|---|---|---|
| Stasiun ke-13 | SPKLU PLN Geowisata Inn, 2 port | SPKLU PLN ULP Bandung Barat, 1 port |
| Total port | **14** | 13 |

Dua belas stasiun lainnya identik. Temuan ini dicatat sebagai keterbatasan dan
**tidak diperbaiki**, karena mengubahnya akan membuat seluruh data hasil
eksperimen yang sudah ada tidak lagi dapat direproduksi.

## Langkah 2 — menyaring transaksi

`step2_filter_transaksi.py` membaca 402.119 transaksi nasional secara mengalir
(berkas 48 MB tidak dimuat sekaligus ke memori) dan menyimpan transaksi yang
`ID_SPKLU`-nya termasuk 77 stasiun Bandung.

Hasilnya **22.514 transaksi**.

### Tambahan: tiga kolom waktu baru

Ekstraksi lama hanya menyimpan jam bulat karena membuang menit dan detik:

```python
hh = int(waktu.split(":")[0])     # "13:47:22" menjadi 13
```

Akibatnya 22.514 transaksi hanya menempati **24 nilai berbeda**. Untuk keperluan
pemodelan distribusi, skrip ini menambahkan tiga kolom tanpa mengubah kolom
lama:

| Kolom baru | Isi |
|---|---|
| `minute` | menit transaksi, 0 sampai 59 |
| `second` | detik transaksi, 0 sampai 59 |
| `hour_frac` | jam sebagai pecahan: `hour + minute/60 + second/3600` |

`hour_frac` punya **19.169 nilai berbeda**. Dengan resolusi itu, fungsi
distribusi kumulatif empiris menjadi mulus sehingga uji Kolmogorov-Smirnov,
Cramer-von Mises, dan Anderson-Darling sah dipakai. Pada jam bulat, kurva
kumulatif berbentuk tangga dengan 24 anak tangga besar sehingga jaraknya
terhadap kurva kontinu manapun otomatis besar; statistik ujinya menjadi bias dan
peringkat antar distribusi tidak lagi informatif.

Kolom `hour` yang lama tetap ada, sehingga kedua versi bisa dibandingkan
berdampingan. Keduanya memang dianalisis terpisah di `../fitting/`.

**Keluaran:** `output/bandung_trx.csv`, `output/bandung_trx_summary.json`,
`output/filtering_transaksi_report.json`.

### Hasil verifikasi

Skrip membandingkan keluarannya dengan berkas turunan yang benar-benar dipakai
simulasi (`simulation_data/_derived/`, hanya dibaca, tidak pernah ditulis):

| Yang diperiksa | Hasil |
|---|---|
| Daftar 77 stasiun beserta koordinat dan chargebox | identik |
| Jumlah baris transaksi | 22.514 lawan 22.514 |
| Seluruh 17 kolom bersama, sel demi sel | identik, nol perbedaan |
| Histogram jam dan histogram durasi yang dibaca sampler | identik |

## Langkah 3 — analisis data eksploratif

`step3_eda.py` menghasilkan 15 grafik dan satu laporan di `eda/`, mencakup
seluruh kolom termasuk yang tidak dipakai simulasi (hari, bulan, jenis konektor,
pengguna). Grafik khusus variabel yang dipakai simulasi ada terpisah di
`../visualization/`.

Angka pokok hasil penyaringan:

| Besaran | Nilai |
|---|---|
| Transaksi | 22.514 |
| Stasiun yang bertransaksi | 34 dari 77 aktif |
| Pengguna unik | 7.025 |
| Energi per sesi | rata-rata 23,43 kWh |
| Durasi sesi | median 45 menit, rata-rata 75,66 menit, maksimum 1.502 menit |
| Kemiringan durasi | +3,82 (miring kanan) |
| SoC saat datang | rata-rata 28,3 persen |
| Harga per kWh | median 2.466,8 rupiah |
| Jam puncak dan lembah | 13:00 dan 03:00 |

## Isi folder

```
filtering/
├── step1_filter_spklu.py          menyaring stasiun
├── step2_filter_transaksi.py      menyaring transaksi + kolom waktu baru
├── step3_eda.py                   analisis data eksploratif
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
