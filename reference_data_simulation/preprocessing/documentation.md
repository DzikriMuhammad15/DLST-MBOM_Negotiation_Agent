# `preprocessing/` — menyiapkan data sebelum pencocokan distribusi

Tahap ini berjalan **setelah** `../filtering/` dan **sebelum** `../fitting/`.
Masukannya 22.514 transaksi Bandung hasil penyaringan; keluarannya sampel
analisis yang siap dicocokkan ke keluarga distribusi teoretis.

```bash
python reference_data_simulation/preprocessing/preprocess.py   # sekitar 20 detik
```

Dua variabel yang disiapkan adalah dua variabel yang benar-benar disampel engine
untuk membentuk AppUserBooking:

| Keluaran | Isi | Satuan |
|---|---|---|
| `output/durasi_mentah.csv` | lama satu sesi pengisian | menit |
| `output/jam_mentah_menit.csv` | waktu mulai sesi dalam sehari | jam, 0 sampai 24 |

Berkas pendamping: `output/sampel_analisis.csv` (kedua variabel beserta kolom
asalnya dan kolom diagnostik), `output/preprocessing_report.json` (angka setiap
langkah), dan enam gambar diagnostik di `plots/`.

## Mengapa tahap ini perlu

Pencocokan distribusi pada `../fitting/` mengunci titik awal setiap keluarga di
nol dan tidak menduga parameter lokasi. Syarat itu membuat dua sifat data mentah
menjadi penghalang yang harus diselesaikan lebih dulu.

Pertama, **durasi tercatat sebagai bilangan bulat menit**, seratus persen tanpa
kecuali. Fungsi distribusi kumulatif empirisnya karena itu berbentuk tangga
dengan anak tangga selebar satu menit. Jarak Kolmogorov-Smirnov terhadap kurva
kontinu manapun otomatis membesar hanya karena bentuk tangga itu, sehingga
peringkat antar keluarga tidak lagi mencerminkan kualitas kecocokan.

Kedua, **283 transaksi berdurasi tepat nol**. Enam dari sepuluh keluarga
kandidat bernilai kepadatan nol atau tak hingga di titik nol, sehingga fungsi
kemungkinannya bernilai minus tak hingga. Tanpa penanganan, keenam keluarga itu
gugur bukan karena bentuknya tidak cocok, melainkan karena satu titik data.

## Lima langkah

Ringkasan pengaruh setiap langkah terhadap banyaknya baris:

| Langkah | Tindakan | Baris dibuang | Sisa |
|---|---|---:|---:|
| 1 | Audit integritas | 0 | 22.514 |
| 2 | Pemeriksaan domain | 0 | 22.514 |
| 3 | Dequantisasi | 0 | 22.514 |
| 4 | Penyaringan kesahihan fisik | 35 | 22.479 |
| 5 | Pemeriksaan pencilan | 0 | 22.479 |

Sampel analisis akhir memuat **22.479 transaksi**, yaitu 99,84 persen data
masukan.

---

### Langkah 1 — audit integritas

Memeriksa baris kembar dan nilai hilang. Tidak membuang apa pun.

| Yang diperiksa | Hasil |
|---|---:|
| Baris kembar seluruh kolom | 0 |
| Baris kembar pada kunci stasiun, chargebox, tanggal, jam, menit, detik, pengguna | 0 |
| Nilai hilang pada durasi | 0 |
| Nilai hilang pada jam mulai | 0 |

**Alasan langkah ini ada meskipun tidak membuang apa pun.** Baris kembar akan
menggandakan bobot satu pengamatan pada fungsi kemungkinan dan mengecilkan
jarak Kolmogorov-Smirnov secara semu. Hasil nol harus dilaporkan, sebab tanpa
laporan pembaca tidak bisa membedakan antara "sudah diperiksa dan bersih" dan
"tidak pernah diperiksa".

### Langkah 2 — pemeriksaan domain

Memeriksa apakah nilai berada di dalam daerah definisi yang disyaratkan
keluarga kandidat. Tidak membuang apa pun.

| Yang diperiksa | Hasil |
|---|---:|
| Durasi bernilai negatif | 0 |
| Durasi bernilai tepat nol | 283 |
| Durasi berupa bilangan bulat | 100,00 persen |
| Jam mulai di luar selang 0 sampai 24 | 0 |
| Jam mulai bernilai tepat nol | 0 |

**Alasan.** Kesepuluh keluarga kandidat bersupport tak-negatif dengan titik awal
dikunci di nol. Satu saja nilai negatif membuat kepadatannya nol di titik itu
dan fungsi kemungkinannya minus tak hingga, sehingga pendugaan gagal total. Dua
temuan penting muncul di sini, yaitu 283 durasi bernilai nol dan kenyataan bahwa
seluruh durasi berupa bilangan bulat. Keduanya ditangani pada langkah berikutnya.

### Langkah 3 — dequantisasi

Nilai tercatat diganti dengan satu nilai yang ditarik seragam di dalam selang
pencatatannya.

| Variabel | Selang pencatatan | Nilai pengganti |
|---|---|---|
| Durasi | satu menit | `k + U(0, 1)` menit |
| Jam mulai | satu detik | `h + U(0, 1)/3600` jam |

Penarikan memakai PCG64 dengan benih tetap 20260811 yang dicatat pada berkas
laporan, sehingga langkah ini dapat diulang persis.

**Alasan pertama: data ini tersensor selang, bukan bernilai titik.** Durasi yang
tercatat sebagai `k` sebenarnya berarti "durasi sebenarnya berada di dalam
selang `[k, k+1)`". Mencocokkan kepadatan kontinu langsung ke nilai bulat
memperlakukan seluruh selang itu seolah menumpuk pada satu titik. Akibatnya
parameter skala terduga bias dan statistik Kolmogorov-Smirnov, Cramer-von Mises,
serta Anderson-Darling menggelembung karena bentuk tangganya, bukan karena
modelnya salah. Menarik satu nilai seragam di dalam selang adalah cara baku
menangani data terbulatkan, dan tidak bias di bawah anggapan nilai sebenarnya
tersebar merata di dalam selang pencatatan. Argumen yang sama sudah dipakai pada
`../filtering/` langkah 2 ketika kolom `hour_frac` dibuat.

**Alasan kedua: masalah durasi nol selesai tanpa membuang data.** Setelah
dequantisasi, durasi terkecil menjadi 0,000865 menit dan **tidak ada lagi nilai
tepat nol**. Keenam keluarga yang tadinya gugur menjadi dapat diduga pada sampel
yang sama persis dengan empat keluarga lainnya, sehingga perbandingannya adil.
Tidak satu baris pun dibuang untuk mencapai itu.

**Alasan ketiga: selang `[k, k+1)` memang selang yang benar.** Pencatatan durasi
memangkas ke bawah, bukan membulatkan ke terdekat. Buktinya ada pada cacah dua
nilai terkecil:

| Nilai tercatat | Cacah |
|---:|---:|
| 0 menit | 283 |
| 1 menit | 141 |

Rasionya 2,007. Seandainya pencatatan membulatkan ke terdekat, bin di nol hanya
selebar setengah menit yaitu `[0 , 0,5)` sedangkan bin di satu selebar satu menit
penuh, sehingga cacah di nol semestinya kira-kira **setengah** cacah di satu.
Yang teramati justru dua kali lipat. Pola itu hanya konsisten dengan bin
berlebar sama, yaitu pemangkasan ke bawah, ditambah kelebihan sesi yang memang
sangat pendek.

Efek sampingnya, 3.345 nilai jam mulai yang tadinya kembar menjadi seluruhnya
unik, sehingga fungsi distribusi kumulatif empirisnya benar-benar mulus.

Gambar `01_durasi_pembulatan_menit.png` menunjukkan pola cacah yang mendasari
argumen ini, dan `02_durasi_dequantisasi.png` menunjukkan ujung kiri sebelum dan
sesudah dequantisasi.

### Langkah 4 — penyaringan kesahihan fisik

Satu-satunya langkah yang membuang baris. Aturannya:

```
daya terhitung = kwh / (durasi dalam jam)
buang bila     daya terhitung > 1,25 x daya terpasang stasiun
```

**Alasan aturannya.** Sebuah pengisi daya tidak dapat menyalurkan daya melebihi
daya terpasangnya. Transaksi yang melanggar berarti energinya terlalu besar
untuk durasi yang tercatat, sehingga **durasinya tidak dapat dipercaya**. Karena
durasi justru variabel yang hendak dimodelkan, baris seperti itu harus keluar.

**Alasan ambangnya 1,25 dan bukan angka lain.** Ambang itu diturunkan dari
datanya sendiri, bukan dipilih sembarang:

| Kuantil rasio daya terhitung terhadap daya terpasang | Nilai |
|---|---:|
| Kuantil 50 | 0,3600 |
| Kuantil 90 | 0,9285 |
| Kuantil 99 | 1,0045 |
| Maksimum | 29,4414 |

Kuantil 99 berada tepat di 1,0045, artinya **batas fisik itu dipatuhi hampir
sempurna oleh sembilan puluh sembilan persen data**. Ambang 1,25 memberi kelonggaran
dua puluh lima persen untuk toleransi meteran dan untuk sisa pembulatan durasi,
sehingga yang tersaring hanyalah pelanggaran yang benar-benar besar, bukan
transaksi yang kebetulan berada di dekat batas.

**Hasilnya.** 35 transaksi dibuang, yaitu 0,1555 persen. Sebanyak 31 di antara
35 itu berdurasi tercatat lima menit ke bawah dan 24 di antaranya berdurasi
tercatat nol, yang menguatkan tafsir bahwa ketidakkonsistenannya memang muncul
di tempat pembulatan menit paling menggigit. Gambar
`03_kesahihan_fisik.png` menunjukkan sebaran rasio beserta letak ambangnya.

### Langkah 5 — pemeriksaan pencilan ekor kanan, diputuskan tidak membuang

| Besaran | Nilai |
|---|---:|
| Kemiringan durasi | +3,8205 |
| Q1 dan Q3 | 27,72 dan 83,50 menit |
| Pagar Tukey (Q3 + 1,5 IQR) | 167,18 menit |
| Pagar ekstrem (Q3 + 3 IQR) | 250,85 menit |
| Di atas pagar Tukey | 2.495 transaksi |
| Di atas pagar ekstrem | 1.142 transaksi |

**Keputusan: tidak ada yang dibuang.** Tiga alasan.

Pertama, **pagar Tukey tidak sah dipakai di sini**. Pagar itu diturunkan dengan
anggapan sebaran kira-kira setangkup. Pada sebaran dengan kemiringan +3,82,
pagar itu menandai ekor kanan karena bentuk sebarannya, bukan karena datanya
salah. Membuang 2.495 transaksi berdasarkan aturan yang anggapannya sudah
dilanggar bukan pembersihan data, melainkan mengubah sebaran agar cocok dengan
alat ujinya.

Kedua, **pemeriksaan fisik menunjukkan sesi panjang itu sah**:

| Kelompok | Cacah | Median energi | Median daya terhitung | Median rasio terhadap daya terpasang |
|---|---:|---:|---:|---:|
| Di atas 2 jam | 3.743 | 18,82 kWh | 6,51 kW | 0,2902 |
| Di atas 4 jam | 1.261 | 21,34 kWh | 3,65 kW | 0,1575 |
| Di atas 8 jam | 275 | 18,17 kWh | 1,95 kW | 0,0896 |
| Di atas 12 jam | 45 | 16,50 kWh | 1,18 kW | 0,0539 |

Seluruhnya menyalurkan energi nyata dalam jumlah wajar pada daya jauh di bawah
daya terpasang. Pola itu adalah kendaraan yang tetap tertancap setelah pengisian
selesai, perilaku yang lazim dan sering terjadi pada pengisi daya publik, bukan
kesalahan pengukuran. Gambar `04_sesi_panjang_daya.png` memperlihatkannya.

Ketiga, **ekor itu justru yang hendak diuji**. Empat dari sepuluh keluarga
kandidat, yaitu Weibull, Log-Logistik, Lognormal, dan Pareto, ada di daftar
tepat karena kemampuannya menggambarkan ekor tebal. Membuangnya lebih dulu akan
membuat pengujian kesepuluh keluarga itu kehilangan maknanya.

## Sampel analisis akhir

| Besaran | Durasi (menit) | Jam mulai (jam) |
|---|---:|---:|
| Banyak pengamatan | 22.479 | 22.479 |
| Minimum | 0,000865 | 0,001415 |
| Median | 45,5113 | 14,1846 |
| Rerata | 76,2686 | 14,0056 |
| Simpangan baku | 95,1425 | 5,3320 |
| Maksimum | 1.502,2205 | 23,9763 |
| Kemiringan | +3,8205 | −0,2811 |

Nilai maksimum durasi 1.502,2205 menit dipakai oleh sampler simulasi pada
`clean_code` sebagai **batas jepitan** penarikan durasi. Angka itu dibaca
langsung dari `output/preprocessing_report.json`, tidak ditulis ulang di kode.

## Tambahan: laju permintaan harian

`laju_permintaan.py` berdiri terpisah dari pipeline lima langkah di atas. Ia
tidak menyiapkan data untuk pencocokan, melainkan menurunkan **banyaknya
permintaan booking** yang dipakai skenario simulasi pada project `clean_code`.

```bash
python reference_data_simulation/preprocessing/laju_permintaan.py   # sekitar 5 detik
```

Dasarnya sebaran **jumlah transaksi per hari** seluruh SPKLU Bandung sepanjang
2024, yaitu 366 nilai harian dari 22.514 transaksi:

| Statistik harian | Transaksi per hari | Dikali 3 hari | Dipakai sebagai |
|---|---:|---:|---|
| Rata-rata | 61,51 | **185** | `medium_occupancy` |
| Kuantil 95 | 122 | **366** | `high_occupancy` |

Keduanya statistik dari sebaran yang sama, sehingga penjelasannya konsisten: yang
satu beban rata-rata, yang satu beban puncak.

**Dua hal yang harus disebut terang-terangan.**

Pertama, lajunya **tumbuh lebih dari empat kali lipat** sepanjang 2024, dari 26,9
transaksi per hari pada Januari menjadi 116,3 pada Desember, sedangkan jumlah
stasiun aktif hampir tidak berubah. Rata-rata setahun karena itu adalah campuran
antara awal tahun yang sepi dan akhir tahun yang ramai.

Kedua, dunia simulasi hanya punya **14 port** sedangkan Bandung nyata melayani
volume itu dengan **236 port**. Menaruh volume seluruh Bandung ke 14 port berarti
memampatkan kepadatannya **16,857 kali**. Pemampatan itu disengaja, sebab tanpa
kepadatan tidak ada rebutan slot yang dapat dinegosiasikan. Okupansi nyata
seluruh Bandung sepanjang 2024 hanya 1,37 persen. Jangan menyajikan angka 185 dan
366 seolah diambil mentah dari data tanpa menyebut pemampatan ini.

**Keluaran:** `output/laju_permintaan.json` dan `plots/07_laju_harian.png`.

## Isi folder

```
preprocessing/
├── preprocess.py                   pipeline lima langkah
├── laju_permintaan.py              penurunan N skenario dari laju harian
├── documentation.md                berkas ini
├── output/
│   ├── durasi_mentah.csv                 22.479 nilai, menit
│   ├── jam_mentah_menit.csv              22.479 nilai, jam
│   ├── sampel_analisis.csv               kedua variabel + kolom asal + diagnostik
│   ├── preprocessing_report.json         angka setiap langkah
│   └── laju_permintaan.json              laju harian + N tiap skenario
└── plots/
    ├── 01_durasi_pembulatan_menit.png    bukti pencatatan bilangan bulat
    ├── 02_durasi_dequantisasi.png        ujung kiri sebelum dan sesudah
    ├── 03_kesahihan_fisik.png            sebaran rasio daya dan ambangnya
    ├── 04_sesi_panjang_daya.png          daya terhitung pada sesi panjang
    ├── 05_durasi_akhir.png               durasi pada sampel akhir
    ├── 06_jam_akhir.png                  jam mulai pada sampel akhir
    └── 07_laju_harian.png                laju transaksi harian sepanjang 2024
```
