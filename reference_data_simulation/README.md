# `reference_data_simulation/` — data referensi simulasi dan pemodelan distribusinya

Folder ini memuat **seluruh rantai dari data asli sampai model distribusi**, dan
sengaja dibuat swasembada: berkas mentahnya ikut disalin ke dalam, sehingga
setiap angka dapat ditelusuri dan diulang dari nol tanpa bergantung pada folder
lain di project.

Tujuannya menjawab satu pertanyaan: **distribusi teoretis apa yang paling
menggambarkan data asli untuk variabel yang disampel simulasi?**

> **Skrip di folder ini tidak pernah mengubah kode simulasi.** Seluruhnya hanya
> membaca data dan menulis ke dalam folder ini sendiri. Berkas turunan yang
> dipakai simulasi (`simulation_data/_derived/`) hanya dibaca untuk diverifikasi,
> tidak pernah ditulis.

## Isi folder

```
reference_data_simulation/
├── real_data/        salinan mentah seluruh data referensi, apa adanya
├── filtering/        penyaringan Bandung + data hasil + analisis eksploratif
├── visualization/    sebaran variabel yang dipakai simulasi
├── preprocessing/    penyiapan sampel analisis sebelum pencocokan
└── fitting/          pencocokan sepuluh keluarga distribusi + metrik + gambar
```

Tiap subfolder punya `documentation.md` sendiri yang menjelaskan isinya secara
rinci, termasuk alasan setiap keputusan.

## Cara menjalankan seluruh rantai

Urutannya penting: tiap tahap memakai keluaran tahap sebelumnya.

```bash
# 1. penyaringan (sekitar 3 menit, membaca berkas 48 MB)
python reference_data_simulation/filtering/step1_filter_spklu.py
python reference_data_simulation/filtering/step2_filter_transaksi.py
python reference_data_simulation/filtering/step3_eda.py

# 2. sebaran variabel simulasi (sekitar 20 detik)
python reference_data_simulation/visualization/plot_variabel_simulasi.py

# 3. penyiapan sampel analisis (sekitar 20 detik)
python reference_data_simulation/preprocessing/preprocess.py

# 4. pencocokan distribusi (sekitar 30 detik) dan gambarnya (sekitar 40 detik)
python reference_data_simulation/fitting/fit_distributions.py
python reference_data_simulation/fitting/plot_fits.py
```

Perlu `numpy`, `pandas`, `scipy`, `matplotlib`, dan `openpyxl`. Kelimanya sudah
ada pada lingkungan project.

## Variabel apa saja yang disampel simulasi

Peta lengkap seluruh variabel dunia simulasi, untuk ketiga keluarga skenario,
ada pada `simulation_variable_description.md` di **akar project**. Ringkas untuk
keperluan folder ini:

| # | Variabel | Sumber data | Dianalisis di sini |
|---|---|---|---|
| 1 | Lama booking | durasi 22.514 transaksi Bandung 2024 | ya, `durasi_mentah` |
| 2 | Jam mulai booking | waktu mulai 22.514 transaksi | ya, `jam_mentah_menit` |
| 3 | Stasiun lalu port booking | volume transaksi per stasiun | tidak, kategorikal, tidak dicocokkan ke keluarga kontinu |
| 4 | Harga per kWh | median 22.514 transaksi | tidak, dipakai sebagai satu angka tetap |
| 5 | Bobot waktu order | TomTom Traffic Index Bandung | tidak, tabel tetap 24 nilai |

Variabel dunia lainnya, yaitu posisi dan SoC awal taksi, tujuan fallback,
kompensasi, serta order penumpang, **tidak berakar pada data historis ini**.
Semuanya tarikan seragam dari asumsi atau hasil rumus, sehingga mencocokkan
bentuk sebarannya ke data nyata tidak relevan.

## Dua cara sampling yang ada

**Penarikan ulang empiris** adalah cara lama. Sampler menarik langsung dari
histogram data asli:

```python
edge  = rng.choices(self._dur_edges, weights=self._dur_weights, k=1)[0]
menit = edge + rng.uniform(0.0, 15.0)
```

Baris pertama memilih satu bin sebanding frekuensinya pada data nyata, baris
kedua menyebar seragam di dalam bin itu. Tidak ada parameter yang diduga dan
tidak ada asumsi bentuk. Kelebihannya, bentuk sebaran hasilnya selalu sama
dengan data asli. Kekurangannya, tidak dapat menghasilkan nilai di luar rentang
yang pernah teramati, dan tidak memberi parameter ringkas yang dapat dilaporkan
atau divariasikan dalam analisis kepekaan.

**Penarikan dari distribusi hasil fit** adalah cara yang dibangun di sini.
Sampler menarik dari keluarga distribusi yang parameternya diduga dari data.
Kelebihan dan kekurangannya kebalikan dari yang di atas.

> **Yang dipakai simulasi berbeda antar project.** Pada `clean_code`, sampler
> durasi dan jam mulai memakai **distribusi hasil fit** sebagai bawaan, dan cara
> empiris tetap tersedia lewat argumen `--empiris`. Pada `nyoba_general`, sampler
> tetap **empiris** tanpa perubahan, sesuai perannya sebagai arsip yang konsisten
> dengan korpus hasil yang sudah ada. Isi folder `reference_data_simulation/`
> sendiri identik di kedua project.

## Jejak verifikasi

Setiap kali skrip penyaringan dijalankan, hasilnya dibandingkan otomatis dengan
berkas turunan yang benar-benar dipakai simulasi. Status terakhir:

| Yang diperiksa | Hasil |
|---|---|
| Daftar 77 stasiun beserta koordinat dan chargebox | identik |
| Jumlah transaksi hasil saringan | 22.514 lawan 22.514 |
| Seluruh 17 kolom bersama, sel demi sel | identik, nol perbedaan |
| Histogram jam dan durasi yang dibaca sampler | identik |
| Nilai harapan durasi setelah pemotongan | 0,98914 jam, sama dengan dokumentasi desain |

Artinya rantai di folder ini benar-benar mereproduksi data yang menjadi dasar
seluruh eksperimen, bukan sekadar mendekatinya.

## Preprocessing, ringkas

Lima langkah membawa 22.514 transaksi menjadi **22.479** sampel analisis, yaitu
99,84 persen data masukan. Hanya satu langkah yang membuang baris.

| Langkah | Tindakan | Dibuang |
|---|---|---:|
| 1 | Audit integritas, nol kembar dan nol nilai hilang | 0 |
| 2 | Pemeriksaan domain, nol nilai negatif | 0 |
| 3 | Dequantisasi, mengembalikan resolusi yang hilang karena pembulatan | 0 |
| 4 | Penyaringan kesahihan fisik, daya terhitung melampaui daya terpasang | 35 |
| 5 | Pemeriksaan pencilan ekor kanan, diputuskan tidak membuang | 0 |

Dua langkah yang paling menentukan hasil:

**Dequantisasi** mengganti tiap nilai dengan satu tarikan seragam di dalam selang
pencatatannya, sebab durasi tercatat sebagai bilangan bulat menit tanpa kecuali.
Selain memulihkan kesahihan uji berbasis fungsi kumulatif, langkah ini
menyelesaikan masalah 283 sesi berdurasi tepat nol tanpa membuang satu baris pun.
Tanpa itu, enam dari sepuluh keluarga kandidat gugur bukan karena bentuknya tidak
cocok melainkan karena satu titik data.

**Pencilan ekor kanan tidak dibuang.** Pagar Tukey akan membuang 2.495 transaksi,
tetapi pagar itu mengandaikan sebaran setangkup sedangkan kemiringan datanya
+3,82. Pemeriksaan fisik menunjukkan sesi panjang tetap menyalurkan energi nyata
pada daya jauh di bawah daya terpasang, yaitu kendaraan yang tetap tertancap
setelah pengisian selesai. Perilaku itu nyata, bukan kesalahan pengukuran.

Rinciannya di `preprocessing/documentation.md`.

## Hasil fit distribusi, ringkas

Sepuluh keluarga bersupport tak-negatif dicocokkan ke dua variabel, dengan
**titik awal dikunci di nol** sehingga parameter lokasi tidak diduga.
Kesepuluhnya: Eksponensial, Gamma, Weibull, Lognormal, Inverse Gaussian, Log-
Logistik, Rayleigh, Chi-Kuadrat, Pareto, dan Half-Normal.

| Variabel | Juara | AIC | Jarak KS | Parameter |
|---|---|---:|---:|---|
| Durasi sesi | **Log-Logistik (Fisk)** | 237.252,5 | 0,0359 | c = 1,74291; skala = 47,06317 |
| Jam mulai sesi | **Weibull** | 141.388,3 | 0,0425 | c = 2,76424; skala = 15,57593 |

Keduanya unggul telak, yaitu 2.452 dan 2.978 satuan AIC di atas peringkat kedua.

**Tetapi tidak ada satu pun yang benar-benar cocok.** Juara durasi masih salah
menempatkan 10,15 persen massa peluang dan juara jam mulai 11,24 persen. Untuk
jam mulai penyebabnya jelas: variabel itu melingkar dan berbatas atas di 24,
memuncak sekitar pukul 13 dan melembah sekitar pukul 03, sedangkan kesepuluh
kandidat tidak berbatas atas dan bermodus tunggal. Weibull menang sebagai yang
paling tidak buruk, bukan sebagai yang benar.

Rinciannya, termasuk empat catatan kejujuran tentang kesamaan Gamma dengan
Chi-Kuadrat dan tentang Pareto yang tidak benar-benar bermula di nol, ada di
`fitting/documentation.md`.

## Temuan yang perlu diketahui

1. **Pencatatan durasi memangkas ke bawah ke menit bulat.** Buktinya cacah pada
   nol adalah 283 dan pada satu adalah 141, yaitu dua kali lipat. Pembulatan ke
   terdekat akan memberi rasio sekitar setengah, bukan dua.
2. **Kolom jam yang dibaca sampler lama kehilangan resolusi menit**, sehingga
   22.514 transaksi hanya menempati 24 nilai. Pengaruhnya ke simulasi kecil
   karena engine hanya mengenal kisi setengah jam, tetapi pengaruhnya ke
   pengujian kecocokan besar. Kolom `hour_frac` beresolusi menit dan detik
   dibuat ulang dari berkas mentah untuk keperluan itu.
3. **Master SPKLU punya dua kolom jumlah konektor yang memberi angka berbeda.**
   Berkas turunan memakai kolom total per stasiun, yang melebih-hitung port pada
   30 dari 77 stasiun. Dampaknya ke dunia simulasi terbatas pada satu stasiun
   dari 13. Dicatat sebagai keterbatasan, tidak diperbaiki, karena mengubahnya
   akan membuat data hasil eksperimen yang sudah ada tidak lagi dapat direproduksi.
4. **Kolom SoC saat datang bernilai nol pada sekitar 36 persen baris.** Angka nol
   sebanyak itu lebih mungkin berarti tidak tercatat daripada baterai benar-benar
   kosong. Variabel ini tidak dipakai simulasi maupun fitting.
5. **Hanya 34 dari 77 stasiun yang pernah bertransaksi** sepanjang 2024. Inilah
   sebabnya sampler stasiun memakai penghalusan Laplace.
