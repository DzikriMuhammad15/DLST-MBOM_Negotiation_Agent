# `fitting/` — pencocokan sepuluh keluarga distribusi

Tahap ini berjalan **setelah** `../preprocessing/`. Masukannya sampel analisis
berisi 22.479 pengamatan; keluarannya parameter terduga, peringkat kecocokan,
dan gambar pembanding untuk setiap keluarga.

```bash
python reference_data_simulation/fitting/fit_distributions.py   # sekitar 30 detik
python reference_data_simulation/fitting/plot_fits.py           # sekitar 40 detik
```

## Yang dicocokkan

Hanya **dua variabel**, yaitu dua variabel yang benar-benar disampel engine untuk
membentuk AppUserBooking:

| Kunci | Variabel | Satuan | n |
|---|---|---|---:|
| `durasi_mentah` | lama satu sesi pengisian | menit | 22.479 |
| `jam_mentah_menit` | waktu mulai sesi dalam sehari | jam, 0 sampai 24 | 22.479 |

## Sepuluh keluarga kandidat dan alasan pemilihannya

Seluruhnya bersupport tak-negatif, sesuai sifat kedua variabel yang tidak pernah
bernilai negatif.

| # | Distribusi | Nama scipy | Alasan diuji |
|---:|---|---|---|
| 1 | Eksponensial | `expon` | Laju kejadian konstan dan sifat tanpa-ingatan. Acuan dasar untuk waktu antar-kejadian. |
| 2 | Gamma | `gamma` | Fleksibel mengatur kemiringan. Eksponensial dan chi-kuadrat adalah bentuk khususnya. |
| 3 | Weibull | `weibull_min` | Baku dalam analisis keandalan. Laju kegagalan bisa menurun, tetap, atau menaik. |
| 4 | Lognormal | `lognorm` | Muncul ketika besaran terbentuk dari perkalian banyak faktor acak bebas. |
| 5 | Inverse Gaussian (Wald) | `invgauss` | Waktu yang dibutuhkan gerak Brown mencapai suatu ambang. |
| 6 | Log-Logistik (Fisk) | `fisk` | Mirip lognormal dengan ekor lebih tebal. Lazim pada analisis ketahanan hidup. |
| 7 | Rayleigh | `rayleigh` | Bentuk khusus Weibull berparameter bentuk dua. Menguji kecukupan satu parameter skala. |
| 8 | Chi-Kuadrat | `chi2` | Jumlah kuadrat peubah normal baku. |
| 9 | Pareto | `pareto` | Ekor sangat tebal, landasan aturan 80/20. Menguji seberapa lebar ekor kanan data. |
| 10 | Half-Normal | `halfnorm` | Nilai mutlak normal berpusat nol. Modusnya selalu tepat di nol. |

## Cara pendugaan

**Titik awal dikunci di nol.** Seluruh keluarga diduga dengan
`dist.fit(x, floc=0)`, sehingga parameter lokasi **tidak diduga** dan setiap
distribusi bermula tepat dari titik nol.

Alasannya dua. Pertama, kedua variabel memang tidak dapat bernilai negatif, jadi
model yang menaruh massa peluang di bawah nol tidak punya arti fisik dan
nilainya tidak dapat disampel simulasi. Kedua, lokasi bebas membuka celah
kemungkinan tak terbatas, yaitu kepadatan yang meledak di batas data sehingga
menghasilkan AIC sangat baik dengan bentuk yang sebenarnya buruk. Gambar
`plots/belajar/parameter_lokasi.png` memperlihatkan kasus nyatanya: dengan
lokasi bebas, Gaussian invers pada durasi menaruh lokasi di −8,71 menit dan
memberi peluang 0,417 persen bagi durasi bernilai negatif.

**Parameter yang dikunci tidak dihitung sebagai parameter.** Jumlah parameter
bebas pada AIC dan BIC dihitung sebagai `jumlah parameter bentuk + 1` untuk
skala. Lokasi tidak ikut dihukum. Tanpa koreksi ini, setiap keluarga akan
dihukum dua satuan AIC untuk parameter yang tidak pernah diduga.

**Pendugaannya kemungkinan maksimum, bukan penyetelan hiperparameter.** Untuk
setiap keluarga, scipy mencari parameter yang memaksimumkan fungsi kemungkinan.
Jadi angka yang tertulis pada tabel sudah merupakan yang terbaik yang dapat
dicapai keluarga itu, bukan satu tebakan di antara banyak tebakan. Yang
dibandingkan antar keluarga adalah puncak masing-masing.

## Metrik yang dilaporkan

| Metrik | Arti | Dipakai untuk |
|---|---|---|
| Log-kemungkinan | Nilai puncak fungsi kemungkinan | Bahan AIC dan BIC |
| AIC | `2k − 2·log L`, dengan `k` = parameter bebas | **Pemeringkatan** |
| BIC | `k·ln n − 2·log L` | Pemeriksaan silang peringkat |
| Jarak KS | Jarak tegak terjauh antara tangga data dan kurva model | Keterbacaan, ditulis pada setiap gambar |
| Cramer-von Mises | Selisih kuadrat sepanjang seluruh rentang | Menimbang badan sebaran |
| Anderson-Darling | Seperti di atas tetapi menimbang ekor lebih berat | Menilai ekor |
| Wasserstein | Jarak pemindahan massa, bersatuan sama dengan data | Besaran salah yang terbaca |
| Massa salah tempat | Jarak variasi total atas bin histogram | Bagian peluang yang salah tempat |

**Mengapa AIC yang dipakai memeringkat.** AIC berbasis fungsi kemungkinan,
sehingga membandingkan keluarga berparameter satu dan berparameter dua secara
setara, dan seluruh keluarga di sini diduga dengan cara yang sama persis. Jarak
KS tetap dilaporkan dan ditulis di setiap gambar karena paling mudah dibaca,
tetapi KS hanya melihat satu titik terjauh sehingga kurang cocok sebagai dasar
pemeringkatan tunggal.

**Catatan tentang nilai p uji Kolmogorov-Smirnov.** Nilai p tidak dilaporkan,
dan itu disengaja. Nilai p baku uji Kolmogorov-Smirnov hanya sah bila parameter
distribusinya ditentukan lebih dulu, bukan diduga dari data yang sama. Di sini
parameternya diduga dari data itu sendiri, sehingga nilai p bakunya terlalu
besar. Lagi pula pada n = 22.479 setiap penyimpangan sekecil apa pun akan
tertolak, sehingga uji hipotesisnya tidak informatif. Yang informatif adalah
**besar** penyimpangannya, dan itulah yang dilaporkan.

## Hasil: durasi sesi

Peringkat menurut AIC, dari terbaik:

| Peringkat | Distribusi | Parameter bebas | Log-kemungkinan | AIC | Jarak KS | Massa salah tempat |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Log-Logistik (Fisk)** | 2 | −118.624,2 | **237.252,5** | **0,0359** | 10,15 persen |
| 2 | Gamma | 2 | −119.850,0 | 239.704,0 | 0,1016 | 17,52 persen |
| 3 | Chi-Kuadrat | 2 | −119.850,0 | 239.704,0 | 0,1016 | 17,52 persen |
| 4 | Eksponensial | 1 | −119.908,9 | 239.819,7 | 0,0938 | 18,45 persen |
| 5 | Weibull | 2 | −119.908,1 | 239.820,3 | 0,0926 | 18,52 persen |
| 6 | Lognormal | 2 | −119.952,6 | 239.909,3 | 0,0810 | 14,50 persen |
| 7 | Half-Normal | 1 | −124.293,0 | 248.588,0 | 0,2606 | 28,40 persen |
| 8 | Inverse Gaussian (Wald) | 2 | −131.105,9 | 262.215,8 | 0,3317 | 35,17 persen |
| 9 | Rayleigh | 1 | −137.135,3 | 274.272,6 | 0,4190 | 45,53 persen |
| 10 | Pareto | 2 | −155.980,7 | 311.965,4 | 0,4979 | 76,38 persen |

Parameter terduga:

| Distribusi | Parameter | Lokasi |
|---|---|---|
| Log-Logistik (Fisk) | c = 1,74291; skala = 47,06317 | 0 (dikunci) |
| Gamma | a = 1,09633; skala = 69,56733 | 0 (dikunci) |
| Chi-Kuadrat | df = 2,19265; skala = 34,78370 | 0 (dikunci) |
| Eksponensial | skala = 76,26857 | 0 (dikunci) |
| Weibull | c = 0,99435; skala = 76,05769 | 0 (dikunci) |
| Lognormal | s = 1,10961; skala = 45,29804 | 0 (dikunci) |
| Half-Normal | skala = 121,93684 | 0 (dikunci) |
| Inverse Gaussian (Wald) | mu = 5,59200; skala = 13,63888 | 0 (dikunci) |
| Rayleigh | skala = 86,22236 | 0 (dikunci) |
| Pareto | b = 0,11935; skala = 0,01041 | 0 (dikunci) |

**Juara: Log-Logistik (Fisk).** Unggul jauh, yaitu 2.452 satuan AIC di atas
peringkat kedua. Jarak KS-nya 0,0359, kira-kira sepertiga dari peringkat kedua.
Kemenangannya masuk akal: Fisk memang keluarga yang menggabungkan badan sebaran
menyerupai lognormal dengan ekor kanan yang lebih tebal, dan itulah bentuk data
durasi yang median 45,5 menit tetapi maksimumnya 1.502 menit.

## Hasil: jam mulai sesi

| Peringkat | Distribusi | Parameter bebas | Log-kemungkinan | AIC | Jarak KS | Massa salah tempat |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Weibull** | 2 | −70.692,1 | **141.388,3** | **0,0425** | 11,24 persen |
| 2 | Rayleigh | 1 | −72.182,2 | 144.366,3 | 0,1193 | 21,12 persen |
| 3 | Gamma | 2 | −73.604,2 | 147.212,3 | 0,0905 | 20,12 persen |
| 4 | Chi-Kuadrat | 2 | −73.604,2 | 147.212,3 | 0,0905 | 20,12 persen |
| 5 | Log-Logistik (Fisk) | 2 | −74.023,2 | 148.050,5 | 0,1144 | 19,14 persen |
| 6 | Half-Normal | 1 | −77.168,6 | 154.339,1 | 0,2762 | 38,51 persen |
| 7 | Lognormal | 2 | −79.054,4 | 158.112,8 | 0,1577 | 31,73 persen |
| 8 | Eksponensial | 1 | −81.811,3 | 163.624,6 | 0,3079 | 48,76 persen |
| 9 | Inverse Gaussian (Wald) | 2 | −92.780,3 | 185.564,6 | 0,3797 | 54,80 persen |
| 10 | Pareto | 2 | −128.468,6 | 256.941,2 | 0,5492 | 89,09 persen |

Parameter terduga:

| Distribusi | Parameter | Lokasi |
|---|---|---|
| Weibull | c = 2,76424; skala = 15,57593 | 0 (dikunci) |
| Rayleigh | skala = 10,59681 | 0 (dikunci) |
| Gamma | a = 4,02187; skala = 3,48235 | 0 (dikunci) |
| Chi-Kuadrat | df = 8,04371; skala = 1,74118 | 0 (dikunci) |
| Log-Logistik (Fisk) | c = 3,57050; skala = 13,48732 | 0 (dikunci) |
| Half-Normal | skala = 14,98616 | 0 (dikunci) |
| Lognormal | s = 0,66223; skala = 12,30509 | 0 (dikunci) |
| Eksponensial | skala = 14,00556 | 0 (dikunci) |
| Inverse Gaussian (Wald) | mu = 1,69279; skala = 8,27365 | 0 (dikunci) |
| Pareto | b = 0,11025; skala = 0,00142 | 0 (dikunci) |

**Juara: Weibull**, unggul 2.978 satuan AIC dari peringkat kedua, dengan jarak KS
0,0425.

## Empat catatan kejujuran

Bagian ini mencatat keterbatasan yang tidak boleh disembunyikan tabel di atas.

**Pertama, Gamma dan Chi-Kuadrat menghasilkan angka yang persis sama.** Bukan
kebetulan dan bukan kesalahan. Chi-kuadrat dengan skala bebas dan lokasi nol
adalah penulisan ulang Gamma, dengan hubungan `a = df/2` dan
`skala_gamma = 2 × skala_chi2`. Periksa pada durasi: `df/2 = 2,19265/2 = 1,09633`
sama dengan `a`, dan `2 × 34,78370 = 69,5674` sama dengan skala Gamma.
Keduanya tetap dilaporkan karena keduanya diminta, tetapi angkanya tidak boleh
dibaca sebagai dua bukti terpisah.

**Kedua, Pareto tidak benar-benar bermula di nol.** Pada scipy, Pareto dengan
lokasi dikunci nol bersupport `x >= skala`, sehingga titik awal efektifnya
berada di parameter skala. Penduga kemungkinan maksimum menaruh skala tepat pada
nilai terkecil sampel, yaitu 0,01041 menit untuk durasi dan 0,00142 jam untuk
jam mulai. Praktisnya sangat dekat ke nol, tetapi secara formal titik awalnya
adalah parameter, bukan nol. Ini sifat keluarga Pareto itu sendiri dan tidak
dapat dihilangkan.

**Ketiga, jarak Wasserstein Pareto bernilai astronomis.** Pada kedua variabel,
parameter bentuk terduga berada di bawah satu, yaitu 0,119 dan 0,110. Distribusi
Pareto dengan bentuk di bawah satu **tidak punya nilai harapan berhingga**,
sehingga jarak Wasserstein-nya memang meledak. Angkanya dibiarkan apa adanya di
berkas hasil karena itu diagnostik yang jujur: Pareto tidak layak untuk kedua
variabel ini, dan angka itulah bukti kuantitatifnya.

**Keempat, tidak ada satu pun keluarga yang benar-benar cocok.** Bahkan juaranya
masih salah menempatkan 10,15 persen massa peluang pada durasi dan 11,24 persen
pada jam mulai. Untuk jam mulai penyebabnya jelas: variabel itu bersifat
melingkar dan berbatas atas di 24, memuncak sekitar pukul 13 dan melembah
sekitar pukul 03, sedangkan kesepuluh keluarga kandidat tidak berbatas atas dan
bermodus tunggal dekat nol atau di badan sebaran. Weibull menang sebagai yang
paling tidak buruk, bukan sebagai yang benar. Akibat langsungnya terlihat saat
sampling: 3,66 persen tarikan Weibull melampaui 24 jam dan harus dijepit.

## Gambar

Empat puluh gambar, satu panel per gambar, dua gambar per keluarga per variabel:

```
plots/
├── durasi_mentah/
│   ├── pdf__fisk.png ... pdf__pareto.png       10 gambar kepadatan peluang
│   └── cdf__fisk.png ... cdf__pareto.png       10 gambar peluang kumulatif
├── jam_mentah_menit/
│   ├── pdf__*.png                              10 gambar
│   └── cdf__*.png                              10 gambar
└── belajar/                                    gambar penjelas konsep
```

Setiap gambar berjudul `Probability Distribution Function` atau
`Cumulative Distribution Function`, berlegenda `data` dan
`hasil fit distribusi <nama>`, dan berketerangan tunggal berupa jarak KS.
Tidak ada peringkat maupun parameter yang ditulis di dalam gambar; keduanya ada
pada tabel dan pada `best_parameters.md`.

Sumbu mendatar gambar durasi dipotong pada kuantil 99,5 persen data agar badan
sebaran terbaca. Pemotongan itu hanya soal tampilan; seluruh 22.479 pengamatan
tetap ikut dalam pendugaan dan dalam seluruh metrik.

## Isi folder

```
fitting/
├── common.py                  target, sepuluh keluarga, pendugaan, metrik
├── fit_distributions.py       penggerak: pendugaan, pemeringkatan, laporan
├── plot_fits.py               gambar satu panel, PDF dan CDF terpisah
├── jelaskan_statistik_ks.py   gambar penjelas tangga empiris dan jarak KS
├── jelaskan_parameter_lokasi.py  gambar penjelas parameter lokasi
├── documentation.md           berkas ini
├── best_parameters.json       parameter terbaik tiap keluarga, terbaca mesin
├── best_parameters.md         hal yang sama dalam bentuk tabel
├── results/
│   ├── metrik__durasi_mentah.csv        seluruh metrik, satu baris per keluarga
│   ├── metrik__jam_mentah_menit.csv
│   ├── parameter__durasi_mentah.json    parameter + peringkat
│   ├── parameter__jam_mentah_menit.json
│   └── RINGKASAN.md                     tabel lengkap kedua variabel
└── plots/                     lihat bagian Gambar di atas
```

## Pemakaian hasil ini oleh simulasi

Pada project `clean_code`, sampler booking membaca `best_parameters.json` dan
memakai keluarga peringkat satu untuk menarik durasi dan jam mulai. Rinciannya
ada pada `../../simulation_variable_description.md` di akar project. Pada project
`nyoba_general`, hasil ini **tidak** dipakai simulasi; sampler di sana tetap
empiris, sesuai perannya sebagai arsip yang konsisten dengan korpus hasil yang
sudah ada.
