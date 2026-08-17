# `plots/belajar/` — gambar penjelas konsep

Folder ini **bukan** hasil fit distribusi. Isinya gambar yang menerangkan konsep
di balik angka-angka pada laporan, supaya tabel metriknya dapat dibaca tanpa
harus percaya begitu saja.

Hasil fit yang sebenarnya ada di folder sebelah, satu subfolder per variabel:
`../durasi_mentah/` dan `../jam_mentah_menit/`.

## Urutan membaca yang disarankan

1. **`tangga_empiris.png`** — apa itu "tangga data".

   Tiga panel bertahap. Panel pertama menaruh lima durasi pada garis bilangan.
   Panel kedua menaikkan kurva sebesar 1/5 pada tiap pengamatan sehingga
   terbentuk tangga. Panel ketiga memperlihatkan kasus nyata dengan 22.479
   pengamatan, yang anak tangganya terlalu rapat untuk terlihat sehingga
   kurvanya tampak mulus. Istilah resminya fungsi distribusi kumulatif empiris.

2. **`statistik_ks.png`** — arti angka "jarak KS" yang tertulis pada setiap
   gambar hasil fit.

   Tangga data dibandingkan dengan kurva distribusi hasil fit. Jarak tegak
   terjauh antara keduanya adalah statistik Kolmogorov-Smirnov. Ada dua bentuk
   jarak, yaitu D+ dan D−, karena tangga punya dua ketinggian pada setiap anak
   tangga. Gambar ini menyertakan tabel perhitungan untuk contoh delapan
   pengamatan sehingga rumusnya dapat diperiksa dengan hitungan tangan.

3. **`parameter_lokasi.png`** — apa itu parameter lokasi dan mengapa dikunci nol.

   Panel kiri memperlihatkan bentuk yang sama digeser ke beberapa nilai lokasi.
   Panel kanan memperlihatkan akibatnya bila lokasi dibiarkan bebas diduga:
   pada durasi, Gaussian invers menaruh lokasi di −8,71 menit sehingga model itu
   memberi peluang 0,417 persen bagi durasi bernilai negatif. Durasi negatif
   tidak punya arti fisik dan tidak dapat disampel simulasi. Itulah sebabnya
   seluruh pendugaan pada `../../fit_distributions.py` mengunci lokasi di nol
   dan tidak menjadikannya parameter.

## Cara membuat ulang

```bash
python reference_data_simulation/fitting/jelaskan_statistik_ks.py
python reference_data_simulation/fitting/jelaskan_parameter_lokasi.py
```

Keduanya memakai parameter hasil fit yang sungguhan dan data yang sungguhan,
bukan angka karangan. `jelaskan_statistik_ks.py` membaca keluarga peringkat satu
langsung dari `../../results/parameter__durasi_mentah.json`, sehingga gambarnya
ikut berubah bila hasil pemeringkatan berubah.
