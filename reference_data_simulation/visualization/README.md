# `visualization/` — sebaran variabel data asli yang dipakai simulasi

Folder ini hanya memuat variabel yang **benar-benar masuk ke simulasi**.
Variabel lain yang tercatat pada data tetapi tidak dipakai (energi per sesi,
SoC, hari, bulan, jenis konektor, pengguna) ada di `../filtering/eda/`.

## Cara menjalankan

```bash
python reference_data_simulation/visualization/plot_variabel_simulasi.py
```

Perlu `../filtering/output/` sudah terisi lebih dulu.

## Variabel yang dipakai simulasi

### Disampel acak per booking

| # | Variabel | Sumber data | Cara sampler memakainya |
|---|---|---|---|
| 1 | Jam mulai | histogram jam dari 22.514 transaksi | jam ditarik berbobot dari histogram; hari ditarik seragam; paruh jam ditarik seragam |
| 2 | Durasi | histogram durasi bin 15 menit | bin ditarik berbobot; menit ditarik seragam di dalam bin; hasilnya dibulatkan ke kisi setengah jam lalu dipotong pada 2 jam |
| 3 | Stasiun lalu port | volume transaksi 2024 pada 13 stasiun dunia simulasi | stasiun ditarik berbobot dengan penghalusan Laplace; port ditarik seragam di dalam stasiun |

### Dipakai sebagai angka tetap, bukan disampel

| # | Variabel | Sumber data | Cara dipakai |
|---|---|---|---|
| 4 | Harga per kWh | 22.514 transaksi | hanya **median**-nya yang dipakai, sebagai tarif tetap semua port |
| 5 | Bobot waktu order | TomTom Traffic Index Bandung | 24 bobot per jam, masuk langsung ke konfigurasi |

**Bukan dari data** (keputusan peneliti, ada di dokumentasi desain): posisi awal
taksi seragam pada kotak kota, SoC awal taksi seragam 0,35 sampai 0,90, tarif
perjalanan 3.000 rupiah per km, dan jumlah booking yang ditetapkan lewat target
tingkat keterisian 22,5 persen.

## Grafik yang dihasilkan

| Berkas | Isi |
|---|---|
| `01_jam_mulai_bulat.png` | Jam mulai versi jam bulat, yaitu versi yang benar-benar dibaca sampler |
| `02_jam_mulai_menit.png` | Jam mulai dengan resolusi menit, bentuk harian yang sama tetapi jauh lebih halus |
| `03_durasi_mentah.png` | Durasi dalam menit, nilai mentah, dengan penanda ambang pemotongan |
| `04_durasi_pasca_engine.png` | Durasi setelah aturan engine, hanya empat nilai tersisa |
| `05_volume_stasiun.png` | Volume transaksi 2024 pada 13 stasiun dunia simulasi |
| `06_bobot_stasiun_laplace.png` | Peluang pemilihan stasiun setelah penghalusan Laplace |
| `07_harga_per_kwh.png` | Sebaran harga per kWh dengan penanda median yang dipakai simulasi |
| `08_bobot_waktu_order.png` | Bobot waktu kemunculan order penumpang per jam |

Angka-angka pokoknya juga ditulis ke `ringkasan_variabel_simulasi.json`.

## Dua hal yang paling terlihat dari grafik

**Pemotongan durasi mengubah bentuk sebaran secara mendasar.** Data mentah
miring kanan dengan ekor sangat panjang (maksimum 1.502 menit). Setelah aturan
engine, yang tersisa hanya empat nilai dengan proporsi berikut:

| Durasi setelah aturan engine | Jumlah transaksi | Proporsi |
|---|---:|---:|
| 0,5 jam | 11.141 | 49,5 persen |
| 1,0 jam | 5.001 | 22,2 persen |
| 1,5 jam | 2.092 | 9,3 persen |
| 2,0 jam | 4.280 | **19,0 persen** |

Perhatikan bentuknya: menurun, menurun, lalu **naik lagi di ujung**. Kenaikan
itu bukan sifat data melainkan tumpukan seluruh sesi 105 menit ke atas pada satu
nilai. Nilai harapan durasi setelah pemotongan adalah **0,98914 jam**, dan
angka inilah yang dipakai menghitung jumlah booking pada konfigurasi simulasi.

**Volume stasiun sangat timpang.** Tiga belas stasiun dunia simulasi hanya
mencatat 2.142 sesi sepanjang 2024, dan tiga di antaranya nyaris atau sama
sekali tidak bertransaksi (Icon Plus 0, Rancaekek 0, Teuan Cafe 1). Tanpa
penghalusan Laplace, ketiganya tidak akan pernah terpilih sama sekali.
