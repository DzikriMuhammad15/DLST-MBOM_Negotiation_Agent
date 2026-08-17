# `_antrean/` — menjalankan korpus dengan stop dan continue yang aman

Kampanye penuh berisi **1.198 run**, yaitu 78 untuk v_1 dan 1.120 untuk v_2.
Waktunya sekitar 35 jam, jadi terputus di tengah adalah hal yang normal dan
sudah diperhitungkan.

## Perintah

```bash
bash _antrean/antrean_semua.sh     # v_1 lalu v_2, aman diulang berapa kali pun
bash _antrean/progres.sh           # laporan kemajuan, aman saat antrean jalan
bash _antrean/hentikan.sh          # berhenti dengan aman
```

Melanjutkan setelah berhenti cukup menjalankan `antrean_semua.sh` lagi. Tidak
ada argumen yang perlu diingat, dan tidak ada berkas status yang perlu disunting.

Bisa juga dijalankan terpisah lewat `antrean_v1.sh` atau `antrean_v2.sh`.
Sebagian antrean v_2 dapat dibatasi dengan peubah lingkungan, misalnya
`RUN_LO=1 RUN_HI=5 bash _antrean/antrean_v2.sh` untuk mencoba lima ulangan dulu.

## Cara kerja jaminan keamanannya

**Penanda selesai** satu run adalah `metrices/aggregate_metrics.json`. Berkas itu
ditulis paling akhir oleh keempat jenis run, yaitu negosiasi, greedy, tanpa
pemesanan, dan pemesanan statis. Kalau ada, run dilewati.

**Tidak ada data parsial yang masuk korpus.** Sebelum satu run dijalankan,
foldernya dihapus lebih dulu. Jadi sisa run yang terpotong karena laptop mati
atau karena dihentikan akan tertimpa bersih pada saat continue, bukan bercampur.

**Tidak ada yang terlewat.** Antreannya deterministik dan menelusuri seluruh sel
setiap kali dijalankan. Yang sudah selesai dilewati, yang belum dikerjakan.
`progres.sh` menghitung langsung dari penanda di disk, bukan dari catatan
terpisah yang bisa basi.

**Urutan v_2 diselang per nomor run**, bukan diselesaikan per skenario. Berhenti
kapan pun meninggalkan jumlah ulangan yang seimbang di seluruh sel, sehingga
korpus separuh jalan tetap bisa dianalisis.

**Urutan v_1 per dunia sampai tuntas**, sebab satu dunia adalah satu baris tabel
di Bab V. Berhenti di tengah meninggalkan baris yang utuh, bukan setengah jadi.

**Penjaga disk.** Antrean berhenti sendiri dengan kode keluar 2 bila ruang kosong
turun di bawah `MIN_FREE_GB`, bawaannya 20 GB. Ambangnya bisa diubah, misalnya
`MIN_FREE_GB=30 bash _antrean/antrean_semua.sh`. Ini penting, sebab riwayat
proyek ini pernah kehilangan run 31 sampai 40 akibat pembersihan disk otomatis
ketika kapasitas menyentuh 94 persen.

**Kegagalan tidak menghentikan antrean.** Run yang gagal dicatat dan dilompati,
lalu dicoba lagi pada continue berikutnya sebab penandanya tidak pernah terbentuk.
Galat lengkapnya masuk ke `logs/v1.err` dan `logs/v2.err`.

## Cakupan yang dijalankan

### v_1, non-*real-case*, 78 run

Tiga belas dunia OFAT, seluruhnya berseed 42. Per dunia ada enam sel, yaitu tiga
pasangan agen negosiasi pada mode PADDED (`exp06` DLST-MBOM, `exp08` DLST-MBOM
Nego, `exp09` DLST murni), satu greedy lawan greedy, satu tanpa pemesanan, dan
satu pemesanan statis.

🔴 **v_1 memakai DUNIA KORPUS LAMA, tanpa saklar.** Seluruh run v_1 melewati
titik masuk di `v_1/`, yaitu `main_bulk_v1.py`, `main_greedy_v1.py`, dan
`main_nonneg_v1.py`, yang memasang tambalan tiga undian lebih dulu. Tidak ada
cara mematikannya, dan itu disengaja supaya tidak mungkin ada run v_1 yang
tercampur dunia baru.

Alasannya, kode sekarang menghasilkan dunia yang bergeser dari korpus lama.
Perbedaannya sudah ditemukan, yaitu pemilih tujuan fallback versi lama
menghabiskan tepat tiga undian rng lebih banyak. Dengan tambalan itu, mekanisme
pemesanan statis mereproduksi Tabel V.1 laporan **persis pada seluruh lima
kolom** untuk empat dunia. Rinciannya di `v_1/overrides.py`.

⚠️ Dunia 30 hari mendekat tetapi belum persis, dan delapan dunia lain belum
diverifikasi. Jangan menjanjikan seluruh korpus akan sama.

⚠️ Tidak ada kode di luar `v_1/` yang disentuh, dan ini sudah diverifikasi
dengan `diff -r` terhadap `nyoba_general`, yaitu `simulation/` nol berkas
berbeda.

⚠️ **Mode ENCODER tidak dijalankan.** Laporan sengaja tidak menyebut mode sama
sekali. Jangan menambahkannya walaupun skrip bulk-nya mendukung.

### v_2, *real-case*, 1.120 run

Empat keluarga skenario dikali 40 ulangan. Per skenario ada tujuh sel, yaitu
empat agen negosiasi, satu greedy, dan dua mekanisme non-negosiasi.

Log negosiasi memakai `--lean --online-log-full` mengikuti korpus lama, sehingga
kurva loss dan jejak aktor tetap terekam untuk analisis kolaps di Bab V. Greedy
dan non-negosiasi memakai `--lean` saja karena tidak punya pelatihan.

Sampler yang dipakai adalah distribusi hasil fit, yaitu bawaannya. `--empiris`
sengaja **tidak** dipakai.

## Kenapa v_1 diperlakukan berbeda

`run_experiment_bulk_per_config.py` **tidak bisa melanjutkan**. Bila folder
tujuannya sudah ada, ia membuat folder baru bersufiks angka, dan satu bulk berisi
banyak sub-eksperimen tanpa titik simpan di tengah.

Karena itu di sini ia dipanggil satu sub-eksperimen sekali lewat `--only` dan
`--modes PADDED` ke folder singgah `v_1/data/_singgah`, lalu hasilnya dipindahkan
ke `v_1/data/experiment_bulk/<dunia>/`. Titik simpannya menjadi per
sub-eksperimen. `bulk_results.csv` tiap panggilan ikut disalin sebagai
`bulk_results_exp<NN>.csv` supaya tidak ada keluaran yang hilang.

## Menghentikan, dan pelajaran yang mahal

`hentikan.sh` mematikan bash lebih dulu baru python. Kalau urutannya terbalik,
skrip antreannya masih hidup dan langsung menjalankan run berikutnya.

Polanya memakai awalan umum `_antrean`, bukan nama skrip tertentu. Orkestrator
berlapis menghasilkan beberapa proses bash dengan CommandLine berbeda antara
induk dan anak, dan mematikan yang cocok nama induk saja tidak cukup. Anaknya
tetap hidup dan terus memunculkan python baru. Skripnya menyapu sampai tiga
putaran lalu memverifikasi, dan hanya berkata bersih bila kedua cacahnya nol.

## ⚠️ Menguji skrip ini tanpa merusak korpus

`PY` bisa ditimpa lewat peubah lingkungan untuk menguji mekanika antrean dengan
interpreter tiruan. **Jangan pernah melakukannya terhadap folder data yang
sebenarnya.** Penanda tiruan akan tercampur ke korpus dan antrean akan menganggap
run itu sudah selesai. Pernah terjadi, 52 penanda palsu masuk ke `v_1/data` dan
harus dibersihkan satu per satu.

Sejak itu ada pengaman. `antrean_v1.sh` membaca akar datanya dari `V1_DATA`,
jadi uji kering **wajib** dialihkan ke folder lain:

```bash
PY=/path/ke/stub V1_DATA=v_1/data_uji_kering bash _antrean/antrean_v1.sh
```

Dengan begitu korpus sungguhan di `v_1/data` tidak mungkin tersentuh. Kalau
terlanjur, deteksinya
`grep -rl '"stub"' v_1/data --include=aggregate_metrics.json` lalu hapus
foldernya.

## Pemindahan hasil v_1 dan kegagalan yang pernah terjadi

Pada Windows, `mv` sebuah folder gagal dengan `Permission denied` bila ada satu
saja berkas di dalamnya yang masih dipegang proses lain, misalnya pemindai
antivirus yang baru menyentuh berkas log ratusan MB. Ini pernah terjadi pada
`hz_n30` `exp06` dan **menghapus hasil run yang sudah jadi**, yaitu kehilangan
hampir empat jam komputasi.

Sekarang pemindahannya ditangani `selamatkan_singgah`, yang mencoba `mv` sampai
lima kali dengan jeda sepuluh detik, lalu jatuh ke salin-dan-hapus. Kalau tetap
gagal, folder singgah **tidak dihapus**, dan percobaan berikutnya akan
menyelamatkan isinya lebih dulu tanpa menjalankan ulang simulasinya.

## Keluaran

```
v_1/data/experiment_bulk/<dunia>/exp06_padded_.../
v_1/data/greedy/<dunia>/
v_1/data/as_is/<dunia>/
v_1/data/booking/<dunia>/
v_2/data_<skenario>/negotiation/real_case_v2/run{n}_padded_<agen>/
v_2/data_<skenario>/non_negotiation/real_case_v2/<mech>/run{n}/
```

Metrik tambahan v_2 yang belum dipakai Bab V, yaitu `booking_generation.json`
dan `pergeseran_fallback`, tetap ditulis apa adanya di tiap run. Pembahasannya
menunggu penyusunan Bab V.
