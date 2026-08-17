# `v_1/` — Korpus Non-*Real-Case*, Data Bab V.1

Folder ini berisi **config 13 dunia** sintetis OFAT (*one-factor-at-a-time*) yang
dipakai untuk menghasilkan korpus non-*real-case* di laporan.

Isinya sengaja dipangkas pada 2026-08-12 supaya **persis** memuat apa yang benar
benar muncul di laporan, tidak lebih. Dasarnya Tabel IV.18 pada Bab IV yang
mendaftar tiga belas skenario, dan Tabel V.1 pada Bab V yang berisi 27 baris,
yaitu tiga belas dunia dikali dua mekanisme non-negosiasi.

Tiga config lama **dihapus** karena tidak pernah dirujuk laporan, yaitu
`ex_entropy05_n10_upd1_s42`, `ms_ref_n10_upd1_s7`, dan `ms_ref_n10_upd1_s123`.
Tidak ada satu pun tabel, gambar, atau kalimat di Bab V yang membahas koefisien
entropi maupun ulangan seed.

## Tiga belas dunia dan padanannya di laporan

| Kelompok | Label di laporan | Berkas config |
|---|---|---|
| CONTROL | 10 hari (REF) | `ref_n10_upd1_s42` |
| DEFAULT | Default tanpa-belajar | `control_default_s42` |
| HZ | 1 hari | `hz_n1_upd1_s42` |
| HZ | 5 hari | `hz_n5_upd1_s42` |
| HZ | 30 hari | `hz_n30_upd1_s42` |
| UF | Tiap 2 episode | `uf_n10_upd2_s42` |
| UF | Tiap 5 episode | `uf_n10_upd5_s42` |
| SC | Kelangkaan tinggi (6:2) | `sc_scarce_n10_upd1_s42` |
| SC | Sumber daya longgar (2:6) | `sc_abundant_n10_upd1_s42` |
| SOC | SoC rendah | `soc_low_n10_upd1_s42` |
| SOC | SoC tinggi | `soc_high_n10_upd1_s42` |
| CC | Lawan bergiliran | `cc_alternating_n10_upd1_s42` |
| CC | Lawan beku | `cc_fixedopp_n10_upd1_s42` |

## 🔴 v_1 memakai DUNIA KORPUS LAMA

Kode simulasi sekarang menghasilkan dunia yang **bergeser** dari korpus
non-*real-case* lama. Penyebabnya sudah ditemukan, yaitu pemilih tujuan fallback
versi lama menghabiskan **tepat tiga undian rng lebih banyak**. Karena rangkaian
bilangan acak berurutan, tiga undian yang hilang menggeser seluruh undian
sesudahnya, sehingga **order penumpang yang terbangkit berbeda**.

`overrides.py` mengembalikannya, dan seluruh perintah di bawah memakai titik
masuk di folder ini yang memasangnya otomatis. **Tidak ada saklar untuk
mematikannya**, supaya tidak mungkin ada run v_1 yang tercampur dunia baru.

Buktinya, pada dunia korpus lama mekanisme pemesanan statis mereproduksi Tabel
V.1 laporan **persis pada seluruh lima kolom** untuk empat dunia.

| Dunia | Fleet | SPKLU | Order | Jam-port | Harga |
|---|---:|---:|---:|---:|---:|
| Default tanpa-belajar | 1,32 | 0,72 | 91 | 11,5 | 2.734 |
| 1 hari | 1,32 | 0,72 | 91 | 11,5 | 2.734 |
| 5 hari | 7,27 | 3,09 | 452 | 51,0 | 2.734 |
| 10 hari (REF) | 14,35 | 6,24 | 906 | 104,0 | 2.748 |

⚠️ Batasnya harus disebut terang-terangan. Dunia 30 hari mendekat tetapi belum
persis. Delapan dunia lain belum diverifikasi. Mekanisme tanpa pemesanan tidak
akan cocok, sebab perbaikan bug jarak 7 Juli membuatnya berperilaku seperti
pemesanan statis, dan itu tidak dikembalikan. Dunia pulih juga tidak menjamin
hasil negosiasi ikut pulih, sebab kode agen berubah dan versi lamanya hilang.

Tidak ada kode di luar `v_1/` yang disentuh, terverifikasi dengan `diff -r`.

## Soal mode PADDED

Seluruh korpus ini dijalankan pada mode **PADDED**. Mode ENCODER tidak
dijalankan dan tidak pernah disebut di laporan. Laporan memang tidak menyinggung
mode sama sekali, dan itu keputusan penulisan yang disengaja supaya tidak
memancing pertanyaan lanjutan tentang varian yang tidak diuji. Karena itu
**setiap perintah di bawah wajib memakai `--modes PADDED`**, dan jangan
menambahkan ENCODER walaupun skrip bulk-nya mendukung.

## Korpus #1, agen negosiasi

Dijalankan lewat `run_experiment_bulk_per_config.py` di root. Skrip itu mampu
menjalankan matriks penuh sembilan pasangan agen dikali dua mode, tetapi laporan
hanya memuat **tiga pasangan pada mode PADDED**, yaitu

| Sub-eksperimen | Pasangan | Nama di laporan |
|---|---|---|
| `exp06` | DLSTMBOM lawan RLNegotiator | DLST-MBOM |
| `exp08` | DLSTMBOMNewMechanism lawan RLNegotiator | DLST-MBOM Nego |
| `exp09` | DLST lawan RLNegotiator | DLST murni |

```bash
python run_experiment_bulk_per_config.py \
    --name hz_n1_upd1_s42 \
    --config v_1/configs/hz_n1_upd1_s42.json \
    --only 6 8 9 --modes PADDED \
    --out-root v_1/data/experiment_bulk
```

Ulangi untuk ketiga belas config, dengan `--name` mengikuti nama berkas tanpa
`.json`. Seluruhnya menjadi 13 dikali 3, yaitu **39 sub-eksperimen**, dan pada
korpus lama waktunya terukur sekitar 25 menit.

Perlu diketahui, Tabel IV.17 pada Bab IV mendaftar **enam** kombinasi agen,
sedangkan Bab V hanya melaporkan tiga yang tercantum di atas. Tiga kombinasi
sisanya, yaitu RLNegotiator lawan RLNegotiator, MBOM, dan MBOMNewMechanism,
tidak dijalankan atas keputusan peneliti. Ketidakcocokan antar-bab itu diketahui
dan menunggu keputusan terpisah.

## Korpus #2, greedy lawan greedy

Mengisi Tabel V.8.

```bash
python metaheuristic/run_greedy.py \
    --config v_1/configs/hz_n1_upd1_s42.json \
    --name hz_n1_upd1_s42 \
    --exp-root v_1/data/greedy
```

Ulangi untuk ketiga belas dunia. Tanpa `--exp-root` keluarannya jatuh ke
`metaheuristic/data/non_rc/<name>/` mengikuti konvensi asli.

## Korpus #3, tanpa pemesanan dan pemesanan statis

Mengisi Tabel V.1.

```bash
python v_1/main_nonneg_v1.py --mech as_is \
    --name hz_n1_upd1_s42 --config v_1/configs/hz_n1_upd1_s42.json --seed 42 \
    --out-root v_1/data/as_is

python v_1/main_nonneg_v1.py --mech booking \
    --name hz_n1_upd1_s42 --config v_1/configs/hz_n1_upd1_s42.json --seed 42 \
    --out-root v_1/data/booking
```

Seluruh sisa config memakai seed 42, sesuai akhiran nama berkasnya, sehingga
`--seed 42` berlaku untuk ketiga belas dunia. Dua config beda seed sudah dihapus
bersama pemangkasan di atas.

## Total run korpus non-*real-case*

| Korpus | Rincian | Run |
|---|---|---:|
| #1 negosiasi | 13 dunia dikali 3 pasangan, PADDED | 39 |
| #2 greedy lawan greedy | 13 dunia | 13 |
| #3 tanpa pemesanan | 13 dunia | 13 |
| #3 pemesanan statis | 13 dunia | 13 |
| **Total** | | **78** |
