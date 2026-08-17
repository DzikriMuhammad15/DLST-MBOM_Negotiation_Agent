# `network_scaling/` — Eksperimen Scaling Arsitektur Jaringan Fleet

Menguji pengaruh ukuran jaringan (`hidden_dim` dan `hidden_layers`) sisi **fleet**
terhadap performa negosiasi, pada dunia real-case Bandung yang **sama persis
dengan `v_2/`**. Sisi SPKLU sengaja dibiarkan bawaan agar lingkungan dan lawan
tetap identik; satu-satunya variabel yang berubah adalah kapasitas jaringan fleet.

**Status**: korpus ini belum dipakai di laporan tesis, masih dalam pertimbangan.
Disiapkan di sini supaya eksperimennya bisa dilanjutkan.

## Dua keluarga skenario

Mengikuti `v_2/` tetapi **tanpa varian mahalkompen**, sebab yang diuji di sini
ukuran jaringan agen, bukan aturan kompensasi.

| Skenario | Permintaan booking | Okupansi yang muncul |
|---|---:|---:|
| `medium_occupancy` | 185 | sekitar 26 persen |
| `high_occupancy` | 366 | sekitar 47 persen |

Angka 185 dan 366 adalah rata-rata dan kuantil 95 jumlah transaksi harian
seluruh SPKLU Bandung 2024, dikali horizon tiga hari. Penjelasan lengkapnya di
`../v_2/README.md`.

## Langkah 1 — Generate config

Config sumbernya `v_2/configs/`, jadi generator v_2 harus dijalankan lebih dulu.

```bash
python v_2/gen_configs.py                                                     # sekali saja
python network_scaling/gen_configs.py --hidden 512 --layers 3
python network_scaling/gen_configs.py --hidden 512 --layers 3 --variant high_occupancy
```

Keluaran: `network_scaling/configs/run{n}_<skenario>_h{H}x{L}.json` untuk lima
ulangan pertama. Lima, bukan empat puluh seperti `v_2/`, karena satu kombinasi
ukuran jaringan sudah mahal dijalankan.

Tidak ada config yang disertakan di repo ini, sebab arsitektur mana yang dipakai
masih keputusan terbuka. Jalankan generator dengan pilihan Anda sendiri.

## Langkah 2 — Jalankan eksperimen

```bash
python network_scaling/main_nego_ns.py --agent-fleet DLST --hidden 512 --layers 3 --run 1 --lean
python network_scaling/main_nego_ns.py --agent-fleet DLSTMBOM --hidden 512 --layers 3 --run 1 \
    --variant high_occupancy --lean --online-log
```

Keluaran: `network_scaling/data/<skenario>_h{H}x{L}/negotiation/real_case_v2/run{n}_padded_<agen>/`,
terisolasi per kombinasi ukuran jaringan sehingga tidak bercampur dengan `v_2/`.

`--agent-fleet` mendukung `DLST`, `DLSTMBOM`, dan `DLSTMBOMNewMechanism`.
`DLSTMBOMDirected` bukan bagian dari eksperimen scaling ini.

## Sampler dan metrik

Sama persis dengan `v_2/`:

- Sampler booking bawaan menarik durasi dan jam mulai dari distribusi teoretis
  hasil pencocokan; `--empiris` mengembalikan histogram data seperti cara lama.
- Kebijakan penempatan booking sama, yaitu durasi tidak pernah diundi ulang.
- Setiap run menulis `metrices/booking_generation.json`,
  `metrices/distribusi_durasi.png`, `metrices/distribusi_jam.png`, serta
  `pergeseran_fallback` di dalam `aggregate_metrics.json`.

Rincian arti tiap metrik ada di `../v_2/README.md`.

## Opsi lanjutan

`--opp-hidden` dan `--opp-layers` menskalakan sisi SPKLU juga. Keduanya harus
dipakai bersamaan, dan config-nya harus digenerate dengan pilihan yang sama.
