# `v_2/` — Real-case Bandung (*fixed_mechanism*)

Dunia real-case Bandung terkalibrasi: 50 taksi, 13 SPKLU dengan 14 port, 18
marking area, horizon 3 hari, tarif 6.300 rupiah per km.

## Empat keluarga skenario

Sejak 2026-08-12, istilah okupansi 22,5 persen dan 50 persen **tidak dipakai
lagi**. Yang dikendalikan sekarang adalah **permintaan booking**, dan okupansi
menjadi keluaran yang diukur.

| Skenario | Permintaan booking | Harga lantai saat digeser | Okupansi yang muncul |
|---|---:|---|---:|
| `medium_occupancy` | 185 | dibiarkan | sekitar 26 persen |
| `high_occupancy` | 366 | dibiarkan | sekitar 47 persen |
| `mahalkompen_medium_occupancy` | 185 | dilarang | sekitar 26 persen |
| `mahalkompen_high_occupancy` | 366 | dilarang | sekitar 47 persen |

**Asal angka 185 dan 366.** Keduanya statistik dari sebaran jumlah transaksi per
hari seluruh SPKLU Bandung sepanjang 2024, yaitu 366 nilai harian dari 22.514
transaksi, lalu dikalikan horizon tiga hari:

- **185** = rata-rata harian 61,51 transaksi × 3 hari, yaitu beban rata-rata.
- **366** = kuantil 95 harian 122 transaksi × 3 hari, yaitu beban puncak.

Dasarnya di `../reference_data_simulation/`.

**Mengapa permintaan, bukan okupansi.** Dalam pemodelan sistem berkapasitas,
yang ditetapkan adalah proses permintaan sedangkan tingkat keterpakaian adalah
keluaran. Memaksa okupansi mencapai suatu angka berarti membengkokkan permintaan
demi mengejar keluaran. Karena itu kolom terakhir tabel di atas adalah perkiraan
hasil, bukan sasaran; angka sebenarnya dicatat per run.

**Catatan penting**: volume seluruh Bandung dimampatkan ke 14 port dunia
simulasi, yaitu sekitar 16,9 kali kepadatan nyata. Itu pilihan pemodelan yang
disengaja supaya ada rebutan slot untuk dinegosiasikan, dan harus disebut apa
adanya, bukan disajikan seolah diambil mentah dari data.

## Regenerasi config

Satu generator untuk keempat skenario, menghasilkan **160 config** yaitu 40 run
per skenario.

```bash
python v_2/gen_configs.py                              # keempat skenario
python v_2/gen_configs.py --skenario high_occupancy    # satu skenario saja
```

Seed run ke-n adalah `7 + 104729 * n`, **sama lintas skenario** sehingga keempat
skenario berpasangan pada dunia yang seed-nya identik. `initial_taxi_state`
diturunkan deterministik dari seed. Seluruh field dunia lain diambil dari
`configs/_template_dunia.json`.

## Dua sampler booking

| Cara | Argumen | Lama booking | Jam mulai booking |
|---|---|---|---|
| **Distribusi hasil fit** (bawaan) | tanpa argumen | Log-Logistik, dijepit 1.502,22 menit, dibulatkan ke ATAS ke kelipatan 30 menit | Weibull, dijepit 24 jam, dibulatkan ke kelipatan 30 menit TERDEKAT, maksimum grid 47 |
| **Penarikan ulang empiris** | `--empiris` | undian berbobot histogram, dibulatkan ke grid TERDEKAT, dipotong 4 grid | jam bulat berbobot histogram, paruh jam seragam |

`--empiris` **hanya** mengganti sumber distribusinya. Kebijakan penempatan sama
pada kedua mode, sehingga keduanya dapat dibandingkan secara adil.

## Kebijakan penempatan booking

Permintaan diundi **sekali**, lalu yang dicari hanyalah tempatnya. **Durasi tidak
pernah diundi ulang.**

1. Coba port dan jam persis seperti yang diundi.
2. Masih pada jam yang sama, coba port lain menurut urutan acak berbobot volume
   stasiun.
3. Geser jam, besaran kecil dulu, sampai 4 grid ke depan atau ke belakang. Pada
   tiap jam baru seluruh port dicoba lagi.
4. Gagal semua, permintaan itu **ditolak**. Tidak ada undian permintaan kedua.

Cara lama mengundi ulang port, durasi, dan jam sekaligus, sehingga booking
panjang yang gagal digantikan booking pendek dan menghilang diam-diam dari dunia
simulasi. Terukur pada N = 366: penolakan turun dari 20,5 menjadi 7,4 persen dan
penyimpangan sebaran durasi turun dari 12,35 menjadi 3,54 persen.

## Menjalankan eksperimen

```bash
# Agen negosiasi
python v_2/main_nego.py --agent-fleet DLST --run 1
python v_2/main_nego.py --agent-fleet DLSTMBOM --run 1 --skenario high_occupancy
python v_2/main_nego.py --agent-fleet DLSTMBOMNewMechanism --run 1 --skenario mahalkompen_medium_occupancy
python v_2/main_nego.py --agent-fleet DLSTMBOMDirected --run 1 --skenario mahalkompen_high_occupancy

# Greedy (metaheuristik)
python v_2/main_greedy.py --run 1 --skenario high_occupancy

# As-Is / Booking (non-negosiasi statis)
python v_2/main_nonneg_v2.py --mech booking --run 1 --skenario medium_occupancy
python v_2/main_nonneg_v2.py --mech as_is   --run 1 --skenario high_occupancy
```

`--run` menerima 1 sampai 40. Tambahkan `--lean` untuk mematikan trace dan log
detail. Tambahkan `--empiris` untuk memakai sampler lama.

Keluaran: `v_2/data_<skenario>/negotiation/real_case_v2/run{n}_padded_<agen>/`
dan `v_2/data_<skenario>/non_negotiation/real_case_v2/<mech>/run{n}/`.

## Metrik tambahan per run

Selain `metrices/aggregate_metrics.json` yang sudah ada, setiap run v_2 menulis:

**`metrices/booking_generation.json`** — nasib setiap permintaan booking saat
dunia dirakit:

| Kunci | Arti |
|---|---|
| `permintaan` | N yang diminta config |
| `terpasang` | yang berhasil masuk dunia |
| `tanpa_menggeser` | berhasil pada port dan jam persis seperti diundi |
| `menggeser_pindah_port` | jam sama, port lain |
| `menggeser_geser_jam` | jamnya digeser |
| `ditolak_kapasitas_penuh` | seluruh port penuh pada seluruh jam yang dicoba |
| `ditolak_di_luar_horizon` | durasinya tidak muat sebelum simulasi berakhir |
| `ditolak_tanpa_fallback` | tertempatkan tetapi tidak ada tujuan fallback yang muat |
| `okupansi_persen` | keterisian port yang dihasilkan |
| `massa_salah_tempat_*` | jarak sebaran terpasang terhadap sebaran hasil fit |
| `ambang_derau_*` | jarak yang muncul dari derau pencuplikan saja pada n yang sama |

Dua kunci terakhir harus dibaca berpasangan. Satu run hanya memuat beberapa
ratus booking sedangkan sebarannya punya puluhan kelas, sehingga sebagian jarak
yang terukur murni kebetulan. Massa salah tempat baru bermakna bila jelas
melampaui ambang derau.

**`metrices/distribusi_durasi.png`** dan **`metrices/distribusi_jam.png`** —
pembanding satu panel antara sebaran hasil fit dan sebaran yang benar-benar
terjadi pada run tersebut.

**`aggregate_metrics.json` → `pergeseran_fallback`** — berapa kali booking
benar-benar dipindahkan ke rencana fallback-nya karena slotnya diambil taksi
**saat negosiasi berjalan**. Ini berbeda dari pergeseran saat dunia dirakit yang
dicatat pada `booking_generation.json`. Ringkasan `booking_generation` juga
disisipkan ke berkas yang sama supaya seluruh angka satu run ada di satu tempat.

## Catatan

Field `negotiation.prune_floor_price_on_shift` sudah **dibakukan di dalam config**
skenario mahalkompen, bukan override runtime.

Baseline greedy pada project ini menghasilkan `deal_success_rate = 0,0`. Itu
**bukan** akibat perubahan di atas: korpus greedy yang tersimpan di
`nyoba_general` juga menunjukkan angka yang sama. Sifat baseline itu sendiri.
