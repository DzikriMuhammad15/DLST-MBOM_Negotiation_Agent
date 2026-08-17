# clean_code — Kode Independen untuk Regenerasi Korpus #1, #2, #3, #7, #8, #9, #10, #12

Proyek ini adalah **salinan bersih dan independen** dari sebagian kode di project
`nyoba_general` (thesis "Model-Based Opponent Modelling dalam Adaptive Strategy
Templates Menggunakan Deep Learnable..."), berisi HANYA kode + config yang
diperlukan untuk meregenerasi 8 korpus data yang benar-benar dipakai di laporan
tesis (dan mungkin dipakai selanjutnya):

| Korpus | Deskripsi | Folder di sini |
|---|---|---|
| #1 | Non-*real-case* (agen negosiasi), 13 dunia OFAT | `v_1/` (config) + `run_experiment_bulk_per_config.py` (root) |
| #2 | Non-*real-case* (Greedy) | `v_1/` (config) + `metaheuristic/run_greedy.py` |
| #3 | Non-*real-case* (As-Is/Booking) | `v_1/` (config) + `non_negotiation_mechanism/*/main_*.py` |
| #7 | Real-case *fixed_mechanism* — `medium_occupancy` | `v_2/` |
| #8 | Real-case *fixed_mechanism* — `high_occupancy` | `v_2/` (`--skenario high_occupancy`) |
| #9 | Real-case *fixed_mechanism* — `mahalkompen_medium_occupancy` | `v_2/` (`--skenario ...`) |
| #10 | Real-case *fixed_mechanism* — `mahalkompen_high_occupancy` | `v_2/` (`--skenario ...`) |
| #12 | *Network scaling* (arsitektur jaringan fleet) | `network_scaling/` |

**Ini BUKAN hasil refactor/tulis-ulang logic** — seluruh kode inti (`negotiation/`,
`simulation/`) dan skrip eksperimen disalin **verbatim** dari `nyoba_general`.
Satu-satunya perubahan adalah:
1. Penataan ulang folder (lihat "Pemetaan Nama" di bawah) supaya lebih rapi/jelas.
2. Penyesuaian path impor yang mengikuti penataan ulang itu (baris `sys.path`/
   `os.path.join(ROOT, "...")` yang menunjuk nama folder lama).
3. Pemangkasan cakupan: varian horizon 30 hari (korpus non-prioritas, di luar 8
   korpus di atas) tidak disertakan.

## ⚠️ Project ini SUDAH BERBEDA dari `nyoba_general`

Pada awalnya project ini dimaksudkan menghasilkan angka yang identik dengan
`nyoba_general`. **Kesetaraan itu sudah tidak berlaku lagi** dan tidak lagi
menjadi sasaran. Dua perubahan yang menyebabkannya, keduanya sengaja dan hanya
diterapkan di sini:

1. **Sampler booking berganti.** Lama dan jam mulai AppUserBooking secara bawaan
   kini ditarik dari distribusi teoretis hasil pencocokan, bukan dari histogram
   data. Cara lama tetap tersedia lewat argumen `--empiris`. Lihat
   `simulation_variable_description.md`.
2. **Bug pergeseran booking ke masa lalu diperbaiki.** Dulu sebuah booking dapat
   digeser ke slot yang secara kronologis sudah lewat, sebab 45,7 persen rencana
   fallback dibangkitkan menghadap ke belakang. `world_assembly.py` kini hanya
   membangkitkan tujuan fallback pada atau sesudah span asli booking berakhir,
   dan `action_library.py` menolak pergeseran ke grid yang sudah lewat sebagai
   lapis kedua. Perbaikan ini mengubah jalannya negosiasi dan menambah jumlah
   pergeseran yang benar-benar dapat dipakai. **Perbaikan yang sama juga
   diterapkan di `nyoba_general` atas permintaan peneliti.**

Karena itu, **jangan memakai project ini untuk membuktikan reproduksi korpus
tesis**. Yang masih dijamin adalah logika simulasi dan negosiasinya sendiri tidak
diubah selain dua hal di atas.

## ⚠️ Independensi

Project ini **TIDAK mengimpor apa pun dari `nyoba_general`**. Semua kode dan
config yang dibutuhkan sudah disalin ke sini. Anda bisa memindahkan folder
`clean_code/` ke komputer lain (atau menghapus `nyoba_general` sepenuhnya) dan
project ini akan tetap berjalan, asalkan environment Python-nya terpasang
(lihat "Environment" di bawah).

## Struktur Direktori

```
clean_code/
├── negotiation/            # kode inti: kelas agen (DLST, DLST-MBOM, DLST-MBOM Nego,
│                            #   DLSTMBOMDirected, RLNegotiator), reward_shaper, trainers
├── simulation/              # kode inti: world assembly, engine/scheduler, action_library,
│                            #   config, metrics, run_pipeline
├── dashboard/                # HANYA config_io.py (deep-merge + validasi config JSON,
│                            #   dipakai main.py). Bukan server web (app.py/sim_session.py/
│                            #   static/ SENGAJA TIDAK disalin, murni backend FastAPI).
├── metaheuristic/            # agen Greedy (dipakai v_1/ non-RC & v_2/ real-case)
├── non_negotiation_mechanism/ # mekanisme As-Is & Booking (dipakai v_1/ & v_2/)
├── real_case_common/          # kode BERSAMA khusus dunia real-case Bandung (dipakai
│                            #   v_2/ & network_scaling/): booking empiris, sampler,
│                            #   generator config dunia dasar
├── main.py, run_experiment.py, run_experiment_bulk_per_config.py  # pipeline dasar (root)
├── simulation_data/          # subset data mentah Bandung (SPKLU, transaksi, time-weight)
├── v_1/                      # config 13 dunia OFAT non-real-case (korpus #1/#2/#3)
├── v_2/                      # entry point + config real-case fixed_mechanism (korpus #7-#10)
├── network_scaling/           # entry point + generator config network scaling (korpus #12)
├── reference_data_simulation/ # data referensi asli + penyaringan + EDA + pemodelan
│                            #   distribusi variabel yang disampel simulasi.
│                            #   TIDAK dipakai saat menjalankan simulasi — folder ini
│                            #   analisis data, bukan bagian pipeline eksperimen.
├── pyproject.toml / requirements.txt / uv.lock / .python-version   # environment
└── README.md (file ini)
```

### `reference_data_simulation/` sekilas

Folder swasembada berisi seluruh rantai dari data mentah sampai model
distribusi. Berkas mentahnya (`DATA TRX SPKLU 2024.xlsx` 48 MB, master SPKLU,
bukti HTML TomTom) ikut disalin ke dalam supaya setiap angka dapat ditelusuri
dan diulang tanpa bergantung folder lain.

| Subfolder | Isi |
|---|---|
| `real_data/` | salinan mentah seluruh data referensi, apa adanya |
| `filtering/` | kode penyaringan Bandung, data hasil, dan 15 grafik analisis eksploratif |
| `visualization/` | 8 grafik sebaran variabel yang dipakai simulasi |
| `preprocessing/` | lima langkah penyiapan sampel analisis, 22.514 menjadi 22.479 baris |
| `fitting/` | pencocokan 10 keluarga distribusi bersupport tak-negatif ke 2 variabel, tabel metrik, dan 40 gambar perbandingan |

Rantai penyaringannya **mereproduksi persis** data yang mendasari seluruh
eksperimen: 77 stasiun identik, 22.514 transaksi identik sel demi sel, dan
histogram yang dibaca sampler identik. Kesamaan itu diperiksa otomatis setiap
kali skripnya dijalankan.

Hasil pencocokannya, dengan titik awal setiap keluarga dikunci di nol:

| Variabel | Distribusi juara | AIC | Jarak KS |
|---|---|---:|---:|
| Lama sesi pengisian | Log-Logistik (Fisk) | 237.252,5 | 0,0359 |
| Jam mulai sesi | Weibull | 141.388,3 | 0,0425 |

> **Hasil ini dipakai simulasi di project ini.** Lama dan jam mulai booking
> secara bawaan ditarik dari kedua distribusi di atas. Penarikan ulang empiris
> yang lama tetap tersedia lewat argumen `--empiris` pada keempat entry point
> real-case. Peta lengkap asal-usul seluruh variabel dunia simulasi, untuk
> ketiga keluarga skenario, ada di `simulation_variable_description.md`.

## Pemetaan Nama (folder lama di `nyoba_general` -> folder baru di sini)

| Nama di `nyoba_general` | Nama di `clean_code` | Alasan |
|---|---|---|
| `eksperiment_real_case_v2/` (hanya kode: `v2_common.py`, `overrides.py`, `booking_sampler.py`, `build_world_config.py`) | `real_case_common/` | Nama lebih menjelaskan perannya sebagai kode BERSAMA (dipakai `v_2/` & `network_scaling/`), bukan satu eksperimen tersendiri. `main.py`/`run_nonneg.py` milik folder ini TIDAK disalin karena keduanya driver korpus `#6` (real_case_v2 lama, di luar cakupan). |
| `eksperiment_last_pisan/` | `v_2/` | Sesuai penamaan yang Anda minta (data Bab V.2 laporan). |
| `network_scaling_eksperiment/` | `network_scaling/` | Nama tetap deskriptif, hanya dipendekkan; skrip analisis/viz (`extract_*.py`, `make_fig_*.py`) TIDAK disalin (di luar cakupan "generasi data mentah"). |
| `desain_eksperiment/configs/` | `v_1/configs/` | Config dunia OFAT non-real-case sekarang menyatu di folder `v_1/` (Bab V.1 laporan). Sejak 2026-08-12 dipangkas dari 16 menjadi 13 dunia, mengikuti isi laporan. |
| (tidak ada, folder baru) | `v_1/` | Sebelumnya tidak ada folder terpusat — 3 mekanisme (negosiasi/Greedy/As-Is-Booking) untuk non-RC dijalankan langsung dari root/`metaheuristic`/`non_negotiation_mechanism` dengan config dari `desain_eksperiment/`. Di sini confignya dipusatkan ke `v_1/configs/`. |

## Environment

Project ini punya file dependency sendiri (`pyproject.toml`, `requirements.txt`,
`uv.lock`, `.python-version`), independen dari `.venv` `nyoba_general`. Butuh
**Python 3.12** (lihat `.python-version`). Dependency utama: `torch`, `numpy`,
`scipy`, `matplotlib`, `negmas==0.15.7` (versi terkunci), `networkx`, `osmnx`,
`pytest`. `fastapi`/`uvicorn`/`websockets` juga ikut ter-*install* (ada di
`pyproject.toml`) meski tidak benar-benar dipakai — `dashboard/` di sini HANYA
berisi `config_io.py` (util murni Python untuk *parsing*/validasi config,
dipakai `main.py`), bukan server FastAPI-nya.

### Opsi A — `uv` (direkomendasikan, sesuai `uv.lock` yang sudah disalin)

```bash
cd clean_code
uv sync
```

`uv` akan otomatis memasang Python 3.12 (kalau belum ada) dan seluruh dependency
persis sesuai versi yang terkunci di `uv.lock` — cara paling reproducible.
Jalankan skrip lewat `uv run`, misalnya:

```bash
uv run python v_2/main_nego.py --agent-fleet DLST --run 1
```

### Opsi B — `venv` + `pip` biasa

```bash
cd clean_code
python -m venv .venv

# Windows (PowerShell / Command Prompt)
.venv\Scripts\activate
# Windows (Git Bash) / Linux / macOS
source .venv/Scripts/activate      # Git Bash di Windows
source .venv/bin/activate          # Linux/macOS

pip install --upgrade pip
pip install -r requirements.txt
```

Pastikan `python --version` menunjukkan 3.12.x sebelum membuat `.venv` (pakai
`pyenv`/`py -3.12` bila komputer Anda punya beberapa versi Python terpasang).

### Verifikasi Environment Terpasang Benar

```bash
python -c "import torch, numpy, scipy, matplotlib, negmas, networkx; print('OK, semua dependency inti bisa diimpor')"
python -c "import negotiation, simulation; print('OK, kode inti clean_code bisa diimpor')"
```

Kalau baris kedua gagal dengan `ModuleNotFoundError: No module named 'negotiation'`,
pastikan Anda menjalankannya dari root `clean_code/` (bukan dari dalam
subfolder seperti `v_2/`), karena `negotiation`/`simulation` diimpor sebagai
paket top-level relatif terhadap direktori kerja saat ini.

## Cara Pakai

Semua perintah di bawah dijalankan dari root `clean_code/` (bukan dari dalam
subfolder), dengan environment yang sudah dipasang (lihat "Environment" di
atas). Detail lengkap tiap korpus (semua *flag*, semua agen, semua varian) ada
di README masing-masing subfolder — bagian ini hanya ringkasan cara mulai.

### Root project — `main.py` & `run_experiment.py` (pipeline dasar, dipakai bebas di luar 8 korpus)

Dua skrip ini bukan bagian dari salah satu korpus, tapi berguna untuk
uji-coba/*debug* satu simulasi dengan config apa pun:

```bash
# satu simulasi cepat dengan config default, output tersebar di data/{runs,logs,registry,history}
python main.py
python main.py --seed 42

# satu eksperimen "rapi" (semua artefak terkumpul di data/experiment/<name>/)
python run_experiment.py --name coba_dulu
```

`run_experiment_bulk_per_config.py` (juga di root) adalah *entry point*
langsung untuk **korpus #1** — lihat bagian "v_1" di bawah.

### `v_1/` — Korpus #1, #2, #3 (non-*real-case*, Bab V.1)

```bash
# #1 — agen negosiasi (matriks 9 pasangan x 2 mode utk 1 dunia)
python run_experiment_bulk_per_config.py --name hz_n1_upd1_s42 \
    --config v_1/configs/hz_n1_upd1_s42.json --out-root v_1/data/experiment_bulk

# #2 — Greedy
python metaheuristic/run_greedy.py --config v_1/configs/hz_n1_upd1_s42.json \
    --name hz_n1_upd1_s42 --exp-root v_1/data/greedy

# #3 — As-Is & Booking
python non_negotiation_mechanism/as_is_mechanism/main_as_is.py \
    --name hz_n1_upd1_s42 --config v_1/configs/hz_n1_upd1_s42.json --seed 42 --out-root v_1/data/as_is
python non_negotiation_mechanism/booking_mechanism/main_booking.py \
    --name hz_n1_upd1_s42 --config v_1/configs/hz_n1_upd1_s42.json --seed 42 --out-root v_1/data/booking
```

Ulangi untuk 13 dunia di `v_1/configs/` (ganti `--name`/`--config`; seluruhnya
berseed 42). Korpus #1 hanya memakai tiga pasangan agen pada mode PADDED,
sehingga perintahnya wajib menyertakan `--only 6 8 9 --modes PADDED`. Detail
lengkap dan padanan tiap dunia dengan label di laporan -> `v_1/README.md`.

### `v_2/` — real-case *fixed_mechanism*

Empat keluarga skenario. Istilah okupansi 22,5 persen dan 50 persen **tidak
dipakai lagi**; yang dikendalikan sekarang adalah **permintaan booking**, dan
okupansi menjadi keluaran yang diukur.

| Skenario | Permintaan | Okupansi yang muncul |
|---|---:|---:|
| `medium_occupancy` | 185 | sekitar 26 persen |
| `high_occupancy` | 366 | sekitar 47 persen |
| `mahalkompen_medium_occupancy` | 185 | sekitar 26 persen |
| `mahalkompen_high_occupancy` | 366 | sekitar 47 persen |

Angka 185 dan 366 adalah rata-rata dan kuantil 95 jumlah transaksi harian
seluruh SPKLU Bandung 2024, dikali horizon tiga hari.

```bash
python v_2/gen_configs.py                    # sekali saja: 160 config (4 x 40 run)

python v_2/main_nego.py --agent-fleet DLST --run 1
python v_2/main_nego.py --agent-fleet DLSTMBOM --run 1 --skenario high_occupancy
python v_2/main_nego.py --agent-fleet DLSTMBOMDirected --run 1 --skenario mahalkompen_high_occupancy

python v_2/main_greedy.py --run 1 --skenario high_occupancy

python v_2/main_nonneg_v2.py --mech booking --run 1 --skenario medium_occupancy
python v_2/main_nonneg_v2.py --mech as_is   --run 1 --skenario high_occupancy
```

`--run` menerima 1 sampai 40. Detail lengkap di `v_2/README.md`.

**Argumen `--empiris`.** Seluruh perintah `v_2/` dan `network_scaling/` menerima
`--empiris`, yang memilih sampler booking LAMA (undian berbobot histogram,
durasi dipotong 2 jam). Tanpa argumen itu, lama dan jam mulai booking ditarik
dari **distribusi teoretis hasil pencocokan**, yaitu Log-Logistik untuk durasi
dan Weibull untuk jam mulai. Flag ini **hanya** mengganti sumber distribusinya;
kebijakan penempatan booking sama pada kedua mode.

**Metrik tambahan per run.** Setiap run `v_2/` dan `network_scaling/` menulis
`metrices/booking_generation.json` (berapa booking berhasil tanpa digeser,
berapa setelah digeser, berapa ditolak, beserta okupansi yang dihasilkan), dua
gambar pembanding sebaran hasil fit terhadap sebaran yang terjadi, serta
`pergeseran_fallback` di dalam `aggregate_metrics.json` (berapa kali booking
digeser ke fallback oleh taksi saat negosiasi). Rinciannya di `v_2/README.md`.

### `network_scaling/` — Korpus #12 (belum dipakai laporan, masih dipertimbangkan)

```bash
# langkah 1: generate config utk ukuran jaringan pilihan
python network_scaling/gen_configs.py --hidden 512 --layers 3

# langkah 2: jalankan eksperimen
python network_scaling/main_nego_ns.py --agent-fleet DLST --hidden 512 --layers 3 --run 1 --lean
```

Detail lengkap (kedua keluarga skenario, opsi ukuran jaringan lawan) -> `network_scaling/README.md`.

## Cakupan yang SENGAJA tidak disalin

- **Kode analisis/agregasi** (`aggregate_analysis/build_ledger.py`, `viz_*.py`,
  dsb.) — di luar permintaan ("hanya kode generasi data mentah"). Output di
  sini tetap berupa `metrices/aggregate_metrics.json` per *run*, format
  identik dengan `nyoba_general`, jadi bisa dianalisis dengan tooling apa pun
  yang membaca format itu.
- **Korpus #4, #5, #6, #11, #13, #14, #15, #16** (lihat `laporan/korpus_override_info.md`
  di `nyoba_general` untuk detail lengkap peta korpus) — di luar 8 korpus yang
  diminta. Beberapa datanya sudah HILANG PERMANEN di sumber aslinya (#4/#5);
  lainnya adalah korpus tuning/eksplorasi (#13) yang hasilnya sudah menyatu ke
  kode inti (`negotiation/agents/dlst_mbom_directed_agent.py`) tanpa perlu
  menyalin datanya sendiri.
- Varian **horizon 30 hari** (bagian dari korpus #14) — tidak pernah dibuatkan
  generatornya di `v_2/` (4 generator skenario hanya menghasilkan varian
  40-run per keluarga skenario, tidak ada varian 30-hari sama sekali).
- **Server web `dashboard/`** (`app.py`, `sim_session.py`, `static/`) — REST
  API + WebSocket + UI untuk visualisasi interaktif, tidak diperlukan untuk
  generasi data. Hanya `dashboard/config_io.py` yang disalin (lihat catatan
  di bawah).

## Riwayat Perbaikan

- **2026-08-XX — `ModuleNotFoundError: No module named 'dashboard'`**: `main.py`
  (dipakai SEMUA skrip untuk memuat config) ternyata memanggil
  `dashboard.config_io.config_from_dict`/`validate_config` untuk *deep-merge*
  dan validasi config JSON. Awalnya `dashboard/` sama sekali tidak disalin
  (dikira murni komponen web). Diperbaiki: `dashboard/__init__.py` +
  `dashboard/config_io.py` disalin (modul ini murni Python, hanya bergantung
  pada `simulation.config`, TIDAK memakai FastAPI sama sekali) — `app.py`/
  `sim_session.py`/`static/` (server web sungguhan) tetap TIDAK disalin karena
  memang tidak diperlukan. Sudah diverifikasi: `from dashboard.config_io import
  config_from_dict, validate_config` dan `load_config_from_path(...)` kini
  berhasil tanpa error di `clean_code`.

## Verifikasi

Seluruh file `.py` di project ini sudah dicek lolos `python -m py_compile`
(tidak ada *syntax error*), dan rantai impor `main.py` -> `dashboard.config_io`
sudah dicek berhasil dimuat.

Perbandingan angka terhadap `nyoba_general` **tidak lagi relevan** dan tidak
dikerjakan, sebab project ini memang sudah sengaja berbeda (lihat bagian
"Project ini SUDAH BERBEDA dari `nyoba_general`" di atas).

Yang sudah diverifikasi langsung pada project ini:

- Penjaga pergeseran booking ke masa lalu: sapuan 48 grid pada 3 seed
  menghasilkan **nol** opsi yang menggeser booking mundur, sedangkan tanpa
  penjaga ada 27 sampai 45. Doctest 5 modul terkait lolos seluruhnya.
- Jalur `--empiris` menghasilkan tarikan dan dunia yang **identik bit demi bit**
  dengan cara pembangunan sampler versi sebelumnya.
- Rantai `reference_data_simulation/` dapat dijalankan mandiri di sini dan
  menghasilkan berkas yang identik dengan salinannya di `nyoba_general`, yaitu
  bukti bahwa seluruh analisis datanya deterministik.
- `network_scaling/gen_configs.py` berjalan untuk kedua keluarga skenario, dan
  `v_2/gen_configs.py` menghasilkan 160 config yang sudah diperiksa isinya.
