# real_case_common/ — Kode Bersama Dunia Real-Case Bandung

Dipakai oleh `v_2/` dan `network_scaling/` (BUKAN oleh `v_1/`, yang murni
sintetis/non-*real-case*). Isi:

- `v2_common.py` — util path (`fix_sys_path`), pembangun sampler
  (`build_sampler`), penyetel tipe agen (`apply_agents`).
- `overrides.py` — `install_v2_bookings` (monkeypatch generator booking bawaan
  jadi versi empiris Bandung) + `normalize_time_weight` (perbaikan key JSON).
- `booking_sampler.py` — `EmpiricalBookingSampler`, membaca distribusi nyata
  dari `simulation_data/_derived/`.
- `build_world_config.py` — generator dunia dasar (13 SPKLU, 50 taksi, harga,
  N booking) dari data mentah Bandung; hasilnya (`configs/run{1,2,3}.json`)
  sudah disertakan langsung, skrip ini untuk regenerasi/verifikasi saja.
- `DOKUMENTASI_SAMPLING_DAN_PROVENANCE.md` — dokumentasi lengkap provenance
  tiap angka dunia (mana yang data-grounded, mana yang keputusan peneliti).

Folder ini TIDAK punya *entry point* (`main.py`) sendiri — pemanggilan selalu
lewat skrip di `v_2/` atau `network_scaling/` yang meng-impor modul di sini via
`sys.path.insert(0, .../real_case_common)`.
