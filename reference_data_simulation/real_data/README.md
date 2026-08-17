# `real_data/` — data referensi asli, apa adanya

Seluruh berkas di folder ini adalah **salinan mentah** dari sumbernya, tanpa
satupun perubahan isi. Folder ini sengaja dibuat swasembada supaya seluruh
rantai analisis dapat diulang dari nol tanpa bergantung pada folder lain di
project.

| Berkas | Isi | Periode | Ukuran | Dipakai untuk |
|---|---|---|---:|---|
| `DATA TRX SPKLU 2024.xlsx` | 402.119 transaksi charging SPKLU se-Indonesia. Per transaksi: stasiun, chargebox, tier daya, waktu, kWh, rupiah, durasi, SoC awal dan akhir, jenis konektor, pengguna | 2024 penuh | 48,1 MB | seluruh distribusi jam mulai, durasi, harga, energi, SoC |
| `MASTER SPKLU PER 14 MAR 2025_.xlsx` | Master lokasi SPKLU: nama, koordinat, chargebox, daya, jumlah konektor, status aktif | per 14 Mar 2025 | 2,8 MB | identitas, koordinat, dan jumlah port stasiun |
| `Master SPKLU 20250311 (Aktif dan Non Aktif)__.xlsx` | Master pembanding, memuat stasiun aktif maupun non-aktif | per 11 Mar 2025 | 0,7 MB | triangulasi hasil ekstraksi |
| `tomtom_traffic_index_city_bandung_raw_2026-07-07.html` | Halaman mentah TomTom Traffic Index kota Bandung, memuat blok `weekHour` berisi tingkat kemacetan per pasangan hari dan jam (168 titik) | data 2025 | 1,7 MB | bobot waktu kemunculan order penumpang |
| `tomtom_bandung_weekhour.json` | Hasil ekstraksi blok `weekHour` | — | 19 KB | turunan dari berkas HTML di atas |
| `time_weight_24.json` | 24 bobot per jam, hasil perataan tujuh hari lalu dinormalkan terhadap puncaknya | — | 1,3 KB | masuk langsung ke konfigurasi simulasi |
| `extract_tomtom_weekhour.py` | Skrip yang mengubah berkas HTML menjadi kedua JSON di atas | — | 4,6 KB | reproduksi bobot waktu |

## Struktur kolom yang dipakai

**`DATA TRX SPKLU 2024.xlsx`, sheet `TRX 2024`**

| Kolom | Keterangan |
|---|---|
| `ID_SPKLU` | Pengenal stasiun. Dipakai untuk menyaring transaksi wilayah Bandung |
| `TANGGAL` | Cap waktu lengkap, contohnya `01/01/2024 13:47:22,2960` |
| `DURASI` | Lama sesi dalam menit |
| `KWH` | Energi tersalur |
| `RPKWH` | Total rupiah. Harga per kWh dihitung sebagai `RPKWH / KWH` |
| `FIRST_SOC`, `LAST_SOC` | Persentase daya baterai saat datang dan saat selesai |
| `DAYA_CHARGEBOX`, `TYPE_CONNECTOR`, `ID_USER` | Atribut pelengkap |

**`MASTER SPKLU PER 14 MAR 2025_.xlsx`, sheet `SPKLU AKTIF PER MESIN (2)`**

Satu baris per chargebox. Kolom yang dipakai: `ID_SPKLU`, `NAMA_SPKLU`,
`LATITUDE`, `LONGITUDE`, `IS_ACTIVE_SPKLU`, `DAYA_CHARGEBOX`, `CHARGEBOX_ID`,
`JML_CONNECTOR_SPKLU`, `JML_CONNECTOR_CHARGER`.

## Catatan

Empat berkas mentah pertama pada tabel di atas (tiga xlsx dan satu html)
**tidak pernah ditulis ulang** oleh skrip manapun. Seluruh skrip pada
`../filtering/` hanya membacanya dan menulis hasilnya ke `../filtering/output/`.

Dua berkas JSON TomTom adalah turunan, bukan mentah. Keduanya disimpan di sini
supaya berdampingan dengan bukti HTML-nya. `extract_tomtom_weekhour.py` membaca
berkas HTML pada folder ini dan menulis ulang kedua JSON itu di tempat, jadi
menjalankannya akan menyegarkan keduanya dari bukti mentah yang sama:

```bash
python reference_data_simulation/real_data/extract_tomtom_weekhour.py
```

Skrip itu memakai `Path(__file__).parent` sehingga seluruhnya bekerja di dalam
folder ini dan tidak menyentuh apapun di luarnya.
