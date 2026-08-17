# Skema Data Run Non-Negosiasi (as_is & booking)

> Format 1 "run" = 1 simulasi (TIDAK ada kombinasi agen negosiasi). Mirror struktur **satu
> eksperimen** di dalam bulk negosiasi, MINUS semua yang berkaitan agen/mekanisme negosiasi.
> Berlaku untuk dua mekanisme: `as_is` (siapa cepat dia dapat) & `booking` (booking statis
> tanpa negosiasi). Dummy/target sintetis mengikuti skema ini; kode simulasi riil (tahap
> berikutnya) HARUS menghasilkan format sama.

## Struktur folder per run
```
<dataset>/<run_name>/
  run_meta.json                 # provenance (analog experiment_meta.json)
  logs/simulation/<sim_id>.jsonl # tick + event alokasi (BUKAN negotiation_*)
  metrices/aggregate_metrics.json# 3 blok: summary, numeric_flat, aggregate
  registry/   (kosong)          # tak ada agen tersimpan
  runs/       (kosong)          # PNG per-negosiasi tak relevan
  history/    (kosong)
  # TIDAK ada logs/online_training/ (tak ada pembelajaran)
```
`<run_name>`: config lama → nama bulk apa adanya (mis. `ref_n10_upd1_s42`); real-case →
`real_case_<nama_config>` (mis. `real_case_rc_ref_d3_t400_s42`).

## Event log simulasi
- `tick`: SAMA dgn negosiasi (world snapshot: taxis soc/state, hour, day).
- Pengganti `negotiation_*` = **event alokasi charging**:
  - `charge_request` (taxi butuh charging; punya target port).
  - as_is: `charge_wait` (port terisi → menunggu; ada `wait_min`), `charge_start`, `charge_done`,
    `charge_giveup`/`emergency` (menyerah / ke SPKLU darurat).
  - booking: `booking_attempt`, `booking_success` (slot kosong → reserve harga base) /
    `booking_fail` (semua slot relevan sudah dibook app-user → tak ada displacement),
    `charge_start` (di slot terbook).
  Tiap event punya `period` (indeks event alokasi, analog period negosiasi), `taxi_id`,
  `port_id`, `mechanism`, dan field bisnis terkait.

## Metrik (`aggregate` block) — pemetaan dari negosiasi + tambahan bisnis
| Key | Makna non-negosiasi |
|-----|---------------------|
| `avg_deal_rate` | **allocation_success_rate**: fraksi charge_request yang berhasil charging |
| `avg_utility_fleet` / `_deal` | utilitas fleet: −(biaya charging harga-base + energi perjalanan + penalti timing + penalti tunggu/order-hilang) |
| `avg_utility_spklu` / `_deal` | revenue SPKLU (utilisasi port oleh fleet) |
| `avg_episode_length` | as_is: rata-rata langkah tunggu; booking: ~0 |
| `negotiations_total` | **allocation_events_total** (nama key dipertahankan agar viz kompatibel) |
| `social_welfare_deal` | uF_deal + uS_deal |
| **`avg_wait_time_min`** (baru) | rata-rata waktu tunggu (menit) — 0 utk booking |
| **`orders_served` / `orders_lost`** (baru) | order penumpang terlayani / hilang |
| **`port_utilization`** (baru) | fraksi port-time terpakai fleet |
| **`compensation_paid`** (baru) | 0 utk kedua mekanisme (displacement hanya di negosiasi) |
| `mechanism` (baru, di run_meta) | "as_is" \| "booking" |

TIDAK ADA: u_*_deal per-agen negosiasi selain fleet/spklu, updF/updS (tak ada trainer),
fleet_inf_*_ms (tak ada inferensi agen), exp1..18 (satu run = satu sim).

## Hipotesis yang diuji dataset ini (ordering bisnis)
`as_is` (terburuk) < `booking` (menengah) < `negosiasi` (terbaik — data riil terpisah).
Dummy = realistis/paling-mungkin (ordering ada tapi moderat + noise; kompres saat dunia
longgar). Target = ordering tegas mendukung premis (tetap realistis). Lihat
[`initial.md`](initial.md) & [`logical_framing.md`](logical_framing.md) untuk dasar teorinya.
