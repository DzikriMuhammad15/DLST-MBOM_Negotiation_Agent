# Baseline METAHEURISTIC — Greedy dua-sisi (desain v2 §5)

> Tujuan tesis: menunjukkan bahwa agen **rule-based greedy dua-sisi** sulit
> mencapai kesepakatan (lebih kolaps daripada agen RL) → memotivasi perlunya
> strategi belajar/adaptif (RL/DLST/hybrid MBOM). Prediksi pre-register **V2**:
> deal_rate greedy << RL di mayoritas skenario.

## Aturan agen (JUJUR — guard kejujuran context.md §14.5)

`greedy_agent.GreedyNegotiator` (satu kelas dipakai KEDUA sisi, masing-masing
menilai dgn ufun sisinya sendiri):

| Peran | Aturan |
|---|---|
| Propose | SELALU tawarkan bundle ber-utility-SENDIRI tertinggi di action_library. Tawaran konstan sepanjang negosiasi (tanpa konsesi). |
| Respond | ACCEPT hanya bila `utility_sendiri(offer) ≥ utility bundle terbaik KE-2 sendiri` (`accept_rank=2`; ambang konstan terhadap waktu). Library berisi 1 bundle → ambang = terbaik. |

**Guard kejujuran (WAJIB dipahami pembaca hasil):**
- Ambang aspirasi = besaran alami milik agen sendiri (utility terbaik ke-2),
  BUKAN konstanta absurd yang dipilih supaya gagal. Agen greedy tetap MAU
  menerima: tiap sisi punya daerah penerimaan nyata (≥ terbaik-ke-2-nya).
- Kolaps (bila terjadi) bersifat **EMERGENT**: bundle terbaik fleet cenderung
  berharga rendah/geser menguntungkan fleet, bundle terbaik SPKLU cenderung
  berharga tinggi — irisan daerah penerimaan sering kosong. Tidak ada sabotase
  kode (tanpa bug disengaja, tanpa reject-semua).
- Payload, metrik, dan instrumentasi inference identik agen lain
  (subclass `BaseNegotiator`, pola `RuleBasedAgent`).

**Limitasi yang diakui:** baseline rule-based dgn konsesi-terjadwal (mis.
Boulware — sudah ada sbg `Rule` di exp01 bulk lama) adalah baseline heuristik
LEBIH KUAT; greedy di sini sengaja mewakili "aspirasi kaku dua pihak", bukan
heuristik terbaik yang mungkin. Klaim yang sah: *rule saklek dua pihak sulit
deal*, bukan *semua rule-based gagal*.

## Registrasi aditif (kode lama TIDAK diubah)

`registry_patch.install_greedy_agent_type()` me-wrap
`agent_factory.construct_agent_shell` + `agent_factory._build_trainers`
(build_agent & rebuild per-episode me-resolve keduanya via global modul →
patch berlaku di semua jalur). Tipe: `"GreedyMH"`, tanpa trainer (pola Rule).

## Menjalankan

```bash
# 16 skenario non-RC (1 exp per skenario; pasangan Greedy-vs-Greedy, PADDED —
# greedy tanpa jaringan => mode-invariant, 1 pasangan cukup):
bash metaheuristic/_run_queue_mh.sh          # resume-safe

# atau satuan:
python metaheuristic/run_greedy.py --config desain_eksperiment/configs/ref_n10_upd1_s42.json --name ref_n10_upd1_s42

# dunia real-case v2 (3 run; dunia identik nego/non-neg v2):
python metaheuristic/run_greedy.py --v2-run 1
```

Output `metaheuristic/data/{non_rc/<skenario>|real_case_v2/run{n}_padded_greedy}/`
— format identik eksperimen negosiasi (experiment_meta.json [marker lengkap,
ditulis terakhir] + logs/ + metrices/aggregate_metrics.json) → kompatibel
build_ledger (extend aditif) & fix_analysis. **Folder tanpa
experiment_meta.json = parsial/korup → hapus sebelum resume.**
