# Spesifikasi `DLSTMBOMDirected` (LOCKED — 2026-07-20)

Agen baru = `DLSTMBOMNewMechanism` (IOP action-conditioned / fan-out N·N) + **Directed
Exploration (L1)** memakai fan-out untuk mengarahkan aksi, **tanpa** membengkakkan
input (concat diringkas). Tujuan: mencegah kolaps (aktor beku) lintas ukuran network.

Basis mekanisme kolaps (terverifikasi kode): gradien aktor `∇θ = (∂Q/∂a)·(∂μ/∂θ)`
mati karena **#1** Q datar (reward gagal buta-aksi), **#2** saturasi, **#3** tanpa noise,
**#4** aktor beku tak memasok data deal. L1 menutup #2,#3,#4; stuck-rescue menutup #2.

> **CAKUPAN v1 (keputusan peneliti 2026-07-20):** implementasi = L1 + L2(lebur) +
> stuck-rescue + handover + §1a + §5a + L3 + L5(A,B1,B2,B4). **L4 di-OMIT**. **§1a =
> SATUKAN** (freq persistent melayani IOP + Eq.10, cross-episode). **Shadow §5 = episode
> evaluasi** (tiap K episode). **L5 tanpa B3** (weight decay). Semua switch default OFF.

---

## 1. Basis & komponen opponent model (LOCKED)

Tiga komponen opponent model (sudah ada di kode), dipakai di tempat berbeda:

| Komponen | Peran di `DLSTMBOMDirected` |
|---|---|
| `FrequencyOpponentModel` (`U_o`) | reward respond (Eq.10, AS-IS) — tak diubah |
| Rollout engine (`compute_best_response`, zero-sum) | melatih IOP (recursive imagination) — tak diubah |
| **IOP fan-out (NewMechanism)** `iop_respond (N,2)`, `iop_propose (N,N)` | **dipakai GANDA**: (a) skor Directed Exploration L1 [mentah], (b) input actor [DIRINGKAS] |

N = `k_max` (ruang aksi propose tetap).

### 1a. Frequency opponent model PERSISTENT (DLSTMBOMDirected-ONLY) — LOCKED

Di `DLSTMBOMNewMechanism`, model frequency pengumpan-IOP (`self.mbom_opponent_model`,
[dlst_mbom_new_mechanism_agent.py:53](dlst_mbom_new_mechanism_agent.py#L53)) dibuat **fresh
di `__init__`** + di-`reset()` tiap episode ([:228-231](dlst_mbom_new_mechanism_agent.py#L228))
→ `freq` **kosong tiap awal episode** (karena agen di-instansiasi ulang per episode,
hard-constraint runtime; trainer persist, `freq` tidak).

**Untuk `DLSTMBOMDirected`**, karena **opponent SAMA di semua episode** (asumsi stasioner),
`freq` pengumpan-IOP **DI-PERSIST-kan**:
- **Embed** `FrequencyOpponentModel` di trainer bag → ikut dibawa
  `rebuild_agent_shell_with_trainers` ([agent_factory.py:640-660](agent_factory.py#L640)).
- **Override `reset_episode`**: JANGAN reset model ini (pertahankan `freq` lintas episode).
- **Bind dari trainers saat attach**, bukan bikin baru di `__init__`.
- **Config-gated** (`directed_persistent_freq`, default OFF → base fresh-freq reproducible).
  **HANYA** `DLSTMBOMDirected`; `DLSTMBOMNewMechanism` tak berubah.

**Efek**: `freq` warm di awal (hampir) tiap episode → **cold-start Sumber A ~lenyap**
(tinggal episode pertama training). Ini yang paling menyembuhkan `Φ`/`Score`/`floor` di awal.

**Dua model di kode** (lihat tabel §1): reward-side ([agent_factory.py:571](agent_factory.py#L571),
persist di `respond_trainer`, umpan Eq.10 `U_o`) & agent-side/IOP ([:53], fresh).
**KEPUTUSAN v1 — SATUKAN**: satu `FrequencyOpponentModel` persistent melayani IOP **dan**
Eq.10 reward. Konsekuensi: **`U_o` Eq.10 jadi cross-episode** (lebih akurat) — ini **deviasi
dari AS-IS reward** yang **WAJIB didokumentasikan di tesis** (DLSTMBOMDirected-only, config-gated).

**Asumsi WAJIB**: perilaku opponent **stasioner lintas episode**. Kalau preferensi opponent
berubah per-episode, persisting akan **membaurkan** profil berbeda → keliru. (Peneliti
menyatakan opponent sama untuk setup ini.)

---

## 2. Representasi state / input actor (LOCKED)

**Prinsip:** fan-out MENTAH untuk mengarahkan aksi (L1); untuk concat cukup **RINGKASAN**
(hindari ledakan `N²` yang menggelembungkan network → memperparah kolaps).

### 2a. Fitur taktis DLST (per-head, dari kode)
Base bersama `_common_attrs` (6): `[u_norm, rel_time, O_best, O_avg, O_sd, has_offer]`

| Head | Dim taktis | Tambahan |
|---|---|---|
| threshold | 6 | base saja |
| respond | 10 | base + `[u_fixed, u_bar, u_future, quantile]` |
| propose | 10 | base + `[u_bar, U_boulware, U_pareto, U_bopp]` |

### 2b. Ringkasan opponent (8 fitur, dari KEDUA blok fan-out)
Dari `iop_respond (N,2)` — "tawaranku diterima?":
```
[ max_o P_accept(o),  U_fleet(offer skor-terbaik-diterima),
  P_accept(offer sekarang),  entropy(P_accept) ]
```
Dari `iop_propose (N,N)` — "lawan akan counter apa?":
```
[ U_fleet(counter paling mungkin),  U_opp(counter itu),
  entropy(distribusi counter),  gap = U_opp(counter) − U_opp(offer saya) ]
```
Aturan: **encode argmax lewat ATRIBUT (utilitas), bukan indeks bundle.**

### 2c. State final per head (taktis + ringkasan-8), di-concat ke MASING-MASING head
```
threshold : 6  + 8 = 14
respond   : 10 + 8 = 18
propose   : 10 + 8 = 18
```
Default: kedua ringkasan ke semua head. (Opsional head-specific: propose paling butuh
ringkasan-respond; respond paling butuh ringkasan-propose.)

---

## 3. L1 — Directed Exploration (LOCKED)

Off-policy-safe (DDPG replay → behavior policy bebas). Reward tetap dihitung reward_shaper
dari episode nyata (C6 stamp tetap sah).

### 3a. Sisi PROPOSE (per langkah)
```
1. Bentuk state s (§2c, head=propose)
2. μ(s) = (δ,c,p)
3. fan-out iop_respond → P_accept[o] untuk semua N bundle
4. Sampel M kandidat aksi:
     - μ(s) + 𝒩(0, ν²)      (lokal; ν di-anneal)
     - acak lebar            (cakupan; jaring pengaman saat μ beku di sudut buruk)
5. Tiap kandidat → template → offer o → Score(o) = U_fleet(o) · P_accept[o]
6. a_dir = argmax Score        ← AMBIL MAX LANGSUNG (tanpa ε-switch)
7. Pancarkan a_dir → tawarkan o → lawan respons → simpan (s, a_dir, r, s', done)
```
Handover: `ν` menyusut → kandidat mengerucut ke `μ(s)` → `a_dir ≈ μ(s)` → aktor mandiri.
(Anneal `ν` = competence-gated, lihat §5.)

### 3b. Sisi RESPOND (per tawaran masuk)
Pakai **utilitas KITA** atas tawaran masuk (bukan fan-out):
```
o_in masuk; U_in = U_fleet(o_in)
effective_bar = min( u_bar_agen , floor )        (u_bar dari threshold head)
kalau U_in ≥ effective_bar → cari (δ,c,p) respond yang menghasilkan ACCEPT → simpan (accept, +u)
kalau U_in <  effective_bar → tolak/counter
```
- **floor** = `γ · U_fleet(o*)`, γ ≈ 0.9, di mana `o* = argmax_o U_fleet·P_accept`
  (grounded ke opponent model: "terima kalau ~90% sebagus deal terbaik yang bisa
  kuupayakan sendiri"). Opsional konsesi-waktu: `floor(t) = floor·(1 − δ·rel_time)`.
- **Peran floor:** roda bantu yang MELATIH threshold head; setelah `u_bar` sehat, dilepas.

### 3c. Sisi THRESHOLD
Dilatih pada data deal yang dihasilkan L1; `u_bar`-nya membaik → mengambil alih saat floor
dilepas (§5).

---

## 4. Stuck-detection / rescue (LOCKED)

Lapisan cadangan darurat bila anneal biasa gagal.
```
Metrik  : deal_rate = (#deal dalam W episode) / W        (mis. W=20)
Deteksi : deal_rate < 0.05 selama W episode → STUCK
Rescue  : (a) ν ← ν_boost (mis. 0.05 → 0.40)             [banjiri deal]
          (b) UN-SATURASI aktor: W_last ← α·W_last (α<1)  [kecilkan z → μ'(z) hidup]
Wajib (a)+(b): aktor jenuh tak merespons data deal tanpa un-saturasi (∂μ/∂θ≈0).
Eskalasi: masih macet → α lebih kecil / ν lebih besar.
```

---

## 5. Handover / anneal — COMPETENCE-GATED (LOCKED)

Bukan hard-coded episode (rapuh: waktu-lepas tergantung ukuran network). Pakai sinyal
terukur bahwa kebijakan murni sudah kompeten.
```
shadow_deal_rate = deal-rate agen MURNI (tanpa floor/directed)

PENGUKURAN (KEPUTUSAN v1 — EPISODE EVALUASI): tiap K episode, jalankan 1 episode
  inferensi PENUH tanpa floor & tanpa directed (μ deterministik) → catat deal/tidak ke
  ring buffer shadow. shadow_deal_rate dihitung setelah terkumpul ≥ W_shadow sampel.
  (K = shadow_eval_every, mis. 5; sampel shadow terpisah dari deal_rate nyata.)

Lepas floor & kecilkan pengaruh directed  ⟺  shadow_deal_rate ≥ 0.70 selama W episode
Selama belajar: effective_bar = min(u_bar, floor); saat lepas: effective_bar = u_bar (murni)

Backstop (RAMBU, bukan pemicu):
  min_episode = 30   : jangan lepas sebelum ini
  max_episode = 400  : kalau lewat ini masih macet → picu STUCK-RESCUE (§4), bukan lepas buta
```
`ν` (lebar noise directed) mengikuti gating yang sama.

---

## 5a. Persistensi state lintas-episode (KRITIS — LOCKED)

**Hard-constraint runtime**: tiap episode = **agen baru + environment baru**
(`rebuild_agent_shell_with_trainers`, [agent_factory.py:640-660](agent_factory.py#L640));
JANGAN diubah. Konsekuensi mengikat: **SEMUA state lintas-episode TAK BOLEH disimpan di
object agen** (akan hilang tiap instansiasi ulang) → **wajib di-embed di trainer bag**
(persist, dibawa lintas episode), lalu re-bind saat attach — pola sama dengan persistent
freq §1a.

**State WAJIB persist** (kalau reset tiap episode → window tak pernah > 1 → §4/§5 TAK
PERNAH terpicu = **rusak SENYAP**, tanpa error):

| State | Dipakai | Bentuk | Efek kalau ter-reset |
|---|---|---|---|
| `mbom_opponent_model` (freq) | Φ/L1/§1a | dict | cold tiap episode (ditangani §1a) |
| deal_rate window | stuck-rescue §4 | ring buffer bool, len W | rescue tak pernah nyala |
| shadow_deal_rate window | handover §5 | ring buffer bool (dari episode evaluasi tiap K) | floor tak pernah lepas |
| shadow_eval_counter | jadwal episode evaluasi §5 | int (mod K) | eval shadow tak terjadwal |
| ν (noise width) | anneal §3a/§5 | float | anneal tak pernah mengerucut |
| floor_active | §3b/§5 | bool | floor tak konsisten |
| episode_index | backstop §5 | int | backstop min/max rusak |
| rescue_active | §4 | flag | rescue tak bisa eskalasi |

**Mekanisme**: satu objek **`directed_controller`** persistent di trainer bag. Di
`on_negotiation_end`: hitung outcome episode (deal? shadow-deal?) → **push ke ring buffer**.
Keputusan rescue/gate/anneal **membaca controller** (bukan agen fresh). Semua config-gated;
controller absen = perilaku base `DLSTMBOMNewMechanism`.

> **Catatan verifikasi**: uji bahwa `deal_rate`/`shadow` window benar-benar bertambah
> lintas episode (assert `len(window)` naik) — kegagalan persistensi tak memunculkan
> error, hanya mematikan §4/§5 diam-diam.

---

## 6. L3 — Potential-based reward shaping (LOCKED)

Menyerang **Lock #1 (critic datar)** LANGSUNG di level reward, dengan jaminan
**policy-invariance** (Ng, Harada & Russell 1999) → optimum objektif AS-IS Eq.9/Eq.10
TIDAK berubah; yang berubah hanya kepadatan sinyal belajar. Ini kandidat **kontribusi
ilmiah** (satu-satunya lock yang disentuh di reward, bukan input/eksplorasi).

### 6a. Bentuk shaping
```
r'_t = r_t + β·F(s_t, s_{t+1}),   F = γ·(1 − done_t)·Φ(s_{t+1}) − Φ(s_t)
Φ(terminal) = 0   (WAJIB untuk invariance; dipaksa lewat done_t)
```
- `γ` = **discount MDP** (sama dengan critic, ~0,95) — BUKAN `γ` floor (0,90) di §3b.
- `β` = skala shaping (default 0,30). Invariance tahan untuk `β` berapa pun (karena
  `β·(γΦ'−Φ)` = PBRS dengan potensi `βΦ`); `β` cuma menyeimbangkan magnitudo vs reward
  mentah `[0,1]/−1`.
- **Telescoping**: `Σ_t γ^t F_t = γ^T Φ(s_T) − Φ(s_0) = −Φ(s_0)` (konstanta bergantung
  state-awal saja) → return total tak berubah argmax-nya; yang berubah cuma distribusi
  kredit antar-langkah.

### 6b. Potensi Φ (grounded & TERVALIDASI ke kode)
```
Φ(s) = max_o  U_fleet_norm(o) · P_accept(o | s)          o ∈ action_library
P_accept(o|s) = iop_respond[o, 0]     (kolom 0 = ACCEPT — terverifikasi label training
                                       dlst_mbom_agent.py:417 action=0 vs :390 action=1)
```
- Sama dengan `o*` di floor (§3b) → satu opponent-model, tiga guna (skor L1, floor, Φ).
- **Fungsi state murni**: `U_fleet(o)` & library tetap; ketergantungan-state masuk lewat
  `P_accept(o|s)`. Rantai tervalidasi:
  `freq → estimate_utility(o)=U_lawan → basis dim-3 → IOP net + Bayesian mixer + softmax → P_accept`.
- `freq` (`FrequencyOpponentModel.self.freq`) bertambah tiap tawaran lawan
  (`opponent_model.py:93`), jadi Φ bergerak antar-langkah walau prosedur `max` identik.
- **Catatan akurasi (tesis)**: monotonisitas `P_accept↑` terhadap `U_lawan↑` bersifat
  **emergent/learned** (label accept langka & muncul di state ber-`U_lawan` tinggi),
  BUKAN aksiomatik.

### 6c. Kenapa menyalakan ∂Q/∂a tanpa perlu deal
Reward mentah `−1` buta-aksi (identik antar aksi → critic datar). `F` **peka-aksi lewat
transisi**: aksi beda → tawaran beda → reaksi lawan beda → `s'` beda → `Φ(s')` beda →
`r'` beda → `∂y/∂a ≠ 0` → critic membentuk kemiringan → gradien aktor hidup **bahkan di
episode gagal**. (L1 menyalakan critic dengan MENYEDIAKAN data deal; L3 dari SETIAP langkah.)

### 6d. Syarat-batas (cold-start) — kenapa L4 wajib menyertai
`total_bids = 0` → `estimate_utility = 0,5` seragam (`opponent_model.py:145-146`) →
`P_accept` datar → `Φ = 0,5·max U_fleet` KONSTAN → `F ≈ 0` → L3 lumpuh. Jendela ini
ditutup **L1** (floor/directed) dan dipersingkat **L4**. Defense-in-depth: L1 menutup
periode Φ-datar; L3 mengambil alih begitu `freq` terisi; L4 mempersingkat jendelanya.

### 6e. Integrasi kode
Additif di `episode_to_batch` ([ddpg_trainer.py:174](../trainers/ddpg_trainer.py)) SETELAH
`compute_reward_sequence`, **config-gated (default OFF)**. Fungsi reward mentah
`_compute_propose_mode1`/`_compute_respond_dlst` TAK disentuh → baseline bit-identik. `Φ`
dihitung dari opponent-model object (bukan dari vektor actor dim-10). `Φ(terminal)=0` via
`dones[t]` — konsisten dengan critic yang sudah menol-kan `Q` di `done`.

---

## 7. L5 — Hygiene anti-kolaps (LOCKED)

Prevention di level **arsitektur/optimizer** (melengkapi stuck-rescue §4 yang *recovery*).
Komponen DIPAKAI: **A (σ_base), B1 (init kecil), B2 (LayerNorm), B4 (grad clip)**.
**B3 weight decay TIDAK dipakai** (keputusan peneliti).

### 7a. A — `σ_base`: lantai noise permanen → tutup Lock #3
```
σ_eff = max(ν_annealed, σ_base),   σ_base = 0.05
aksi dipancarkan = μ(s) + 𝒩(0, σ_eff²)
```
`ν` meluruh ke `σ_base` (BUKAN 0) → selalu ada variasi aksi tipis → critic tak pernah
kehilangan sinyal `∂Q/∂a`, **bahkan setelah handover** (permanen). Contoh: μ=0.60 → aksi
0.58/0.63/0.59… (bukan beku di 0.600).

### 7b. B1 — init kecil lapisan-akhir aktor → titik-awal responsif
`DDPGActorStrategy.head` ([ddpg_networks.py:108](../models/ddpg_networks.py#L108)) &
`DDPGActorThreshold.net[-1]` ([:51](../models/ddpg_networks.py#L51)):
```python
nn.init.uniform_(head.weight, -3e-3, 3e-3);  nn.init.uniform_(head.bias, -3e-3, 3e-3)
```
Efek: `z≈0` di awal → `tanh'/sigmoid'` maksimal. **CATATAN JUJUR**: hanya titik-awal;
TIDAK menahan drift saturasi (kolaps kita = drift training-time, `z→7`). Init default
PyTorch pun tak lahir-jenuh (`z~0.37`). Jadi B1 = **lever terlemah**, pelengkap murah.

### 7c. B2 — LayerNorm sebelum aktivasi → tahan drift (INTI L5)
Sisip `nn.LayerNorm(hidden_dim)` **setelah tiap `Linear`, sebelum `ReLU`** di backbone
(`DDPGActorStrategy`/`DDPGActorThreshold`); opsional pada `h` sebelum `head`. Memaksa
pre-activation mean-0/std-1 → `z` tetap ~`[−3,3]` → `tanh'/sigmoid'` tak pernah mati,
**sepanjang training**. Ini penahan drift utama (B1 cuma titik-awal).

### 7d. B4 — gradient clipping → cegah lompatan ke saturasi
Di `ddpg_trainer.update()` sebelum `optimizer.step()`:
```python
actor_loss.backward()
torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
optimizer.step()
```
Batasi `‖grad‖` → satu langkah tak bisa melempar `z` ke zona jenuh sekaligus.

### 7e. Defense-in-depth & batasan
| Lock | Prevention (L5) | Recovery (§4) |
|---|---|---|
| #2 saturasi | B1 titik-awal + **B2 LayerNorm** + B4 clip | un-saturasi `W_last←α·W_last` |
| #3 tanpa noise | **A `σ_base`** lantai | boost `ν` |

B1/B2 mengubah **arsitektur**, B4 mengubah **update**, A mengubah **sampling aksi** →
semua **config-gated, default OFF, DLSTMBOMDirected-ONLY** → baseline `DLSTMBOMNewMechanism`
bit-identik. B2 mengubah dinamika konvergensi → **verifikasi tak merusak sel non-kolaps**.

---

## 8. Hyperparameter (default awal, untuk dituning)

| Param | Arti | Default |
|---|---|---|
| M | jumlah kandidat aksi / langkah | 16–32 |
| ν₀ | lebar noise awal (skala aksi) | 0.30–0.50 |
| ν_boost | lebar noise saat rescue | 0.40 |
| σ_base | noise dasar (L5 hygiene) selalu ada | 0.05 |
| γ (floor) | fraksi nilai deal-terbaik-realistis (§3b) | 0.90 |
| γ_MDP (shaping) | discount L3 `F=γΦ'−Φ` (= discount critic) | 0.95 |
| β (shaping) | skala potential-based reward shaping (L3, §6a) | 0.30 |
| shadow-gate | ambang shadow_deal_rate lepas-floor | 0.70 |
| W | jendela episode (deal-rate / gating) | 20 |
| stuck-thr | ambang deal_rate → STUCK | 0.05 |
| α (un-sat) | skala bobot lapisan-akhir saat rescue | 0.2 |
| min/max ep | backstop | 30 / 400 |
| init_scale (B1) | bound uniform init lapisan-akhir aktor | 3e-3 |
| clip_norm (B4) | max `‖grad‖` aktor (grad clipping) | 1.0 |
| K (shadow_eval_every) | tiap berapa episode jalankan episode evaluasi shadow (§5) | 5 |
| W_shadow | min sampel shadow sebelum gating aktif (§5) | 10 |

---

## 9. Titik integrasi di kode

1. `dlst_mbom_new_mechanism_agent.py` → subclass `DLSTMBOMDirected`.
2. **Concat ringkas** — override `_build_augmented_state_combined`: pakai ringkasan-8
   (§2b), bukan `iop_respond(2N)+iop_propose(N²)` penuh.
3. **L1 propose** — bungkus `sample_action_propose`: fan-out → sampel M → skor → a_dir (§3a).
4. **L1 respond + floor** — bungkus `respond`/`sample_action_respond`: effective_bar,
   accept-if-U_in≥bar (§3b).
5. **Gating & stuck** — state per-agen (deal_rate window, shadow eval, ν, floor-active),
   di-update di `on_negotiation_start`/akhir episode.
6. **Un-saturasi** — util skala bobot lapisan-akhir actor (§4b).
7. **Config switches** — semua default OFF → baseline `DLSTMBOMNewMechanism` reproducible.
8. Off-policy: simpan aksi yang BENAR-BENAR dipancarkan (a_dir / accept), reward via shaper.
9. **L3 shaping** — di `episode_to_batch` (ddpg_trainer.py) SETELAH `compute_reward_sequence`:
   `rewards[t] += β·(γ·(1−dones[t])·Φ(s_{t+1}) − Φ(s_t))`, `Φ` via util opponent-model
   (accept-kolom-0). Config-gated (`directed_shaping_enabled`, default OFF).
10. **L5 hygiene** — A: `σ_eff=max(ν,σ_base)` di `_sample_actor`. B1: re-init `head`/`net[-1]`
   di konstruksi actor (varian Directed). B2: sisip `LayerNorm` di `DDPGActorStrategy`/
   `DDPGActorThreshold` backbone. B4: `clip_grad_norm_` di `ddpg_trainer.update()`. Semua
   config-gated default OFF (`directed_hygiene`), Directed-only.

---

## 10. Status layer

| Layer | Status |
|---|---|
| **L1 Directed Exploration** (propose+respond+threshold+floor+concat ringkas) | **LOCKED** (§2–3) |
| **L2 Best-response residual** | **DILEBUR ke L1** (warm-start via ν₀ lebar; bukan komponen terpisah) |
| **Stuck-rescue + un-saturasi** (bagian L5) | **LOCKED** (§4) |
| **Handover competence-gated** | **LOCKED** (§5) |
| **L3 Potential-based reward shaping** | **LOCKED** (§6) — Φ=max_o U_fleet·P_accept(kolom-0); PBRS policy-invariant; lumpuh saat cold-start (self-heal seiring waktu) |
| **Persistent freq (§1a) + persistensi state lintas-episode (§5a)** | **LOCKED** — freq & controller (deal/shadow window, ν, floor_active, ep_idx, rescue) di trainer bag |
| **L4 De-cold-start opponent model** | **TIDAK diimplementasikan (v1 OMIT)** — freq self-healing dari OFFER lawan (update tiap `respond`, tak butuh deal) + persistent freq §1a + stuck-rescue §4 (recovery) sudah menutup. Prior zero-sum/korpus-seed/pretrain-IOP ditunda; tambah kalau verifikasi menunjukkan perlu |
| **L5 Hygiene** (§7) | **LOCKED** — A `σ_base` + B1 init kecil + B2 LayerNorm + B4 grad-clip; **B3 weight decay TIDAK dipakai**. B2 inti (tahan drift), B1 pelengkap terlemah |

---

## 11. Verifikasi (rencana)

Matriks {DLST, MBOM, **DLSTMBOMDirected**} × {kecil 128×2, besar 512×3} × 5 ulangan.
Target: **0/5 kolaps** (deal-rate<0.5) di semua sel, khususnya besar.
Diagnostik: deal-count di window↑, `∂Q/∂a` tak-nol lebih awal, fraksi unit saturated↓,
shadow_deal_rate. Ablation: matikan tiap sub-komponen → tunjukkan lock mana terbuka.
