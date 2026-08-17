"""DLSTMBOMDirected — agen anti-kolaps (subclass ``DLSTMBOMNewMechanism``).

Cakupan v1 (lihat ``negotiation/agents/DLSTMBOMDirected_SPEC.md``, LOCKED):
L1 directed exploration, persistent frequency model (§1a, SATUKAN IOP+Eq.10),
cross-episode ``DirectedController`` (§5a), stuck-rescue (§4), handover
competence-gated (§5), L3 potential-based reward shaping (via ``DirectedDDPGTrainer``),
L5 hygiene (A ``sigma_base`` + B1 init kecil + B2 LayerNorm; B4 grad-clip sudah AS-IS;
B3 weight-decay TIDAK dipakai). **L4 di-OMIT.**

Semua perilaku baru **config-gated** lewat ``DirectedConfig`` (default OFF), sehingga dengan
semua switch OFF kelas ini **bit-identik** dengan ``DLSTMBOMNewMechanism``. Progres per-fase
dilacak di ``directed_impl/TRACKER.md``.

Fase terpasang: 0 (skeleton) · 1 (persistensi §1a satukan-freq + §5a controller).
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    from negmas import ResponseType
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    ResponseType = None  # type: ignore

from negotiation.agents.dlst_mbom_new_mechanism_agent import DLSTMBOMNewMechanism


# =============================================================================
# Config switches & hyperparameter (spec §8) — semua switch default OFF
# =============================================================================
@dataclass
class DirectedConfig:
    """Switch & hyperparameter DLSTMBOMDirected (spec §8). **Semua switch default OFF**
    → dengan semua OFF, ``DLSTMBOMDirected`` bit-identik dengan ``DLSTMBOMNewMechanism``.

    Layer switches (per-fase; lihat directed_impl/TRACKER.md):
      l1              — L1 directed exploration (propose sampling + respond floor + concat ringkas-8)
      persistent_freq — §1a SATUKAN freq persistent (IOP + Eq.10, cross-episode)
      stuck_rescue    — §4 stuck-detection + un-saturasi
      handover        — §5 handover competence-gated (shadow via episode evaluasi)
      shaping         — L3 potential-based reward shaping (DirectedDDPGTrainer)
      hygiene         — L5 A(sigma_base)+B1(init kecil)+B2(LayerNorm)

    Example:
        >>> DirectedConfig().l1
        False
        >>> DirectedConfig().sigma_base
        0.05
    """

    # ── layer switches (default OFF → baseline) ──────────────────────────────
    l1: bool = False
    persistent_freq: bool = False
    stuck_rescue: bool = False
    handover: bool = False
    shaping: bool = False
    hygiene: bool = False

    # ── hyperparameter (spec §8) ─────────────────────────────────────────────
    M: int = 24                 # kandidat aksi / langkah (16-32)
    nu0: float = 0.40           # lebar noise directed awal
    nu_boost: float = 0.40      # lebar noise saat rescue
    sigma_base: float = 0.05    # L5-A lantai noise permanen
    beta: float = 0.30          # L3 skala shaping
    gamma_shaping: float = 0.95 # L3 discount MDP (= discount critic)
    floor_gamma: float = 0.90   # §3b fraksi nilai deal-terbaik-realistis
    shadow_gate: float = 0.70   # §5 ambang shadow_deal_rate lepas-floor
    W: int = 20                 # jendela episode (deal-rate/gating)
    W_shadow: int = 10          # min sampel shadow sebelum gating aktif
    K_shadow_eval: int = 5      # cadence episode evaluasi shadow
    stuck_thr: float = 0.05     # ambang deal_rate -> STUCK
    alpha_unsat: float = 0.20   # skala bobot lapisan-akhir saat rescue
    init_scale: float = 3e-3    # L5-B1 bound uniform init lapisan-akhir
    min_episode: int = 30       # backstop handover
    max_episode: int = 400      # backstop handover

    def any_on(self) -> bool:
        """True bila ≥1 switch layer aktif (mempercepat jalur baseline saat semua OFF)."""
        return any((self.l1, self.persistent_freq, self.stuck_rescue,
                    self.handover, self.shaping, self.hygiene))


# =============================================================================
# Cross-episode controller (§5a) — DIPERSIST dengan hidup pada objek trainer
# (trainer bag dibawa lintas episode oleh rebuild_agent_shell_with_trainers).
# =============================================================================
class DirectedController:
    """State lintas-episode (§5a). Disimpan sebagai atribut pada trainer persistent,
    sehingga bertahan saat agen di-instansiasi ulang tiap episode.

    Fase 1: wadah + ``episode_index``. Logika deal/shadow/rescue/anneal menyusul Fase 3.

    Example:
        >>> c = DirectedController(DirectedConfig(W=4))
        >>> c.deal_window.maxlen, c.floor_active, c.episode_index
        (4, True, 0)
    """

    def __init__(self, cfg: DirectedConfig) -> None:
        self.cfg = cfg
        self.deal_window: deque = deque(maxlen=cfg.W)           # outcome deal nyata (bool)
        self.shadow_window: deque = deque(maxlen=max(cfg.W, cfg.W_shadow))  # outcome shadow
        self.shadow_eval_counter: int = 0
        self.nu: float = float(cfg.nu0)
        self.floor_active: bool = True
        self.episode_index: int = 0
        self.rescue_active: bool = False


class DLSTMBOMDirected(DLSTMBOMNewMechanism):
    """Agen DLST+MBOM anti-kolaps. Semua switch OFF → identik ``DLSTMBOMNewMechanism``.

    Example:
        >>> DLSTMBOMDirected.__name__
        'DLSTMBOMDirected'
        >>> issubclass(DLSTMBOMDirected, DLSTMBOMNewMechanism)
        True
    """

    def __init__(self, *args: Any, directed: Optional[DirectedConfig] = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dcfg: DirectedConfig = directed if directed is not None else DirectedConfig()
        self._directed_controller: Optional[DirectedController] = None
        self._shadow_mode: bool = False  # episode ini = shadow-eval (μ murni) ? (§5)

    # ------------------------------------------------------------------
    # §1a + §5a — bind persistensi saat attach trainer (hook _attach_trainers)
    # ------------------------------------------------------------------
    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """Attach trainer (base), lalu bind persistent freq (§1a) & controller (§5a).

        Saat semua switch OFF → hanya super().attach_trainer → identik base.
        """
        super().attach_trainer(dict_of_trainer)
        if not self.dcfg.any_on():
            return
        if self.dcfg.persistent_freq:
            self._bind_persistent_freq()
        self._bind_directed_controller()

    def _bind_persistent_freq(self) -> None:
        """§1a SATUKAN: pakai frequency model reward-side (persist di respond_trainer,
        umpan Eq.10) sebagai model IOP agen. Efek: satu model untuk IOP + Eq.10, cross-episode,
        dan update tawaran lawan (``_on_real_offer_received``) kini mengisi Eq.10 juga.
        """
        rt = self.trainers.get("respond_trainer")
        rs = getattr(rt, "reward_shaper", None) if rt is not None else None
        pm = getattr(rs, "opponent_model", None) if rs is not None else None
        if pm is not None:
            self.mbom_opponent_model = pm

    def _bind_directed_controller(self) -> None:
        """§5a: DirectedController hidup pada trainer persistent (propose_trainer) → bertahan
        lintas episode. Idempoten: kalau sudah ada, dipakai ulang (bukan dibuat baru)."""
        host = self.trainers.get("propose_trainer") or self.trainers.get("respond_trainer")
        if host is None:
            return
        ctrl = getattr(host, "_directed_controller", None)
        if ctrl is None:
            ctrl = DirectedController(self.dcfg)
            host._directed_controller = ctrl
        self._directed_controller = ctrl

    # ------------------------------------------------------------------
    # reset_episode — jaga freq tetap hangat saat persistent (§1a)
    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset per-episode. Saat ``persistent_freq`` ON → JANGAN reset frequency model
        (lewati langkah ``mbom_opponent_model.reset()`` milik NewMechanism dengan memanggil
        reset_episode GRANDPARENT). Saat OFF → identik base.
        """
        if self.dcfg.persistent_freq:
            super(DLSTMBOMNewMechanism, self).reset_episode()  # skip mbom reset
        else:
            super().reset_episode()

    # ==================================================================
    # L1 §2 — representasi state: concat RINGKAS-8 (ganti fan-out 2N+N²)
    # ==================================================================
    RINGKASAN_DIM = 8  # spec §2b (4 dari iop_respond + 4 dari iop_propose)

    def _l1_state_on(self) -> bool:
        """L1 mengubah representasi STATE (ringkasan-8). Ditentukan CONFIG (tetap; TIDAK
        di-toggle shadow — dim network fixed saat build)."""
        d = getattr(self, "dcfg", None)
        return d is not None and d.l1

    def _l1_act_on(self) -> bool:
        """L1 mengubah AKSI (directed propose + floor). OFF saat shadow-eval episode (μ murni)."""
        return self._l1_state_on() and not getattr(self, "_shadow_mode", False)

    def _enriched_dim(self, dlst_dim: int) -> int:
        """L1 ON → dim = dlst_dim + 8 (ringkasan). OFF → base (fan-out 2N+N²)."""
        if self._l1_state_on():
            return dlst_dim + self.RINGKASAN_DIM
        return super()._enriched_dim(dlst_dim)

    def _fanout_batches(self, iop_state_basis: Any):
        """Fan-out IOP atas ``action_library`` NYATA → (resp (real_N,2), prop (real_N,k_max)).
        Replikasi jalur base ``_build_augmented_state_combined`` SEBELUM padding (nilai
        kandidat nyata; hindari baris padding). Dipakai bersama: ringkasan-8, Φ (Fase 2.2),
        skor L1 propose (Fase 2.3)."""
        relative_time = float(iop_state_basis[1]) if len(iop_state_basis) > 1 else 0.0
        lib = getattr(self, "action_library", None) or []
        if not lib or torch is None:
            return None, None
        basis = [
            self.build_opponent_agent_state_basis(
                self.build_negmas_state_from_offer(offer, relative_time))
            for offer in lib
        ]
        X = torch.tensor(np.stack(basis, axis=0), dtype=torch.float32, device=self.device)
        resp = self._fanout_probs(X, "iop_trainers_respond",
                                  self.iop_respond_action_dim, self.bayesian_mixer_respond)
        prop = self._fanout_probs(X, "iop_trainers_propose",
                                  self.iop_propose_action_dim, self.bayesian_mixer_propose)
        return np.asarray(resp), np.asarray(prop)

    @staticmethod
    def _norm_entropy(v: Any) -> float:
        """Entropi ternormalisasi [0,1] (dibagi log N). Sebaran suatu vektor non-negatif."""
        v = np.asarray(v, dtype=np.float64)
        s = v.sum()
        if s <= 1e-12 or v.size <= 1:
            return 0.0
        p = v / s
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)) / math.log(v.size))

    @staticmethod
    def _summarize8(resp: Any, prop: Any, ufleet: Any, uopp: Any,
                    cur_idx: Optional[int]) -> "np.ndarray":
        """Ringkasan-8 (spec §2b). ``resp``:(N,2) ``prop``:(N,K) ``ufleet``/``uopp``:(N,).

        4 dari iop_respond: [max P_accept, U_fleet(o*), P_accept(offer sekarang), entropy(P_accept)]
        4 dari iop_propose: [U_fleet(counter*), U_opp(counter*), entropy(counter), gap]
        PINNING (didokumentasikan): ``o* = argmax U_fleet·P_accept``; distribusi counter =
        ``prop[o*, :N]`` (baris offer skor-terbaik-diterima); ``counter* = argmax``; entropy
        dinormalisasi ``log N``; ``gap = U_opp(counter*) − U_opp(offer sekarang)``.
        """
        n = int(ufleet.shape[0])
        if n == 0:
            return np.zeros(8, dtype=np.float32)
        p_acc = resp[:, 0]
        o_star = int(np.argmax(ufleet * p_acc))
        valid_cur = cur_idx is not None and 0 <= cur_idx < n
        counter = prop[o_star, :n]
        c_star = int(np.argmax(counter)) if counter.size else 0
        u_cur = float(uopp[cur_idx]) if valid_cur else 0.0
        out = np.array([
            float(p_acc.max()),                                        # f1 max P_accept
            float(ufleet[o_star]),                                     # f2 U_fleet(o*)
            float(p_acc[cur_idx]) if valid_cur else 0.0,               # f3 P_accept(sekarang)
            DLSTMBOMDirected._norm_entropy(p_acc),                     # f4 entropy(P_accept)
            float(ufleet[c_star]),                                     # f5 U_fleet(counter*)
            float(uopp[c_star]),                                       # f6 U_opp(counter*)
            DLSTMBOMDirected._norm_entropy(counter),                   # f7 entropy(counter)
            float(uopp[c_star]) - u_cur,                               # f8 gap
        ], dtype=np.float32)
        return out

    def _build_augmented_state_combined(self, dlst_state_vec: Any,
                                        iop_state_basis: Any) -> "np.ndarray":
        """L1 ON → ``[dlst | ringkasan-8]``. OFF → base ``[dlst | iop_respond(2N) | iop_propose(N²)]``."""
        if not self._l1_state_on():
            return super()._build_augmented_state_combined(dlst_state_vec, iop_state_basis)
        resp, prop = self._fanout_batches(iop_state_basis)
        if resp is None:
            ring = np.zeros(self.RINGKASAN_DIM, dtype=np.float32)
        else:
            n = resp.shape[0]
            ufleet = np.array(
                [self._normalize_utility(self._index_user_utility(i)) for i in range(n)],
                dtype=np.float64)
            uopp = np.array(
                [float(self.mbom_opponent_model.estimate_utility(self.make_offer(i)))
                 for i in range(n)], dtype=np.float64)
            cur = (self._offer_to_action_idx(self.last_received_offer)
                   if self.last_received_offer is not None else None)
            ring = self._summarize8(resp, prop, ufleet, uopp, cur)
        return np.concatenate(
            [np.asarray(dlst_state_vec, dtype=np.float32), ring]).astype(np.float32)

    # ==================================================================
    # L1 §3a — PROPOSE directed exploration (M kandidat → skor → argmax)
    #          + cache Φ ke step.meta (§6 L3 memakainya nanti)
    # ==================================================================
    def _propose_split(self):
        """split_sizes (n_delta, n_choice, n_param) dari actor propose (untuk range aksi)."""
        tr = self.trainers.get("propose_trainer")
        actor = tr.models.get("actor", {}).get("model") if tr is not None else None
        return getattr(actor, "split_sizes", None)

    def _action_bounds(self, n: int, split):
        """Batas per komponen: δ,c ∈ [0,1] (sigmoid), p ∈ [-1,1] (tanh)."""
        lo = np.zeros(n, dtype=np.float64)
        hi = np.ones(n, dtype=np.float64)
        if split:
            nd, nc, npar = split
            lo[nd + nc: nd + nc + npar] = -1.0
        else:
            lo[:] = -1.0  # tak tahu split → pakai batas terlebar
        return lo, hi

    def _sample_candidates(self, mu: "np.ndarray", nu: float):
        """M kandidat aksi (spec §3a): μ + (μ+𝒩(0,ν²)) lokal + acak lebar, di-clip ke range."""
        M = max(1, int(self.dcfg.M))
        n = mu.shape[0]
        split = self._propose_split()
        lo, hi = self._action_bounds(n, split)
        cands = [mu.copy()]
        n_local = (M - 1) // 2
        for _ in range(n_local):
            cands.append(mu + np.random.normal(0.0, nu, size=n))
        for _ in range(M - 1 - n_local):
            cands.append(np.random.uniform(lo, hi))
        return [np.clip(np.asarray(c, dtype=np.float64), lo, hi) for c in cands]

    def _directed_select(self, mu, t, ctx_bid, ufleet, p_acc, nu):
        """Skor tiap kandidat: template→idx, Score=U_fleet(idx)·P_accept(idx); argmax (tanpa ε)."""
        n = ufleet.shape[0]
        best_a, best_idx, best_score = mu, None, -1e18
        for a in self._sample_candidates(mu, nu):
            idx = self.bidding_strategy.generate(t, a.tolist(), ctx_bid)
            if idx is None or idx < 0 or idx >= n:
                continue
            score = float(ufleet[idx] * p_acc[idx])
            if score > best_score:
                best_score, best_idx, best_a = score, int(idx), a
        if best_idx is None:  # fallback: template dari μ
            best_idx = self.bidding_strategy.generate(t, mu.tolist(), ctx_bid)
            best_a = mu
        return best_a, int(best_idx)

    def propose(self, state: Any, dest: Any = None) -> Any:
        """L1 ON → directed exploration (M kandidat, argmax U_fleet·P_accept) + cache Φ.
        OFF (atau shadow-eval) → base ``propose`` (identik AS-IS, μ murni)."""
        if not self._l1_act_on():
            return super().propose(state, dest)

        offer_opponent = self._state_offer(state)
        t = self._relative_time(state)
        u_bar = self._predict_threshold(state, offer_opponent)
        obs_propose = self.build_state_propose(state, offer_opponent)
        mu = np.asarray(self.sample_action_propose(obs_propose), dtype=np.float64)
        ctx_bid = {
            "u_bar": u_bar,
            "opponent_last_offer": self.last_received_offer,
            "opponent_last_idx": self._offer_to_action_idx(self.last_received_offer)
            if self.last_received_offer is not None else None,
        }

        # Fan-out sekali: P_accept & U_fleet per offer → skor kandidat + Φ.
        resp, _prop = self._fanout_batches(self._state_basis_for_iop(state, offer_opponent))
        phi = None
        if resp is not None and len(self.action_library):
            n = resp.shape[0]
            p_acc = np.asarray(resp[:, 0], dtype=np.float64)
            ufleet = np.array(
                [self._normalize_utility(self._index_user_utility(i)) for i in range(n)],
                dtype=np.float64)
            nu = self._directed_controller.nu if self._directed_controller is not None else self.dcfg.nu0
            if self.dcfg.hygiene:  # L5-A: lantai noise permanen (σ_eff = max(ν, σ_base))
                nu = max(float(nu), float(self.dcfg.sigma_base))
            a_dir, idx = self._directed_select(mu, t, ctx_bid, ufleet, p_acc, float(nu))
            phi = float(np.max(ufleet * p_acc))
        else:  # fan-out tak tersedia → fallback template μ (perilaku ~ base)
            a_dir = mu
            idx = self.bidding_strategy.generate(t, mu.tolist(), ctx_bid)

        offer = self.make_offer(idx)
        current_u = self._offer_utility(offer)
        self.current_state_data_propose = {
            "state": obs_propose,
            "action": a_dir.tolist() if hasattr(a_dir, "tolist") else list(a_dir),
            "utility": current_u,
            "value": None,
            "log_prob": None,
            "phi": phi,  # dibaca oleh flush_pending_propose → step.meta["phi"]
        }
        self._emit_trace_step(
            "propose", state, obs_propose, idx,
            actor_output={"actor_vector": self.current_state_data_propose["action"]},
            extra={"u_bar": float(u_bar), "offer": offer, "utility": current_u},
        )
        return offer

    def flush_pending_propose(self, state: Any) -> None:
        """Base flush (append Step), lalu sisipkan Φ dari pending ke ``step.meta['phi']``
        (alignment terjamin: step yang baru di-append = pending yang barusan di-flush)."""
        phi = None
        if self._l1_act_on() and self.current_state_data_propose:
            phi = self.current_state_data_propose.get("phi")
        super().flush_pending_propose(state)
        if phi is not None and self.memory_propose:
            last = self.memory_propose[-1]
            meta = dict(last.meta) if isinstance(last.meta, dict) else {}
            meta["phi"] = float(phi)
            last.meta = meta

    # ==================================================================
    # L1 §3b — RESPOND floor: roda bantu yang melatih threshold head
    #   effective_bar = min(u_bar, floor);  floor = γ · U_fleet(o*)
    #   accept bila U_in ≥ effective_bar (menurunkan ambang → banjiri deal)
    # ==================================================================
    def _floor_active(self) -> bool:
        """Floor aktif sampai handover melepasnya (§5, Fase 3). Default True."""
        c = self._directed_controller
        return c.floor_active if c is not None else True

    def _compute_floor(self, state: Any, offer: Any) -> float:
        """floor = γ · U_fleet(o*), o* = argmax_o U_fleet(o)·P_accept(o) (grounded opponent model)."""
        resp, _prop = self._fanout_batches(self._state_basis_for_iop(state, offer))
        if resp is None or not len(getattr(self, "action_library", []) or []):
            return 0.0
        n = resp.shape[0]
        p_acc = np.asarray(resp[:, 0], dtype=np.float64)
        ufleet = np.array(
            [self._normalize_utility(self._index_user_utility(i)) for i in range(n)],
            dtype=np.float64)
        o_star = int(np.argmax(ufleet * p_acc))
        return float(self.dcfg.floor_gamma * ufleet[o_star])

    def respond(self, state: Any, source: Any = None) -> Any:
        """L1 ON & floor aktif → jika base REJECT tapi U_in ≥ min(u_bar, floor) → ACCEPT.
        OFF → base ``respond`` (identik; base tetap menjalankan _on_real_offer_received & rekam)."""
        resp = super().respond(state, source)
        if not (self._l1_act_on() and self._floor_active()):
            return resp
        if ResponseType is None or resp != ResponseType.REJECT_OFFER:
            return resp
        offer = self._state_offer(state)
        if offer is None:
            return resp
        u_in = self._normalize_utility(self._offer_utility(offer))
        u_bar = float(getattr(self, "_last_threshold_value", 1.0))
        effective_bar = min(u_bar, self._compute_floor(state, offer))
        if u_in >= effective_bar:
            if self.current_state_data_respond:
                self.current_state_data_respond["decision"] = 0  # konsistensi rekaman
            return ResponseType.ACCEPT_OFFER
        return resp

    # ==================================================================
    # Fase 3 — Stuck-rescue (§4) + Handover competence-gated (§5)
    #   state lintas-episode dari DirectedController (§5a, persist di trainer)
    # ==================================================================
    def on_negotiation_start(self, state: Any) -> None:
        """Base start + jadwalkan shadow-eval episode (μ murni) tiap K episode (§5)."""
        super().on_negotiation_start(state)
        self._shadow_mode = False
        if getattr(self, "rollout_level", None) is not None:
            return
        c = self._directed_controller
        if c is None or not self.dcfg.handover or not c.floor_active:
            return  # shadow-eval hanya relevan selama floor masih aktif (untuk gating lepas)
        c.shadow_eval_counter += 1
        if c.shadow_eval_counter >= max(1, int(self.dcfg.K_shadow_eval)):
            c.shadow_eval_counter = 0
            self._shadow_mode = True

    def on_negotiation_end(self, state: Any) -> None:
        """Base end (simpan episode), lalu update controller: catat outcome deal/shadow →
        cek stuck-rescue (§4) & handover (§5). Tak berlaku saat rollout."""
        super().on_negotiation_end(state)
        if getattr(self, "rollout_level", None) is not None:
            return
        c = self._directed_controller
        if c is None:
            return
        deal = getattr(state, "agreement", None) is not None
        c.episode_index += 1
        if self._shadow_mode:
            c.shadow_window.append(bool(deal))
        else:
            c.deal_window.append(bool(deal))
        self._maybe_rescue(c)
        self._maybe_handover(c)

    def _maybe_rescue(self, c: "DirectedController") -> None:
        """§4: deal_rate < stuck_thr selama W → ν←ν_boost + UN-SATURASI aktor (keduanya wajib)."""
        if not self.dcfg.stuck_rescue or len(c.deal_window) < self.dcfg.W:
            return
        deal_rate = sum(c.deal_window) / len(c.deal_window)
        if deal_rate < self.dcfg.stuck_thr:
            c.nu = float(self.dcfg.nu_boost)
            c.rescue_active = True
            self._unsaturate_actors(self.dcfg.alpha_unsat)
            c.deal_window.clear()  # jendela baru untuk mengukur pemulihan (hindari re-trigger langsung)
        else:
            c.rescue_active = False

    def _maybe_handover(self, c: "DirectedController") -> None:
        """§5: lepas floor & ciutkan ν saat shadow_deal_rate ≥ gate (≥W_shadow sampel) &
        episode ≥ min_episode. Backstop: lewat max_episode masih macet → picu stuck-rescue."""
        if not self.dcfg.handover or not c.floor_active:
            return
        enough = len(c.shadow_window) >= self.dcfg.W_shadow
        shadow_rate = (sum(c.shadow_window) / len(c.shadow_window)) if c.shadow_window else 0.0
        if (c.episode_index >= self.dcfg.min_episode and enough
                and shadow_rate >= self.dcfg.shadow_gate):
            c.floor_active = False
            c.nu = float(self.dcfg.sigma_base)  # ciutkan directed ke lantai (handover)
            return
        if c.episode_index > self.dcfg.max_episode and self.dcfg.stuck_rescue:
            # backstop: bukan lepas buta — picu rescue
            c.nu = float(self.dcfg.nu_boost)
            c.rescue_active = True
            self._unsaturate_actors(self.dcfg.alpha_unsat)

    def _unsaturate_actors(self, alpha: float) -> None:
        """§4b: skala bobot lapisan-akhir aktor (head / net[-1]) dengan α<1 → kecilkan z →
        μ'(z) hidup lagi. Berlaku ke 3 head (propose/respond/threshold)."""
        if torch is None:
            return
        a = float(alpha)
        for key in ("propose_trainer", "respond_trainer", "threshold_trainer"):
            tr = self.trainers.get(key)
            actor = tr.models.get("actor", {}).get("model") if tr is not None else None
            if actor is None:
                continue
            final = getattr(actor, "head", None)
            if final is None:
                net = getattr(actor, "net", None)
                final = net[-1] if (net is not None and len(net) > 0) else None
            if final is not None and hasattr(final, "weight"):
                with torch.no_grad():
                    final.weight.mul_(a)
                    if getattr(final, "bias", None) is not None:
                        final.bias.mul_(a)
