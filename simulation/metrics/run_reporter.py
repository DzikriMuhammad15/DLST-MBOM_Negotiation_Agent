"""Orkestrasi pelaporan "runs" real-time per periode (KOMPONEN N5 — LLR-N5.*).

Modul ini menyediakan :class:`RunReporter`, lapisan simulasi yang:

* menulis JSON periode negosiasi (deal / no-deal) secara REAL-TIME tiap
  negosiasi selesai (KR-3, KR-5, KR-11);
* menulis JSON "no-negotiation" + alasan saat periode dilewati (KR-4, KR-7);
* mengakumulasi deret utility per-periode (per-hari & per-sim) — periode tanpa
  negosiasi masuk dengan utility ``0.0`` (KR-7) — dan mengirimkannya ke
  ``collector.record_period_utilities`` (pemetaan #C4 ditentukan di sini);
* meregenerasi PNG metrik per-hari saat pergantian hari dan PNG per-simulasi
  (kumulatif) saat run selesai (KR-6, KR-11), dengan metrik bisnis 06-03 per-hari
  dihitung via SNAPSHOT + DIFF di batas hari (#C1).

Aturan lapisan & invariant:

* **INV-1** — hanya mengimpor lapisan simulasi
  (:mod:`simulation.metrics.run_paths`, :mod:`simulation.metrics.episode_record`,
  :mod:`simulation.metrics.plots`); TIDAK mengimpor ``negotiation/*`` maupun
  ``simulation/world``. Payload Episode dibaca via duck typing di ``episode_record``.
* **INV-5** — TIDAK pernah memutasi WorldState atau agent; hanya MEMBACA hasil
  (collector / payload) dan MENULIS file/grafik.
* **INV-6** — seluruh IO & rendering DIBUNGKUS try/except agar kegagalan tidak
  menjatuhkan run; kegagalan dicatat ringan ke stderr. Flag
  ``config.reporting.enabled`` / ``write_period_json`` / ``render_day_png`` /
  ``render_sim_png`` dihormati.

Lihat ``.claude/low_level_requirement_creator/low_level_requirement.txt``
KOMPONEN N5 (LLR-N5.1..N5.7).
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence

from simulation.metrics.episode_record import (
    STATUS_DEAL,
    STATUS_NO_DEAL,
    build_no_negotiation_record,
    build_period_record,
    extract_phase_utilities,
    reconstruct_step_sequence,
    write_period_json,
)
from simulation.metrics.run_paths import RunPaths

# Pemetaan utility no-deal (#C4): penalty NYATA yang sama dengan terminal_utility
# agent untuk no-deal (reward_shaper failure_penalty / project lama). Dipakai
# sebagai fallback bila ufun final None pada periode no-deal.
NO_DEAL_UTILITY: float = -1.0

# Pemetaan utility periode tanpa-negosiasi (#C4 / KR-7).
NO_NEGOTIATION_UTILITY: float = 0.0

# Kunci fase pada dict payload yang diteruskan scheduler (M3-G).
_PHASE_PROPOSE: str = "propose"
_PHASE_RESPOND: str = "respond"


def _log_failure(context: str, error: BaseException) -> None:
    """Catat kegagalan reporter ke stderr tanpa menjatuhkan run (INV-6).

    Args:
        context: Deskripsi singkat operasi yang gagal (mis. ``"write period json"``).
        error: Exception yang tertangkap.

    Returns:
        ``None``. Efek samping: menulis baris peringatan ke ``sys.stderr``.

    Contoh:
        >>> import io, sys
        >>> buf, old = io.StringIO(), sys.stderr
        >>> sys.stderr = buf
        >>> _log_failure("demo", ValueError("boom"))
        >>> sys.stderr = old
        >>> "RunReporter" in buf.getvalue() and "demo" in buf.getvalue()
        True
    """
    try:
        sys.stderr.write(
            f"[RunReporter] WARNING: gagal saat {context}: "
            f"{type(error).__name__}: {error}\n"
        )
        # Jejak ringkas untuk diagnosis (tidak menghentikan run).
        traceback.print_exc(file=sys.stderr)
    except Exception:
        # Logging tidak boleh pernah menjatuhkan run.
        pass


def _terminal_utility_of(payloads: Dict[str, Any]) -> Optional[float]:
    """Ambil ``terminal_utility`` aktual dari payload propose suatu pihak.

    ``terminal_utility`` merepresentasikan U(agreement) bila deal atau marker
    no-deal (-1.0) menurut agent. Dibaca via duck typing dari payload propose
    (fase utama yang merekam outcome); bila tak ada, coba payload respond.

    Args:
        payloads: Dict ``{"propose": Episode|None, "respond": Episode|None,
            "utility_threshold": Episode|None}`` untuk satu pihak.

    Returns:
        ``terminal_utility`` sebagai ``float``, atau ``None`` bila tak tersedia.

    Contoh:
        >>> class _E:
        ...     terminal_utility = 0.42
        >>> _terminal_utility_of({"propose": _E(), "respond": None})
        0.42
        >>> _terminal_utility_of({"propose": None, "respond": None}) is None
        True
    """
    for phase in (_PHASE_PROPOSE, _PHASE_RESPOND):
        episode = payloads.get(phase)
        if episode is None:
            continue
        value = getattr(episode, "terminal_utility", None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


class RunReporter:
    """Orkestrator pelaporan run: tulis JSON periode + regen PNG per-hari/per-sim.

    Memegang sebuah :class:`~simulation.metrics.run_paths.RunPaths`, objek config
    simulasi (membaca ``config.reporting``), dan referensi ke ``collector`` untuk
    membaca ``summary``/``loss_history`` dan menyimpan deret utility per-periode
    (``record_period_utilities``). Kelas ini TIDAK memutasi world/agent (INV-5):
    seluruh operasinya adalah membaca hasil lalu menulis file/grafik, dengan setiap
    operasi IO/render dibungkus try/except (INV-6).

    Tanggung jawab utama:

    * :meth:`record_period` — JSON deal/no-deal real-time + deret utility (#C4).
    * :meth:`record_no_negotiation` — JSON no-negotiation + utility 0.0 (KR-7).
    * :meth:`capture_day_boundary_snapshot` / :meth:`on_day_completed` — snapshot &
      diff metrik bisnis per-hari (#C1) + regen PNG per-hari.
    * :meth:`on_run_finished` — regen PNG per-simulasi (kumulatif) + jaring pengaman
      PNG hari terakhir.

    Attributes:
        run_paths: Penyusun jalur folder/file run.
        config: Objek config simulasi (sumber ``reporting``).
        collector: Sumber summary/loss_history + tujuan record_period_utilities
            (boleh ``None`` untuk pemakaian tanpa collector).
        reporting: Alias ``config.reporting`` (``ReportingConfig``).

    Contoh:
        >>> import tempfile
        >>> from simulation.config.default import default_config
        >>> from simulation.metrics.run_paths import RunPaths
        >>> rp = RunPaths(tempfile.mkdtemp(), sim_id="demo")
        >>> reporter = RunReporter(rp, default_config(), collector=None)
        >>> reporter.sim_id
        'demo'
    """

    def __init__(
        self,
        run_paths: RunPaths,
        config: Any,
        collector: Any = None,
    ) -> None:
        """Inisialisasi reporter dengan RunPaths, config, dan collector opsional.

        Args:
            run_paths: Instance :class:`RunPaths` (sim_id + struktur folder).
            config: Objek config simulasi yang punya atribut ``reporting``.
            collector: Objek collector (mengekspos ``summary``, ``loss_history``,
                ``numeric_metrics_flat``, ``diff_numeric_metrics``,
                ``record_period_utilities``); boleh ``None``.

        Contoh:
            >>> import tempfile
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> rp = RunPaths(tempfile.mkdtemp(), sim_id="d2")
            >>> r = RunReporter(rp, default_config())
            >>> r.run_dir.endswith("d2")
            True
        """
        self.run_paths: RunPaths = run_paths
        self.config: Any = config
        self.collector: Any = collector
        self.reporting: Any = config.reporting

        # Counter periode per hari (mulai 0 tiap hari) — LLR-N5.4 / N5.7.
        self._period_index_by_day: Dict[int, int] = {}

        # Record periode untuk PNG per-hari & per-sim — LLR-N5.7.
        self._periods_by_day: Dict[int, List[Dict[str, Any]]] = {}
        self._all_periods: List[Dict[str, Any]] = []

        # Deret utility per-periode (per-hari & per-sim) — KR-7 / #C4.
        self._fleet_util_all: List[float] = []
        self._spklu_util_all: List[float] = []
        self._fleet_util_by_day: Dict[int, List[float]] = {}
        self._spklu_util_by_day: Dict[int, List[float]] = {}

        # Hari yang PNG-nya sudah diregen (cegah regen ganda).
        self._rendered_days: set = set()

        # #C1 snapshot+diff: snapshot metrik bisnis kumulatif di AWAL tiap hari.
        # Metrik HARI d = snapshot(awal d+1) - snapshot(awal d).
        self._metrics_snapshot_at_day_start: Dict[int, Dict[str, float]] = {}
        self._last_metrics_snapshot: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # LLR-N5.7 — util kecil (identitas run)
    # ------------------------------------------------------------------

    @property
    def sim_id(self) -> str:
        """Kembalikan sim_id run (delegasi ke ``run_paths.sim_id``).

        Returns:
            String sim_id.

        Contoh:
            >>> import tempfile
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> r = RunReporter(RunPaths(tempfile.mkdtemp(), sim_id="x"), default_config())
            >>> r.sim_id
            'x'
        """
        return self.run_paths.sim_id

    @property
    def run_dir(self) -> str:
        """Kembalikan jalur folder run (delegasi ke ``run_paths.run_dir``).

        Returns:
            String jalur ``run_dir``.

        Contoh:
            >>> import tempfile, os
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> root = tempfile.mkdtemp()
            >>> r = RunReporter(RunPaths(root, sim_id="y"), default_config())
            >>> os.path.basename(r.run_dir)
            'y'
        """
        return self.run_paths.run_dir

    # ------------------------------------------------------------------
    # LLR-N5.4 — counter periode per hari
    # ------------------------------------------------------------------

    def _next_period_index(self, day_index: int) -> int:
        """Kembalikan indeks periode berikutnya untuk ``day_index`` (mulai 0).

        Args:
            day_index: Indeks hari.

        Returns:
            Indeks periode (0 untuk periode pertama hari itu, lalu naik 1).

        Contoh:
            >>> import tempfile
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> r = RunReporter(RunPaths(tempfile.mkdtemp(), sim_id="z"), default_config())
            >>> r._next_period_index(0), r._next_period_index(0), r._next_period_index(1)
            (0, 1, 0)
        """
        current = self._period_index_by_day.get(day_index, 0)
        self._period_index_by_day[day_index] = current + 1
        return current

    # ------------------------------------------------------------------
    # Penyimpanan deret in-memory (LLR-N5.7) + cap memori
    # ------------------------------------------------------------------

    def _store_period_record(
        self, day_index: int, record: Dict[str, Any]
    ) -> None:
        """Simpan record periode ke deret per-hari & per-sim (untuk regen PNG).

        Menghormati ``reporting.keep_period_payloads_in_memory`` &
        ``reporting.max_periods_in_memory``: bila menyimpan banyak record dapat
        melampaui batas, record terlama di-drop (FIFO) agar memori terbatas
        (risiko HLR "volume riwayat terbatas").

        Args:
            day_index: Indeks hari periode.
            record: Record periode (penuh atau ringkas) untuk PNG.

        Returns:
            ``None``. Efek samping: memutakhirkan ``_periods_by_day`` &
            ``_all_periods``.
        """
        to_store = record
        if not getattr(self.reporting, "keep_period_payloads_in_memory", True):
            to_store = self._slim_record(record)

        day_list = self._periods_by_day.setdefault(day_index, [])
        day_list.append(to_store)
        self._all_periods.append(to_store)

        cap = int(getattr(self.reporting, "max_periods_in_memory", 0) or 0)
        if cap > 0:
            if len(day_list) > cap:
                del day_list[: len(day_list) - cap]
            if len(self._all_periods) > cap:
                del self._all_periods[: len(self._all_periods) - cap]

    @staticmethod
    def _slim_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Bentuk ringkas record (hanya yang dibutuhkan PNG) untuk hemat memori.

        Mempertahankan field yang dipakai modul plots: ``status``, ``steps``
        (untuk propose utils per aktor), ``agreement``, ``final_utility``,
        ``reason_code``. ``action_library`` (paling besar) dibuang dari salinan
        in-memory; JSON di disk tetap utuh (ditulis terpisah).

        Args:
            record: Record periode penuh.

        Returns:
            Salinan ringkas record.

        Contoh:
            >>> RunReporter._slim_record({"status": "deal", "steps": [],
            ...     "agreement": None, "action_library": [1, 2, 3]})["action_library"]
            []
        """
        return {
            "status": record.get("status"),
            "steps": record.get("steps", []),
            "agreement": record.get("agreement"),
            "final_utility": record.get("final_utility"),
            "reason_code": record.get("reason_code"),
            "action_library": [],
        }

    def _append_utilities(
        self, day_index: int, fleet_utility: float, spklu_utility: float
    ) -> None:
        """Tambahkan satu titik utility ke deret per-hari & per-sim + collector.

        Args:
            day_index: Indeks hari periode.
            fleet_utility: Nilai utility fleet (sudah final, #C4).
            spklu_utility: Nilai utility SPKLU (sudah final, #C4).

        Returns:
            ``None``. Efek samping: memutakhirkan deret in-memory & memanggil
            ``collector.record_period_utilities`` (bila collector ada).
        """
        self._fleet_util_all.append(fleet_utility)
        self._spklu_util_all.append(spklu_utility)
        self._fleet_util_by_day.setdefault(day_index, []).append(fleet_utility)
        self._spklu_util_by_day.setdefault(day_index, []).append(spklu_utility)

        if self.collector is not None:
            try:
                self.collector.record_period_utilities(
                    fleet_utility, spklu_utility
                )
            except Exception as error:  # INV-6
                _log_failure("collector.record_period_utilities", error)

    # ------------------------------------------------------------------
    # LLR-N5.2 — penulisan periode negosiasi (deal / no-deal)
    # ------------------------------------------------------------------

    def record_period(
        self,
        day_index: int,
        sim_time_min: float,
        hour: int,
        minute: int,
        status: str,
        action_library: Sequence[Any],
        turn_order: Optional[List[Dict[str, Any]]],
        fleet_payloads: Dict[str, Any],
        spklu_payloads: Dict[str, Any],
        agreement: Any,
        deal_bundle: Any,
        ufun_fleet_final: Optional[float],
        ufun_spklu_final: Optional[float],
        candidates_count: int,
    ) -> Optional[str]:
        """Tulis JSON periode deal/no-deal real-time + catat deret utility (#C4).

        Merekonstruksi urutan langkah dari DUA payload terpisah propose/respond
        tiap pihak (INV-2) memakai ``turn_order``, membangun record periode, lalu
        menulisnya ke ``run_paths.period_json_path`` (bila ``write_period_json``).
        Deret utility per-periode dihitung DI SINI dengan pemetaan #C4 lalu
        disimpan apa adanya oleh collector:

        * ``status == "deal"`` -> ``ufun_*_final`` (= ``U(agreement)`` aktual);
        * ``status == "no_deal"`` -> ``ufun_*_final`` aktual (= -1.0 dari agent);
          bila ``None`` -> fallback ``-1.0`` (penalty no-deal). TIDAK di-clamp ke 0.

        Args:
            day_index: Indeks hari periode.
            sim_time_min: Waktu simulasi (menit) saat periode.
            hour: Jam simulasi.
            minute: Menit simulasi.
            status: ``"deal"`` atau ``"no_deal"``.
            action_library: Bundle yang ditawarkan negosiasi ini.
            turn_order: Deskriptor urutan giliran dari runner (PKG-05), atau ``None``.
            fleet_payloads: Dict ``{"propose","respond","utility_threshold"}`` fleet.
            spklu_payloads: Dict ``{"propose","respond","utility_threshold"}`` SPKLU.
            agreement: Outcome NegMAS ``(idx,)`` atau ``None``.
            deal_bundle: :class:`Bundle` yang menang (atau ``None``).
            ufun_fleet_final: Utility akhir fleet (mentah dari agent) atau ``None``.
            ufun_spklu_final: Utility akhir SPKLU (mentah dari agent) atau ``None``.
            candidates_count: Jumlah taksi kandidat.

        Returns:
            Jalur file JSON yang ditulis, atau ``None`` (penulisan dimatikan/gagal).

        Contoh:
            >>> import tempfile, os
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> r = RunReporter(RunPaths(tempfile.mkdtemp(), sim_id="rp"),
            ...                 default_config())
            >>> p = r.record_period(0, 30.0, 0, 30, "no_deal", [], None,
            ...     {"propose": None, "respond": None, "utility_threshold": None},
            ...     {"propose": None, "respond": None, "utility_threshold": None},
            ...     None, None, None, None, 1)
            >>> os.path.isfile(p)
            True
            >>> r._fleet_util_all
            [-1.0]
        """
        period_index = self._next_period_index(day_index)

        # Langkah 2-3: ekstraksi utility per fase (None-safe) + rekonstruksi urutan.
        fleet_propose = extract_phase_utilities(fleet_payloads.get(_PHASE_PROPOSE))
        fleet_respond = extract_phase_utilities(fleet_payloads.get(_PHASE_RESPOND))
        spklu_propose = extract_phase_utilities(spklu_payloads.get(_PHASE_PROPOSE))
        spklu_respond = extract_phase_utilities(spklu_payloads.get(_PHASE_RESPOND))
        steps_sequence = reconstruct_step_sequence(
            turn_order,
            fleet_propose,
            fleet_respond,
            spklu_propose,
            spklu_respond,
        )

        # Langkah 4: bangun record periode.
        record = build_period_record(
            self.sim_id,
            day_index,
            period_index,
            sim_time_min,
            hour,
            minute,
            status,
            action_library,
            steps_sequence,
            agreement,
            deal_bundle,
            ufun_fleet_final,
            ufun_spklu_final,
            candidates_count,
        )

        # Langkah 5: tulis JSON (bila flag aktif) — INV-6.
        path: Optional[str] = None
        if getattr(self.reporting, "enabled", True) and getattr(
            self.reporting, "write_period_json", True
        ):
            try:
                path = self.run_paths.period_json_path(
                    day_index, period_index, sim_time_min, status=status
                )
                write_period_json(
                    path, record, int(getattr(self.reporting, "json_indent", 2))
                )
            except Exception as error:  # INV-6: IO gagal tak menjatuhkan run.
                _log_failure("write period json", error)
                path = None

        # Langkah 6: simpan record untuk PNG.
        self._store_period_record(day_index, record)

        # Langkah 7: hitung utility per-periode (#C4) lalu catat ke deret.
        fleet_util = self._period_utility_for(status, ufun_fleet_final)
        spklu_util = self._period_utility_for(status, ufun_spklu_final)
        self._append_utilities(day_index, fleet_util, spklu_util)

        return path

    @staticmethod
    def _period_utility_for(
        status: str, ufun_final: Optional[float]
    ) -> float:
        """Tentukan utility deret satu pihak untuk satu periode negosiasi (#C4).

        * ``deal`` -> ``ufun_final`` aktual (= ``U(agreement)``);
        * ``no_deal`` -> ``ufun_final`` aktual (= -1.0 dari agent), atau fallback
          ``-1.0`` bila ``None``. TIDAK di-clamp ke 0.
        * status lain -> ``ufun_final`` bila ada, else ``0.0`` (defensif; periode
          tanpa-negosiasi ditangani :meth:`record_no_negotiation`).

        Args:
            status: Status periode (``"deal"`` / ``"no_deal"``).
            ufun_final: Utility akhir mentah dari agent, atau ``None``.

        Returns:
            Nilai utility ``float`` untuk deret.

        Contoh:
            >>> RunReporter._period_utility_for("deal", 0.8)
            0.8
            >>> RunReporter._period_utility_for("no_deal", None)
            -1.0
            >>> RunReporter._period_utility_for("no_deal", -1.0)
            -1.0
        """
        if status == STATUS_DEAL:
            if ufun_final is None:
                return 0.0
            return float(ufun_final)
        if status == STATUS_NO_DEAL:
            if ufun_final is None:
                return NO_DEAL_UTILITY
            return float(ufun_final)
        if ufun_final is None:
            return 0.0
        return float(ufun_final)

    # ------------------------------------------------------------------
    # LLR-N5.3 — penulisan periode no-negotiation
    # ------------------------------------------------------------------

    def record_no_negotiation(
        self,
        day_index: int,
        sim_time_min: float,
        hour: int,
        minute: int,
        reason_code: str,
        candidates_count: int = 0,
    ) -> Optional[str]:
        """Tulis JSON periode tanpa-negosiasi + catat utility 0.0 (KR-4, KR-7).

        Membangun record no-negotiation (alasan + pesan), menulisnya ke disk (bila
        flag), menyimpannya untuk PNG, lalu menambahkan titik utility ``0.0`` ke
        deret per-hari & per-sim serta ke collector (KR-7).

        Args:
            day_index: Indeks hari periode.
            sim_time_min: Waktu simulasi (menit) saat periode.
            hour: Jam simulasi.
            minute: Menit simulasi.
            reason_code: Salah satu konstanta ``REASON_*`` (LLR-A6).
            candidates_count: Jumlah taksi kandidat (0 untuk no-candidates).

        Returns:
            Jalur file JSON yang ditulis, atau ``None``.

        Contoh:
            >>> import tempfile, os
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> from simulation.metrics.episode_record import REASON_NO_CANDIDATES
            >>> r = RunReporter(RunPaths(tempfile.mkdtemp(), sim_id="nn"),
            ...                 default_config())
            >>> p = r.record_no_negotiation(0, 60.0, 1, 0, REASON_NO_CANDIDATES)
            >>> os.path.isfile(p)
            True
            >>> r._fleet_util_all, r._spklu_util_all
            ([0.0], [0.0])
        """
        period_index = self._next_period_index(day_index)
        record = build_no_negotiation_record(
            self.sim_id,
            day_index,
            period_index,
            sim_time_min,
            hour,
            minute,
            reason_code,
            candidates_count,
        )

        path: Optional[str] = None
        if getattr(self.reporting, "enabled", True) and getattr(
            self.reporting, "write_period_json", True
        ):
            try:
                path = self.run_paths.period_json_path(
                    day_index,
                    period_index,
                    sim_time_min,
                    status=record["status"],
                )
                write_period_json(
                    path, record, int(getattr(self.reporting, "json_indent", 2))
                )
            except Exception as error:  # INV-6
                _log_failure("write no-negotiation json", error)
                path = None

        self._store_period_record(day_index, record)
        self._append_utilities(
            day_index, NO_NEGOTIATION_UTILITY, NO_NEGOTIATION_UTILITY
        )
        return path

    # ------------------------------------------------------------------
    # LLR-N5.5 — snapshot/diff batas hari (#C1) + regen PNG per-hari
    # ------------------------------------------------------------------

    def capture_day_boundary_snapshot(self, day_index: int) -> None:
        """Ambil snapshot metrik bisnis kumulatif di AWAL ``day_index`` (#C1).

        Snapshot ini = batas akhir hari sebelumnya. Metrik HARI d dihitung sebagai
        ``snapshot(awal d+1) - snapshot(awal d)`` di :meth:`on_day_completed`.
        Tidak melakukan apa pun (aman) bila collector tak menyediakan
        ``numeric_metrics_flat`` (INV-5/INV-6).

        Args:
            day_index: Indeks hari yang BARU dimulai (batas awalnya direkam).

        Returns:
            ``None``. Efek samping: mengisi ``_metrics_snapshot_at_day_start`` &
            ``_last_metrics_snapshot``.

        Contoh:
            >>> import tempfile
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> from simulation.metrics.collector import MetricsCollector
            >>> r = RunReporter(RunPaths(tempfile.mkdtemp(), sim_id="snap"),
            ...                 default_config(), collector=MetricsCollector())
            >>> r.capture_day_boundary_snapshot(1)
            >>> isinstance(r._metrics_snapshot_at_day_start.get(1), dict)
            True
        """
        if self.collector is None:
            return
        flat_fn = getattr(self.collector, "numeric_metrics_flat", None)
        if flat_fn is None:
            return
        try:
            snapshot = dict(flat_fn())
        except Exception as error:  # INV-6
            _log_failure("capture day boundary snapshot", error)
            return
        self._metrics_snapshot_at_day_start[day_index] = snapshot
        self._last_metrics_snapshot = snapshot

    def _current_metrics_snapshot(self) -> Dict[str, float]:
        """Kembalikan snapshot metrik kumulatif TERKINI (untuk batas hari terakhir).

        Args:
            None.

        Returns:
            Peta datar metrik kumulatif saat ini (kosong bila tak tersedia).
        """
        if self.collector is None:
            return {}
        flat_fn = getattr(self.collector, "numeric_metrics_flat", None)
        if flat_fn is None:
            return {}
        try:
            return dict(flat_fn())
        except Exception as error:  # INV-6
            _log_failure("read current metrics snapshot", error)
            return {}

    def on_day_completed(self, day_index: int) -> None:
        """Regen PNG metrik per-hari ``day_index`` + diff metrik bisnis (#C1, KR-6).

        No-op bila ``reporting.enabled`` / ``render_day_png`` mati, atau hari sudah
        diregen. Metrik bisnis dunia per-hari dihitung via DIFF snapshot(awal d+1)
        − snapshot(awal d); bila batas d+1 belum tersimpan (mis. hari terakhir saat
        run selesai) dipakai snapshot kumulatif terkini. Metrik berbasis
        period-record (utility per-periode) difilter murni per-hari dari deret
        in-memory. Seluruh render dibungkus try/except (INV-6).

        Guard "hari yang TERJADI": bila hari ini tidak merekam satu pun period
        record (deal / no-deal / no-negotiation), method ini SKIP tanpa membuat
        ``day_dir(day_index)`` maupun merender PNG. Ini mencegah folder hari
        "hantu" (mis. ``day_1`` saat run 1-hari yang clock-nya menyentuh batas
        hari di akhir, padahal hari itu tak pernah berjalan). Konsisten dengan
        kontrak "folder hari = hari yang TERJADI" (LLR-N5.5/N5.6): jaring pengaman
        di :meth:`on_run_finished` hanya melooping hari ber-record, sehingga hari
        kosong memang tak pernah dirender.

        Args:
            day_index: Indeks hari yang baru selesai.

        Returns:
            ``None``. Efek samping: menulis PNG ke ``run_paths.day_dir(day_index)``
            & menandai ``_rendered_days``. Tidak ada efek samping (folder/PNG) bila
            hari itu kosong (0 period record).

        Contoh:
            >>> import tempfile
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> from simulation.metrics.collector import MetricsCollector
            >>> r = RunReporter(RunPaths(tempfile.mkdtemp(), sim_id="day"),
            ...                 default_config(), collector=MetricsCollector())
            >>> r.on_day_completed(0)  # tanpa data periode -> aman, tak crash
        """
        if not getattr(self.reporting, "enabled", True):
            return
        if not getattr(self.reporting, "render_day_png", True):
            return
        if day_index in self._rendered_days:
            return
        # Guard hari "hantu": hari tanpa satu pun period record TIDAK menghasilkan
        # folder/PNG (folder hari = hari yang TERJADI). day_dir() TIDAK dipanggil di
        # sini sebelum guard, sehingga folder tak pernah dibuat untuk hari kosong.
        if not self._periods_by_day.get(day_index):
            return

        # Metrik bisnis per-hari via DIFF (#C1).
        prev = self._metrics_snapshot_at_day_start.get(day_index, {})
        curr = self._metrics_snapshot_at_day_start.get(day_index + 1)
        if curr is None:
            curr = self._current_metrics_snapshot()
        day_business: Optional[Dict[str, float]] = None
        if self.collector is not None:
            diff_fn = getattr(self.collector, "diff_numeric_metrics", None)
            if diff_fn is not None:
                try:
                    day_business = diff_fn(curr, prev)
                except Exception as error:  # INV-6
                    _log_failure("diff numeric metrics", error)
                    day_business = None

        loss_history = getattr(self.collector, "loss_history", None)
        try:
            from simulation.metrics import plots

            out_dir = self.run_paths.day_dir(day_index)
            plots.render_all(
                periods=self._periods_by_day.get(day_index, []),
                summary=None,
                loss_history=loss_history,
                fleet_util_series=self._fleet_util_by_day.get(day_index, []),
                spklu_util_series=self._spklu_util_by_day.get(day_index, []),
                out_dir=out_dir,
                config_reporting=self.reporting,
                scope_label=f"day_{day_index}",
                day_business_flat=day_business,
            )
        except Exception as error:  # INV-6: render gagal tak menjatuhkan run.
            _log_failure(f"render day_{day_index} png", error)

        self._rendered_days.add(day_index)

    # ------------------------------------------------------------------
    # LLR-N5.6 — regen PNG per-simulasi saat run selesai
    # ------------------------------------------------------------------

    def on_run_finished(self) -> None:
        """Regen PNG per-simulasi (kumulatif) + jaring pengaman PNG hari terakhir.

        No-op bila ``reporting.enabled`` / ``render_sim_png`` mati. PNG per-sim
        memakai nilai KUMULATIF (``collector.summary()`` + deret per-sim). Setelah
        itu, untuk tiap hari ber-record yang belum diregen, panggil
        :meth:`on_day_completed` (yang ber-fallback ke snapshot kumulatif terkini
        untuk diff batas hari terakhir, #C1). Seluruh render dibungkus try/except
        (INV-6).

        Args:
            None.

        Returns:
            ``None``. Efek samping: menulis PNG per-sim ke ``run_dir`` & PNG hari
            yang tertinggal.

        Contoh:
            >>> import tempfile
            >>> from simulation.config.default import default_config
            >>> from simulation.metrics.run_paths import RunPaths
            >>> from simulation.metrics.collector import MetricsCollector
            >>> r = RunReporter(RunPaths(tempfile.mkdtemp(), sim_id="fin"),
            ...                 default_config(), collector=MetricsCollector())
            >>> r.on_run_finished()  # tanpa data -> aman, tak crash
        """
        if not getattr(self.reporting, "enabled", True):
            return
        if not getattr(self.reporting, "render_sim_png", True):
            return

        summary: Optional[Dict[str, Any]] = None
        if self.collector is not None:
            summary_fn = getattr(self.collector, "summary", None)
            if summary_fn is not None:
                try:
                    summary = summary_fn()
                except Exception as error:  # INV-6
                    _log_failure("read summary", error)
                    summary = None
        loss_history = getattr(self.collector, "loss_history", None)

        try:
            from simulation.metrics import plots

            out_dir = self.run_paths.ensure_run_dir()
            plots.render_all(
                periods=self._all_periods,
                summary=summary,
                loss_history=loss_history,
                fleet_util_series=self._fleet_util_all,
                spklu_util_series=self._spklu_util_all,
                out_dir=out_dir,
                config_reporting=self.reporting,
                scope_label="run",
                day_business_flat=None,
            )
        except Exception as error:  # INV-6
            _log_failure("render run (per-simulation) png", error)

        # Jaring pengaman: regen PNG hari yang belum diregen (run berhenti sebelum
        # ganti hari). on_day_completed sudah ber-fallback snapshot terkini (#C1).
        for day_index in sorted(self._periods_by_day.keys()):
            if day_index not in self._rendered_days:
                self.on_day_completed(day_index)
