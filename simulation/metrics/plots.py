"""Render semua grafik PNG untuk pelaporan "runs" (KOMPONEN N4 — LLR-N4.0..N4.7).

Modul ini menghasilkan SELURUH PNG metrik & log negosiasi: ringkasan tanh
konsesi, figur per-episode 4-subplot, utility per-periode per pihak, tiga
kelompok metrik bisnis 06-03 (per-simulasi kumulatif & per-hari diff), dan
grafik loss per ``(side, phase)`` terpisah. Orkestrator :func:`render_all`
memanggil seluruh render dan dipakai oleh ``RunReporter`` (KOMPONEN N5).

Invariant yang ditegakkan modul ini:

* **INV-9** — matplotlib WAJIB backend non-interaktif ``"Agg"``: modul ini
  memanggil ``matplotlib.use("Agg")`` SEBELUM ``import matplotlib.pyplot``.
  Tidak ada ``plt.show()``; setiap figur ditutup via ``plt.close(fig)``.
* **INV-3** — gaya/perhitungan figur 4-subplot & ringkasan tanh PERSIS seperti
  ``.claude/references/project_old_reference/evaluation/episode_plots.py``
  (``plot_episode`` & ``plot_per_episode_summary``). Rumus konsesi diambil dari
  :mod:`simulation.metrics.concession_metrics` (port persis, KOMPONEN N3).
* **INV-1** — hanya bergantung pada ``matplotlib`` / ``numpy`` /
  :mod:`simulation.metrics.concession_metrics`; TIDAK mengimpor ``negotiation/*``
  maupun ``simulation/world``.
* **INV-6** — fungsi render BOLEH melempar exception agar bisa di-log; pemanggil
  (``RunReporter``, KOMPONEN N5) membungkus tiap pemanggilan dengan try/except
  sehingga kegagalan grafik tidak menjatuhkan run. Khusus :func:`render_all`
  membungkus tiap sub-render dengan try/except (sesuai LLR-N4.7).

Catatan: dpi & ukuran figur diterima sebagai ARGUMEN fungsi (default dari
``ReportingConfig``: ``png_dpi``, ``figure_width_in``, ``figure_height_in``);
modul ini TIDAK mengimpor config (lapisan pemanggil yang meneruskannya).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # INV-9: backend non-interaktif WAJIB sebelum import pyplot

import matplotlib.pyplot as plt  # noqa: E402  (harus setelah matplotlib.use)
import numpy as np  # noqa: E402

from simulation.metrics.concession_metrics import (  # noqa: E402
    compute_per_episode_tanh_metrics,
)


# ---------------------------------------------------------------------------
# Konstanta lokal
# ---------------------------------------------------------------------------

#: Status periode yang BUKAN negosiasi (figur per-episode dilewati).
_STATUS_NO_NEGOTIATION = "no_negotiation"

#: Aktor (pihak) yang dirender per-figur tanh / per-episode.
_ACTORS: Tuple[str, str] = ("fleet", "spklu")

#: Kombinasi (side, phase) loss yang dirender TERPISAH (KR-9). Termasuk fase
#: ``utility_threshold`` untuk strategi DLST.
_LOSS_COMBINATIONS: Tuple[Tuple[str, str], ...] = (
    ("fleet", "propose"),
    ("fleet", "respond"),
    ("spklu", "propose"),
    ("spklu", "respond"),
    ("fleet", "utility_threshold"),
    ("spklu", "utility_threshold"),
)

#: Ambang CRI* yang digambar sebagai garis acuan pada subplot kecepatan konsesi
#: (gaya referensi lama: theta=0.1). Sumber visual saja, bukan perhitungan.
_CRI_STAR_REFERENCE_THETA = 0.1


# ---------------------------------------------------------------------------
# Helper internal
# ---------------------------------------------------------------------------


def _save(fig: Any, path: str, dpi: int) -> str:
    """Simpan satu figur ke ``path`` lalu tutup figur (INV-9).

    Membuat direktori induk bila belum ada, menyimpan dengan ``bbox_inches=
    "tight"``, lalu memanggil ``plt.close(fig)`` agar memori figur dilepas dan
    tidak ada window interaktif yang dibuka.

    Args:
        fig: Objek ``matplotlib.figure.Figure`` yang akan disimpan.
        path: Jalur file PNG tujuan (absolut atau relatif).
        dpi: Resolusi dots-per-inch saat menyimpan.

    Returns:
        ``path`` (jalur file yang ditulis).

    Example:
        >>> import os, tempfile
        >>> fig = plt.figure()
        >>> p = os.path.join(tempfile.mkdtemp(), "x.png")
        >>> _save(fig, p, 100) == p
        True
        >>> os.path.exists(p)
        True
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _log_render_error(context: str, error: Exception) -> None:
    """Catat kegagalan render ke ``stderr`` tanpa menjatuhkan run (INV-6).

    Args:
        context: Deskripsi singkat tahap render yang gagal.
        error: Exception yang ditangkap.

    Example:
        >>> _log_render_error("demo", ValueError("boom"))  # doctest: +SKIP
    """
    print(f"[plots] render gagal ({context}): {error}", file=sys.stderr)


def _propose_utils_for_actor(period: Dict[str, Any], actor: str) -> List[float]:
    """Ambil deret propose-utility milik ``actor`` dari satu period record (LLR-N4.1).

    Memfilter ``period["steps"]`` untuk langkah ber-``kind == "propose"`` milik
    ``actor`` dengan ``utility`` non-``None``, dalam urutan kemunculan.

    Args:
        period: Period record (hasil KOMPONEN N2 build_period_record).
        actor: ``"fleet"`` atau ``"spklu"``.

    Returns:
        Daftar ``float`` utility proposal aktor tersebut.

    Example:
        >>> period = {"steps": [
        ...     {"actor": "fleet", "kind": "propose", "utility": 0.9},
        ...     {"actor": "spklu", "kind": "respond", "utility": 0.3},
        ...     {"actor": "fleet", "kind": "propose", "utility": 0.7},
        ... ]}
        >>> _propose_utils_for_actor(period, "fleet")
        [0.9, 0.7]
        >>> _propose_utils_for_actor(period, "spklu")
        []
    """
    utils: List[float] = []
    for step in period.get("steps", []):
        if (
            step.get("kind") == "propose"
            and step.get("actor") == actor
            and step.get("utility") is not None
        ):
            utils.append(float(step["utility"]))
    return utils


def _all_utils_in_order(period: Dict[str, Any]) -> List[float]:
    """Kembalikan semua utility per langkah dalam urutan step (propose+respond).

    Args:
        period: Period record dengan field ``steps`` terurut.

    Returns:
        Daftar ``float`` seluruh utility non-``None`` dalam urutan langkah.

    Example:
        >>> period = {"steps": [
        ...     {"kind": "propose", "utility": 0.9},
        ...     {"kind": "respond", "utility": None},
        ...     {"kind": "respond", "utility": 0.4},
        ... ]}
        >>> _all_utils_in_order(period)
        [0.9, 0.4]
    """
    utils: List[float] = []
    for step in period.get("steps", []):
        u = step.get("utility")
        if u is not None:
            utils.append(float(u))
    return utils


def _is_negotiation_period(period: Dict[str, Any]) -> bool:
    """True bila period adalah negosiasi (deal/no_deal), bukan no_negotiation.

    Args:
        period: Period record dengan field ``status``.

    Returns:
        ``False`` bila ``status == "no_negotiation"``, ``True`` lainnya.

    Example:
        >>> _is_negotiation_period({"status": "deal"})
        True
        >>> _is_negotiation_period({"status": "no_negotiation"})
        False
    """
    return period.get("status") != _STATUS_NO_NEGOTIATION


# ---------------------------------------------------------------------------
# LLR-N4.2 — Per-episode 4-subplot (gaya episode_plots.plot_episode)
# ---------------------------------------------------------------------------


def plot_period_episode(
    period: Dict[str, Any],
    actor: str,
    episode_idx: int,
    tanh_metrics: Dict[str, Any],
    out_dir: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Optional[str]:
    """Render figur per-episode 4-subplot untuk satu periode & aktor (LLR-N4.2).

    Gaya & perhitungan PERSIS ``episode_plots.plot_episode`` (INV-3):
      1. Trajektori propose-utility aktor + kurva fit tanh.
      2. Semua utility per langkah: propose (aktor) vs respond (received).
      3. Cumulative utility (``np.cumsum`` seluruh utility terurut step).
      4. Profil kecepatan konsesi dari tanh
         (``speed = |a*b| / cosh(a*xs - c)^2`` ternormalkan + garis theta).

    Periode ``no_negotiation`` dilewati (mengembalikan ``None``).

    Args:
        period: Period record (KOMPONEN N2).
        actor: ``"fleet"`` / ``"spklu"`` — pihak yang dijadikan learner figur ini.
        episode_idx: Indeks episode (untuk judul & nama file).
        tanh_metrics: Hasil
            :func:`simulation.metrics.concession_metrics.compute_per_episode_tanh_metrics`
            untuk ``actor`` di periode ini (berisi ``tanh_params``,
            ``fitted_curve``, ``_x_dense``, ``cri``, ``cri_star``, ``burstiness``).
        out_dir: Direktori tujuan PNG.
        dpi: Resolusi simpan.
        figsize: ``(width_in, height_in)`` ukuran figur.

    Returns:
        Jalur PNG yang dibuat, atau ``None`` bila periode no_negotiation.

    Example:
        >>> import tempfile
        >>> period = {"status": "deal", "agreement": {"index": 0}, "steps": [
        ...     {"actor": "fleet", "kind": "propose", "utility": 0.9},
        ...     {"actor": "spklu", "kind": "respond", "utility": 0.3},
        ...     {"actor": "fleet", "kind": "propose", "utility": 0.6},
        ... ]}
        >>> from simulation.metrics.concession_metrics import (
        ...     compute_per_episode_tanh_metrics)
        >>> tm = compute_per_episode_tanh_metrics([0.9, 0.6])
        >>> p = plot_period_episode(period, "fleet", 0, tm,
        ...     tempfile.mkdtemp(), 90, (8.0, 6.0))
        >>> p.endswith("episode_fleet_0000.png")
        True
    """
    if not _is_negotiation_period(period):
        return None

    steps = period.get("steps", [])
    agreement = period.get("agreement")
    propose_utils = tanh_metrics.get("propose_utils", [])
    all_utils = _all_utils_in_order(period)
    tanh_params = tanh_metrics.get("tanh_params")
    fitted_curve = tanh_metrics.get("fitted_curve")
    x_dense = tanh_metrics.get("_x_dense")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Episode {episode_idx} ({actor})  |  "
        f"Agreement: {'YES' if agreement is not None else 'NO'}  |  "
        f"Steps: {len(steps)}",
        fontsize=12,
        fontweight="bold",
    )

    # ---- Subplot 1: Learner proposal utility over turns ----
    ax1 = axes[0, 0]
    if propose_utils:
        ax1.plot(propose_utils, "bo-", label="Proposal utility", linewidth=2)
    if fitted_curve is not None and x_dense is not None:
        ax1.plot(x_dense, fitted_curve, "r--", label="Tanh fit", linewidth=1.5)

    params_str = ""
    if tanh_params is not None:
        a, b, c, d = tanh_params
        params_str = (
            f"a={a:.3f}, b={b:.3f}\n"
            f"CRI={tanh_metrics['cri']:.3f}, "
            f"τ={tanh_metrics['burstiness']:.3f}"
        )
    ax1.set_title(f"Proposal Utility Trajectory\n{params_str}", fontsize=9)
    ax1.set_xlabel("Proposal step")
    ax1.set_ylabel("Utility")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Subplot 2: All utilities in step order ----
    ax2 = axes[0, 1]
    if all_utils:
        propose_xs = [
            i
            for i, s in enumerate(steps)
            if s.get("kind") == "propose" and s.get("utility") is not None
        ]
        respond_xs = [
            i
            for i, s in enumerate(steps)
            if s.get("kind") == "respond" and s.get("utility") is not None
        ]
        p_utils = [
            s["utility"]
            for s in steps
            if s.get("kind") == "propose" and s.get("utility") is not None
        ]
        r_utils = [
            s["utility"]
            for s in steps
            if s.get("kind") == "respond" and s.get("utility") is not None
        ]

        if propose_xs and p_utils:
            ax2.plot(propose_xs, p_utils, "b^-", label="Propose", markersize=6)
        if respond_xs and r_utils:
            ax2.plot(
                respond_xs,
                r_utils,
                "rs-",
                label="Respond (received)",
                markersize=6,
            )

    ax2.set_title("All Steps: Utility per Turn", fontsize=9)
    ax2.set_xlabel("Step index")
    ax2.set_ylabel("Utility")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Subplot 3: Cumulative utility (proxy for reward) ----
    ax3 = axes[1, 0]
    if all_utils:
        cumsum = np.cumsum(all_utils)
        ax3.plot(cumsum, "g-o", label="Cumulative utility", linewidth=2)
    ax3.set_title("Cumulative Utility (Reward Proxy)", fontsize=9)
    ax3.set_xlabel("Step index")
    ax3.set_ylabel("Cumulative utility")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Subplot 4: Concession speed profile from tanh ----
    ax4 = axes[1, 1]
    if tanh_params is not None and x_dense is not None:
        a, b, c, d = tanh_params
        xs = np.array(x_dense)
        # Instantaneous speed = |a*b| * sech²(ax - c)
        speed = abs(a * b) / np.cosh(a * xs - c) ** 2
        max_sp = speed.max() if speed.max() > 0 else 1.0
        speed_norm = speed / max_sp
        ax4.fill_between(xs, speed_norm, alpha=0.4, color="purple")
        ax4.plot(xs, speed_norm, "purple", linewidth=1.5)
        ax4.axhline(
            _CRI_STAR_REFERENCE_THETA,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="θ=0.1 (CRI* threshold)",
        )
        cri_str = (
            f"CRI={tanh_metrics['cri']:.3f}  "
            f"CRI*={tanh_metrics['cri_star']:.3f}"
        )
        ax4.set_title(f"Concession Speed Profile\n{cri_str}", fontsize=9)
    else:
        ax4.set_title(
            "Concession Speed Profile\n(insufficient data)", fontsize=9
        )

    ax4.set_xlabel("Step")
    ax4.set_ylabel("Normalised speed")
    handles, _labels = ax4.get_legend_handles_labels()
    if handles:
        ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"episode_{actor}_{episode_idx:04d}.png")
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# LLR-N4.3 — Tanh metrics summary time-series
# ---------------------------------------------------------------------------


def plot_tanh_summary(
    all_tanh_metrics: List[Dict[str, Any]],
    episode_indices: List[int],
    actor: str,
    out_dir: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Optional[str]:
    """Render ringkasan deret waktu metrik tanh untuk satu aktor (LLR-N4.3).

    Empat subplot — CRI, CRI*, burstiness (τ), anchor distance — PERSIS gaya
    ``episode_plots.plot_per_episode_summary`` (INV-3), termasuk garis rata-rata
    per panel.

    Args:
        all_tanh_metrics: Daftar dict metrik tanh per episode (berisi ``cri``,
            ``cri_star``, ``burstiness``, ``anchor_distance``).
        episode_indices: Indeks episode untuk sumbu-x (sejajar
            ``all_tanh_metrics``).
        actor: ``"fleet"`` / ``"spklu"`` (untuk suffix nama file & judul).
        out_dir: Direktori tujuan PNG.
        dpi: Resolusi simpan.
        figsize: ``(width_in, height_in)`` ukuran figur.

    Returns:
        Jalur PNG yang dibuat, atau ``None`` bila ``all_tanh_metrics`` kosong.

    Example:
        >>> import tempfile
        >>> from simulation.metrics.concession_metrics import (
        ...     compute_per_episode_tanh_metrics)
        >>> tms = [compute_per_episode_tanh_metrics([0.9, 0.7, 0.5, 0.3, 0.1])]
        >>> p = plot_tanh_summary(tms, [0], "fleet",
        ...     tempfile.mkdtemp(), 90, (8.0, 6.0))
        >>> p.endswith("tanh_metrics_summary_fleet.png")
        True
    """
    if not all_tanh_metrics:
        return None

    cris = [m["cri"] for m in all_tanh_metrics]
    cri_stars = [m["cri_star"] for m in all_tanh_metrics]
    burstiness = [m["burstiness"] for m in all_tanh_metrics]
    anchors = [m["anchor_distance"] for m in all_tanh_metrics]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Per-Episode Tanh Concession Metrics ({actor})",
        fontsize=12,
        fontweight="bold",
    )

    def _plot(
        ax: Any,
        values: List[float],
        title: str,
        ylabel: str,
        color: str = "steelblue",
    ) -> None:
        """Gambar satu panel ringkasan (deret + garis rata-rata)."""
        ax.plot(
            episode_indices,
            values,
            "o-",
            color=color,
            markersize=4,
            linewidth=1.5,
        )
        if values:
            ax.axhline(
                np.mean(values),
                color="red",
                linestyle="--",
                linewidth=1,
                label=f"Mean={np.mean(values):.3f}",
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], cris, "CRI (Concession-Rigidity Index)", "CRI", "steelblue")
    _plot(axes[0, 1], cri_stars, "CRI* (Data-driven)", "CRI*", "darkorange")
    _plot(axes[1, 0], burstiness, "Burstiness (τ)", "τ", "green")
    _plot(
        axes[1, 1],
        anchors,
        "Anchor Distance (first proposal utility)",
        "Utility",
        "purple",
    )

    fig.tight_layout()
    path = os.path.join(out_dir, f"tanh_metrics_summary_{actor}.png")
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# LLR-N4.4 — Utility per episode/periode per pihak
# ---------------------------------------------------------------------------


def plot_period_utilities(
    fleet_series: List[float],
    spklu_series: List[float],
    out_dir: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Optional[str]:
    """Render grafik utility per-periode untuk fleet & SPKLU (LLR-N4.4 / KR-6,7).

    Garis fleet & SPKLU terhadap indeks periode. Periode tanpa-negosiasi muncul
    sebagai 0.0 (KR-7) karena deret yang dipassing sudah memuat 0 (lihat M5.3 /
    N5.3); fungsi ini tidak mengubah nilai.

    Args:
        fleet_series: Deret utility periodik pihak fleet.
        spklu_series: Deret utility periodik pihak SPKLU.
        out_dir: Direktori tujuan PNG.
        dpi: Resolusi simpan.
        figsize: ``(width_in, height_in)`` ukuran figur.

    Returns:
        Jalur PNG yang dibuat, atau ``None`` bila kedua deret kosong.

    Example:
        >>> import tempfile
        >>> p = plot_period_utilities([0.5, -1.0, 0.0], [0.4, -1.0, 0.0],
        ...     tempfile.mkdtemp(), 90, (8.0, 4.0))
        >>> p.endswith("utility_per_period.png")
        True
    """
    if not fleet_series and not spklu_series:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    if fleet_series:
        ax.plot(
            range(len(fleet_series)),
            fleet_series,
            "o-",
            color="steelblue",
            markersize=4,
            linewidth=1.5,
            label="Fleet",
        )
    if spklu_series:
        ax.plot(
            range(len(spklu_series)),
            spklu_series,
            "s-",
            color="darkorange",
            markersize=4,
            linewidth=1.5,
            label="SPKLU",
        )
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Utility per Negotiation Period", fontsize=11, fontweight="bold")
    ax.set_xlabel("Period index")
    ax.set_ylabel("Terminal utility")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "utility_per_period.png")
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# LLR-N4.5 — Tiga kelompok metrik 06-03 (per-sim kumulatif & per-hari diff)
# ---------------------------------------------------------------------------


def _business_panels_from_summary(
    summary: Dict[str, Any],
) -> Tuple[
    List[Tuple[str, float]],
    List[Tuple[str, float]],
    List[Tuple[str, float]],
]:
    """Susun tiga panel metrik bisnis dari ``collector.summary()`` kumulatif.

    Args:
        summary: Struktur bertingkat hasil ``MetricsCollector.summary()`` dengan
            kunci ``spklu_business`` / ``fleet_business`` / ``negotiation_system``.

    Returns:
        Tuple ``(spklu_items, fleet_items, system_items)``; tiap item adalah
        ``(label, value)``.

    Example:
        >>> from simulation.metrics.collector import MetricsCollector
        >>> s = MetricsCollector().summary()
        >>> sp, fl, sy = _business_panels_from_summary(s)
        >>> ("total_revenue", 0.0) in sp
        True
    """
    spklu = summary["spklu_business"]
    fleet = summary["fleet_business"]
    system = summary["negotiation_system"]

    spklu_items: List[Tuple[str, float]] = [
        ("total_revenue", float(spklu["total_revenue"])),
        ("total_compensation_paid", float(spklu["total_compensation_paid"])),
        ("shifted_bookings_count", float(spklu["shifted_bookings_count"])),
        (
            "port_hours_used",
            float(spklu["port_utilization"]["port_hours_used"]),
        ),
        ("avg_price_per_kwh", float(spklu["avg_price_per_kwh"])),
    ]
    fleet_items: List[Tuple[str, float]] = [
        ("passenger_revenue", float(fleet["passenger_revenue"])),
        ("charging_cost", float(fleet["charging_cost"])),
        ("unproductive_km", float(fleet["unproductive_km"])),
        ("orders_served", float(fleet["orders_served"])),
        ("orders_unserved", float(fleet["orders_unserved"])),
        ("taxis_died", float(fleet["taxis_died"])),
        ("emergency_rescue_count", float(fleet["emergency_rescue_count"])),
        ("slot_forfeit_count", float(fleet["slot_forfeit_count"])),
    ]
    system_items: List[Tuple[str, float]] = [
        ("negotiations_total", float(system["negotiations_total"])),
        ("deal_success_rate", float(system["deal_success_rate"])),
        ("partial_deal_rate", float(system["partial_deal_rate"])),
        ("avg_cycles_waited", float(system["avg_cycles_waited"])),
        (
            "fleet_avg_utility",
            float(system["fleet"]["avg_utility_per_negotiation"]),
        ),
        (
            "spklu_avg_utility",
            float(system["spklu"]["avg_utility_per_negotiation"]),
        ),
    ]
    return spklu_items, fleet_items, system_items


def _business_panels_from_flat(
    flat: Dict[str, float],
) -> Tuple[
    List[Tuple[str, float]],
    List[Tuple[str, float]],
    List[Tuple[str, float]],
]:
    """Susun tiga panel metrik bisnis PER-HARI dari peta datar diff (#C1).

    Memakai akumulator dunia hasil ``diff_numeric_metrics`` (selisih per-hari).
    Metrik rasio/rata-rata non-diffable diturunkan dari count per-hari bila
    memungkinkan (mis. ``deal_success_rate = deals_diff / negotiations_diff``);
    karena flat hanya memuat akumulator, panel sistem menampilkan count per-hari
    & total utility per-hari (rasio tidak dipaksakan bila tak ada count deal).

    Args:
        flat: Peta datar ``{path: value}`` hasil
            ``diff_numeric_metrics(curr, prev)`` untuk satu hari.

    Returns:
        Tuple ``(spklu_items, fleet_items, system_items)``; tiap item
        ``(label, value)``.

    Example:
        >>> flat = {
        ...     "spklu_business.total_revenue": 100.0,
        ...     "fleet_business.charging_cost": 80.0,
        ...     "negotiation_system.negotiations_total": 3.0,
        ... }
        >>> sp, fl, sy = _business_panels_from_flat(flat)
        >>> ("total_revenue", 100.0) in sp
        True
    """

    def _g(key: str) -> float:
        return float(flat.get(key, 0.0))

    spklu_items: List[Tuple[str, float]] = [
        ("total_revenue", _g("spklu_business.total_revenue")),
        (
            "total_compensation_paid",
            _g("spklu_business.total_compensation_paid"),
        ),
        ("shifted_bookings_count", _g("spklu_business.shifted_bookings_count")),
        (
            "port_hours_used",
            _g("spklu_business.port_utilization.port_hours_used"),
        ),
    ]
    fleet_items: List[Tuple[str, float]] = [
        ("passenger_revenue", _g("fleet_business.passenger_revenue")),
        ("charging_cost", _g("fleet_business.charging_cost")),
        ("unproductive_km", _g("fleet_business.unproductive_km")),
        ("orders_served", _g("fleet_business.orders_served")),
        ("orders_unserved", _g("fleet_business.orders_unserved")),
        ("taxis_died", _g("fleet_business.taxis_died")),
        (
            "emergency_rescue_count",
            _g("fleet_business.emergency_rescue_count"),
        ),
        ("slot_forfeit_count", _g("fleet_business.slot_forfeit_count")),
    ]
    system_items: List[Tuple[str, float]] = [
        ("negotiations_total", _g("negotiation_system.negotiations_total")),
        ("fleet_total_utility", _g("negotiation_system.fleet.total_utility")),
        ("spklu_total_utility", _g("negotiation_system.spklu.total_utility")),
        (
            "payload_count",
            _g("negotiation_system.negotiation_payload_count"),
        ),
    ]
    return spklu_items, fleet_items, system_items


def _draw_business_panel(
    ax: Any, items: List[Tuple[str, float]], title: str
) -> None:
    """Gambar satu panel bar-chart metrik bisnis pada axis ``ax``.

    Args:
        ax: Axis matplotlib tujuan.
        items: Daftar ``(label, value)``.
        title: Judul panel.

    Example:
        >>> fig, ax = plt.subplots()
        >>> _draw_business_panel(ax, [("a", 1.0), ("b", 2.0)], "demo")
        >>> plt.close(fig)
    """
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    positions = range(len(labels))
    ax.bar(positions, values, color="steelblue")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


def plot_business_metrics(
    summary: Optional[Dict[str, Any]],
    day_business_flat: Optional[Dict[str, float]],
    scope_label: str,
    out_dir: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Optional[str]:
    """Render tiga kelompok metrik bisnis 06-03 dalam satu figur (LLR-N4.5).

    Dua jalur sumber data (#C1):

    * **Per-simulasi** (``scope_label == "run"``): pakai ``summary`` KUMULATIF
      dari ``collector.summary()`` (struktur bertingkat).
    * **Per-hari** (``scope_label`` diawali ``"day_"``): pakai
      ``day_business_flat`` (peta datar selisih ``diff_numeric_metrics``) yang
      memuat akumulator dunia HANYA hari itu.

    Tiga panel: SPKLU, Fleet, Sistem (bar chart). Judul menyertakan
    ``scope_label`` (kumulatif vs per hari). Tidak ada rumus yang diubah —
    hanya pelaporan/agregasi.

    Args:
        summary: ``collector.summary()`` kumulatif (dipakai bila scope ``run``).
        day_business_flat: Peta datar diff per-hari (dipakai bila scope ``day_*``).
        scope_label: ``"run"`` atau ``"day_<i>"``.
        out_dir: Direktori tujuan PNG.
        dpi: Resolusi simpan.
        figsize: ``(width_in, height_in)`` ukuran figur.

    Returns:
        Jalur PNG ``business_metrics.png``, atau ``None`` bila kedua sumber data
        kosong/tak relevan untuk scope.

    Example:
        >>> import tempfile
        >>> from simulation.metrics.collector import MetricsCollector
        >>> p = plot_business_metrics(MetricsCollector().summary(), None,
        ...     "run", tempfile.mkdtemp(), 90, (10.0, 8.0))
        >>> p.endswith("business_metrics.png")
        True
    """
    use_day = scope_label.startswith("day_")
    if use_day:
        if day_business_flat is None:
            return None
        spklu_items, fleet_items, system_items = _business_panels_from_flat(
            day_business_flat
        )
        scope_note = f"{scope_label} (per hari)"
    else:
        if summary is None:
            return None
        spklu_items, fleet_items, system_items = _business_panels_from_summary(
            summary
        )
        scope_note = f"{scope_label} (kumulatif)"

    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(
        f"Business metrics — {scope_note}",
        fontsize=12,
        fontweight="bold",
    )
    _draw_business_panel(axes[0], spklu_items, "SPKLU business")
    _draw_business_panel(axes[1], fleet_items, "Fleet business")
    _draw_business_panel(axes[2], system_items, "Negotiation / system")

    fig.tight_layout()
    path = os.path.join(out_dir, "business_metrics.png")
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# LLR-N4.6 — Loss per update() per pihak-per-fase (TERPISAH dari utility)
# ---------------------------------------------------------------------------


def plot_losses(
    loss_history: Any,
    out_dir: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> List[str]:
    """Render grafik loss per ``(side, phase)`` TERPISAH (LLR-N4.6 / KR-8,9).

    Untuk setiap kombinasi di ``[(fleet,propose), (fleet,respond),
    (spklu,propose), (spklu,respond), (fleet,utility_threshold),
    (spklu,utility_threshold)]``:

    * ``series = loss_history.get(side, phase)``.
    * Bila series KOSONG: SKIP (tidak membuat figur) — KR-9: kombinasi kosong
      tidak menghasilkan figur. Tanpa exception.
    * Bila ada: figur tunggal dengan sumbu-x = indeks update (``range(len)``),
      sumbu-y = loss. Sumbu-x adalah indeks ``update()`` (BUKAN episode), menegaskan
      pemisahan dari grafik utility.

    Tidak ada figur yang memplot loss & utility bersama (KR-9).

    Args:
        loss_history: Objek dengan metode ``get(side, phase) -> List[float]``
            (KOMPONEN N6 ``LossHistory``).
        out_dir: Direktori tujuan PNG.
        dpi: Resolusi simpan.
        figsize: ``(width_in, height_in)`` ukuran figur.

    Returns:
        Daftar jalur PNG yang dibuat (mungkin kosong bila semua series kosong).

    Example:
        >>> import tempfile
        >>> from simulation.metrics.loss_history import LossHistory
        >>> lh = LossHistory()
        >>> lh.record("fleet", "propose", 0.5)
        >>> lh.record("fleet", "propose", 0.3)
        >>> paths = plot_losses(lh, tempfile.mkdtemp(), 90, (8.0, 4.0))
        >>> len(paths)
        1
        >>> paths[0].endswith("loss_fleet_propose.png")
        True
    """
    created: List[str] = []
    for side, phase in _LOSS_COMBINATIONS:
        series = loss_history.get(side, phase)
        if not series:
            continue  # KR-9: kombinasi kosong -> tidak ada figur

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            range(len(series)),
            series,
            "-",
            color="crimson",
            linewidth=1.5,
        )
        ax.set_title(
            f"Loss — {side} {phase} (per update())",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("update() index")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(out_dir, f"loss_{side}_{phase}.png")
        created.append(_save(fig, path, dpi))
    return created


# ---------------------------------------------------------------------------
# LLR-N4.7 — Orkestrator render-set
# ---------------------------------------------------------------------------


def _figsize_from_config(config_reporting: Any) -> Tuple[float, float]:
    """Ambil ``(width_in, height_in)`` dari ``ReportingConfig`` (LLR-N4.7).

    Args:
        config_reporting: Objek ``ReportingConfig`` (atau yang punya
            ``figure_width_in`` & ``figure_height_in``).

    Returns:
        Tuple ``(figure_width_in, figure_height_in)``.

    Example:
        >>> class _Cfg:
        ...     figure_width_in = 14.0
        ...     figure_height_in = 8.0
        >>> _figsize_from_config(_Cfg())
        (14.0, 8.0)
    """
    return (
        float(config_reporting.figure_width_in),
        float(config_reporting.figure_height_in),
    )


def _collect_actor_tanh_metrics(
    negotiation_periods: List[Dict[str, Any]],
    actor: str,
    theta: float,
    min_points: int,
) -> List[Dict[str, Any]]:
    """Hitung metrik tanh per-episode untuk satu aktor (burstiness populasi).

    Pass-1: fit semua episode untuk mengumpulkan ``all_params``. Pass-2: hitung
    ulang dengan ``all_params`` agar burstiness ternormalkan tingkat populasi
    (sesuai LLR-N4.7 "all_params untuk burstiness"). Rumus identik N3 (INV-3).

    Args:
        negotiation_periods: Periode negosiasi (deal/no_deal saja).
        actor: ``"fleet"`` / ``"spklu"``.
        theta: Ambang CRI* (``config.reporting.cri_star_theta``).
        min_points: Minimum titik fit (``config.reporting.min_points_for_tanh_fit``).

    Returns:
        Daftar dict metrik tanh per episode (sejajar ``negotiation_periods``).

    Example:
        >>> periods = [{"status": "deal", "steps": [
        ...     {"actor": "fleet", "kind": "propose", "utility": u}
        ...     for u in (0.9, 0.7, 0.5, 0.3, 0.1)]}]
        >>> tms = _collect_actor_tanh_metrics(periods, "fleet", 0.1, 4)
        >>> len(tms)
        1
    """
    propose_utils_list = [
        _propose_utils_for_actor(p, actor) for p in negotiation_periods
    ]

    # Pass-1: kumpulkan parameter fit populasi untuk burstiness.
    all_params: List[Optional[Tuple[float, float, float, float]]] = []
    for utils in propose_utils_list:
        m = compute_per_episode_tanh_metrics(
            utils, all_params=None, theta=theta, min_points=min_points
        )
        all_params.append(m["tanh_params"])

    # Pass-2: hitung final dengan burstiness ternormalkan populasi.
    metrics: List[Dict[str, Any]] = []
    for utils in propose_utils_list:
        metrics.append(
            compute_per_episode_tanh_metrics(
                utils,
                all_params=all_params,
                n_steps=len(utils),
                theta=theta,
                min_points=min_points,
            )
        )
    return metrics


def render_all(
    periods: List[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    loss_history: Any,
    fleet_util_series: List[float],
    spklu_util_series: List[float],
    out_dir: str,
    config_reporting: Any,
    scope_label: str = "run",
    day_business_flat: Optional[Dict[str, float]] = None,
) -> Dict[str, List[str]]:
    """Orkestrasi seluruh render PNG ke ``out_dir`` (LLR-N4.7).

    Untuk tiap aktor (``fleet`` & ``spklu``): kumpulkan metrik tanh per-episode
    (via :func:`_collect_actor_tanh_metrics` dengan ``all_params`` untuk
    burstiness), render figur per-episode 4-subplot tiap periode negosiasi, lalu
    ringkasan tanh. Setelah itu render utility per-periode, metrik bisnis 3 grup,
    dan loss per fase.

    Sumber metrik bisnis (#C1): bila ``scope_label == "run"`` pakai ``summary``
    kumulatif (``day_business_flat`` diabaikan); bila ``scope_label`` diawali
    ``"day_"`` pakai ``day_business_flat`` (selisih per-hari). Loss/tanh/utility
    selalu dari deret yang dipassing (per-hari atau per-sim sesuai pemanggil N5).

    Tiap sub-render dibungkus try/except (log ke stderr, lanjut) agar kegagalan
    satu grafik tidak menjatuhkan run (INV-6).

    Args:
        periods: Daftar period record (deal/no_deal/no_negotiation).
        summary: ``collector.summary()`` kumulatif (untuk scope ``run``).
        loss_history: Objek ``LossHistory`` (KOMPONEN N6).
        fleet_util_series: Deret utility periodik fleet (sudah memuat 0 no-nego).
        spklu_util_series: Deret utility periodik SPKLU.
        out_dir: Direktori tujuan PNG (per-hari atau per-sim).
        config_reporting: ``ReportingConfig`` (sumber dpi/figsize/theta/min_points).
        scope_label: ``"run"`` (kumulatif) atau ``"day_<i>"`` (per hari).
        day_business_flat: Peta datar diff per-hari (dipakai bila scope ``day_*``).

    Returns:
        Dict ringkasan jalur yang dibuat dengan kunci: ``episodes``,
        ``tanh_summaries``, ``utilities``, ``business``, ``losses``.

    Example:
        >>> import tempfile
        >>> from simulation.metrics.loss_history import LossHistory
        >>> from simulation.metrics.collector import MetricsCollector
        >>> class _Cfg:
        ...     png_dpi = 90
        ...     figure_width_in = 8.0
        ...     figure_height_in = 6.0
        ...     cri_star_theta = 0.1
        ...     min_points_for_tanh_fit = 4
        >>> period = {"status": "deal", "agreement": {"index": 0}, "steps": [
        ...     {"actor": "fleet", "kind": "propose", "utility": u}
        ...     for u in (0.9, 0.7, 0.5)]}
        >>> out = render_all([period], MetricsCollector().summary(),
        ...     LossHistory(), [0.9], [0.5], tempfile.mkdtemp(), _Cfg())
        >>> "business" in out
        True
    """
    dpi = int(config_reporting.png_dpi)
    figsize = _figsize_from_config(config_reporting)
    theta = float(config_reporting.cri_star_theta)
    min_points = int(config_reporting.min_points_for_tanh_fit)

    created: Dict[str, List[str]] = {
        "episodes": [],
        "tanh_summaries": [],
        "utilities": [],
        "business": [],
        "losses": [],
    }

    negotiation_periods = [p for p in periods if _is_negotiation_period(p)]

    for actor in _ACTORS:
        try:
            tanh_metrics_list = _collect_actor_tanh_metrics(
                negotiation_periods, actor, theta, min_points
            )
        except Exception as error:  # INV-6: jangan jatuhkan run
            _log_render_error(f"tanh_metrics[{actor}]", error)
            continue

        for episode_idx, (period, tanh_metrics) in enumerate(
            zip(negotiation_periods, tanh_metrics_list)
        ):
            try:
                path = plot_period_episode(
                    period,
                    actor,
                    episode_idx,
                    tanh_metrics,
                    out_dir,
                    dpi,
                    figsize,
                )
                if path is not None:
                    created["episodes"].append(path)
            except Exception as error:
                _log_render_error(
                    f"plot_period_episode[{actor}#{episode_idx}]", error
                )

        try:
            summary_path = plot_tanh_summary(
                tanh_metrics_list,
                list(range(len(tanh_metrics_list))),
                actor,
                out_dir,
                dpi,
                figsize,
            )
            if summary_path is not None:
                created["tanh_summaries"].append(summary_path)
        except Exception as error:
            _log_render_error(f"plot_tanh_summary[{actor}]", error)

    try:
        util_path = plot_period_utilities(
            fleet_util_series, spklu_util_series, out_dir, dpi, figsize
        )
        if util_path is not None:
            created["utilities"].append(util_path)
    except Exception as error:
        _log_render_error("plot_period_utilities", error)

    try:
        business_summary = None if scope_label.startswith("day_") else summary
        business_flat = (
            day_business_flat if scope_label.startswith("day_") else None
        )
        business_path = plot_business_metrics(
            business_summary,
            business_flat,
            scope_label,
            out_dir,
            dpi,
            figsize,
        )
        if business_path is not None:
            created["business"].append(business_path)
    except Exception as error:
        _log_render_error("plot_business_metrics", error)

    try:
        created["losses"].extend(
            plot_losses(loss_history, out_dir, dpi, figsize)
        )
    except Exception as error:
        _log_render_error("plot_losses", error)

    return created
