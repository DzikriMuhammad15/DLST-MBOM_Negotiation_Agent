"""Metrik konsesi (tanh / CRI / CRI* / burstiness) untuk satu episode negosiasi.

Modul ini adalah **port PERSIS** rumus dari
`.claude/references/project_old_reference/evaluation/episode_plots.py`
(bagian "Tanh model & fitting" + CRI / CRI* / burstiness). Rumus TIDAK diubah
(INV-3): modul ini hanya menangkap / menghitung metrik, tidak menyentuh
reward/loss.

Satu-satunya adaptasi dibanding referensi lama adalah **sumber input**
`compute_per_episode_tanh_metrics`: ia menerima `propose_utils` (list float)
langsung, bukan `episode_payload` (dict). Ini menjaga INV-1: modul ini TIDAK
mengimpor `negotiation/*` maupun `simulation/world` — ekstraksi utility per
langkah sudah dilakukan di lapisan lain (lihat KOMPONEN N2).

Ketergantungan:
- ``numpy`` WAJIB.
- ``scipy`` OPSIONAL — bila tidak terpasang, ``SCIPY_AVAILABLE`` bernilai
  ``False`` dan ``fit_tanh_curve`` mengembalikan ``None`` (deret/figur tetap
  aman karena pemanggil menangani ``None``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tanh model & fitting
# ---------------------------------------------------------------------------


def tanh_model(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Evaluasi model tanh ``y(x) = d + b * tanh(a*x - c)`` (persis referensi).

    Args:
        x: Titik-titik abscissa (array numpy atau skalar yang kompatibel numpy).
        a: Parameter kecuraman (steepness).
        b: Parameter amplitudo (setengah rentang).
        c: Parameter pergeseran fase.
        d: Parameter offset vertikal.

    Returns:
        ``np.ndarray`` hasil ``d + b * tanh(a*x - c)``.

    Example:
        >>> import numpy as np
        >>> float(tanh_model(np.array([0.0]), 1.0, 1.0, 0.0, 0.0)[0])
        0.0
        >>> vals = tanh_model(np.array([0.0, 100.0]), 1.0, 1.0, 0.0, 5.0)
        >>> round(float(vals[1]), 4)
        6.0
    """
    return d + b * np.tanh(a * x - c)


def fit_tanh_curve(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """Fit ``y(x) = d + b * tanh(a*x - c)`` via nonlinear least squares.

    Implementasi PERSIS referensi lama: butuh ``scipy`` (lihat
    ``SCIPY_AVAILABLE``) dan minimal 4 titik data. Tebakan awal diturunkan dari
    data (``b0 = ptp(y)/2``, ``d0 = median(y)``, ``a0 = 1.0``, ``c0 =
    median(x)``); ``curve_fit`` dipanggil dengan ``maxfev=5000``.

    Args:
        x: Array abscissa.
        y: Array ordinat (utility) yang akan dicocokkan.

    Returns:
        Tuple ``(a, b, c, d)`` bila berhasil; ``None`` bila ``scipy`` tak ada,
        titik kurang dari 4, atau fitting gagal.

    Example:
        >>> import numpy as np
        >>> x = np.arange(8, dtype=float)
        >>> y = np.tanh(0.5 * x - 2.0)
        >>> res = fit_tanh_curve(x, y)
        >>> res is None or len(res) == 4
        True
    """
    if not SCIPY_AVAILABLE:
        return None
    if len(x) < 4:
        return None

    # Derive initial guesses from data
    y_range = float(np.ptp(y)) if np.ptp(y) > 1e-10 else 1e-4
    b0 = y_range / 2.0
    d0 = float(np.median(y))
    a0 = 1.0
    c0 = float(np.median(x))

    try:
        popt, _ = curve_fit(
            tanh_model,
            x,
            y,
            p0=[a0, b0, c0, d0],
            maxfev=5000,
        )
        return tuple(float(v) for v in popt)  # (a, b, c, d)
    except Exception:
        return None


def compute_burstiness(
    a: float,
    b: float,
    all_a: List[float],
    all_b: List[float],
) -> float:
    """Hitung burstiness ``tau = |a_scaled| * b_scaled`` (persis referensi).

    Penskalaan adalah normalisasi min-max atas seluruh negosiasi yang ter-fit.
    Bila populasi kurang dari 2 nilai untuk ``a`` atau ``b``, kembalikan 0.0.

    Args:
        a: Parameter ``a`` episode ini.
        b: Parameter ``b`` episode ini.
        all_a: Daftar nilai ``a`` dari seluruh episode (populasi).
        all_b: Daftar nilai ``b`` dari seluruh episode (populasi).

    Returns:
        Nilai burstiness ``float``.

    Example:
        >>> compute_burstiness(1.0, 1.0, [1.0], [1.0])
        0.0
        >>> round(compute_burstiness(2.0, 2.0, [0.0, 2.0], [0.0, 2.0]), 4)
        1.0
    """
    if len(all_a) < 2 or len(all_b) < 2:
        return 0.0

    a_min, a_max = min(all_a), max(all_a)
    b_min, b_max = min(all_b), max(all_b)

    def _scale(v: float, lo: float, hi: float) -> float:
        denom = max(1e-8, hi - lo)
        return (abs(v) - lo) / denom

    a_scaled = _scale(a, a_min, a_max)
    b_scaled = _scale(b, b_min, b_max)
    return float(a_scaled * b_scaled)


def compute_cri(a: float, T: int) -> float:
    """Hitung Concession-Rigidity Index ``CRI = 1 - 1.32 / (|a| * T)``.

    Dipotong (clamp) ke rentang ``[0, 1]`` (persis referensi). ``T`` di-floor ke
    minimal 1 lewat ``max(T, 1)``.

    Args:
        a: Parameter kecuraman tanh.
        T: Jumlah langkah proposal (durasi).

    Returns:
        Nilai CRI ``float`` di ``[0, 1]``.

    Example:
        >>> compute_cri(0.0, 5)
        1.0
        >>> 0.0 <= compute_cri(2.0, 10) <= 1.0
        True
    """
    denom = abs(a) * max(T, 1)
    if denom < 1e-10:
        return 1.0
    return float(np.clip(1.0 - 1.32 / denom, 0.0, 1.0))


def compute_cri_star(
    a: float,
    b: float,
    T: int,
    theta: float = 0.1,
) -> float:
    """Hitung CRI* berbasis data (persis referensi lama).

    Langkah (sampling pada T titik integer):
    1. Profil kecepatan ternormalisasi ``s_hat(x) = sech^2(a*x)`` di ``[0, T]``.
    2. Jendela aktif ``W = {x : s_hat(x) >= theta}``.
    3. ``CRI* = 1 - len(W) / T``, dipotong ke ``[0, 1]``.

    Catatan: nilai default ``theta=0.1`` menjaga fungsi tetap murni. Pemanggil
    biasanya meneruskan ``config.reporting.cri_star_theta``.

    Args:
        a: Parameter kecuraman tanh.
        b: Parameter amplitudo tanh (tidak memengaruhi bentuk ternormalisasi).
        T: Jumlah titik sampling (durasi).
        theta: Ambang aktivitas kecepatan ternormalisasi.

    Returns:
        Nilai CRI* ``float`` di ``[0, 1]``.

    Example:
        >>> compute_cri_star(1.0, 1.0, 1)
        1.0
        >>> 0.0 <= compute_cri_star(0.5, 1.0, 10) <= 1.0
        True
    """
    if T < 2:
        return 1.0

    xs = np.linspace(0, T, T)
    # Speed proportional: |ab| * sech²(ax - c)  -> max at x = c/a
    # sech² at max = 1, so normalised speed = sech²(ax - c)
    speed = 1.0 / np.cosh(a * xs) ** 2  # normalised (c=0 for profile shape)
    max_s = float(speed.max())
    if max_s < 1e-10:
        return 1.0
    speed_hat = speed / max_s

    active = int((speed_hat >= theta).sum())
    return float(np.clip(1.0 - active / T, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-episode tanh metrics
# ---------------------------------------------------------------------------


def compute_per_episode_tanh_metrics(
    propose_utils: List[float],
    all_params: Optional[List[Optional[Tuple[float, float, float, float]]]] = None,
    n_steps: int = 0,
    theta: float = 0.1,
    min_points: int = 4,
) -> Dict[str, Any]:
    """Hitung metrik konsesi berbasis tanh untuk SATU episode.

    Adaptasi dari ``compute_per_episode_tanh_metrics`` referensi lama: input
    berupa ``propose_utils`` (list float) langsung — BUKAN dict
    ``episode_payload`` — sehingga modul ini tidak perlu mengimpor
    ``negotiation/*`` (INV-1). Rumus inti (fit, CRI, CRI*, burstiness, fitted
    curve) IDENTIK referensi (INV-3).

    Args:
        propose_utils: Daftar utility proposal dalam urutan langkah.
        all_params: Daftar ``(a, b, c, d)`` dari seluruh episode (boleh berisi
            ``None``), dipakai untuk penskalaan burstiness tingkat populasi.
            Bila ``None``, burstiness tetap 0.0.
        n_steps: Jumlah langkah total episode (untuk field ``n_steps`` keluaran).
        theta: Ambang CRI* (lihat ``compute_cri_star``).
        min_points: Jumlah titik minimum agar fitting dilakukan (default 4).

    Returns:
        Dict dengan kunci: ``tanh_params``, ``burstiness``, ``cri``,
        ``cri_star``, ``anchor_distance``, ``propose_utils``, ``fitted_curve``,
        ``_x_dense``, ``n_steps``.

    Example:
        >>> res = compute_per_episode_tanh_metrics([0.9, 0.7, 0.5, 0.3, 0.1])
        >>> 0.0 <= res["cri"] <= 1.0
        True
        >>> res["anchor_distance"]
        0.9
        >>> short = compute_per_episode_tanh_metrics([0.5, 0.4])
        >>> short["tanh_params"] is None
        True
    """
    result: Dict[str, Any] = {
        "tanh_params": None,
        "burstiness": 0.0,
        "cri": 0.0,
        "cri_star": 0.0,
        "anchor_distance": propose_utils[0] if propose_utils else 0.0,
        "propose_utils": propose_utils,
        "fitted_curve": None,
        "_x_dense": None,
        "n_steps": n_steps,
    }

    if len(propose_utils) < min_points:
        return result

    x = np.arange(len(propose_utils), dtype=float)
    y = np.array(propose_utils, dtype=float)

    params = fit_tanh_curve(x, y)
    if params is None:
        return result

    a, b, c, d = params
    T = len(propose_utils)

    result["tanh_params"] = params
    result["cri"] = compute_cri(a, T)
    result["cri_star"] = compute_cri_star(a, b, T, theta)

    # Fitted curve for plotting
    x_dense = np.linspace(0, T - 1, max(T * 4, 50))
    result["fitted_curve"] = tanh_model(x_dense, a, b, c, d).tolist()
    result["_x_dense"] = x_dense.tolist()

    # Burstiness needs population-level normalisation
    if all_params is not None:
        all_a = [p[0] for p in all_params if p is not None]
        all_b = [p[1] for p in all_params if p is not None]
        if all_a and all_b:
            result["burstiness"] = compute_burstiness(a, b, all_a, all_b)

    return result
