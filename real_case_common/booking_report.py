# -*- coding: utf-8 -*-
"""Laporan pembentukan AppUserBooking untuk satu run eksperimen.

Menulis tiga berkas ke ``<run_dir>/metrices/``:

``booking_generation.json``
    Berapa permintaan booking yang berhasil ditempatkan **tanpa menggeser**,
    berapa yang berhasil **setelah digeser**, dan berapa yang **ditolak**.
    Pergeseran di sini adalah pergeseran saat DUNIA DIRAKIT, bukan pergeseran ke
    fallback saat negosiasi berjalan. Berkas ini juga memuat okupansi yang
    dihasilkan dan jarak sebaran terpasang terhadap sebaran permintaan.

``distribusi_durasi.png`` dan ``distribusi_jam.png``
    Pembanding satu panel antara sebaran hasil fit dan sebaran yang benar-benar
    terjadi pada run tersebut.

Selain itu, ringkasannya disisipkan ke ``aggregate_metrics.json`` di bawah kunci
``booking_generation``, bersama ``pergeseran_fallback`` yang mencatat berapa kali
booking benar-benar digeser ke rencana fallback-nya oleh taksi selama negosiasi.

Dipakai oleh ``v_2/main_*.py`` dan ``network_scaling/main_nego_ns.py`` setelah
eksperimen selesai.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import overrides as _ov

# Banyaknya tarikan untuk membangun sebaran acuan hasil fit. Cukup besar supaya
# kurvanya mulus, dan berbenih tetap supaya gambar dapat diulang.
N_ACUAN = 200_000
BENIH_ACUAN = 20260812

BIRU, ORANYE = "#2a78d6", "#eb6834"
TINTA, TINTA2, REDUP, GARIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def _proporsi(nilai: Sequence[int], n_kelas: int, mulai: int = 0) -> List[float]:
    cacah = [0] * n_kelas
    for v in nilai:
        k = int(v) - mulai
        if 0 <= k < n_kelas:
            cacah[k] += 1
    total = sum(cacah)
    return [c / total for c in cacah] if total else cacah


def _tvd(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q)) * 100.0


def _ambang_derau(p_acuan: Sequence[float], n: int, ulangan: int = 400) -> float:
    """Berapa besar TVD yang muncul dari derau pencuplikan SAJA.

    Satu run hanya memuat beberapa ratus booking, sedangkan sebaran acuannya
    punya puluhan kelas. Sebagian dari jarak yang terukur karena itu murni
    kebetulan, bukan penyimpangan sistematis. Fungsi ini menarik ``n`` contoh
    dari sebaran acuan berulang kali lalu mengukur jaraknya terhadap acuan itu
    sendiri, sehingga menghasilkan pembanding yang jujur: TVD yang terukur baru
    bermakna kalau jelas melampaui angka ini.
    """
    if n <= 0:
        return 0.0
    try:
        import numpy as np
    except ImportError:                                          # pragma: no cover
        return float("nan")
    p = np.asarray(p_acuan, dtype=float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    rng = np.random.default_rng(BENIH_ACUAN)
    cacah = rng.multinomial(n, p, size=ulangan) / float(n)
    return float(np.mean(0.5 * np.abs(cacah - p).sum(axis=1)) * 100.0)


def _acuan(sampler, n_days: int) -> Dict[str, List[int]]:
    """Sebaran acuan langsung dari sampler, tanpa penempatan."""
    rng = random.Random(BENIH_ACUAN)
    durasi = [sampler.sample_duration_grids(rng) for _ in range(N_ACUAN)]
    rng = random.Random(BENIH_ACUAN)
    jam = [sampler.sample_start_grid(rng, n_days) % 48 for _ in range(N_ACUAN)]
    return {"durasi": durasi, "jam": jam}


def _gambar(path: str, judul: str, xlabel: str, kelas: Sequence[int],
            p_acuan: Sequence[float], p_terjadi: Sequence[float],
            label_acuan: str, tvd: float, derau: float, n: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 10, "font.family": "sans-serif",
        "axes.edgecolor": GARIS, "axes.labelcolor": TINTA2, "text.color": TINTA,
        "xtick.color": REDUP, "ytick.color": REDUP, "axes.titlecolor": TINTA,
        "axes.grid": True, "grid.color": GARIS, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    })
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(kelas, p_terjadi, color=BIRU, alpha=0.65, width=0.9,
           label="terjadi pada simulasi")
    ax.step(kelas, p_acuan, where="mid", color=ORANYE, lw=2.0, label=label_acuan)
    ax.set_title(judul, fontsize=11, fontweight="bold", loc="left", pad=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("proporsi booking")
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False)
    ax.text(0.985, 0.72,
            f"massa salah tempat = {tvd:.2f} persen\n"
            f"ambang derau pencuplikan = {derau:.2f} persen\n"
            f"n terpasang = {n}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=TINTA2)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _sisipkan_ke_aggregate(metrices_dir: str, ringkas: Dict[str, Any]) -> Optional[int]:
    """Sisipkan ringkasan ke aggregate_metrics.json; kembalikan cacah pergeseran fallback."""
    path = os.path.join(metrices_dir, "aggregate_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        agg = json.load(fh)
    geser = None
    try:
        geser = int(agg["summary"]["spklu_business"]["shifted_bookings_count"])
    except (KeyError, TypeError, ValueError):
        geser = None
    agg["booking_generation"] = ringkas
    agg["pergeseran_fallback"] = {
        "keterangan": "Berapa kali booking benar-benar dipindahkan ke rencana "
                      "fallback-nya karena slotnya diambil taksi saat negosiasi. "
                      "Berbeda dari pergeseran saat dunia dirakit, yang dicatat "
                      "pada booking_generation.",
        "jumlah_pergeseran_ke_fallback": geser,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=1, ensure_ascii=False)
    return geser


def tulis_laporan(run_dir: str, sampler, n_days: int,
                  label: str = "") -> Dict[str, Any]:
    """Tulis laporan pembentukan booking + dua gambar pembanding.

    Args:
        run_dir: folder run eksperimen (``<exp_root>/<name>``).
        sampler: sampler yang dipakai merakit dunia.
        n_days: panjang horizon, untuk membangun sebaran acuan jam.
        label: keterangan bebas, misalnya nama skenario.

    Returns:
        Ringkasan yang ditulis. Kosong bila statistik pemasangan tidak tersedia.
    """
    stat = dict(_ov.LAST_INSTALL_STATS)
    if not stat:
        print("[laporan-booking] statistik pemasangan tidak tersedia, dilewati")
        return {}

    metrices_dir = os.path.join(run_dir, "metrices")
    os.makedirs(metrices_dir, exist_ok=True)

    dur_pasang = stat.pop("durasi_terpasang_grid", [])
    jam_pasang = stat.pop("jam_terpasang_grid", [])
    dur_minta = stat.pop("durasi_diminta_grid", [])
    jam_minta = stat.pop("jam_diminta_grid", [])

    acuan = _acuan(sampler, n_days)
    n_kelas_dur = max([1] + list(acuan["durasi"]) + list(dur_pasang))
    p_dur_acuan = _proporsi(acuan["durasi"], n_kelas_dur, mulai=1)
    p_dur_terjadi = _proporsi(dur_pasang, n_kelas_dur, mulai=1)
    p_jam_acuan = _proporsi(acuan["jam"], 48, mulai=0)
    p_jam_terjadi = _proporsi(jam_pasang, 48, mulai=0)

    tvd_dur = _tvd(p_dur_acuan, p_dur_terjadi)
    tvd_jam = _tvd(p_jam_acuan, p_jam_terjadi)
    derau_dur = _ambang_derau(p_dur_acuan, len(dur_pasang))
    derau_jam = _ambang_derau(p_jam_acuan, len(jam_pasang))

    empiris = stat.get("sampler") == "empiris"
    label_acuan = "hasil fit distribusi" if not empiris else "sebaran empiris data"

    # Sumbu durasi dipotong supaya badan sebaran terbaca; ekornya sangat tipis.
    batas = n_kelas_dur
    kumulatif = 0.0
    for k, p in enumerate(p_dur_acuan, start=1):
        kumulatif += p
        if kumulatif >= 0.995:
            batas = k
            break
    kelas_dur = list(range(1, batas + 1))
    _gambar(os.path.join(metrices_dir, "distribusi_durasi.png"),
            "Durasi booking: hasil fit dibanding yang terjadi",
            "durasi (grid setengah jam)", kelas_dur,
            p_dur_acuan[:batas], p_dur_terjadi[:batas],
            label_acuan, tvd_dur, derau_dur, len(dur_pasang))
    _gambar(os.path.join(metrices_dir, "distribusi_jam.png"),
            "Jam mulai booking: hasil fit dibanding yang terjadi",
            "grid setengah jam dalam sehari (0 sampai 47)", list(range(48)),
            p_jam_acuan, p_jam_terjadi,
            label_acuan, tvd_jam, derau_jam, len(jam_pasang))

    ringkas: Dict[str, Any] = dict(stat)
    ringkas.update({
        "label": label,
        "n_days": int(n_days),
        "massa_salah_tempat_durasi_persen": round(tvd_dur, 4),
        "massa_salah_tempat_jam_persen": round(tvd_jam, 4),
        "ambang_derau_durasi_persen": round(derau_dur, 4),
        "ambang_derau_jam_persen": round(derau_jam, 4),
        "keterangan_ambang_derau":
            "TVD yang muncul dari derau pencuplikan saja pada n yang sama. "
            "Massa salah tempat baru bermakna bila jelas melampaui angka ini.",
        "keterangan": "menggeser = booking dipindahkan dari port atau jam yang "
                      "diundi, saat DUNIA DIRAKIT. Bukan pergeseran ke fallback "
                      "saat negosiasi (lihat pergeseran_fallback).",
    })
    detail = {
        "ringkasan": ringkas,
        "histogram_durasi_grid": {
            "kelas": list(range(1, n_kelas_dur + 1)),
            "acuan": [round(v, 8) for v in p_dur_acuan],
            "terjadi": [round(v, 8) for v in p_dur_terjadi],
            "diminta": [round(v, 8) for v in _proporsi(dur_minta, n_kelas_dur, mulai=1)],
        },
        "histogram_jam_grid": {
            "kelas": list(range(48)),
            "acuan": [round(v, 8) for v in p_jam_acuan],
            "terjadi": [round(v, 8) for v in p_jam_terjadi],
            "diminta": [round(v, 8) for v in _proporsi(jam_minta, 48, mulai=0)],
        },
    }
    with open(os.path.join(metrices_dir, "booking_generation.json"),
              "w", encoding="utf-8") as fh:
        json.dump(detail, fh, indent=1, ensure_ascii=False)

    geser = _sisipkan_ke_aggregate(metrices_dir, ringkas)
    print(f"[laporan-booking] {ringkas['terpasang']}/{ringkas['permintaan']} terpasang "
          f"(tanpa geser {ringkas['tanpa_menggeser']}, menggeser "
          f"{ringkas['menggeser_total']}, ditolak {ringkas['ditolak_total']}), "
          f"okupansi {ringkas['okupansi_persen']:.2f}%, "
          f"pergeseran ke fallback {geser if geser is not None else 'n/a'}")
    return ringkas
