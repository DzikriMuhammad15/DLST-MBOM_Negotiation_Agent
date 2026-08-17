# -*- coding: utf-8 -*-
"""Sampler empiris AppUserBooking untuk dunia real-case Bandung (desain v2 §2.5).

Semua distribusi DIBACA dari file turunan data nyata `simulation_data/_derived/`
(prinsip grounding desain v2 §2.0 — DILARANG hardcode angka dunia):

- **Jam mulai**   ~ ``hour_hist``   (SELURUH Bandung, 22.514 trx 2024)
- **Durasi**      ~ ``durasi_hist_bin15`` (SELURUH Bandung), konversi grid ½-jam
  nearest (``int(menit/30 + 0.5)``, min 1), **cap [1, 4] grid = 0,5–2 jam**
  (mengikuti batas engine ``_MIN/_MAX_BOOKING_DURATION``)
- **Stasiun/port** ~ bobot volume trx 2024 per 13 stasiun terpilih
  (``bandung_trx.csv``) **+ smoothing Laplace** (alpha=1) → port dibagi rata
  di dalam stasiun
- **Harga base**  = median ``price_per_kwh`` seluruh trx Bandung (tarif PLN
  nyata, flat)

Kelas ini juga menghitung **N target okupansi** (keputusan skenario peneliti
2026-07-07: okupansi 22,5%) dari mean durasi-tersampel-tercap yang EKSAK
(ekspektasi analitik atas histogram, bukan angka desain ~0,99 yang dibulatkan):

    N = round(okupansi × (n_port × n_hari × 24 jam) ÷ E[durasi_tercap_jam])

Dipakai oleh ``build_world_config.py`` (menghitung N & bobot utk dokumentasi)
dan ``overrides.py`` (menyampel booking saat world assembly).
"""
from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DERIVED_DIR = os.path.join(_ROOT, "simulation_data", "_derived")

# Konstanta struktural grid (identik engine world_assembly).
GRID_MINUTES = 30
GRIDS_PER_DAY = 48
MIN_DURATION_GRIDS = 1   # 0,5 jam — batas engine
MAX_DURATION_GRIDS = 4   # 2 jam — batas engine
LAPLACE_ALPHA = 1.0      # smoothing bobot stasiun (3 stasiun ~0 trx)


def _minutes_to_grids(minutes: float) -> int:
    """Konversi menit → grid ½-jam terdekat (min 1), TANPA cap.

    ``int(m/30 + 0.5)`` = pembulatan half-up; min 1 grid (engine tidak
    mengenal durasi 0).
    """
    return max(MIN_DURATION_GRIDS, int(minutes / GRID_MINUTES + 0.5))


def _capped_grids(minutes: float) -> int:
    """Konversi menit → grid dengan cap [1, 4] (0,5–2 jam, batas engine)."""
    return min(MAX_DURATION_GRIDS, _minutes_to_grids(minutes))


class EmpiricalBookingSampler:
    """Sampler booking dari distribusi transaksi charging nyata Bandung 2024.

    Args:
        station_ports: mapping ``station_id_config -> list port_id`` untuk 13
            stasiun dunia v2 (urutan config).
        station_csv_ids: mapping ``station_id_config -> id_spklu`` di
            ``bandung_trx.csv`` (untuk membaca volume nyata per stasiun).
        derived_dir: lokasi folder ``_derived`` (default: repo).
    """

    def __init__(
        self,
        station_ports: Dict[str, Sequence[str]],
        station_csv_ids: Dict[str, str],
        derived_dir: str = DERIVED_DIR,
    ) -> None:
        self.station_ports = {k: list(v) for k, v in station_ports.items()}
        self.station_csv_ids = dict(station_csv_ids)
        self.derived_dir = derived_dir

        summary_path = os.path.join(derived_dir, "bandung_only_trx_summary.json")
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        self.n_rows_bandung = int(summary["rows_bandung"])

        # Histogram jam (0..23) — dadu berbobot CDF.
        self.hour_hist: Dict[int, int] = {
            int(h): int(c) for h, c in summary["hour_hist"].items()
        }
        self._hours = sorted(self.hour_hist)
        self._hour_weights = [self.hour_hist[h] for h in self._hours]

        # Histogram durasi bin 15 menit (edge bawah). Bin terakhir (240) =
        # akumulasi ekor >= 240 menit; utk sampling diperlakukan [240, 255) —
        # tidak berpengaruh karena cap 2 jam memotong jauh di bawahnya.
        self.durasi_hist: Dict[int, int] = {
            int(e): int(c) for e, c in summary["durasi_hist_bin15"].items()
        }
        self._dur_edges = sorted(self.durasi_hist)
        self._dur_weights = [self.durasi_hist[e] for e in self._dur_edges]

        # Volume trx nyata per stasiun terpilih (dibaca dari CSV per-transaksi).
        self.station_volumes: Dict[str, int] = self._load_station_volumes()

        # Bobot Laplace per stasiun: (volume + alpha) / sum(volume + alpha).
        denom = sum(self.station_volumes[s] + LAPLACE_ALPHA for s in self.station_ports)
        self.station_weights: Dict[str, float] = {
            s: (self.station_volumes[s] + LAPLACE_ALPHA) / denom
            for s in self.station_ports
        }
        self._stations = list(self.station_ports)
        self._station_w = [self.station_weights[s] for s in self._stations]

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------
    def _load_station_volumes(self) -> Dict[str, int]:
        """Hitung jumlah trx 2024 per stasiun terpilih dari ``bandung_trx.csv``."""
        wanted = {csv_id: sid for sid, csv_id in self.station_csv_ids.items()}
        counts = {sid: 0 for sid in self.station_ports}
        csv_path = os.path.join(self.derived_dir, "bandung_trx.csv")
        with open(csv_path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sid = wanted.get(row["id_spklu"])
                if sid is not None:
                    counts[sid] += 1
        return counts

    @staticmethod
    def real_base_price(derived_dir: str = DERIVED_DIR) -> float:
        """Median ``price_per_kwh`` seluruh trx Bandung (tarif PLN nyata, flat).

        Dibulatkan 1 desimal (2466.8) — konsisten angka desain/EDA.
        """
        prices: List[float] = []
        with open(os.path.join(derived_dir, "bandung_trx.csv"), "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    prices.append(float(row["price_per_kwh"]))
                except (ValueError, KeyError):
                    continue
        prices.sort()
        n = len(prices)
        med = prices[n // 2] if n % 2 else 0.5 * (prices[n // 2 - 1] + prices[n // 2])
        return round(med, 1)

    # ------------------------------------------------------------------
    # Ekspektasi analitik (deterministik — untuk N & dokumentasi)
    # ------------------------------------------------------------------
    def expected_capped_duration_grids(self) -> float:
        """E[durasi grid ter-cap] eksak dari histogram (uniform dalam bin).

        Bin 15-menit sejajar dengan ambang pembulatan grid (45/75/105 menit),
        sehingga seluruh isi satu bin memetakan ke SATU nilai grid — ekspektasi
        cukup dihitung per-bin (tanpa integrasi numerik).
        """
        total = sum(self._dur_weights)
        acc = 0.0
        for edge, count in zip(self._dur_edges, self._dur_weights):
            mid = edge + 7.5  # titik mana pun dalam bin memetakan ke grid sama
            acc += count * _capped_grids(mid)
        return acc / total

    def expected_capped_duration_hours(self) -> float:
        """E[durasi jam ter-cap] = E[grid] × 0,5 jam."""
        return self.expected_capped_duration_grids() * (GRID_MINUTES / 60.0)

    def n_target_bookings(
        self, n_ports: int, n_days: int, occupancy_target: float
    ) -> int:
        """N booking dari target okupansi (keputusan skenario, desain v2 §2.5).

        N = round(okupansi × n_port × n_hari × 24 ÷ E[durasi_jam_tercap]).
        """
        capacity_port_hours = n_ports * n_days * 24
        return int(round(occupancy_target * capacity_port_hours
                         / self.expected_capped_duration_hours()))

    # ------------------------------------------------------------------
    # Sampling (dipakai override world assembly; rng = rng dunia engine)
    # ------------------------------------------------------------------
    def sample_start_grid(self, rng: random.Random, n_days: int) -> int:
        """Grid mulai: hari ~ U{0..n_days-1} (bentuk harian diulang), jam ~
        ``hour_hist``, menit ~ U dalam jam → grid ½-jam."""
        day = rng.randrange(max(1, int(n_days)))
        hour = rng.choices(self._hours, weights=self._hour_weights, k=1)[0]
        half = rng.randrange(2)  # menit uniform dalam jam → paruh-jam 0/1
        return day * GRIDS_PER_DAY + hour * 2 + half

    def sample_duration_grids(self, rng: random.Random) -> int:
        """Durasi: bin ~ ``durasi_hist``, menit ~ U dalam bin, → grid cap [1,4]."""
        edge = rng.choices(self._dur_edges, weights=self._dur_weights, k=1)[0]
        minutes = edge + rng.uniform(0.0, 15.0)
        return _capped_grids(minutes)

    def sample_station(self, rng: random.Random) -> str:
        """Stasiun ~ bobot Laplace volume 2024."""
        return rng.choices(self._stations, weights=self._station_w, k=1)[0]

    def sample_port(self, rng: random.Random) -> str:
        """Port: pilih stasiun berbobot, lalu port uniform dalam stasiun."""
        station = self.sample_station(rng)
        return rng.choice(self.station_ports[station])

    def urutan_port(self, rng: random.Random) -> List[str]:
        """Seluruh port dalam urutan acak BERBOBOT, tanpa pengembalian.

        Dipakai saat penempatan booking perlu mencoba port lain: port yang
        stasiunnya lebih ramai pada data nyata cenderung dicoba lebih dulu,
        sehingga preferensi lokasi tetap terjaga meskipun port pertama penuh.

        Memakai kunci Efraimidis-Spirakis ``U^(1/w)`` yang menghasilkan
        pengambilan sampel berbobot tanpa pengembalian dalam satu langkah.
        """
        kunci: List[Tuple[float, str]] = []
        for station, w_station in zip(self._stations, self._station_w):
            ports = self.station_ports[station]
            w_port = max(1e-12, w_station / max(1, len(ports)))
            for port_id in ports:
                kunci.append((rng.random() ** (1.0 / w_port), port_id))
        kunci.sort(reverse=True)
        return [port_id for _, port_id in kunci]

    # ------------------------------------------------------------------
    # Dokumentasi / audit
    # ------------------------------------------------------------------
    def provenance(self) -> Dict[str, object]:
        """Ringkasan angka utk audit & DOKUMENTASI_SAMPLING_DAN_PROVENANCE.md."""
        return {
            "sampler": "empiris",
            "rows_bandung": self.n_rows_bandung,
            "hour_hist": {str(h): self.hour_hist[h] for h in self._hours},
            "durasi_hist_bin15": {str(e): self.durasi_hist[e] for e in self._dur_edges},
            "station_volumes_2024": dict(self.station_volumes),
            "laplace_alpha": LAPLACE_ALPHA,
            "station_weights": {s: round(w, 6) for s, w in self.station_weights.items()},
            "expected_capped_duration_grids": round(self.expected_capped_duration_grids(), 6),
            "expected_capped_duration_hours": round(self.expected_capped_duration_hours(), 6),
            "duration_grid_rule": "grid = min(4, max(1, int(menit/30 + 0.5))) — nearest ½-jam, cap 0,5–2 jam",
        }
