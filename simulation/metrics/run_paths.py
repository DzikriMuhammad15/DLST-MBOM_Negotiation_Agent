"""Pusat tunggal pembuatan ``sim_id`` + jalur folder ``data/runs``.

Modul ini menyediakan:

* :func:`make_sim_id` — id run berbasis timestamp (KR-1).
* :class:`RunPaths` — penyusun jalur folder/file run (KR-2).
* :func:`make_run_paths` — helper membangun :class:`RunPaths` dari config.

Hanya memakai pustaka standar (``os``, ``datetime``, ``uuid``). Modul ini TIDAK
mengimpor ``negotiation/*``, ``torch``, ``numpy``, atau ``matplotlib`` (tetap di
lapisan simulasi). Folder dibuat secara *lazy* (saat dibutuhkan), bukan saat
konstruksi (INV-6). Folder ``data/registry/`` & ``data/history/`` tidak pernah
disentuh oleh modul ini (INV-7).

Lihat ``.claude/low_level_requirement_creator/low_level_requirement.txt``
KOMPONEN N1 (LLR-N1.1, N1.2, N1.3) dan LLR-A4.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, List, Optional, Tuple
from uuid import uuid4

# Format timestamp sim_id mengikuti pola kode lama
# (.claude/references/project_old_reference/utils/io.py::timestamp()), LLR-A4.
SIM_ID_TIME_FORMAT: str = "%Y%m%d_%H%M%S"

# Prefiks folder per-hari di dalam run_dir (KR-2: day_<index>).
DAY_DIR_PREFIX: str = "day_"

# Panjang suffix anti-tabrakan (hex dari uuid4) bila folder run sudah ada (LLR-A4).
COLLISION_SUFFIX_LEN: int = 4


def make_sim_id() -> str:
    """Bangun id run berbasis timestamp lokal sekarang.

    Format ``YYYYMMDD_HHMMSS`` (panjang 15: 8 + 1 + 6), mengikuti pola kode lama.

    Returns:
        String sim_id, mis. ``"20260626_153000"``.

    Contoh:
        >>> sid = make_sim_id()
        >>> len(sid) == 15
        True
        >>> sid[8] == "_"
        True
    """
    return datetime.now().strftime(SIM_ID_TIME_FORMAT)


class RunPaths:
    """Penyusun jalur folder & file untuk satu run di bawah ``runs_root``.

    Kelas ini memusatkan pembuatan ``sim_id`` dan komposisi jalur folder run,
    folder per-hari (``day_<index>``), serta file periode JSON. Pembuatan folder
    bersifat *lazy*: konstruktor TIDAK membuat folder apa pun di disk; folder
    dibuat saat metode pembentuk dipanggil (mis. :meth:`ensure_run_dir`,
    :meth:`day_dir`). Bila ``run_dir`` sudah ada (dua run pada detik yang sama),
    sebuah suffix unik pendek ditambahkan ke ``sim_id`` agar setiap run memiliki
    folder berbeda (KR-1 / LLR-A4).

    Attributes:
        runs_root: Folder akar tempat folder ``sim_id`` dibuat.
        sim_id: Identitas run (timestamp, mungkin + suffix anti-tabrakan).
        run_dir: ``os.path.join(runs_root, sim_id)``.

    Contoh:
        >>> rp = RunPaths("data/runs", sim_id="20260626_153000")
        >>> rp.run_dir.replace("\\\\", "/")
        'data/runs/20260626_153000'
    """

    def __init__(self, runs_root: str, sim_id: Optional[str] = None) -> None:
        """Inisialisasi RunPaths tanpa membuat folder (lazy).

        Args:
            runs_root: Folder akar runs (mis. ``"data/runs"``).
            sim_id: sim_id eksplisit; bila ``None`` dibuat via :func:`make_sim_id`.

        Bila ``run_dir`` yang dihitung sudah ada di disk, ``sim_id`` diberi suffix
        ``"_" + uuid4().hex[:4]`` berulang sampai folder unik (LLR-A4).

        Contoh:
            >>> rp = RunPaths("data/runs", sim_id="20260626_153000")
            >>> rp.sim_id
            '20260626_153000'
        """
        self.runs_root: str = runs_root
        self.sim_id: str = sim_id if sim_id is not None else make_sim_id()
        self.run_dir: str = os.path.join(self.runs_root, self.sim_id)
        # Anti-tabrakan: bila folder sudah ada, tambah suffix hex sampai unik.
        while os.path.exists(self.run_dir):
            self.sim_id = f"{self.sim_id}_{uuid4().hex[:COLLISION_SUFFIX_LEN]}"
            self.run_dir = os.path.join(self.runs_root, self.sim_id)

    def ensure_run_dir(self) -> str:
        """Buat ``run_dir`` bila belum ada dan kembalikan jalurnya.

        Returns:
            Jalur absolut/relatif ``run_dir``.

        Contoh:
            >>> import tempfile, os
            >>> root = tempfile.mkdtemp()
            >>> rp = RunPaths(root, sim_id="run_a")
            >>> os.path.isdir(rp.ensure_run_dir())
            True
        """
        os.makedirs(self.run_dir, exist_ok=True)
        return self.run_dir

    def day_dir(self, day_index: int) -> str:
        """Kembalikan (dan buat bila perlu) folder hari ``day_<index>``.

        Args:
            day_index: Indeks hari (>= 0).

        Returns:
            Jalur folder ``<run_dir>/day_<index>``.

        Contoh:
            >>> import tempfile, os
            >>> root = tempfile.mkdtemp()
            >>> rp = RunPaths(root, sim_id="run_b")
            >>> path = rp.day_dir(0)
            >>> os.path.basename(path)
            'day_0'
            >>> os.path.isdir(path)
            True
        """
        path = os.path.join(self.run_dir, f"{DAY_DIR_PREFIX}{int(day_index)}")
        os.makedirs(path, exist_ok=True)
        return path

    def period_json_path(
        self,
        day_index: int,
        period_index_in_day: int,
        sim_time_min: float,
        status: str = "period",
    ) -> str:
        """Bangun jalur file JSON untuk satu periode (terurut deterministik).

        Nama file mengikuti format N2 (zero-pad agar sort leksikografis sama
        dengan sort kronologis):
        ``period_{period_index_in_day:04d}_t{int(sim_time_min):07d}_{status}.json``.

        Args:
            day_index: Indeks hari.
            period_index_in_day: Indeks periode dalam hari (mulai 0).
            sim_time_min: Waktu simulasi (menit) saat periode terjadi.
            status: Label status periode (mis. ``"deal"``, ``"no_deal"``,
                ``"no_negotiation"``); default ``"period"``.

        Returns:
            Jalur file JSON di dalam ``day_<index>`` (folder hari dibuat bila
            belum ada karena memanggil :meth:`day_dir`).

        Contoh:
            >>> import tempfile, os
            >>> root = tempfile.mkdtemp()
            >>> rp = RunPaths(root, sim_id="run_c")
            >>> p = rp.period_json_path(1, 3, 720.0, status="deal")
            >>> os.path.basename(p)
            'period_0003_t0000720_deal.json'
        """
        filename = (
            f"period_{int(period_index_in_day):04d}"
            f"_t{int(sim_time_min):07d}_{status}.json"
        )
        return os.path.join(self.day_dir(day_index), filename)

    def day_png_path(self, day_index: int, name: str) -> str:
        """Bangun jalur file PNG metrik di dalam folder hari.

        Args:
            day_index: Indeks hari.
            name: Nama file PNG (mis. ``"utility_per_period.png"``).

        Returns:
            Jalur ``<run_dir>/day_<index>/<name>`` (folder hari dibuat bila perlu).

        Contoh:
            >>> import tempfile, os
            >>> root = tempfile.mkdtemp()
            >>> rp = RunPaths(root, sim_id="run_d")
            >>> p = rp.day_png_path(2, "business_metrics.png")
            >>> os.path.basename(p)
            'business_metrics.png'
        """
        return os.path.join(self.day_dir(day_index), name)

    def sim_png_path(self, name: str) -> str:
        """Bangun jalur file PNG metrik di tingkat run (per-simulasi).

        Args:
            name: Nama file PNG (mis. ``"business_metrics.png"``).

        Returns:
            Jalur ``<run_dir>/<name>`` (``run_dir`` dibuat bila belum ada).

        Contoh:
            >>> import tempfile, os
            >>> root = tempfile.mkdtemp()
            >>> rp = RunPaths(root, sim_id="run_e")
            >>> p = rp.sim_png_path("utility_per_period.png")
            >>> os.path.basename(p)
            'utility_per_period.png'
        """
        self.ensure_run_dir()
        return os.path.join(self.run_dir, name)

    def list_day_dirs(self) -> List[Tuple[int, str]]:
        """Daftar folder hari di ``run_dir`` terurut menaik per ``day_index``.

        Hanya entri bernama ``day_<int>`` yang dipertimbangkan; entri lain
        diabaikan. Bila ``run_dir`` belum ada, kembalikan list kosong.

        Returns:
            List ``(day_index, path)`` terurut menaik berdasarkan ``day_index``.

        Contoh:
            >>> import tempfile, os
            >>> root = tempfile.mkdtemp()
            >>> rp = RunPaths(root, sim_id="run_f")
            >>> _ = rp.day_dir(2); _ = rp.day_dir(0)
            >>> [idx for idx, _ in rp.list_day_dirs()]
            [0, 2]
        """
        result: List[Tuple[int, str]] = []
        if not os.path.isdir(self.run_dir):
            return result
        for entry in os.listdir(self.run_dir):
            full = os.path.join(self.run_dir, entry)
            if not os.path.isdir(full):
                continue
            if not entry.startswith(DAY_DIR_PREFIX):
                continue
            suffix = entry[len(DAY_DIR_PREFIX):]
            try:
                day_index = int(suffix)
            except ValueError:
                continue
            result.append((day_index, full))
        result.sort(key=lambda item: item[0])
        return result

    def agent_bundle_dir(self, side: str) -> str:
        """Kembalikan (dan buat bila perlu) folder penyimpanan agent bundle untuk
        satu sisi di dalam run.

        Path: <run_dir>/agents/<side>/
        Folder dibuat bila belum ada (lazy, seperti day_dir).
        Folder <run_dir>/agents/ dibuat terlebih dahulu bila belum ada.

        Args:
            side: "fleet" atau "spklu".
        Returns:
            Path string <run_dir>/agents/<side>/

        Contoh:
            >>> import tempfile, os
            >>> root = tempfile.mkdtemp()
            >>> rp = RunPaths(root, sim_id="test_bundle")
            >>> p = rp.agent_bundle_dir("fleet")
            >>> os.path.isdir(p)
            True
            >>> "agents" in p
            True
        """
        agents_dir = os.path.join(self.run_dir, "agents")
        path = os.path.join(agents_dir, side)
        os.makedirs(path, exist_ok=True)
        return path

    def list_period_files(self, day_index: int) -> List[str]:
        """Daftar file ``*.json`` di folder hari, terurut nama (== kronologis).

        Nama file periode sudah zero-padded sehingga sort leksikografis setara
        dengan sort kronologis. Bila folder hari belum ada, kembalikan list
        kosong (tanpa membuat folder).

        Args:
            day_index: Indeks hari.

        Returns:
            List jalur file ``.json`` terurut menaik berdasarkan nama.

        Contoh:
            >>> import tempfile, os
            >>> root = tempfile.mkdtemp()
            >>> rp = RunPaths(root, sim_id="run_g")
            >>> d = rp.day_dir(0)
            >>> open(os.path.join(d, "period_0001_t0000060_deal.json"), "w").close()
            >>> open(os.path.join(d, "period_0000_t0000000_deal.json"), "w").close()
            >>> [os.path.basename(p) for p in rp.list_period_files(0)]
            ['period_0000_t0000000_deal.json', 'period_0001_t0000060_deal.json']
        """
        day_path = os.path.join(self.run_dir, f"{DAY_DIR_PREFIX}{int(day_index)}")
        if not os.path.isdir(day_path):
            return []
        files = [
            os.path.join(day_path, entry)
            for entry in os.listdir(day_path)
            if entry.endswith(".json")
            and os.path.isfile(os.path.join(day_path, entry))
        ]
        files.sort()
        return files


def make_run_paths(config: Any) -> RunPaths:
    """Bangun :class:`RunPaths` dari objek config.

    Membaca ``config.reporting.runs_root`` (lihat ``ReportingConfig`` di
    ``simulation/config/config.py``). sim_id dibuat otomatis (timestamp).

    Args:
        config: Objek config yang punya atribut ``reporting.runs_root``.

    Returns:
        Instance :class:`RunPaths` baru.

    Contoh:
        >>> class _Reporting:
        ...     runs_root = "data/runs"
        >>> class _Config:
        ...     reporting = _Reporting()
        >>> rp = make_run_paths(_Config())
        >>> rp.runs_root
        'data/runs'
    """
    return RunPaths(config.reporting.runs_root)
