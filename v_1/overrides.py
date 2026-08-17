# -*- coding: utf-8 -*-
"""Tambalan LOKAL v_1 supaya dunianya sama dengan korpus non-*real-case* lama.

Berlaku **hanya** untuk v_1, dipasang saat dijalankan lewat titik masuk di
folder ini. Tidak ada satu baris pun kode bersama yang disunting, dan itu sudah
diverifikasi dengan ``diff -r`` terhadap ``nyoba_general``, yaitu ``simulation/``
nol berkas berbeda.

APA YANG BERBEDA
================

Hanya satu hal. Pemilih tujuan fallback versi lama **menghabiskan tepat tiga
undian rng lebih banyak** daripada versi sekarang. Karena rangkaian bilangan
acak berurutan, tiga undian yang hilang itu menggeser seluruh undian sesudahnya.
Dunia punya dua pemesanan sehingga fungsinya dipanggil dua kali, jadi totalnya
enam posisi bergeser.

Yang bergeser bukan pemesanannya, melainkan **order penumpang yang terbangkit
sepanjang simulasi**, yaitu titik jemput, tujuan, dan waktunya. Itulah yang
membuat pendapatan dan jumlah order berbeda.

Pilihan tujuan fallback sendiri terbukti **tidak berpengaruh** di dunia v_1.
Dua aturan tujuan yang berlawanan, yaitu terdekat dan terjauh, memberi hasil
identik baik pada non-negosiasi maupun negosiasi. Sebabnya dunia ini hanya punya
dua pemesanan sehingga praktis tak pernah bertabrakan dengan permintaan taksi.

BAGAIMANA ANGKA TIGA ITU DIPEROLEH
==================================

Bukan tebakan. Pada mekanisme non-negosiasi tidak ada pemesanan yang benar-benar
digeser, sehingga satu-satunya jalan bagi perubahan ini untuk memengaruhi hasil
adalah lewat jumlah undian. Ruang pencariannya jadi satu dimensi, lalu disisir
0 sampai 16 dan dibandingkan dengan Tabel V.1 laporan.

Pada tiga undian tambahan, mekanisme pemesanan statis mereproduksi Tabel V.1
**persis pada seluruh lima kolom** untuk empat dunia::

    dunia            fleet   spklu   order   jam-port   harga
    control_default   1,32    0,72      91      11,5    2.734
    hz_n1             1,32    0,72      91      11,5    2.734
    hz_n5             7,27    3,09     452      51,0    2.734
    ref_n10          14,35    6,24     906     104,0    2.748

BATASNYA, HARUS DISEBUT TERANG-TERANGAN
=======================================

- Dunia 30 hari **mendekat tetapi belum persis**, yaitu 43,62 lawan 43,67 dan
  2.723 lawan 2.699 order. Penyimpangan menumpuk sepanjang tiga puluh hari.
- Delapan dunia lain **belum diverifikasi**, sebab pembandingnya belum ada.
- Mekanisme **tanpa pemesanan tidak akan cocok**. Perbaikan bug jarak pada
  7 Juli membuatnya berperilaku seperti pemesanan statis, dan baris "Tanpa
  pemesanan" di Tabel V.1 lahir dari versi yang masih bermasalah. Itu TIDAK
  dikembalikan, sebab mengembalikannya berarti memasang bug lagi.
- Dunia pulih **tidak berarti negosiasi ikut pulih**. Pada dunia referensi,
  DLST-MBOM Nego kembali cocok di 1,000 sedangkan DLST murni tetap 0,631 lawan
  0,988. Sisa selisih itu diduga dari kode agen, dan tidak dapat dikembalikan
  sebab kode Juli sudah tidak ada di perangkat ini.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulation.world import world_assembly as _wa  # noqa: E402

#: Selisih jumlah undian rng antara versi lama dan versi sekarang.
UNDIAN_TAMBAHAN = 3

_ASLI = _wa._choose_fallback_destination


def _fallback_dunia_lama(all_ports, origin_port, slot_start, duration_grids,
                         total_grids, shift_range, rng):
    """Logika pemilihan tujuan yang sekarang, dengan undian versi lama."""
    for _ in range(UNDIAN_TAMBAHAN):
        rng.random()
    return _ASLI(all_ports, origin_port, slot_start, duration_grids,
                 total_grids, shift_range, rng)


def pasang_dunia_lama() -> None:
    """Menyetel perakitan dunia v_1 ke keadaan korpus lama.

    Wajib dipanggil sebelum config dimuat dan dunia dirakit. Ketiga titik masuk
    di folder ini sudah melakukannya secara otomatis.
    """
    _wa._choose_fallback_destination = _fallback_dunia_lama
    print(f"[v_1] dunia korpus lama AKTIF ({UNDIAN_TAMBAHAN} undian tambahan)")
