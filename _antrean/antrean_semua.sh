#!/usr/bin/env bash
# Orkestrator seluruh kampanye, yaitu v_1 lebih dulu lalu v_2.
#
# v_1 didahulukan karena murah, sekitar setengah jam, sehingga korpus Bab V.1
# sudah utuh sebelum masuk ke v_2 yang memakan puluhan jam. Kalau v_1 berhenti
# karena disk tipis (kode keluar 2), v_2 TIDAK dimulai.
#
# STOP     : bash _antrean/hentikan.sh
# CONTINUE : jalankan ulang skrip ini, apa adanya. Keduanya melewati yang sudah
#            selesai, jadi aman diulang berapa kali pun.
set -u

ROOT="c:/Users/ACER/Documents/Magister/thesis_final/clean_code"
LOGDIR="$ROOT/_antrean/logs"
mkdir -p "$LOGDIR"

echo "########## KAMPANYE MULAI $(date '+%Y-%m-%d %H:%M:%S')"

bash "$ROOT/_antrean/antrean_v1.sh"
kode=$?
if [ "$kode" -ne 0 ]; then
  echo "########## v_1 berhenti dengan kode $kode, v_2 TIDAK dimulai"
  exit "$kode"
fi

bash "$ROOT/_antrean/antrean_v2.sh"
kode=$?

echo "########## KAMPANYE BERAKHIR $(date '+%Y-%m-%d %H:%M:%S') kode=$kode"
exit "$kode"
