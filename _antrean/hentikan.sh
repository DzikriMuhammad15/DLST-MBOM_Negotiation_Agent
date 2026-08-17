#!/usr/bin/env bash
# Menghentikan kampanye dengan aman.
#
# Urutannya sengaja bash dulu baru python. Kalau python dimatikan lebih dulu,
# skrip antreannya masih hidup dan langsung menjalankan run berikutnya.
#
# PELAJARAN LAMA. Orkestrator berlapis punya BEBERAPA proses bash dengan
# CommandLine berbeda, yaitu induk dan anak. Mematikan yang cocok nama induk
# saja TIDAK CUKUP. Karena itu polanya memakai awalan umum "_antrean" yang
# mencakup antrean_semua, antrean_v1, dan antrean_v2 sekaligus.
#
# Run yang sedang berjalan saat dihentikan akan tertinggal tanpa penanda
# metrices/aggregate_metrics.json, dan folder itu dihapus otomatis saat
# continue. Tidak ada data parsial yang masuk korpus.
set -u

POLA_BASH='_antrean'
POLA_PY='clean_code'

ps_filter() { # $1 nama proses  $2 pola CommandLine
  printf "Get-CimInstance Win32_Process -Filter \"Name='%s'\" -EA SilentlyContinue | Where-Object { \$_.CommandLine -match '%s' -and \$_.CommandLine -notmatch 'shell-snapshot' -and \$_.CommandLine -notmatch 'hentikan' }" "$1" "$2"
}

matikan() {
  powershell -NoProfile -Command "$(ps_filter "$1" "$2") | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -EA SilentlyContinue; 'dimatikan PID ' + \$_.ProcessId }" 2>/dev/null
}

cacah() {
  powershell -NoProfile -Command "@($(ps_filter "$1" "$2")).Count" 2>/dev/null | tr -d '\r'
}

for putaran in 1 2 3; do
  echo "--- putaran $putaran"
  matikan bash.exe   "$POLA_BASH"
  matikan python.exe "$POLA_PY"
  powershell -NoProfile -Command "Start-Sleep -Seconds 2" >/dev/null 2>&1

  sisa_bash=$(cacah bash.exe   "$POLA_BASH")
  sisa_py=$(cacah python.exe   "$POLA_PY")
  echo "    tersisa: bash=${sisa_bash:-?} python=${sisa_py:-?}"

  if [ "${sisa_bash:-1}" = "0" ] && [ "${sisa_py:-1}" = "0" ]; then
    echo "BERHENTI BERSIH. Jalankan _antrean/antrean_semua.sh lagi untuk melanjutkan."
    exit 0
  fi
done

echo "MASIH ADA PROSES HIDUP setelah tiga putaran."
echo "Jangan anggap sudah berhenti. Periksa manual sebelum melanjutkan."
exit 1
