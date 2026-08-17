#!/usr/bin/env bash
# Antrean korpus v_1 non-real-case, 13 dunia OFAT = 78 run.
#
# Per dunia ada enam sel, yaitu tiga pasangan agen negosiasi pada mode PADDED,
# satu greedy lawan greedy, satu tanpa pemesanan, dan satu pemesanan statis.
# Urutannya per dunia sampai tuntas, sehingga berhenti kapan pun meninggalkan
# baris tabel Bab V yang utuh, bukan setengah jadi.
#
# CATATAN PENTING soal run_experiment_bulk_per_config.py. Skrip itu tidak bisa
# melanjutkan, dan bila foldernya sudah ada ia membuat folder baru bersufiks
# angka. Karena itu di sini ia dipanggil SATU sub-eksperimen sekali ke folder
# singgah, lalu hasilnya dipindahkan ke folder dunia yang benar. Dengan begitu
# titik simpannya per sub-eksperimen, bukan per bulk.
#
# STOP     : bash _antrean/hentikan.sh
# CONTINUE : jalankan ulang skrip ini, apa adanya.
set -u

ROOT="c:/Users/ACER/Documents/Magister/thesis_final/clean_code"
PY="${PY:-$ROOT/.venv/Scripts/python.exe}"
LOGDIR="$ROOT/_antrean/logs"
cd "$ROOT" || exit 1
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
mkdir -p "$LOGDIR"

MIN_FREE_GB=${MIN_FREE_GB:-20}

# DUNIA KORPUS LAMA. Seluruh run v_1 melewati titik masuk di v_1/ yang memasang
# tambalannya sendiri, sehingga dunianya sama dengan korpus non-real-case lama.
# Tidak ada saklar untuk mematikannya, dan itu disengaja supaya tidak mungkin
# ada run v_1 yang tercampur dunia baru. Tidak ada kode di luar v_1/ yang
# disentuh.

# Akar data bisa dialihkan untuk uji kering, supaya penanda tiruan TIDAK PERNAH
# bisa mencemari korpus sungguhan. Pernah terjadi, 52 penanda palsu masuk.
DATA="${V1_DATA:-v_1/data}"

# Tiga belas dunia yang benar-benar muncul di laporan. Seluruhnya berseed 42.
DUNIA=(
  ref_n10_upd1_s42
  control_default_s42
  hz_n1_upd1_s42
  hz_n5_upd1_s42
  hz_n30_upd1_s42
  uf_n10_upd2_s42
  uf_n10_upd5_s42
  sc_scarce_n10_upd1_s42
  sc_abundant_n10_upd1_s42
  soc_low_n10_upd1_s42
  soc_high_n10_upd1_s42
  cc_alternating_n10_upd1_s42
  cc_fixedopp_n10_upd1_s42
)

# Tiga pasangan agen yang dilaporkan. Mode ENCODER TIDAK dijalankan.
PASANGAN=(6 8 9)

BULK_ROOT="$DATA/experiment_bulk"
SINGGAH="$DATA/_singgah"

ok=0; skip=0; fail=0

free_gb() { df -BG . 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4+0}'; }

gerbang_disk() {
  local f; f=$(free_gb)
  if [ -n "$f" ] && [ "$f" -lt "$MIN_FREE_GB" ]; then
    echo "=== BERHENTI, DISK TIPIS: sisa ${f} GB kurang dari ${MIN_FREE_GB} GB"
    echo "=== terhenti sebelum $1"
    echo "RINGKASAN ok=$ok skip=$skip fail=$fail"
    exit 2
  fi
}

# Memindahkan hasil dari folder singgah ke folder dunia.
#
# Pada Windows, ``mv`` sebuah folder GAGAL dengan "Permission denied" bila ada
# satu saja berkas di dalamnya yang masih dipegang proses lain, misalnya
# pemindai antivirus yang baru saja menyentuh berkas log ratusan MB, atau
# perintah pemantauan yang membaca lognya. Kegagalan itu sempat MENGHAPUS hasil
# run yang sudah jadi, yaitu kehilangan hampir empat jam komputasi. Karena itu
# pemindahan dicoba berulang, lalu jatuh ke salin-dan-hapus, dan folder singgah
# TIDAK PERNAH dihapus selama isinya belum benar-benar sampai tujuan.
#
# $1 dunia  $2 nomor pasangan berdigit dua. Mengembalikan 0 bila hasil lengkap
# sudah berada di tujuan.
selamatkan_singgah() {
  local d="$1" nn="$2"
  local asal="$SINGGAH/$d"
  local tujuan="$BULK_ROOT/$d"
  compgen -G "$asal/exp${nn}_padded_*/metrices/aggregate_metrics.json" >/dev/null || return 1

  mkdir -p "$tujuan"
  local percobaan
  for percobaan in 1 2 3 4 5; do
    if mv "$asal"/exp${nn}_padded_* "$tujuan/" 2>>"$LOGDIR/v1.err"; then
      break
    fi
    echo "[TUNDA] pemindahan $d/exp$nn gagal, percobaan $percobaan, menunggu 10 detik"
    powershell -NoProfile -Command "Start-Sleep -Seconds 10" >/dev/null 2>&1
  done

  # Jalan terakhir, salin lalu hapus. Lebih lambat tetapi tidak butuh rename
  # atas folder yang berkasnya sedang dipegang.
  if ! compgen -G "$tujuan/exp${nn}_padded_*/metrices/aggregate_metrics.json" >/dev/null; then
    echo "[SALIN] pemindahan $d/exp$nn tetap gagal, beralih ke salin-dan-hapus"
    cp -r "$asal"/exp${nn}_padded_* "$tujuan/" 2>>"$LOGDIR/v1.err"
  fi

  if compgen -G "$tujuan/exp${nn}_padded_*/metrices/aggregate_metrics.json" >/dev/null; then
    [ -f "$asal/bulk_results.csv" ] && \
      cp "$asal/bulk_results.csv" "$tujuan/bulk_results_exp${nn}.csv"
    rm -rf "$SINGGAH"
    echo "[SELAMAT] hasil $d/exp$nn berhasil diselamatkan dari folder singgah"
    return 0
  fi
  return 1
}

# $1 dunia  $2 nomor pasangan
jalankan_nego() {
  local d="$1" n="$2"
  local nn; nn=$(printf '%02d' "$n")
  local tujuan="$BULK_ROOT/$d"

  # Penanda selesai dicari lewat pola, sebab akhiran nama sub-eksperimen
  # ditentukan run_experiment_bulk_per_config.py sendiri.
  if compgen -G "$tujuan/exp${nn}_padded_*/metrices/aggregate_metrics.json" >/dev/null; then
    skip=$((skip + 1)); return 0
  fi
  gerbang_disk "$d / exp$nn"

  rm -rf "$tujuan"/exp${nn}_padded_*
  # Sisa singgah dari percobaan sebelumnya dibuang, KECUALI kalau isinya justru
  # hasil lengkap yang gagal dipindah. Itu diselamatkan lebih dulu di bawah.
  selamatkan_singgah "$d" "$nn" && { ok=$((ok + 1)); return 0; }
  rm -rf "$SINGGAH"
  mkdir -p "$SINGGAH"

  echo "[JALAN] $d / nego exp$nn   (disk sisa $(free_gb) GB, $(date '+%H:%M:%S'))"
  if "$PY" v_1/main_bulk_v1.py \
        --name "$d" --config "v_1/configs/$d.json" \
        --only "$n" --modes PADDED \
        --out-root "$SINGGAH" >/dev/null 2>>"$LOGDIR/v1.err"; then
    if selamatkan_singgah "$d" "$nn"; then
      ok=$((ok + 1))
    else
      # Hasilnya SENGAJA dibiarkan di folder singgah, bukan dihapus, supaya
      # komputasi yang sudah jadi bisa diselamatkan pada continue berikutnya.
      echo "[GAGAL] $d / exp$nn, hasil tertahan di folder singgah"
      fail=$((fail + 1))
    fi
  else
    echo "[GAGAL] $d / nego exp$nn"
    fail=$((fail + 1))
    rm -rf "$SINGGAH"
  fi
}

# $1 dunia  $2 sel: greedy | as_is | booking
jalankan_pendamping() {
  local d="$1" sel="$2" out cmd
  case "$sel" in
    greedy)
      out="$DATA/greedy/$d"
      cmd="$PY v_1/main_greedy_v1.py --config v_1/configs/$d.json --name $d --exp-root $DATA/greedy"
      ;;
    as_is)
      out="$DATA/as_is/$d"
      cmd="$PY v_1/main_nonneg_v1.py --mech as_is --name $d --config v_1/configs/$d.json --seed 42 --out-root $DATA/as_is"
      ;;
    booking)
      out="$DATA/booking/$d"
      cmd="$PY v_1/main_nonneg_v1.py --mech booking --name $d --config v_1/configs/$d.json --seed 42 --out-root $DATA/booking"
      ;;
  esac

  if [ -f "$out/metrices/aggregate_metrics.json" ]; then
    skip=$((skip + 1)); return 0
  fi
  gerbang_disk "$d / $sel"

  rm -rf "$out"
  echo "[JALAN] $d / $sel   (disk sisa $(free_gb) GB, $(date '+%H:%M:%S'))"
  if $cmd >/dev/null 2>>"$LOGDIR/v1.err"; then
    ok=$((ok + 1))
  else
    echo "[GAGAL] $d / $sel"
    fail=$((fail + 1))
  fi
}

# Gerbang config. 13 dunia, tidak lebih dan tidak kurang.
for d in "${DUNIA[@]}"; do
  if [ ! -f "v_1/configs/$d.json" ]; then
    echo "[GAGAL] config hilang: v_1/configs/$d.json"; exit 1
  fi
done
n_cfg=$(ls v_1/configs/*.json 2>/dev/null | wc -l)
if [ "$n_cfg" -ne 13 ]; then
  echo "[PERINGATAN] v_1/configs berisi $n_cfg berkas, seharusnya 13."
fi

echo "===== ANTREAN v_1 mulai $(date '+%Y-%m-%d %H:%M:%S')"
echo "      13 dunia, disk sisa $(free_gb) GB, ambang $MIN_FREE_GB GB"

for d in "${DUNIA[@]}"; do
  for n in "${PASANGAN[@]}"; do
    jalankan_nego "$d" "$n"
  done
  jalankan_pendamping "$d" greedy
  jalankan_pendamping "$d" booking
  jalankan_pendamping "$d" as_is
done

echo "===== ANTREAN v_1 SELESAI ok=$ok skip=$skip fail=$fail, disk sisa $(free_gb) GB"
