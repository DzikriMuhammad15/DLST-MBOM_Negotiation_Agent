#!/usr/bin/env bash
# Antrean korpus v_2 real-case, 4 keluarga skenario x 40 ulangan = 1.120 run.
#
# AMAN DIHENTIKAN KAPAN SAJA. Penanda selesai satu run adalah
# metrices/aggregate_metrics.json. Run yang sudah punya penanda dilewati, dan
# folder tanpa penanda dihapus lebih dulu sebelum dijalankan, sehingga sisa run
# parsial tidak pernah tercampur ke dalam korpus.
#
# Urutannya diselang per NOMOR RUN, bukan per skenario. Berhenti kapan pun
# menghasilkan jumlah ulangan yang seimbang di seluruh sel.
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

RUN_LO=${RUN_LO:-1}
RUN_HI=${RUN_HI:-40}
MIN_FREE_GB=${MIN_FREE_GB:-20}

SKENARIO=(medium_occupancy high_occupancy
          mahalkompen_medium_occupancy mahalkompen_high_occupancy)
AGEN=(DLST DLSTMBOM DLSTMBOMNewMechanism DLSTMBOMDirected)

ok=0; skip=0; fail=0

free_gb() { df -BG . 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4+0}'; }

slug() {
  case "$1" in
    DLST)                 echo dlst ;;
    DLSTMBOM)             echo dlstmbom ;;
    DLSTMBOMNewMechanism) echo dlstmbomnew ;;
    DLSTMBOMDirected)     echo directed ;;
  esac
}

# $1 skenario  $2 sel  $3 nomor run
# sel: nego:<AGEN> | greedy | nonneg:<mech>
jalankan_satu() {
  local sc="$1" sel="$2" r="$3" out cmd
  case "$sel" in
    nego:*)
      local agen="${sel#nego:}"
      out="v_2/data_${sc}/negotiation/real_case_v2/run${r}_padded_$(slug "$agen")"
      cmd="$PY v_2/main_nego.py --agent-fleet $agen --run $r --skenario $sc --lean --online-log-full"
      ;;
    greedy)
      out="v_2/data_${sc}/negotiation/real_case_v2/run${r}_padded_greedy"
      cmd="$PY v_2/main_greedy.py --run $r --skenario $sc --lean"
      ;;
    nonneg:*)
      local mech="${sel#nonneg:}"
      out="v_2/data_${sc}/non_negotiation/real_case_v2/${mech}/run${r}"
      cmd="$PY v_2/main_nonneg_v2.py --mech $mech --run $r --skenario $sc --lean"
      ;;
  esac

  if [ -f "$out/metrices/aggregate_metrics.json" ]; then
    skip=$((skip + 1)); return 0
  fi

  local f; f=$(free_gb)
  if [ -n "$f" ] && [ "$f" -lt "$MIN_FREE_GB" ]; then
    echo "=== BERHENTI, DISK TIPIS: sisa ${f} GB kurang dari ${MIN_FREE_GB} GB"
    echo "=== terhenti sebelum $sc / $sel / run$r"
    echo "RINGKASAN ok=$ok skip=$skip fail=$fail"
    exit 2
  fi

  rm -rf "$out"
  echo "[JALAN] $sc / $sel / run$r   (disk sisa ${f} GB, $(date '+%H:%M:%S'))"
  if $cmd >/dev/null 2>>"$LOGDIR/v2.err"; then
    ok=$((ok + 1))
  else
    echo "[GAGAL] $sc / $sel / run$r"
    fail=$((fail + 1))
  fi
}

# Gerbang config. Tanpa 160 config, seluruh antrean sia-sia.
n_cfg=$(ls v_2/configs/run*_*.json 2>/dev/null | wc -l)
if [ "$n_cfg" -lt 160 ]; then
  echo "[GAGAL] config v_2 hanya $n_cfg dari 160. Jalankan dulu: python v_2/gen_configs.py"
  exit 1
fi

echo "===== ANTREAN v_2 mulai $(date '+%Y-%m-%d %H:%M:%S')"
echo "      run $RUN_LO sampai $RUN_HI, disk sisa $(free_gb) GB, ambang $MIN_FREE_GB GB"

for r in $(seq "$RUN_LO" "$RUN_HI"); do
  for sc in "${SKENARIO[@]}"; do
    for agen in "${AGEN[@]}"; do
      jalankan_satu "$sc" "nego:$agen" "$r"
    done
    jalankan_satu "$sc" "greedy" "$r"
    jalankan_satu "$sc" "nonneg:booking" "$r"
    jalankan_satu "$sc" "nonneg:as_is" "$r"
  done
done

echo "===== ANTREAN v_2 SELESAI ok=$ok skip=$skip fail=$fail, disk sisa $(free_gb) GB"
