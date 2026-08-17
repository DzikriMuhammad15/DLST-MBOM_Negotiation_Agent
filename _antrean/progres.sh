#!/usr/bin/env bash
# Melaporkan kemajuan kampanye dengan menghitung penanda selesai di disk.
# Aman dijalankan kapan saja, termasuk saat antrean sedang berjalan.
set -u

ROOT="c:/Users/ACER/Documents/Magister/thesis_final/clean_code"
cd "$ROOT" || exit 1

hitung() { # $1 pola glob penanda
  local n=0
  for m in $1; do [ -f "$m" ] && n=$((n + 1)); done
  echo "$n"
}

echo "===== v_1, target 78 run"
tot1=0
for sel in exp06 exp08 exp09; do
  n=$(hitung "v_1/data/experiment_bulk/*/${sel}_padded_*/metrices/aggregate_metrics.json")
  printf '  nego %-6s %2d / 13\n' "$sel" "$n"; tot1=$((tot1 + n))
done
for sel in greedy booking as_is; do
  n=$(hitung "v_1/data/$sel/*/metrices/aggregate_metrics.json")
  printf '  %-11s %2d / 13\n' "$sel" "$n"; tot1=$((tot1 + n))
done
printf '  SUBTOTAL v_1 %d / 78\n' "$tot1"

echo
echo "===== v_2, target 1.120 run"
tot2=0
for sc in medium_occupancy high_occupancy \
          mahalkompen_medium_occupancy mahalkompen_high_occupancy; do
  sub=0
  echo "  --- $sc"
  for slug in dlst dlstmbom dlstmbomnew directed greedy; do
    n=$(hitung "v_2/data_${sc}/negotiation/real_case_v2/run*_padded_${slug}/metrices/aggregate_metrics.json")
    printf '      %-12s %2d / 40\n' "$slug" "$n"; sub=$((sub + n))
  done
  for mech in booking as_is; do
    n=$(hitung "v_2/data_${sc}/non_negotiation/real_case_v2/${mech}/run*/metrices/aggregate_metrics.json")
    printf '      %-12s %2d / 40\n' "$mech" "$n"; sub=$((sub + n))
  done
  printf '      subtotal   %3d / 280\n' "$sub"; tot2=$((tot2 + sub))
done
printf '  SUBTOTAL v_2 %d / 1120\n' "$tot2"

echo
printf 'TOTAL %d / 1198 run\n' "$((tot1 + tot2))"
echo "disk sisa: $(df -BG . 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4+0}') GB"
