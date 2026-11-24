#!/usr/bin/env bash
# Spec 011 line C5 supplement: cmix on one full matrix (L0.E191.w1, 2304 x 5120), not cropped.
#   nohup setsid bash experiments/011-limits-and-models/c5_cmix_full.sh run a &      # codes only
#   nohup setsid bash experiments/011-limits-and-models/c5_cmix_full.sh run b &      # scale byte before every block
#   nohup setsid bash experiments/011-limits-and-models/c5_cmix_full.sh watch &      # progress every 30 min -> timing.md
# Inputs: data/011/c5/in/<name>.<fmt>, written by `c5_generic.py prep` (unchanged). Tool: the same cmix build as C5
# (data/011/tools/cmix, commit 194af9c), default parameters. Encoding and decoding run under /usr/bin/time -v;
# the decoded file is compared byte for byte (cmp) with the input.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NAME=L0.E191.w1
D="$ROOT/data/011/c5"
OD="$D/cmix_full"
CMIX="$ROOT/data/011/tools/cmix/cmix"
TIMING="$ROOT/experiments/011-limits-and-models/timing.md"
mkdir -p "$OD"

run() {
  fmt=$1
  src="$D/in/$NAME.$fmt"; out="$OD/$NAME.$fmt.cmix"; back="$OD/$NAME.$fmt.back"; ev="$OD/$NAME.$fmt.events"
  : > "$ev"
  echo "SRC_SHA256 $(sha256sum "$src" | cut -d' ' -f1) $(stat -c %s "$src")" >> "$ev"
  echo "ENC_START $(date +%s)" >> "$ev"
  /usr/bin/time -v -o "$OD/$NAME.$fmt.enc.time" "$CMIX" -c "$src" "$out" > "$OD/$NAME.$fmt.enc.log" 2>&1
  echo "ENC_END $(date +%s) $(stat -c %s "$out")" >> "$ev"
  echo "DEC_START $(date +%s)" >> "$ev"
  /usr/bin/time -v -o "$OD/$NAME.$fmt.dec.time" "$CMIX" -d "$out" "$back" > "$OD/$NAME.$fmt.dec.log" 2>&1
  echo "DEC_END $(date +%s)" >> "$ev"
  if cmp -s "$src" "$back"; then echo "LOSSLESS 1" >> "$ev"; rm -f "$back"; else echo "LOSSLESS 0" >> "$ev"; fi
  echo "DONE $(date +%s)" >> "$ev"
}

progress_line() {
  fmt=$1; ev="$OD/$NAME.$fmt.events"
  [ -f "$ev" ] || { echo "  - 格式 ($fmt)：未启动"; return; }
  t0=$(awk '/^ENC_START/{print $2}' "$ev"); now=$(date +%s); el=$(( (now - t0) / 60 ))
  if grep -q '^DONE' "$ev"; then echo "  - 格式 ($fmt)：已完成（$(grep '^LOSSLESS' "$ev")）"; return; fi
  if grep -q '^DEC_START' "$ev"; then ph=解码; log="$OD/$NAME.$fmt.dec.log"; f="$OD/$NAME.$fmt.back"
  else ph=编码; log="$OD/$NAME.$fmt.enc.log"; f="$OD/$NAME.$fmt.cmix"; fi
  pct=$(tail -c 200 "$log" 2>/dev/null | tr '\r' '\n' | grep -o 'progress: [0-9.]*%' | tail -1 | cut -d' ' -f2)
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  echo "  - 格式 ($fmt)：${ph}中，cmix 进度 ${pct:-?}，输出文件 ${sz} 字节，从编码开始已用 ${el} 分钟"
}

watch() {
  while true; do
    { echo "- $(date '+%Y-%m-%d %H:%M')，负载 $(cut -d' ' -f1-3 /proc/loadavg)："; progress_line a; progress_line b; } >> "$TIMING"
    if grep -qs '^DONE' "$OD/$NAME.a.events" && grep -qs '^DONE' "$OD/$NAME.b.events"; then break; fi
    sleep 1800
  done
}

case "$1" in
  run) run "$2" ;;
  watch) watch ;;
esac
