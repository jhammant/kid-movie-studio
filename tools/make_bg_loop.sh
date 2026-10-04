#!/usr/bin/env bash
# Loop a short background clip into a long, seamless 1080p plate for the keyer.
#   tools/make_bg_loop.sh <clip> <out.mp4> [copies=3] [xfade=0.6]
# Each copy crossfades into the next so the loop point doesn't jump.
set -euo pipefail
src=$1; out=$2; n=${3:-3}; xf=${4:-0.6}
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$src")
inputs=(); chain=""; prev="[v0]"
for ((i = 0; i < n; i++)); do
  inputs+=(-i "$src")
  chain+="[$i:v]scale=1920:1080:flags=area,fps=25,format=yuv420p,settb=AVTB[v$i];"
done
for ((i = 1; i < n; i++)); do
  off=$(python3 -c "print(round($i * ($dur - $xf), 3))")
  chain+="${prev}[v$i]xfade=transition=fade:duration=$xf:offset=$off[x$i];"
  prev="[x$i]"
done
ffmpeg -v error -y "${inputs[@]}" -filter_complex "${chain%;}" -map "$prev" -an \
  -c:v libx264 -crf 14 -preset fast "$out"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$out"
