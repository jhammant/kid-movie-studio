#!/usr/bin/env bash
# Key a long green-screen take in 3 parallel time slices, join them, then level the kid's audio.
#   tools/render_parallel.sh <src> <clean_plate_src> <bg_loop> <start> <dur> <out.mp4> [green_out.mp4]
# Each slice keys 1s early (--preroll) so the matte has settled before it starts writing:
# no visible seam at the joins. Roughly 3x faster than one keyer.py run.
set -euo pipefail
src=$1; plate=$2; bg=$3; start=$4; dur=$5; out=$6; gout=${7:-}
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
third=$(python3 -c "print(round($dur / 3, 2))")
for k in 0 1 2; do
  s=$(python3 -c "print(round($start + $k * $third, 2))")
  d=$(python3 -c "print(round($dur - 2 * $third, 2) if $k == 2 else $third)")
  b=$(python3 -c "print(round($k * $third, 2))")
  python3 "$here/keyer.py" "$src" "$work/c$k.mp4" --plate "$plate" --start "$s" --dur "$d" \
    --preroll 1.0 --bg "$bg" --bg-start "$b" --green-out "$work/g$k.mp4" > "$work/log$k.txt" 2>&1 &
done
wait
for p in c g; do
  printf "file '%s'\n" "$work/${p}0.mp4" "$work/${p}1.mp4" "$work/${p}2.mp4" > "$work/$p.txt"
  ffmpeg -v error -y -f concat -safe 0 -i "$work/$p.txt" -c copy "$work/${p}_all.mp4"
  python3 "$here/loudness.py" "$work/${p}_all.mp4" -15 --audio-from "$src" --start "$start" --dur "$dur"
done
mv "$work/c_all.mp4" "$out"
[ -n "$gout" ] && mv "$work/g_all.mp4" "$gout"
rm -rf "$work"
