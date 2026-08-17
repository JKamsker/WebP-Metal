#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 INPUT [ITERATIONS=3] [METHOD=4]" >&2
  exit 2
fi

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
input=$1
iterations=${2:-3}
method=${3:-4}
temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/webp-metal-bench.XXXXXX")
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM

case $iterations in
  ''|*[!0-9]*) echo "iterations must be a positive integer" >&2; exit 2 ;;
esac
if [ "$iterations" -lt 1 ]; then
  echo "iterations must be a positive integer" >&2
  exit 2
fi

make -C "$project_dir" metal -j8 >/dev/null
: >"$temporary_dir/cpu-times"
: >"$temporary_dir/metal-times"

i=1
while [ "$i" -le "$iterations" ]; do
  /usr/bin/time -p -o "$temporary_dir/cpu-time" \
    env WEBP_METAL=0 "$project_dir/cwebp-metal" -quiet -lossless -exact \
      -m "$method" "$input" -o "$temporary_dir/cpu.webp"
  awk '$1 == "real" { print $2 }' "$temporary_dir/cpu-time" \
    >>"$temporary_dir/cpu-times"

  /usr/bin/time -p -o "$temporary_dir/metal-time" \
    env WEBP_METAL_MIN_PIXELS=0 "$project_dir/cwebp-metal" -quiet -lossless \
      -exact -m "$method" "$input" -o "$temporary_dir/metal.webp"
  awk '$1 == "real" { print $2 }' "$temporary_dir/metal-time" \
    >>"$temporary_dir/metal-times"
  i=$((i + 1))
done

cpu_average=$(awk '{ total += $1 } END { printf "%.4f", total / NR }' \
  "$temporary_dir/cpu-times")
metal_average=$(awk '{ total += $1 } END { printf "%.4f", total / NR }' \
  "$temporary_dir/metal-times")
speedup=$(awk -v cpu="$cpu_average" -v metal="$metal_average" \
  'BEGIN { printf "%.2f", cpu / metal }')

"$project_dir/dwebp-metal" -quiet "$temporary_dir/cpu.webp" -pam \
  -o "$temporary_dir/cpu.pam"
"$project_dir/dwebp-metal" -quiet "$temporary_dir/metal.webp" -pam \
  -o "$temporary_dir/metal.pam"
cmp "$temporary_dir/cpu.pam" "$temporary_dir/metal.pam"

printf 'Input:             %s\n' "$input"
printf 'Iterations/method: %s / %s\n' "$iterations" "$method"
printf 'CPU average:       %s s\n' "$cpu_average"
printf 'Metal average:     %s s\n' "$metal_average"
printf 'Speedup:           %sx\n' "$speedup"
printf 'CPU output:        %s bytes\n' \
  "$(stat -f %z "$temporary_dir/cpu.webp")"
printf 'Metal output:      %s bytes\n' \
  "$(stat -f %z "$temporary_dir/metal.webp")"
printf 'Decoded pixels:    identical\n'
