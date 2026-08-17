#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
input=${1:-"$project_dir/examples/test_ref.ppm"}
temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/webp-metal-test.XXXXXX")
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM

make -C "$project_dir" metal -j8 >/dev/null

WEBP_METAL=0 "$project_dir/cwebp-metal" -quiet -lossless -exact -m 4 \
  "$input" -o "$temporary_dir/cpu.webp"
WEBP_METAL_MIN_PIXELS=0 "$project_dir/cwebp-metal" -quiet -lossless -exact \
  -m 4 "$input" -o "$temporary_dir/metal.webp"

"$project_dir/dwebp-metal" -quiet "$temporary_dir/cpu.webp" -pam \
  -o "$temporary_dir/cpu.pam"
"$project_dir/dwebp-metal" -quiet "$temporary_dir/metal.webp" -pam \
  -o "$temporary_dir/metal.pam"

cmp "$temporary_dir/cpu.pam" "$temporary_dir/metal.pam"
printf 'PASS: CPU and Metal decode to identical pixels\n'
printf 'CPU WebP:   %s bytes\n' "$(stat -f %z "$temporary_dir/cpu.webp")"
printf 'Metal WebP: %s bytes\n' "$(stat -f %z "$temporary_dir/metal.webp")"
