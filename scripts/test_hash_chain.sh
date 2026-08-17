#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/webp-metal-hash-test.XXXXXX")
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM

make -C "$project_dir" metal -j8 >/dev/null

if [ "$#" -eq 0 ]; then
  set -- "$project_dir/examples/test_ref.ppm"
fi

for input do
  for method in 0 1 2 3 4 5 6; do
    WEBP_METAL_HASH=0 WEBP_METAL_HASH_MIN_PIXELS=0 \
      "$project_dir/cwebp-metal" -quiet -lossless -exact -m "$method" \
      "$input" -o "$temporary_dir/cpu.webp"
    WEBP_METAL_HASH=1 WEBP_METAL_HASH_MIN_PIXELS=0 \
      "$project_dir/cwebp-metal" -quiet -lossless -exact -m "$method" \
      "$input" -o "$temporary_dir/metal.webp"
    cmp "$temporary_dir/cpu.webp" "$temporary_dir/metal.webp"
  done
  printf 'PASS: exact CPU and Metal hash-chain output for methods 0-6: %s\n' \
    "$input"
done
