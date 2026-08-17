#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
  echo "usage: $0 image [image ...]" >&2
  exit 2
fi

root_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
encoder="$root_dir/examples/cwebp"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/webp-metal-lossy.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM

if [ ! -x "$encoder" ]; then
  echo "build the encoder first with: make metal" >&2
  exit 2
fi

for input in "$@"; do
  name=$(basename -- "$input")
  for settings in "25 0" "75 4" "95 6"; do
    set -- $settings
    quality=$1
    method=$2
    cpu="$test_dir/$name-q$quality-m$method-cpu.webp"
    metal="$test_dir/$name-q$quality-m$method-metal.webp"
    WEBP_METAL_LOSSY=0 "$encoder" -quiet -q "$quality" -m "$method" \
      "$input" -o "$cpu"
    WEBP_METAL_LOSSY=1 WEBP_METAL_LOSSY_MIN_PIXELS=0 \
      "$encoder" -quiet -q "$quality" -m "$method" "$input" -o "$metal"
    cmp "$cpu" "$metal"
  done
  echo "lossy bitstream exact: $input"
done
