# WebP-Metal

WebP-Metal is a macOS/Apple-silicon port of the CUDA lossless-encoding work in
the CMU 15-418 final project at `/Users/jonas/Documents/15418-Final-Project`.
It builds a real `cwebp` encoder and accelerates libwebp's cross-color search
with a Metal compute kernel.

The port is intentionally not a line-for-line CUDA translation:

- One Metal threadgroup owns one lossless transform tile.
- 256 threads cooperatively process up to 32x32 pixels and build threadgroup
  histograms with local atomics.
- The Metal device, runtime-compiled pipeline, command queue, and shared
  buffers are cached and reused.
- Apple unified memory is used through `MTLStorageModeShared`; the encoder does
  one upload and one download for the complete transform, rather than copying
  for every candidate.
- Small images and all Metal failures transparently use libwebp's original C
  implementation.

## Build

Requirements:

- macOS with a Metal-capable GPU
- Apple Command Line Tools (`clang`/`clang++`)
- Homebrew `libpng` and `jpeg-turbo` (the standard Homebrew include and library
  paths are already configured)

```sh
make metal -j8
```

This creates `cwebp-metal` and `dwebp-metal` symlinks in the project root.
The Metal shader is compiled once per process by the Metal runtime, so the
offline `metal` compiler and a full Xcode installation are not required.

## Encode

```sh
./cwebp-metal -lossless -m 6 input.png -o output.webp
```

Metal is enabled by default for images of at least 65,536 pixels. Environment
controls:

- `WEBP_METAL=0` disables Metal and provides a CPU baseline.
- `WEBP_METAL_MIN_PIXELS=N` changes the crossover threshold; use `0` to force
  Metal for tests.
- `WEBP_METAL_VERBOSE=1` prints the selected GPU and transform timing.

## Verify and benchmark

```sh
scripts/test.sh
scripts/benchmark.sh /path/to/image.png 5 6
```

`test.sh` encodes through both paths, decodes both files, and requires the
decoded pixels to be byte-identical. `benchmark.sh` reports average complete
CLI runtime, speedup, and output size for CPU and Metal.

On an Apple M4 Max with the reference project's 2876x1572 `mitski.png`, method
4 took 0.96 s on CPU and 0.55 s through Metal (1.75x end-to-end speedup). The
Metal transform itself took about 24 ms. Method 6 took 1.22 s versus 0.72 s
(1.69x). Timings include per-process shader compilation.

## Compression tradeoff

Like the final CUDA implementation, this kernel evaluates tiles independently
so all tiles can run concurrently. It does not use the serial previous-tile and
accumulated-image histogram heuristics from upstream libwebp. Decoded pixels
remain exactly lossless, but the bitstream and compressed size can differ. On
the sample above, Metal output was 0.3% larger at method 4 and 0.5% larger at
method 6. Set `WEBP_METAL=0` when the smallest possible file matters more than
encode latency.

The vendored libwebp 1.0.3 source remains under its original BSD-style license;
see `COPYING` and `PATENTS`.
