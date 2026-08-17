# WebP-Metal

WebP-Metal is a macOS/Apple-silicon port of the CUDA lossless-encoding work in
the CMU 15-418 final project at `/Users/jonas/Documents/15418-Final-Project`.
It builds a real `cwebp` encoder and accelerates lossless cross-color search,
lossless backward-reference matching, and lossy opaque RGB-to-YUV420
conversion with Metal compute kernels. On AArch64 it also backports selected
newer libwebp NEON encoder kernels.

The port is intentionally not a line-for-line CUDA translation:

- One Metal threadgroup owns one lossless transform tile.
- 256 threads cooperatively process up to 32x32 pixels and build threadgroup
  histograms with local atomics.
- Lossless hash-chain candidates are searched independently on the GPU, then
  the original left-extension/skip decisions are replayed on the CPU. This
  makes the backward-reference result byte-exact with the CPU path.
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
./cwebp-metal -q 75 -m 4 input.jpg -o output.webp
```

The optimized CLI enables libwebp multithreading by default. Use `-no_mt` for
a single-thread baseline; `-mt` is still accepted explicitly. This does not
change the public library's `WebPConfig` default.

`WEBP_NEON_INTRA4=0` and `WEBP_NEON_INTRA16=0` disable the corresponding arm64
predictor backports for A/B benchmarking. They are enabled by default. Intra4
improved complete lossy encode time by roughly 0.4-2.4% and intra16 by
0.2-1.2% on the measured set without changing the bitstream.

Walsh-Hadamard DC quantization now uses the existing compatible AArch64 NEON
quantizer instead of scalar C, improving meaningful complete lossy encodes by
0.8-1.5% with byte-identical output. Native AArch64 vector horizontal sums add
a further 0.2-0.5% on several method-4 inputs and are neutral elsewhere.

The method-6 trellis hot loop also includes two exact upstream optimizations.
Reduced scoring work improves complete lossy encodes by 0.3-1.6%; fixed-size
coefficient clears remove two hot-loop `bzero` calls and add another 2.6-6.6%
on the measured set. They have no GPU startup cost and produce byte-identical
files.

`WEBP_NEON_LOSSLESS_PREDICTORS=0` disables the arm64 lossless predictor 9-12
micro-optimizations. They improved complete Metal-enabled lossless encoding by
1.5-2.1% at method 4 on the measured set, with byte-identical output.

Lossless histogram copying no longer transfers the variable-length literal
counts twice. Its complete-encode effect is content-dependent (0.996-1.029x in
the measured set), but it unconditionally removes redundant memory traffic and
is retained for persistent/batch workloads.

Metal is enabled by default for images of at least 65,536 pixels. Environment
controls:

- `WEBP_METAL=0` disables Metal and provides a CPU baseline.
- `WEBP_METAL_MIN_PIXELS=N` changes the crossover threshold; use `0` to force
  Metal for tests.
- `WEBP_METAL_VERBOSE=1` prints the selected GPU and transform timing.

Lossless hash-chain search has separate controls:

- `WEBP_METAL_HASH=0` disables the exact GPU match search.
- `WEBP_METAL_HASH_MIN_PIXELS=N` changes its conservative 4,000,000-pixel
  threshold. Use `0` for warmed batch encoders or forced tests. Below the
  threshold, the benefit is content-dependent and shader startup can dominate
  a one-shot encode.
- `WEBP_METAL_VERBOSE=1` also prints each hash-candidate command timing.

On the M4 Max, the hash accelerator improved complete one-shot method-4
lossless encodes by 1.41x on 4.2-4.5 MP inputs, 1.61x on Corgi (5.9 MP), and
1.76x on Siamese (9.2 MP). Method 6 improved by 1.30x, 1.45x, and 1.62x,
respectively. It produces byte-identical WebP files. The 4 MP default avoids
measurable regressions on small or easy inputs; batch callers may set the
threshold to zero to reuse the lazily compiled pipeline.

Lossy RGB-to-YUV conversion has separate controls because its best use case is
a persistent batch encoder:

- `WEBP_METAL_LOSSY=0` disables the lossy Metal import path.
- `WEBP_METAL_LOSSY_MIN_PIXELS=N` sets its threshold. The conservative default
  is 80,000,000 pixels, which avoids runtime shader-compilation regressions in
  ordinary one-shot CLI encodes. Use `0` in a batch process to reuse the cached
  pipeline and buffers.
- `WEBP_METAL_VERBOSE=1` also prints lossy conversion command timing.

On the M4 Max, warmed `WebPPictureImportRGB` improved by 4.55x at 6 MP and
4.95x at 12 MP. The lossy output is bit-identical to the CPU path. See
`PERFORMANCE_LOG.md` for cold-start crossover data and rejected variants.

## Verify and benchmark

```sh
scripts/test.sh
scripts/test_hash_chain.sh /path/to/image.png [/path/to/another.jpg]
scripts/test_lossy.sh /path/to/image.jpg [/path/to/another.png]
scripts/benchmark.sh /path/to/image.png 5 6
make benchmark-lossy-import
WEBP_METAL_LOSSY=0 build/benchmark_lossy_import 4000 3000 30
WEBP_METAL_LOSSY_MIN_PIXELS=0 build/benchmark_lossy_import 4000 3000 30
```

`test.sh` encodes through both transform paths, decodes both files, and requires
the decoded pixels to be byte-identical. `test_hash_chain.sh` forces CPU and
Metal hash matching at methods 0-6 and requires byte-identical WebP files.
`benchmark.sh` reports average complete CLI runtime, speedup, and output size
for CPU and Metal.

Current cumulative lossless results on the Apple M4 Max, with both Metal
stages enabled and per-process shader compilation included:

| Input | Method | CPU | All Metal | Speedup |
|---|---:|---:|---:|---:|
| `mitski.png` | 4 | 1.030 s | 0.433 s | **2.38x** |
| `corgi.jpeg` | 4 | 1.826 s | 0.872 s | **2.10x** |
| `siamese.jpg` | 4 | 3.398 s | 1.430 s | **2.38x** |
| `mitski.png` | 6 | 1.306 s | 0.604 s | **2.16x** |
| `corgi.jpeg` | 6 | 2.432 s | 1.178 s | **2.06x** |
| `siamese.jpg` | 6 | 4.131 s | 1.760 s | **2.35x** |

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
