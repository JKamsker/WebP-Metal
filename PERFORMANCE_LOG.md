# WebP-Metal performance log

This is the running record of performance work: the invariant tested, the
measurement, and the keep/reject decision. Times are wall-clock averages on an
Apple M4 Max unless stated otherwise. Lower is better; speedup is CPU time
divided by accelerated time.

## Measurement rules

- Compare the same source revision, input, quality, and method.
- Alternate CPU and accelerated trials when measuring complete CLI runtime.
- Use at least five trials for short tests and record cold versus warmed state.
- Lossy changes must preserve the CPU encoder's bitstream unless a quality or
  size tradeoff is explicitly approved and documented.
- Lossless changes must decode to byte-identical pixels. A different compressed
  bitstream is acceptable only when its size impact is measured.
- Keep batch-only improvements, but identify their initialization/amortization
  requirements and retain a safe one-shot fallback.

## Kept: lossless cross-color transform in Metal

The first implementation replaces `VP8LColorSpaceTransform_C` with a
tile-parallel Metal kernel. It caches the device, pipeline, command queue, and
shared buffers. Each 32x32 tile is independent, matching the final CUDA
project's parallelization choice.

### Results

| Scope/input | CPU | Metal | Speedup |
|---|---:|---:|---:|
| Transform, `layout.png` | 119.274 ms | 6.905 ms | **17.27x** |
| Transform, `mitski.png` | 440.427 ms | 10.796 ms | **40.80x** |
| Transform, `corgi.jpeg` | 482.536 ms | 11.509 ms | **41.93x** |
| Complete lossless encode, `mitski.png`, method 4 | 0.960 s | 0.545 s | **1.76x** |
| Complete lossless encode, `mitski.png`, method 6 | 1.220 s | 0.720 s | **1.69x** |

Decision: **kept**. RGB and RGBA tests at methods 0 through 6 decoded to
byte-identical pixels. Independent tile decisions changed compressed size by
-0.15% to +0.28% on the three-image set; see `CUDA_METAL_COMPARISON.md`.

## Kept: opaque RGB-to-YUV420 lossy import in Metal

The lossy path converts opaque RGB/BGR input to WebP's YUV420 planes. The Metal
kernel reproduces libwebp 1.0.3's integer luma equations, gamma lookup and
interpolation, 2x2 chroma subsampling, and edge replication exactly. Alpha,
dithered, sharp-YUV, small-image, invalid-input, and Metal-error cases use the
existing CPU implementation.

### Correctness

At quality 75/method 4, CPU and Metal produced byte-identical `.webp` files for
`corgi.jpeg`, `twinpeaks.jpg`, `siamese.jpg`, `noise.png`, and `mitski.png`.
The 10.5 MP `twinpeaks.jpg` outputs had the same SHA-256:
`1045cdb0817edd9d6b9785c1d6269a71ad05c18f589f679ba38a9705732eaaa2`.

### Warmed batch/library result

`benchmark_lossy_import` times repeated `WebPPictureImportRGB` calls inside one
process, after one warm-up call. It excludes input decoding and VP8 compression.

| Synthetic RGB input | CPU import | Metal import | Speedup |
|---|---:|---:|---:|
| 3000x2000 (6 MP), 50 iterations | 4.957 ms | 1.089 ms | **4.55x** |
| 4000x3000 (12 MP), 30 iterations | 9.909 ms | 2.003 ms | **4.95x** |

This is primarily a batch/persistent-process improvement. The device, runtime-
compiled shader, queue, and buffers are reused after the first call.

### Cold first-call result and default policy

Only the first import call is timed below, including Metal shader compilation
and buffer allocation. The crossover on this M4 Max is approximately 70-80 MP.

| Synthetic RGB input | CPU import | Metal import | Speedup |
|---|---:|---:|---:|
| 4000x3000 (12 MP) | 10.32 ms | 34.58 ms | 0.30x |
| 8192x8192 (67 MP) | 59.31 ms | 60.34 ms | 0.98x |
| 9000x9000 (81 MP) | 71.63 ms | 66.05 ms | **1.08x** |
| 12000x10000 (120 MP) | 108.14 ms | 78.47 ms | **1.38x** |
| 15000x11878 (178 MP) | 158.99 ms | 97.53 ms | **1.63x** |

The default minimum is therefore 80,000,000 pixels. Smaller one-shot encodes
fall back to CPU and avoid the roughly 25 ms shader startup cost. Batch users
should set `WEBP_METAL_LOSSY_MIN_PIXELS=0` to amortize initialization and use
the 4.5-5x warmed conversion path. `WEBP_METAL_LOSSY=0` disables it.

Decision: **kept**, including its batch-only benefit, with a conservative cold-
start threshold. When forcibly enabled for ordinary 4.5-10.5 MP one-image CLI
encodes, runtime compilation regressed total time by 4-18%; that variant was
rejected as the default policy rather than hiding the result.

For the real 15000x11878 `starry_night_crop.jpg` at quality 75/method 4,
three alternating complete CLI trials averaged 87.894 s CPU and 87.414 s
Metal: **1.005x** end to end. The `.webp` files were byte-identical. The small
whole-encoder gain is expected because conversion is under 0.2% of this
encode; macroblock analysis and entropy coding dominate.

## Tried: current libwebp CLI versus vendored 1.0.3

Homebrew `cwebp` 1.5 was compared with the vendored CPU encoder at quality 75,
method 4. Approximate complete times were 0.04 vs 0.053 s (`layout.png`), 0.14
vs 0.13 s (`mitski.png`), and 0.31 vs 0.29 s (`corgi.jpeg`).

Decision: **not adopted as a performance change**. It was not a consistent win
on this small sample and would confound comparisons with the original CUDA
fork. Individual newer optimizations remain candidates for selective porting.

## Kept: multithreaded CLI default

The existing `-mt` option was measured with Metal lossy conversion disabled.

| Input, quality 75/method 4 | Single-thread default | `-mt` | Speedup |
|---|---:|---:|---:|
| `corgi.jpeg` | 0.3025 s | 0.2893 s | **1.05x** |
| `mitski.png` | 0.1396 s | 0.1289 s | **1.08x** |
| `twinpeaks.jpg` | 0.5880 s | 0.5677 s | **1.04x** |
| `siamese.jpg` | 0.4413 s | 0.4193 s | **1.05x** |

The optimized `cwebp-metal` CLI now enables this existing worker path by
default. `-no_mt` restores a single-thread baseline and `-mt` remains accepted.
Lossy and lossless comparisons produced byte-identical files with and without
threading. A post-change seven-run lossy check reproduced 1.04-1.08x on three
inputs; `corgi.jpeg` measured 1.23x in that run but was more variable, so the
original conservative 1.05x result above is the representative figure.

Decision: **kept**. This changes only the optimized CLI default; callers of the
libwebp API retain upstream `WebPConfig` defaults and control `thread_level`
themselves.

## Kept: AArch64 NEON intra4 prediction

A two-second sample of a quality-75/method-6 lossy encode placed 1005 of 1047
main-thread samples (96%) in `VP8EncTokenLoop`. Trellis quantization dominated,
but the scalar `Intra4Preds_C` was also directly visible in the hot call tree.
libwebp 1.6 has an AArch64 table-lookup NEON implementation absent from this
1.0.3 fork, so that isolated kernel was backported and dispatched on arm64.

Ten alternating complete CLI trials, with lossy Metal import disabled:

| Input | Method | Scalar predictor | NEON predictor | Speedup |
|---|---:|---:|---:|---:|
| `corgi.jpeg` | 4 | 0.2931 s | 0.2885 s | **1.016x** |
| `mitski.png` | 4 | 0.1298 s | 0.1274 s | **1.019x** |
| `twinpeaks.jpg` | 4 | 0.5761 s | 0.5675 s | **1.015x** |
| `corgi.jpeg` | 6 | 0.5453 s | 0.5418 s | **1.006x** |
| `mitski.png` | 6 | 0.1918 s | 0.1873 s | **1.024x** |
| `twinpeaks.jpg` | 6 | 1.1454 s | 1.1414 s | **1.004x** |

CPU and NEON produced byte-identical WebP files at qualities 25/75/95 and
methods 0/4/6 on three images. `WEBP_NEON_INTRA4=0` retains a benchmark/fallback
switch.

Decision: **kept**. The gain is small but consistent across all six measured
cases, has no GPU startup cost, and preserves output exactly.

## Next opportunities

- Profile lossy macroblock analysis, residual transforms, quantization, and
  entropy coding; RGB conversion is a small fraction of complete encode time.
- Investigate avoiding the remaining RGB input and planar output copies with
  page-aligned/no-copy shared buffers in integrations that control allocation.
- Profile lossless backward-reference generation and histogram clustering,
  which now dominate after the cross-color transform acceleration.
- Selectively evaluate newer libwebp NEON and encoder changes against this fork,
  keeping only reproducible wins and preserving output/correctness invariants.
