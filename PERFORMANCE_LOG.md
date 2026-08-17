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

## Current kept improvements

| Improvement | Mode/scope | Measured result | Output invariant |
|---|---|---:|---|
| Metal cross-color transform | Lossless stage | **17.27-41.93x** | Decoded pixels exact; size -0.15% to +0.28% |
| Metal cross-color transform | Complete lossless CLI | **1.69-1.76x** | Decoded pixels exact |
| Metal hash-chain search | Lossless stage / complete CLI | **12-72x / 1.02-1.76x** | Bitstream exact |
| Both lossless Metal stages | Complete lossless CLI | **2.06-2.38x** | Decoded pixels exact |
| Metal RGB-to-YUV420 | Lossy warmed batch stage | **4.55-4.95x** | Bitstream exact |
| Metal RGB-to-YUV420 | 178 MP cold stage / complete CLI | **1.63x / 1.005x** | Bitstream exact |
| Multithreaded CLI default | Primarily lossy complete CLI | **1.04-1.08x** representative | Bitstream exact |
| NEON intra4 prediction | Lossy complete CLI | **1.004-1.024x** | Bitstream exact |
| NEON intra16 prediction | Lossy complete CLI | **1.002-1.012x** | Bitstream exact |
| NEON WHT quantization | Lossy complete CLI | **1.000-1.015x** | Bitstream exact |
| AArch64 vector horizontal sums | Lossy method-4 CLI | **1.000-1.005x** | Bitstream exact |
| Reduced trellis scoring work | Lossy method-6 CLI | **1.003-1.016x** | Bitstream exact |
| Fixed-size trellis clears | Lossy method-5/6 CLI | **1.002-1.066x** | Bitstream exact |
| NEON predictors 9-12 | Lossless complete CLI | **0.998-1.021x** | Bitstream exact |
| Remove duplicate literal-histogram copy | Lossless complete CLI | **0.996-1.029x** | Bitstream exact |

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

## Kept: AArch64 NEON intra16 prediction

The matching upstream 16x16 DC, vertical, horizontal, and true-motion
predictors were backported as a separate experiment. Ten alternating complete
CLI trials, with intra4 NEON enabled in both variants and lossy Metal import
disabled:

| Input | Method | Scalar predictor | NEON predictor | Speedup |
|---|---:|---:|---:|---:|
| `corgi.jpeg` | 4 | 0.2964 s | 0.2944 s | **1.007x** |
| `mitski.png` | 4 | 0.1304 s | 0.1288 s | **1.012x** |
| `twinpeaks.jpg` | 4 | 0.5838 s | 0.5804 s | **1.006x** |
| `corgi.jpeg` | 6 | 0.5550 s | 0.5506 s | **1.008x** |
| `mitski.png` | 6 | 0.1914 s | 0.1895 s | **1.010x** |
| `twinpeaks.jpg` | 6 | 1.1421 s | 1.1399 s | **1.002x** |

CPU and NEON output was byte-identical at qualities 25/75/95 and methods
0/4/6 on the three-image set. `WEBP_NEON_INTRA16=0` disables only this kernel.

Decision: **kept**. The 0.2-1.2% whole-encode gain is modest but consistent,
has no initialization or transfer cost, and composes with the intra4 win.

## Kept: AArch64 NEON WHT quantization

The AArch64 encoder already had a bit-exact NEON block quantizer and dispatched
ordinary 4x4 quantization to it, but Walsh-Hadamard DC quantization still used
`QuantizeBlock_C`. Upstream commit `314a142a` confirms that both operations use
the same compatible kernel. The missing dispatch was added as an isolated
one-line experiment.

Fifteen alternating quality-75 complete CLI trials:

| Input | Method 4 speedup | Method 6 speedup |
|---|---:|---:|
| `layout.png` | **1.002x** | 1.000x (neutral) |
| `mitski.png` | **1.010x** | **1.009x** |
| `corgi.jpeg` | **1.015x** | **1.009x** |
| `twinpeaks.jpg` | **1.011x** | **1.008x** |
| `siamese.jpg` | **1.012x** | **1.011x** |

Five images produced byte-identical files at qualities 25/75/95 and methods
4/5/6 (45 combinations).

Decision: **kept**. The four meaningful inputs consistently gain 0.8-1.5%
with no startup cost or output change; the tiny Layout case is neutral.

## Kept: native AArch64 vector horizontal sums

Upstream commit `e68765af` replaces pairwise widen/extract sequences in NEON
SSE and coefficient-flatness scoring with AArch64's native `vaddvq_u32`
horizontal sum. The first fifteen-trial run measured 1.001-1.007x at method 4,
but method 6 was neutral (0.998-1.001x), so the larger method-4 inputs were
rerun for 31 alternating trials:

| Input | Previous reduction | Native `vaddvq_u32` | Speedup |
|---|---:|---:|---:|
| `mitski.png` | 0.12692 s | 0.12696 s | 1.000x (neutral) |
| `corgi.jpeg` | 0.29393 s | 0.29249 s | **1.005x** |
| `twinpeaks.jpg` | 0.57542 s | 0.57406 s | **1.002x** |
| `siamese.jpg` | 0.41784 s | 0.41695 s | **1.002x** |

Five images produced byte-identical files at qualities 25/75/95 and methods
4/5/6 (45 combinations).

Decision: **kept**. This is a small 0.2-0.5% method-4 gain on three substantial
inputs, neutral elsewhere, and a simple exact replacement with no startup cost.

## Kept: reduced lossy trellis scoring work

Profiling showed `TrellisQuantizeBlock` dominating the method-6 token loop.
Moving individual 4x4 trellises to Metal would add dispatch/synchronization
overhead and their neighboring-context dependencies prevent a simple
whole-frame launch. Instead, upstream libwebp commit `93480160` was backported
as a bounded exact experiment. It hoists a score shared by all predecessors,
initializes the first predecessor outside the comparison loop, and avoids
terminal-node cost work when the partial score is already worse.

Eleven alternating complete quality-75 CLI trials, with lossy Metal import
disabled in both binaries:

| Input | Method | Before | Reduced scoring | Speedup |
|---|---:|---:|---:|---:|
| `layout.png` | 4 | 0.0329 s | 0.0331 s | 0.995x (noise/neutral) |
| `mitski.png` | 4 | 0.1246 s | 0.1244 s | **1.002x** |
| `corgi.jpeg` | 4 | 0.2891 s | 0.2877 s | **1.005x** |
| `twinpeaks.jpg` | 4 | 0.5656 s | 0.5660 s | 0.999x (noise/neutral) |
| `siamese.jpg` | 4 | 0.4147 s | 0.4143 s | **1.001x** |
| `layout.png` | 6 | 0.0440 s | 0.0439 s | **1.003x** |
| `mitski.png` | 6 | 0.1861 s | 0.1854 s | **1.004x** |
| `corgi.jpeg` | 6 | 0.5447 s | 0.5420 s | **1.005x** |
| `twinpeaks.jpg` | 6 | 1.1276 s | 1.1170 s | **1.010x** |
| `siamese.jpg` | 6 | 0.6935 s | 0.6828 s | **1.016x** |

Five images produced byte-identical files before and after the change at
qualities 25/75/95 and methods 4/5/6 (45 combinations).

Decision: **kept**. The method-6 gain is small but consistent, exact, has no
startup cost, and specifically improves the dominant lossy hot loop. Method 4
is expected to be neutral because it does not run full trellis optimization.

## Kept: fixed-size trellis coefficient clearing

The adjacent upstream `1a8f0d45` optimization was tested separately. The old
code cleared either 15 or 16 coefficients using a runtime-derived pointer and
length. Apple Clang emitted two calls to `bzero` for every trellis block. An
explicit branch between the fixed 30-byte and 32-byte layouts lets the compiler
inline both clears as stores; disassembly confirms that the two calls disappear.

Fifteen alternating quality-75/method-6 complete CLI trials:

| Input | Runtime-sized clears | Fixed-size clears | Speedup |
|---|---:|---:|---:|
| `layout.png` | 0.04350 s | 0.04080 s | **1.066x** |
| `mitski.png` | 0.18470 s | 0.17364 s | **1.064x** |
| `corgi.jpeg` | 0.53974 s | 0.52194 s | **1.034x** |
| `twinpeaks.jpg` | 1.11488 s | 1.08657 s | **1.026x** |
| `siamese.jpg` | 0.68313 s | 0.65248 s | **1.047x** |

Method 5 also improved in eleven alternating trials: 1.017x on Layout, 1.015x
on Mitski, and 1.002-1.005x on the other three inputs. The effect is larger at
method 6 because full trellis optimization invokes the clearing path far more
often. Five images produced byte-identical files at qualities 25/75/95 and
methods 4/5/6 (45 combinations).

Decision: **kept**. This removes hot-loop function calls, is exact, and provides
a surprisingly material 2.6-6.6% method-6 whole-encoder gain on this machine.

## Rejected: restrict qualifiers in lossy quantization

Upstream commit `b9d2f9cd` adds non-aliasing qualifiers throughout the
quantization/reconstruction call graph and reported a few fewer instructions.
The complete change was tested after the two retained trellis optimizations.
Fifteen alternating quality-75 trials across five images measured
0.996-1.005x at method 4 and 0.999-1.001x at method 6.

Decision: **rejected and removed**. On Apple Clang this is timing noise, with
small regressions in several cases; keeping annotation churn without a measured
benefit would make future backports harder to review.

## Kept: AArch64 NEON lossless predictors 9-12

After Metal acceleration, a method-6 lossless profile put 463 of 843 samples
in residual/predictor search and 328 in hash-chain construction. Predictor
modes 9-12 appeared as scalar C in the call tree. Small arm64 implementations
were added for channel-wise averages, selection, and clamped add/subtract.

Seven alternating complete Metal-enabled lossless CLI trials:

| Input | Method | Scalar predictors | NEON predictors | Speedup |
|---|---:|---:|---:|---:|
| `layout.png` | 4 | 0.1176 s | 0.1156 s | **1.018x** |
| `mitski.png` | 4 | 0.6220 s | 0.6090 s | **1.021x** |
| `corgi.jpeg` | 4 | 1.4206 s | 1.3994 s | **1.015x** |
| `layout.png` | 6 | 0.0499 s | 0.0501 s | 0.998x (noise/neutral) |
| `mitski.png` | 6 | 0.7865 s | 0.7824 s | **1.005x** |
| `corgi.jpeg` | 6 | 1.7263 s | 1.7141 s | **1.007x** |

CPU and NEON paths produced byte-identical lossless WebP files for three
images at methods 0/4/6. `WEBP_NEON_LOSSLESS_PREDICTORS=0` provides the A/B
switch.

Decision: **kept**. Method 4 consistently gains 1.5-2.1%; method 6 is neutral
to 0.7% faster. The single 0.2% negative is below timing resolution on a 50 ms
palette-dominated encode and does not outweigh the repeatable larger-image
benefit.

## Rejected: AArch64 NEON entropy and match scanning

Two small lossless kernels were prototyped after profiling
`CombinedShannonEntropy_C` and `VectorMismatch_C`:

- A four-bin-at-a-time NEON combined-Shannon-entropy loop.
- A four-pixel-at-a-time NEON equality scan for hash-chain matches.

The first entropy version followed the existing SSE2 operation ordering. It
changed floating-point rounding enough to produce a 1,270-byte larger method-6
`corgi.jpeg` stream (+0.0185%), so it was not eligible as an exact
micro-optimization. A second version preserved the scalar evaluator's exact
per-symbol operation order and restored byte-identical output.

Five alternating complete Metal-enabled lossless trials of the exact version:

| Experiment/input | Method 4 speedup | Method 6 speedup |
|---|---:|---:|
| NEON entropy, `layout.png` | 0.969x | 1.008x |
| NEON entropy, `mitski.png` | 0.992x | 0.982x |
| NEON entropy, `corgi.jpeg` | 0.995x | 0.986x |
| NEON mismatch, `layout.png` | 1.004x | 1.002x |
| NEON mismatch, `mitski.png` | 0.997x | 0.993x |
| NEON mismatch, `corgi.jpeg` | 1.000x | 0.999x |

Decision: **rejected and removed**. Exact entropy scoring regressed five of six
cases because logarithm lookup/call work still dominates while NEON adds lane
extraction overhead. Match scanning was neutral to 0.7% slower on meaningful
inputs; most candidate chains reject before four pixels, so vector setup does
not amortize. The original scalar code remains active.

## Kept: remove duplicate literal-histogram copy

`HistogramCopy` copied the complete variable-sized allocation and then copied
the trailing literal-count array a second time after restoring its pointer.
Upstream commit `be5af857` identifies the redundant traffic. Copying only the
fixed structure in the first operation preserves the intended two-allocation
layout and avoids copying the literal array twice.

Nine alternating complete lossless trials with the final Metal defaults:

| Input | Method | Before | Single literal copy | Speedup |
|---|---:|---:|---:|---:|
| `layout.png` | 4 | 0.1153 s | 0.1130 s | **1.021x** |
| `mitski.png` | 4 | 0.4405 s | 0.4282 s | **1.029x** |
| `corgi.jpeg` | 4 | 0.8746 s | 0.8744 s | 1.000x |
| `siamese.jpg` | 4 | 1.4168 s | 1.4231 s | 0.996x |
| `layout.png` | 6 | 0.05005 s | 0.05006 s | 1.000x |
| `mitski.png` | 6 | 0.6072 s | 0.6030 s | **1.007x** |
| `corgi.jpeg` | 6 | 1.1893 s | 1.1880 s | **1.001x** |
| `siamese.jpg` | 6 | 1.7711 s | 1.7673 s | **1.002x** |

Four images produced byte-identical files at every method from 0 to 6. The
small method-4 Siamese regression is within the variability seen in these
one-shot process timings; the operation removes memory traffic unconditionally
and upstream measured 13 MB less copying for a 1600x1600 encode.

Decision: **kept**, including the batch/memory-bandwidth benefit. The measured
whole-encode effect is content-dependent rather than advertised as universal.

## Kept: exact lossless hash-chain matching with Metal

Instrumentation separated the hash-table fill from the best-match traversal.
On representative method-4 inputs, table construction took 0.6-6.4 ms while
the serial match search took 1.5 ms (`layout.png`), 199.5 ms (`mitski.png`),
and 536.0 ms (`corgi.jpeg`). The data-dependent predecessor walk, rather than
building the hash table, was the useful target.

The retained implementation keeps the CPU-built predecessor chain but launches
one Metal thread per base position to compute the same best match. The CPU then
replays libwebp's original left-extension and position-skipping rules from
those candidates. This preserves all ordering-sensitive choices: six diverse
images, including synthetic noise and photographic inputs, produced
byte-identical files with CPU and Metal matching at every method from 0 to 6.
Near-lossless strengths 20, 60, and 100 were also byte-identical at methods 0,
4, and 6.

Nine alternating complete CLI trials on an Apple M4 Max, including lazy shader
compilation in each process:

| Input | Method | CPU hash search | Metal hash search | Speedup |
|---|---:|---:|---:|---:|
| `apple_holiday.png` (4.35 MP) | 4 | 0.4068 s | 0.3967 s | **1.026x** |
| `bon_appetit.png` (4.18 MP) | 4 | 0.6092 s | 0.4312 s | **1.413x** |
| `mitski.png` (4.52 MP) | 4 | 0.6009 s | 0.4255 s | **1.412x** |
| `corgi.jpeg` (5.87 MP) | 4 | 1.3955 s | 0.8670 s | **1.610x** |
| `siamese.jpg` (9.22 MP) | 4 | 2.4586 s | 1.4005 s | **1.756x** |
| `apple_holiday.png` (4.35 MP) | 6 | 0.5440 s | 0.5347 s | **1.017x** |
| `bon_appetit.png` (4.18 MP) | 6 | 0.7614 s | 0.5853 s | **1.301x** |
| `mitski.png` (4.52 MP) | 6 | 0.7559 s | 0.5838 s | **1.295x** |
| `corgi.jpeg` (5.87 MP) | 6 | 1.6715 s | 1.1494 s | **1.454x** |
| `siamese.jpg` (9.22 MP) | 6 | 2.7967 s | 1.7226 s | **1.624x** |

For the main method-4 pass, GPU command time was about 16.5 ms on Mitski and
7.5 ms on Corgi, versus 199.5 ms and 536.0 ms for the instrumented CPU search
(about 12x and 72x at the targeted stage). End-to-end gains are smaller because
hash-table construction, residual transforms, histogram work, and entropy
coding stay on the CPU.

Threshold experiments exposed two important non-wins. Eagerly compiling this
second shader slowed below-threshold encodes by 1-2%, so the hash pipeline is
now compiled lazily. Forcing it on `layout.png` (1.16 MP),
`carbon_emissions.png` (2.17 MP), and `zip.png` (3.15 MP) ranged from a clear
regression on the tiny image to approximately neutral on the simple larger
ones. The default is therefore 4,000,000 pixels. A clean pre-change binary and
the final lazy version were indistinguishable (0.998-1.007x) on those fallback
cases. `WEBP_METAL_HASH_MIN_PIXELS=0` remains useful for persistent batch
encoders, where compilation is amortized and complex smaller images can still
win; that below-threshold benefit is explicitly content-dependent.

Decision: **kept**. This is the largest exact whole-encoder improvement so far.
`WEBP_METAL_HASH=0` provides the A/B/fallback switch.

### Cumulative lossless result

Seven alternating trials compared `WEBP_METAL=0` with the final default,
combining the cross-color and exact hash-search accelerators. Both variants
included the retained NEON code:

| Input | Method | CPU | All Metal | Cumulative speedup |
|---|---:|---:|---:|---:|
| `mitski.png` | 4 | 1.0303 s | 0.4332 s | **2.379x** |
| `corgi.jpeg` | 4 | 1.8264 s | 0.8715 s | **2.096x** |
| `siamese.jpg` | 4 | 3.3979 s | 1.4304 s | **2.375x** |
| `mitski.png` | 6 | 1.3065 s | 0.6040 s | **2.163x** |
| `corgi.jpeg` | 6 | 2.4324 s | 1.1784 s | **2.064x** |
| `siamese.jpg` | 6 | 4.1313 s | 1.7595 s | **2.348x** |

These are the current CPU-versus-Metal whole-encoder figures. The Metal
cross-color transform can choose different—but still exactly decodable—lossless
transform parameters, so cumulative output sizes retain the previously
documented small variation. The new hash-search stage itself is bit-exact.

## Profiled, not yet implemented

### Lossy token/trellis loop

A two-second quality-75/method-6 sample placed 1005 of 1047 main-thread samples
inside `VP8EncTokenLoop`. `VP8Decimate` and `TrellisQuantizeBlock` dominate;
the transform, distortion, and residual-cost helpers already dispatch to NEON
in many cases. Trellis state and per-macroblock decisions are dependency-heavy,
so moving only the arithmetic to Metal would risk command/transfer overhead.

Status: **next major lossy research target**, not yet claimed as an
improvement. A useful experiment needs a batched macroblock interface and an
exact CPU fallback, followed by size/PSNR and total-time checks.

### Remaining lossless entropy scoring

After the Metal transform, a one-second method-6 sample placed 463 of 843
main-thread samples in residual/predictor search; 221 samples were directly in
`CombinedShannonEntropy_C`. The straightforward exact NEON entropy prototype
was slower and has been removed, as recorded above. A profitable next design
would need to reduce or batch the repeated histogram scoring rather than only
vectorizing its final 256-bin loop.

## Next opportunities

1. Design a persistent-process lossy macroblock experiment. The final 4-second
   profile still put roughly half of token-loop samples in
   `TrellisQuantizeBlock`; a useful Metal design must batch work while respecting
   reconstructed-neighbor and coefficient-context dependencies. Per-block GPU
   dispatch is not viable.
2. Prototype the newer upstream lossless histogram-clustering changes
   (`a4183d94`, `00338240`, and `e53e2130`) one logical change at a time. They
   remove or cache substantially more cost computation but require a larger
   backport than the safe duplicate-copy fix retained above.
3. Add a persistent-library end-to-end batch benchmark. The current harness
   isolates warmed lossy import, while CLI benchmarks intentionally include
   shader compilation. A library harness would refine the 4 MP lossless-hash
   threshold and quantify whole-batch gains below it.
4. Build offline `.metallib` assets when full Xcode is available. Removing the
   roughly 20-25 ms runtime compilation cost should lower the profitable
   one-shot thresholds, especially for lossy RGB import.
5. Investigate page-aligned/no-copy shared input and planar output buffers for
   integrations that control allocation, then measure memory and wall time.
6. Add macOS arm64 CI covering lossy qualities 25/75/95, lossless methods 0-6,
   near-lossless strengths, fallback thresholds, and byte/decode invariants.

## Stopping point (2026-08-17)

Both lossy and lossless encoding are accelerated. Current cumulative lossless
CPU-versus-Metal speedup is 2.06-2.38x on the representative 4.5-9.2 MP set.
The largest lossy GPU result is the warmed batch RGB-to-YUV stage at 4.55-4.95x;
exact NEON and trellis improvements additionally reduce complete encode time.
Every retained exact optimization has an A/B measurement and commit, while
rejected experiments remain recorded above. `README.md` contains operational
thresholds and test commands; this file is the experiment ledger and handoff.
