# CUDA vs. Metal WebP Lossless Encoding

## Executive summary

The WebP-Metal implementation was compared with the CUDA results published in
Kevin Geng and Emma Liu's CMU 15-418 final project, [Accelerating the WebP
Encoding/Decoding Pipeline](https://emmaliu.info/15418-Final-Project/final_report.pdf).

On the directly comparable `VP8LColorSpaceTransform` stage, Metal on an Apple
M4 Max produced greater relative speedups than the CUDA implementation reported
for both the NVIDIA GeForce GTX 1080 and Tesla P100. On the three shared test
images measured here, Metal achieved 17.3x to 41.9x transform-stage speedups.

This does not establish that Metal GPUs are universally faster than the tested
NVIDIA GPUs. Each result is relative to a different host CPU and software
platform: the CUDA baselines used Intel Xeon CPUs with SSE, while the Metal
baseline used the considerably newer M4 Max CPU.

### Results at a glance

The directly accelerated `ColorSpaceTransform` stage produced these relative
CPU-to-accelerator speedups:

| Image | M4 Max CPU to Metal | Original CPU to GTX 1080 CUDA | Original CPU to P100 CUDA |
|---|---:|---:|---:|
| `layout.png` | **17.27x** | 6.85x | 15.78x |
| `mitski.png` | **40.80x** | 6.76x | 16.16x |
| `corgi.jpeg` | **41.93x** | 6.34x | 14.87x |
| **Three-image average** | **33.33x** | **6.65x** | **15.60x** |

In broad terms:

- M4 Max CPU to Metal: **17x-42x**, averaging approximately **33x**.
- Original CPU to GTX 1080 CUDA: generally **5x-10x**.
- Original CPU to Tesla P100 CUDA: generally **10x-20x**.
- The CUDA report's exceptionally large `starry_night_cropped.jpg` reached
  12.92x on the GTX 1080 and 28.04x on the P100.

These are transform-stage numbers. For the complete WebP encoder, Metal
achieved **1.76x** at method 4 and **1.69x** at method 6. The CUDA report did
not publish an equivalent whole-encoder speedup.

## Implementations compared

Both implementations accelerate the lossless WebP cross-color transform. This
stage searches for transform coefficients that reduce the entropy of each
32x32 pixel tile before Huffman and LZ77 encoding.

The final CUDA implementation:

- Maps one 32x32 tile to one 1,024-thread CUDA block.
- Runs all image tiles in one kernel launch.
- Uses CUB reductions for `CombinedShannonEntropy`.
- Transfers image and transform data over PCIe.
- Allocates CUDA buffers for each transform invocation.
- Disables the accumulated-image histogram and neighboring-tile dependencies
  so tiles can execute independently and deterministically.

The Metal implementation:

- Maps one 32x32 tile to one 256-thread Metal threadgroup.
- Has each thread process up to four pixels.
- Builds 256-entry histograms with threadgroup atomics.
- Uses `MTLStorageModeShared` on Apple unified memory.
- Compiles and caches the Metal pipeline once per process.
- Caches and grows reusable shared buffers for repeated encodes.
- Uses the same independent-tile heuristic as the final CUDA implementation.
- Falls back to the original C implementation for small images, unavailable
  Metal devices, or command failures.

## Benchmark methodology

The CUDA report measured `ColorSpaceTransform_C` against
`ColorSpaceTransform_CUDA`, including allocation and transfers to and from the
GPU. It averaged five runs and launched an empty CUDA kernel before measurement
to remove first-context initialization overhead.

The Metal comparison followed the same scope:

1. The original `VP8LColorSpaceTransform_C` and the Metal replacement were
   timed directly.
2. Buffer allocation and copies between libwebp memory and Metal shared buffers
   were included.
3. Each reported value is the average of five runs.
4. Each Metal measurement ran in a fresh process, preventing large image
   buffers from being reused between measured runs.
5. A 1x1 warm-up command initialized the Metal command context before timing,
   corresponding to the CUDA report's `cub::PtxVersion()` warm-up.
6. Tests used 32x32 transform tiles, quality 75, and the default method-4
   lossless settings.

All Metal tests ran on an Apple M4 Max. The CUDA report used these systems:

| System | CPU | GPU | GPU interconnect |
|---|---|---|---|
| GHC cluster | Intel Xeon E5-1660 v4 | NVIDIA GeForce GTX 1080 | PCIe, theoretical 16 GB/s |
| PSC Bridges | Intel Broadwell E5-2683 v4 | NVIDIA Tesla P100 | PCIe, theoretical 32 GB/s |
| WebP-Metal | Apple M4 Max | Integrated Apple M4 Max GPU | Unified memory |

## Transform-stage results

Speedup is calculated as CPU transform time divided by GPU transform time. The
CUDA speedups are from Figures 7 and 8 of the final report. Metal timings were
measured locally using the same images from the reference project.

| Image | Pixels | CUDA GTX 1080 | CUDA P100 | Metal M4 Max | M4 CPU | M4 Metal |
|---|---:|---:|---:|---:|---:|---:|
| `layout.png` | 1,155,200 | 6.851x | 15.782x | **17.274x** | 119.274 ms | 6.905 ms |
| `mitski.png` | 4,521,072 | 6.760x | 16.158x | **40.795x** | 440.427 ms | 10.796 ms |
| `corgi.jpeg` | 5,868,726 | 6.339x | 14.865x | **41.927x** | 482.536 ms | 11.509 ms |

Relative to the reported CUDA speedups:

| Image | Metal vs. GTX 1080 speedup ratio | Metal vs. P100 speedup ratio |
|---|---:|---:|
| `layout.png` | 2.52x | 1.09x |
| `mitski.png` | 6.04x | 2.52x |
| `corgi.jpeg` | 6.61x | 2.82x |

These ratios compare platform-relative speedups, not GPU execution times on
identical host hardware. They should not be interpreted as direct cross-GPU
performance ratios.

## End-to-end encoding

The CUDA report's headline results cover only `ColorSpaceTransform`; it does
not publish equivalent complete `cwebp` speedups. Consequently, the following
Metal whole-encoder result cannot be compared directly with the report's
5x-20x CUDA figures.

For the 2876x1572 `mitski.png` image using lossless method 4:

| Encoder | Complete CLI time |
|---|---:|
| CPU | 0.960 s |
| Metal | 0.545 s |
| End-to-end speedup | **1.76x** |

The transform is only one stage of WebP encoding. Input decoding, predictor
selection, backward-reference generation, Huffman coding, and output writing
remain on the CPU. These stages limit complete encoder speedup even when the
cross-color transform becomes much faster.

For method 6 on the same image:

| Encoder | Complete CLI time |
|---|---:|
| CPU | 1.22 s |
| Metal | 0.72 s |
| End-to-end speedup | **1.69x** |

## Compression size and correctness

Removing the cross-tile accumulated histogram and neighboring-tile biases
allows GPU tiles to run concurrently, but can select slightly different
transform coefficients. The WebP remains exactly lossless; only the encoded
bitstream and compressed size can differ.

Method-4 results:

| Image | CPU WebP | Metal WebP | Metal difference |
|---|---:|---:|---:|
| `layout.png` | 6,542 bytes | 6,532 bytes | **-0.153%** |
| `mitski.png` | 931,898 bytes | 934,482 bytes | **+0.277%** |
| `corgi.jpeg` | 6,867,472 bytes | 6,869,382 bytes | **+0.028%** |

The arithmetic mean of these three relative changes is approximately +0.05%.
The CUDA report describes its outputs as approximately 1,000 bytes larger for
images around 1 MB or more, roughly +0.1% on average. The Metal and CUDA
compression effects are therefore of the same small order of magnitude.

Correctness was verified by encoding through both CPU and Metal paths,
decoding both WebP files, and comparing the decoded PAM/PPM pixels byte for
byte. RGB and RGBA inputs and encoding methods 0 through 6 produced identical
decoded pixels.

## Interpretation

Several factors likely contribute to Metal's strong transform-stage results:

- **Unified memory:** shared Metal buffers do not cross a discrete PCIe bus.
  The current implementation still performs one CPU copy into the Metal buffer
  and one copy back into libwebp memory, but avoids separate host-to-device and
  device-to-host PCIe transfers.
- **Smaller threadgroups:** 256 Metal threads process each 1,024-pixel tile,
  allowing each thread to handle several pixels and reducing the scheduling
  pressure of the CUDA implementation's 1,024-thread blocks.
- **Cached infrastructure:** the device, command queue, pipeline, and buffers
  persist across transform calls. The matched cold-process benchmark still
  includes large-buffer allocation, while repeated library use can reuse those
  buffers.
- **Newer platform:** the M4 Max CPU and GPU are several generations newer than
  the Xeon, GTX 1080, and P100 systems used in the 2019 report.

## Limitations

- The GPUs were not tested in the same machine, operating system, compiler, or
  libwebp version.
- Relative speedup depends on host CPU performance as well as GPU performance.
- Only three shared images were instrumented for the matched transform-stage
  comparison.
- Runtime Metal shader compilation occurs once per process. It is outside the
  transform-only timing but included in complete CLI measurements.
- The independent-tile heuristic may modestly increase or decrease compressed
  size depending on image content.
- The vendored code is based on libwebp 1.0.3 to remain close to the original
  CUDA project; results may differ with current libwebp releases.

## Reproducing the Metal results

Build and run the encoder:

```sh
make metal -j8
./cwebp-metal -lossless -exact -m 4 input.png -o output.webp
```

Run the included end-to-end benchmark and lossless verification:

```sh
scripts/benchmark.sh /path/to/input.png 5 4
scripts/test.sh /path/to/input.png
```

Useful environment variables:

- `WEBP_METAL=0` selects the CPU implementation.
- `WEBP_METAL_MIN_PIXELS=0` forces Metal for all image sizes.
- `WEBP_METAL_VERBOSE=1` prints the selected GPU and transform command timing.

## Sources

- Kevin Geng and Emma Liu, [Accelerating the WebP Encoding/Decoding Pipeline -
  Final Report](https://emmaliu.info/15418-Final-Project/final_report.pdf).
- The local reference fork: `/Users/jonas/Documents/15418-Final-Project`.
- The Metal implementation: `src/dsp/lossless_enc_metal.mm`.
- Metal benchmark tooling: `scripts/benchmark.sh` and `scripts/test.sh`.
