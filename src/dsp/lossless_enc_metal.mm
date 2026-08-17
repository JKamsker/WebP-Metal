// Copyright 2026
//
// Metal acceleration for the expensive cross-color search in the WebP
// lossless encoder. This ports the independent-tile algorithm from the CUDA
// 15-418 project, while avoiding its per-kernel allocation and synchronization
// overhead. The pipeline and shared buffers are retained across encodes.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

extern "C" {
#include "src/dsp/lossless.h"
#include "src/enc/backward_references_enc.h"
}

namespace {

constexpr size_t kDefaultMinimumPixels = 256u * 256u;
constexpr size_t kDefaultHashMinimumPixels = 4u * 1000u * 1000u;
constexpr NSUInteger kPreferredThreads = 256;

// Compiling once at runtime keeps this target buildable with the command-line
// tools alone; a full Xcode installation and an offline metallib are optional.
constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
  uint width;
  uint height;
  uint bits;
  uint quality;
  uint tile_columns;
};

inline int signed_byte(int value) {
  const int byte_value = value & 255;
  return byte_value >= 128 ? byte_value - 256 : byte_value;
}

inline int color_delta(int predictor, int color) {
  return (signed_byte(predictor) * signed_byte(color)) >> 5;
}

inline uint transformed_red(uint pixel, int green_to_red) {
  const int green = int((pixel >> 8) & 255u);
  const int red = int((pixel >> 16) & 255u);
  return uint(red - color_delta(green_to_red, green)) & 255u;
}

inline uint transformed_blue(uint pixel, int green_to_blue,
                             int red_to_blue) {
  const int green = int((pixel >> 8) & 255u);
  const int red = int((pixel >> 16) & 255u);
  const int blue = int(pixel & 255u);
  return uint(blue - color_delta(green_to_blue, green)
                   - color_delta(red_to_blue, red)) & 255u;
}

inline float slog2_value(uint value) {
  return value == 0u ? 0.0f : float(value) * log2(float(value));
}

inline float score_histogram(threadgroup atomic_uint* histogram,
                             threadgroup float& result,
                             uint thread_index) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0u) {
    uint sum = 0u;
    float sum_slog = 0.0f;
    for (uint i = 0u; i < 256u; ++i) {
      const uint count = atomic_load_explicit(histogram + i,
                                               memory_order_relaxed);
      sum += count;
      sum_slog += slog2_value(count);
    }

    // The CUDA project intentionally disabled the cross-tile accumulated
    // histogram to make tile execution deterministic. With the second
    // distribution empty, CombinedShannonEntropy(X, X + Y) reduces to this.
    const float entropy = 2.0f * (slog2_value(sum) - sum_slog);
    float spatial = 3.0f *
        float(atomic_load_explicit(histogram, memory_order_relaxed));
    float weight = 2.4f;
    for (uint i = 1u; i < 16u; ++i) {
      const uint lo = atomic_load_explicit(histogram + i,
                                            memory_order_relaxed);
      const uint hi = atomic_load_explicit(histogram + (256u - i),
                                            memory_order_relaxed);
      spatial += weight * float(lo + hi);
      weight *= 0.6f;
    }
    result = entropy - 0.1f * spatial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

inline void clear_histogram(threadgroup atomic_uint* histogram,
                            uint thread_index, uint group_size) {
  for (uint i = thread_index; i < 256u; i += group_size) {
    atomic_store_explicit(histogram + i, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float evaluate_red(device const uint* pixels, const Params params,
                          uint tile_x, uint tile_y, uint tile_width,
                          uint tile_height, int green_to_red,
                          threadgroup atomic_uint* histogram,
                          threadgroup float& histogram_score,
                          uint thread_index, uint group_size) {
  clear_histogram(histogram, thread_index, group_size);
  const uint count = tile_width * tile_height;
  for (uint i = thread_index; i < count; i += group_size) {
    const uint local_x = i % tile_width;
    const uint local_y = i / tile_width;
    const uint x = tile_x + local_x;
    const uint y = tile_y + local_y;
    const uint symbol = transformed_red(pixels[y * params.width + x],
                                         green_to_red);
    atomic_fetch_add_explicit(histogram + symbol, 1u, memory_order_relaxed);
  }
  return score_histogram(histogram, histogram_score, thread_index);
}

inline float evaluate_blue(device const uint* pixels, const Params params,
                           uint tile_x, uint tile_y, uint tile_width,
                           uint tile_height, int green_to_blue,
                           int red_to_blue,
                           threadgroup atomic_uint* histogram,
                           threadgroup float& histogram_score,
                           uint thread_index, uint group_size) {
  clear_histogram(histogram, thread_index, group_size);
  const uint count = tile_width * tile_height;
  for (uint i = thread_index; i < count; i += group_size) {
    const uint local_x = i % tile_width;
    const uint local_y = i / tile_width;
    const uint x = tile_x + local_x;
    const uint y = tile_y + local_y;
    const uint symbol = transformed_blue(pixels[y * params.width + x],
                                          green_to_blue, red_to_blue);
    atomic_fetch_add_explicit(histogram + symbol, 1u, memory_order_relaxed);
  }
  return score_histogram(histogram, histogram_score, thread_index);
}

constant int2 kAxes[8] = {
    int2(0, -1), int2(0, 1), int2(-1, 0), int2(1, 0),
    int2(-1, -1), int2(-1, 1), int2(1, -1), int2(1, 1)};
constant int kDeltas[7] = {16, 16, 8, 4, 2, 2, 2};

kernel void color_space_transform(
    device uint* pixels [[buffer(0)]],
    device uint* transform_image [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]) {
  const uint tile_size = 1u << params.bits;
  const uint tile_column = group_index % params.tile_columns;
  const uint tile_row = group_index / params.tile_columns;
  const uint tile_x = tile_column * tile_size;
  const uint tile_y = tile_row * tile_size;
  const uint tile_width = min(tile_size, params.width - tile_x);
  const uint tile_height = min(tile_size, params.height - tile_y);

  threadgroup atomic_uint histogram[256];
  threadgroup int best_red;
  threadgroup int best_green_blue;
  threadgroup int best_red_blue;
  threadgroup float best_score;
  threadgroup float histogram_score;

  if (thread_index == 0u) {
    best_red = 0;
    best_green_blue = 0;
    best_red_blue = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float score = evaluate_red(pixels, params, tile_x, tile_y, tile_width,
                             tile_height, 0, histogram, histogram_score,
                             thread_index,
                             group_size);
  if (thread_index == 0u) {
    // prev_x and prev_y are zero in the deterministic CUDA algorithm.
    best_score = score - 9.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int red_iterations = 4 + int((7u * params.quality) >> 8);
  for (int iteration = 0; iteration < red_iterations; ++iteration) {
    const int delta = 32 >> iteration;
    for (int sign = -1; sign <= 1; sign += 2) {
      const int candidate = best_red + sign * delta;
      score = evaluate_red(pixels, params, tile_x, tile_y, tile_width,
                           tile_height, candidate, histogram, histogram_score,
                           thread_index,
                           group_size);
      if (thread_index == 0u) {
        if ((candidate & 255) == 0) score -= 6.0f;
        if (candidate == 0) score -= 3.0f;
        if (score < best_score) {
          best_score = score;
          best_red = candidate;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  score = evaluate_blue(pixels, params, tile_x, tile_y, tile_width,
                        tile_height, 0, 0, histogram, histogram_score,
                        thread_index,
                        group_size);
  if (thread_index == 0u) best_score = score - 18.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int blue_iterations = params.quality < 25u ? 1
                            : params.quality > 50u ? 7 : 4;
  for (int iteration = 0; iteration < blue_iterations; ++iteration) {
    for (int axis = 0; axis < 8; ++axis) {
      const int candidate_green =
          best_green_blue + kAxes[axis].x * kDeltas[iteration];
      const int candidate_red =
          best_red_blue + kAxes[axis].y * kDeltas[iteration];
      score = evaluate_blue(pixels, params, tile_x, tile_y, tile_width,
                            tile_height, candidate_green, candidate_red,
                            histogram, histogram_score, thread_index,
                            group_size);
      if (thread_index == 0u) {
        if ((candidate_green & 255) == 0) score -= 6.0f;
        if ((candidate_red & 255) == 0) score -= 6.0f;
        if (candidate_green == 0) score -= 3.0f;
        if (candidate_red == 0) score -= 3.0f;
        if (score < best_score) {
          best_score = score;
          best_green_blue = candidate_green;
          best_red_blue = candidate_red;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (kDeltas[iteration] == 2 && best_green_blue == 0 &&
        best_red_blue == 0) {
      break;
    }
  }

  if (thread_index == 0u) {
    transform_image[group_index] = 0xff000000u |
        (uint(best_red_blue) & 255u) << 16 |
        (uint(best_green_blue) & 255u) << 8 |
        (uint(best_red) & 255u);
  }

  const uint count = tile_width * tile_height;
  for (uint i = thread_index; i < count; i += group_size) {
    const uint local_x = i % tile_width;
    const uint local_y = i / tile_width;
    const uint index = (tile_y + local_y) * params.width + tile_x + local_x;
    const uint pixel = pixels[index];
    const uint new_red = transformed_red(pixel, best_red);
    const uint new_blue = transformed_blue(pixel, best_green_blue,
                                            best_red_blue);
    pixels[index] = (pixel & 0xff00ff00u) | (new_red << 16) | new_blue;
  }
}

)METAL";

constexpr const char* kHashMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct HashParams {
  uint size;
  uint xsize;
  uint iter_max;
  uint window_size;
  uint low_effort;
};

inline uint match_length(device const uint* pixels, uint first, uint second,
                         uint best_length, uint max_length) {
  if (pixels[first + best_length] != pixels[second + best_length]) return 0u;
  uint length = 0u;
  while (length < max_length &&
         pixels[first + length] == pixels[second + length]) {
    ++length;
  }
  return length;
}

kernel void hash_chain_candidates(
    device const uint* pixels [[buffer(0)]],
    device const int* chain [[buffer(1)]],
    device uint* candidates [[buffer(2)]],
    constant HashParams& params [[buffer(3)]],
    uint base_position [[thread_position_in_grid]]) {
  if (base_position >= params.size) return;
  if (base_position == 0u || base_position + 1u >= params.size) {
    candidates[base_position] = 0u;
    return;
  }

  const uint max_length = min(4095u, params.size - 1u - base_position);
  const uint minimum_position = base_position > params.window_size
      ? base_position - params.window_size : 0u;
  const uint good_length = min(max_length, 256u);
  uint best_length = 0u;
  uint best_distance = 0u;
  int iterations = int(params.iter_max);
  int position = chain[base_position];

  if (params.low_effort == 0u) {
    if (base_position >= params.xsize) {
      const uint current = match_length(
          pixels, base_position - params.xsize, base_position,
          best_length, max_length);
      if (current > best_length) {
        best_length = current;
        best_distance = params.xsize;
      }
      --iterations;
    }
    const uint current = match_length(
        pixels, base_position - 1u, base_position, best_length, max_length);
    if (current > best_length) {
      best_length = current;
      best_distance = 1u;
    }
    --iterations;
    if (best_length == 4095u) position = int(minimum_position) - 1;
  }

  uint best_pixel = pixels[base_position + best_length];
  while (position >= int(minimum_position)) {
    --iterations;
    if (iterations == 0) break;
    if (pixels[uint(position) + best_length] == best_pixel) {
      const uint current = match_length(
          pixels, uint(position), base_position, 0u, max_length);
      if (current > best_length) {
        best_length = current;
        best_distance = base_position - uint(position);
        best_pixel = pixels[base_position + best_length];
        if (best_length >= good_length) break;
      }
    }
    position = chain[position];
  }
  candidates[base_position] = (best_distance << 12) | best_length;
}
)METAL";

struct KernelParams {
  uint32_t width;
  uint32_t height;
  uint32_t bits;
  uint32_t quality;
  uint32_t tile_columns;
};

struct HashKernelParams {
  uint32_t size;
  uint32_t xsize;
  uint32_t iter_max;
  uint32_t window_size;
  uint32_t low_effort;
};

struct MetalState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLComputePipelineState> hash_pipeline = nil;
  id<MTLBuffer> pixel_buffer = nil;
  id<MTLBuffer> transform_buffer = nil;
  id<MTLBuffer> chain_buffer = nil;
  size_t pixel_capacity = 0;
  size_t transform_capacity = 0;
  size_t chain_capacity = 0;
  size_t minimum_pixels = kDefaultMinimumPixels;
  size_t hash_minimum_pixels = kDefaultHashMinimumPixels;
  bool verbose = false;
  bool hash_pipeline_attempted = false;
  std::mutex operation_mutex;
};

MetalState* g_state = nullptr;

bool EnvironmentFlag(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  return std::strcmp(value, "0") != 0 &&
         strcasecmp(value, "false") != 0 &&
         strcasecmp(value, "no") != 0;
}

size_t EnvironmentSize(const char* name, size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') return default_value;
  return static_cast<size_t>(parsed);
}

size_t RoundedBufferLength(size_t length) {
  constexpr size_t kPage = 16u * 1024u;
  return std::max(kPage, (length + kPage - 1u) & ~(kPage - 1u));
}

void InitializeMetal() {
  if (!EnvironmentFlag("WEBP_METAL", true)) return;

  @autoreleasepool {
    auto* state = new MetalState();
    state->verbose = EnvironmentFlag("WEBP_METAL_VERBOSE", false);
    state->minimum_pixels = EnvironmentSize("WEBP_METAL_MIN_PIXELS",
                                             kDefaultMinimumPixels);
    state->hash_minimum_pixels = EnvironmentSize(
        "WEBP_METAL_HASH_MIN_PIXELS", kDefaultHashMinimumPixels);
    state->device = MTLCreateSystemDefaultDevice();
    if (state->device == nil) {
      delete state;
      return;
    }

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kMetalSource];
    id<MTLLibrary> library = [state->device newLibraryWithSource:source
                                                         options:nil
                                                           error:&error];
    if (library == nil) {
      std::fprintf(stderr, "WebP-Metal: shader compilation failed: %s\n",
                   error.localizedDescription.UTF8String);
      delete state;
      return;
    }
    id<MTLFunction> function = [library newFunctionWithName:
        @"color_space_transform"];
    if (function == nil) {
      std::fprintf(stderr, "WebP-Metal: color_space_transform not found\n");
      delete state;
      return;
    }
    state->pipeline = [state->device newComputePipelineStateWithFunction:
        function error:&error];
    state->queue = [state->device newCommandQueue];
    if (state->pipeline == nil || state->queue == nil) {
      std::fprintf(stderr, "WebP-Metal: pipeline creation failed: %s\n",
                   error.localizedDescription.UTF8String);
      delete state;
      return;
    }
    if (state->verbose) {
      std::fprintf(stderr,
                   "WebP-Metal: using %s (transform minimum %zu, hash minimum "
                   "%zu pixels)\n", state->device.name.UTF8String,
                   state->minimum_pixels, state->hash_minimum_pixels);
    }
    g_state = state;
  }
}

MetalState* GetMetalState() {
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{ InitializeMetal(); });
  return g_state;
}

bool EnsureHashPipeline(MetalState* state) {
  if (state->hash_pipeline != nil) return true;
  if (state->hash_pipeline_attempted) return false;
  state->hash_pipeline_attempted = true;
  NSError* error = nil;
  NSString* source = [NSString stringWithUTF8String:kHashMetalSource];
  id<MTLLibrary> library = [state->device newLibraryWithSource:source
                                                       options:nil
                                                         error:&error];
  if (library == nil) {
    if (state->verbose) {
      std::fprintf(stderr, "WebP-Metal: hash shader compilation failed: %s\n",
                   error.localizedDescription.UTF8String);
    }
    return false;
  }
  id<MTLFunction> function = [library newFunctionWithName:
      @"hash_chain_candidates"];
  if (function == nil) return false;
  state->hash_pipeline = [state->device newComputePipelineStateWithFunction:
      function error:&error];
  if (state->hash_pipeline == nil && state->verbose) {
    std::fprintf(stderr, "WebP-Metal: hash pipeline creation failed: %s\n",
                 error.localizedDescription.UTF8String);
  }
  return state->hash_pipeline != nil;
}

bool EnsureBuffers(MetalState* state, size_t pixel_bytes,
                   size_t transform_bytes) {
  if (pixel_bytes > state->pixel_capacity) {
    const size_t capacity = RoundedBufferLength(pixel_bytes);
    state->pixel_buffer = [state->device newBufferWithLength:capacity
        options:MTLResourceStorageModeShared];
    if (state->pixel_buffer == nil) return false;
    state->pixel_capacity = capacity;
  }
  if (transform_bytes > state->transform_capacity) {
    const size_t capacity = RoundedBufferLength(transform_bytes);
    state->transform_buffer = [state->device newBufferWithLength:capacity
        options:MTLResourceStorageModeShared];
    if (state->transform_buffer == nil) return false;
    state->transform_capacity = capacity;
  }
  return true;
}

bool EnsureHashBuffers(MetalState* state, size_t bytes) {
  if (!EnsureBuffers(state, bytes, bytes)) return false;
  if (bytes > state->chain_capacity) {
    const size_t capacity = RoundedBufferLength(bytes);
    state->chain_buffer = [state->device newBufferWithLength:capacity
        options:MTLResourceStorageModeShared];
    if (state->chain_buffer == nil) return false;
    state->chain_capacity = capacity;
  }
  return true;
}

extern "C" void VP8LColorSpaceTransform_C(int width, int height, int bits,
                                            int quality, uint32_t* argb,
                                            uint32_t* image);

void ColorSpaceTransformMetal(int width, int height, int bits, int quality,
                              uint32_t* argb, uint32_t* image) {
  MetalState* state = GetMetalState();
  const size_t pixel_count = static_cast<size_t>(width) * height;
  if (state == nullptr || width <= 0 || height <= 0 || bits < 0 || bits > 8 ||
      pixel_count < state->minimum_pixels) {
    VP8LColorSpaceTransform_C(width, height, bits, quality, argb, image);
    return;
  }

  const uint32_t tile_size = 1u << bits;
  const uint32_t tile_columns =
      (static_cast<uint32_t>(width) + tile_size - 1u) >> bits;
  const uint32_t tile_rows =
      (static_cast<uint32_t>(height) + tile_size - 1u) >> bits;
  const size_t tile_count = static_cast<size_t>(tile_columns) * tile_rows;
  const size_t pixel_bytes = pixel_count * sizeof(uint32_t);
  const size_t transform_bytes = tile_count * sizeof(uint32_t);
  const KernelParams params = {
      static_cast<uint32_t>(width), static_cast<uint32_t>(height),
      static_cast<uint32_t>(bits), static_cast<uint32_t>(quality),
      tile_columns};

  std::lock_guard<std::mutex> lock(state->operation_mutex);
  @autoreleasepool {
    if (!EnsureBuffers(state, pixel_bytes, transform_bytes)) {
      VP8LColorSpaceTransform_C(width, height, bits, quality, argb, image);
      return;
    }
    std::memcpy(state->pixel_buffer.contents, argb, pixel_bytes);

    id<MTLCommandBuffer> command_buffer = [state->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
      VP8LColorSpaceTransform_C(width, height, bits, quality, argb, image);
      return;
    }
    [encoder setComputePipelineState:state->pipeline];
    [encoder setBuffer:state->pixel_buffer offset:0 atIndex:0];
    [encoder setBuffer:state->transform_buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];

    const NSUInteger threads = std::min(
        kPreferredThreads, state->pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreadgroups:MTLSizeMake(tile_count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];

    const CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      if (state->verbose) {
        std::fprintf(stderr, "WebP-Metal: command failed: %s\n",
                     command_buffer.error.localizedDescription.UTF8String);
      }
      VP8LColorSpaceTransform_C(width, height, bits, quality, argb, image);
      return;
    }

    std::memcpy(argb, state->pixel_buffer.contents, pixel_bytes);
    std::memcpy(image, state->transform_buffer.contents, transform_bytes);
    if (state->verbose) {
      const double milliseconds =
          (CFAbsoluteTimeGetCurrent() - start) * 1000.0;
      std::fprintf(stderr,
                   "WebP-Metal: transformed %dx%d in %.3f ms (%zu tiles)\n",
                   width, height, milliseconds, tile_count);
    }
  }
}

extern "C" int VP8LHashChainFillMetalCandidates(
    const uint32_t* pixels, const int32_t* chain, int size, int xsize,
    int iter_max, uint32_t window_size, int low_effort,
    uint32_t* candidates) {
  MetalState* state = GetMetalState();
  if (state == nullptr || pixels == nullptr || chain == nullptr ||
      candidates == nullptr || size <= 2 || xsize <= 0 || iter_max <= 0 ||
      static_cast<size_t>(size) < state->hash_minimum_pixels ||
      !EnvironmentFlag("WEBP_METAL_HASH", true)) {
    return 0;
  }

  const size_t bytes = static_cast<size_t>(size) * sizeof(uint32_t);
  const HashKernelParams params = {
      static_cast<uint32_t>(size), static_cast<uint32_t>(xsize),
      static_cast<uint32_t>(iter_max), window_size,
      static_cast<uint32_t>(low_effort != 0)};
  std::lock_guard<std::mutex> lock(state->operation_mutex);
  @autoreleasepool {
    if (!EnsureHashPipeline(state) || !EnsureHashBuffers(state, bytes)) return 0;
    const CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
    std::memcpy(state->pixel_buffer.contents, pixels, bytes);
    std::memcpy(state->chain_buffer.contents, chain, bytes);

    id<MTLCommandBuffer> command_buffer = [state->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) return 0;
    [encoder setComputePipelineState:state->hash_pipeline];
    [encoder setBuffer:state->pixel_buffer offset:0 atIndex:0];
    [encoder setBuffer:state->chain_buffer offset:0 atIndex:1];
    [encoder setBuffer:state->transform_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    const NSUInteger threads = std::min(
        kPreferredThreads, state->hash_pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(size), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      if (state->verbose) {
        std::fprintf(stderr, "WebP-Metal: hash command failed: %s\n",
                     command_buffer.error.localizedDescription.UTF8String);
      }
      return 0;
    }
    std::memcpy(candidates, state->transform_buffer.contents, bytes);
    if (state->verbose) {
      std::fprintf(stderr,
                   "WebP-Metal: hash candidates for %d pixels in %.3f ms\n",
                   size, (CFAbsoluteTimeGetCurrent() - start) * 1000.0);
    }
    return 1;
  }
}

}  // namespace

extern "C" WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitMetal(void) {
  if (GetMetalState() != nullptr) {
    VP8LColorSpaceTransform = ColorSpaceTransformMetal;
  }
}
