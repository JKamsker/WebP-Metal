// Copyright 2026
//
// Metal implementation of libwebp's opaque RGB/BGR to YUV420 conversion.
// The integer conversion and gamma lookup/interpolation match
// ImportYUVAFromRGBA() byte-for-byte.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

extern "C" {
#include "src/enc/vp8i_enc.h"
}

namespace {

// Runtime shader compilation costs roughly 25 ms on the benchmark M4 Max.
// A cold, one-shot conversion crosses over near 70 MP, so default to a
// conservative 80 MP. Persistent/batch encoders should set the threshold to 0.
constexpr size_t kDefaultMinimumPixels = 80u * 1000u * 1000u;
constexpr NSUInteger kPreferredThreads = 256;

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
  uint width;
  uint height;
  uint step;
  uint source_stride;
  uint red_offset;
  uint green_offset;
  uint blue_offset;
  uint uv_width;
  uint y_size;
  uint uv_size;
};

inline uchar rgb_to_y(int r, int g, int b) {
  const int luma = 16839 * r + 33059 * g + 6420 * b;
  return uchar((luma + 32768 + (16 << 16)) >> 16);
}

inline uchar clip_uv(int value) {
  value = (value + (32768 << 2) + (128 << 18)) >> 18;
  return uchar(clamp(value, 0, 255));
}

inline uchar rgb_to_u(int r, int g, int b) {
  return clip_uv(-9719 * r - 19081 * g + 28800 * b);
}

inline uchar rgb_to_v(int r, int g, int b) {
  return clip_uv(28800 * r - 24116 * g - 4684 * b);
}

inline int interpolate_gamma(uint value, device const int* linear_to_gamma) {
  const uint table_position = value >> 9;
  const uint fraction = value & 511u;
  const int v0 = linear_to_gamma[table_position];
  const int v1 = linear_to_gamma[table_position + 1u];
  const int interpolated = v1 * int(fraction) +
                           v0 * int(512u - fraction);
  return (interpolated + 64) >> 7;
}

inline uint source_index(uint x, uint y, constant Params& params) {
  return y * params.source_stride + x * params.step;
}

kernel void rgb_to_yuv420(
    device const uchar* source [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device const ushort* gamma_to_linear [[buffer(2)]],
    device const int* linear_to_gamma [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint pixel_count = params.width * params.height;
  if (index < pixel_count) {
    const uint x = index % params.width;
    const uint y = index / params.width;
    const uint offset = source_index(x, y, params);
    output[index] = rgb_to_y(source[offset + params.red_offset],
                             source[offset + params.green_offset],
                             source[offset + params.blue_offset]);
  }

  const uint uv_height = (params.height + 1u) >> 1;
  const uint uv_count = params.uv_width * uv_height;
  if (index < uv_count) {
    const uint uv_x = index % params.uv_width;
    const uint uv_y = index / params.uv_width;
    const uint x0 = uv_x << 1;
    const uint y0 = uv_y << 1;
    const uint x1 = min(x0 + 1u, params.width - 1u);
    const uint y1 = min(y0 + 1u, params.height - 1u);
    const uint p00 = source_index(x0, y0, params);
    const uint p01 = source_index(x1, y0, params);
    const uint p10 = source_index(x0, y1, params);
    const uint p11 = source_index(x1, y1, params);

    const uint red_linear =
        uint(gamma_to_linear[source[p00 + params.red_offset]]) +
        uint(gamma_to_linear[source[p01 + params.red_offset]]) +
        uint(gamma_to_linear[source[p10 + params.red_offset]]) +
        uint(gamma_to_linear[source[p11 + params.red_offset]]);
    const uint green_linear =
        uint(gamma_to_linear[source[p00 + params.green_offset]]) +
        uint(gamma_to_linear[source[p01 + params.green_offset]]) +
        uint(gamma_to_linear[source[p10 + params.green_offset]]) +
        uint(gamma_to_linear[source[p11 + params.green_offset]]);
    const uint blue_linear =
        uint(gamma_to_linear[source[p00 + params.blue_offset]]) +
        uint(gamma_to_linear[source[p01 + params.blue_offset]]) +
        uint(gamma_to_linear[source[p10 + params.blue_offset]]) +
        uint(gamma_to_linear[source[p11 + params.blue_offset]]);

    const int red = interpolate_gamma(red_linear, linear_to_gamma);
    const int green = interpolate_gamma(green_linear, linear_to_gamma);
    const int blue = interpolate_gamma(blue_linear, linear_to_gamma);
    output[params.y_size + index] = rgb_to_u(red, green, blue);
    output[params.y_size + params.uv_size + index] =
        rgb_to_v(red, green, blue);
  }
}
)METAL";

struct KernelParams {
  uint32_t width;
  uint32_t height;
  uint32_t step;
  uint32_t source_stride;
  uint32_t red_offset;
  uint32_t green_offset;
  uint32_t blue_offset;
  uint32_t uv_width;
  uint32_t y_size;
  uint32_t uv_size;
};

struct MetalState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> source = nil;
  id<MTLBuffer> output = nil;
  id<MTLBuffer> gamma_to_linear = nil;
  id<MTLBuffer> linear_to_gamma = nil;
  size_t source_capacity = 0;
  size_t output_capacity = 0;
  std::mutex mutex;
};

bool EnvironmentFlag(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  return std::strcmp(value, "0") != 0 && strcasecmp(value, "false") != 0 &&
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

void BuildGammaTables(uint16_t gamma_to_linear[256],
                      int32_t linear_to_gamma[33]) {
  constexpr double kGamma = 0.80;
  constexpr int kGammaFix = 12;
  constexpr int kGammaTabFix = 7;
  constexpr int kGammaScale = (1 << kGammaFix) - 1;
  constexpr int kGammaTabSize = 1 << (kGammaFix - kGammaTabFix);
  const double scale = static_cast<double>(1 << kGammaTabFix) / kGammaScale;
  for (int value = 0; value <= 255; ++value) {
    gamma_to_linear[value] = static_cast<uint16_t>(
        std::pow(value / 255.0, kGamma) * kGammaScale + 0.5);
  }
  for (int value = 0; value <= kGammaTabSize; ++value) {
    linear_to_gamma[value] = static_cast<int32_t>(
        255.0 * std::pow(scale * value, 1.0 / kGamma) + 0.5);
  }
}

MetalState* InitializeState() {
  if (!EnvironmentFlag("WEBP_METAL_LOSSY", true)) return nullptr;
  @autoreleasepool {
    auto* state = new MetalState();
    state->device = MTLCreateSystemDefaultDevice();
    if (state->device == nil) {
      delete state;
      return nullptr;
    }
    NSError* error = nil;
    id<MTLLibrary> library = [state->device
        newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                     options:nil error:&error];
    if (library == nil) {
      if (EnvironmentFlag("WEBP_METAL_VERBOSE", false)) {
        std::fprintf(stderr, "WebP-Metal: lossy library failed: %s\n",
                     error.localizedDescription.UTF8String);
      }
      delete state;
      return nullptr;
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@"rgb_to_yuv420"];
    if (function == nil) {
      if (EnvironmentFlag("WEBP_METAL_VERBOSE", false)) {
        std::fprintf(stderr, "WebP-Metal: lossy function not found\n");
      }
      delete state;
      return nullptr;
    }
    state->pipeline = [state->device newComputePipelineStateWithFunction:
        function error:&error];
    state->queue = [state->device newCommandQueue];
    if (state->pipeline == nil || state->queue == nil) {
      if (EnvironmentFlag("WEBP_METAL_VERBOSE", false)) {
        std::fprintf(stderr, "WebP-Metal: lossy pipeline failed: %s\n",
                     error.localizedDescription.UTF8String);
      }
      delete state;
      return nullptr;
    }

    uint16_t gamma_to_linear[256];
    int32_t linear_to_gamma[33];
    BuildGammaTables(gamma_to_linear, linear_to_gamma);
    state->gamma_to_linear = [state->device
        newBufferWithBytes:gamma_to_linear length:sizeof(gamma_to_linear)
                   options:MTLResourceStorageModeShared];
    state->linear_to_gamma = [state->device
        newBufferWithBytes:linear_to_gamma length:sizeof(linear_to_gamma)
                   options:MTLResourceStorageModeShared];
    if (state->gamma_to_linear == nil || state->linear_to_gamma == nil) {
      delete state;
      return nullptr;
    }
    return state;
  }
}

MetalState* GetState() {
  static MetalState* state = InitializeState();
  return state;
}

bool EnsureBuffers(MetalState* state, size_t source_size, size_t output_size) {
  if (source_size > state->source_capacity) {
    state->source_capacity = RoundedBufferLength(source_size);
    state->source = [state->device newBufferWithLength:state->source_capacity
        options:MTLResourceStorageModeShared];
  }
  if (output_size > state->output_capacity) {
    state->output_capacity = RoundedBufferLength(output_size);
    state->output = [state->device newBufferWithLength:state->output_capacity
        options:MTLResourceStorageModeShared];
  }
  return state->source != nil && state->output != nil;
}

}  // namespace

extern "C" int WebPImportRGBToYUVAMetal(
    const uint8_t* red, const uint8_t* green, const uint8_t* blue, int step,
    int source_stride, WebPPicture* picture) {
  if (red == nullptr || green == nullptr || blue == nullptr ||
      picture == nullptr || (step != 3 && step != 4) || source_stride <= 0) {
    return 0;
  }
  const size_t pixel_count =
      static_cast<size_t>(picture->width) * picture->height;
  if (pixel_count < EnvironmentSize("WEBP_METAL_LOSSY_MIN_PIXELS",
                                     kDefaultMinimumPixels) ||
      !EnvironmentFlag("WEBP_METAL_LOSSY", true)) {
    return 0;
  }
  MetalState* state = GetState();
  if (state == nullptr) return 0;

  const uint8_t* base = std::min(red, std::min(green, blue));
  const ptrdiff_t red_offset = red - base;
  const ptrdiff_t green_offset = green - base;
  const ptrdiff_t blue_offset = blue - base;
  if (red_offset < 0 || green_offset < 0 || blue_offset < 0 ||
      red_offset >= step || green_offset >= step || blue_offset >= step) {
    return 0;
  }
  const uint32_t uv_width = (picture->width + 1u) >> 1;
  const uint32_t uv_height = (picture->height + 1u) >> 1;
  const size_t y_size = pixel_count;
  const size_t uv_size = static_cast<size_t>(uv_width) * uv_height;
  const size_t source_size = static_cast<size_t>(source_stride) *
                             picture->height;
  const size_t output_size = y_size + 2u * uv_size;
  const KernelParams params = {
      static_cast<uint32_t>(picture->width),
      static_cast<uint32_t>(picture->height), static_cast<uint32_t>(step),
      static_cast<uint32_t>(source_stride),
      static_cast<uint32_t>(red_offset), static_cast<uint32_t>(green_offset),
      static_cast<uint32_t>(blue_offset), uv_width,
      static_cast<uint32_t>(y_size), static_cast<uint32_t>(uv_size)};

  std::lock_guard<std::mutex> lock(state->mutex);
  @autoreleasepool {
    if (!EnsureBuffers(state, source_size, output_size)) return 0;
    std::memcpy(state->source.contents, base, source_size);
    id<MTLCommandBuffer> command = [state->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) return 0;
    [encoder setComputePipelineState:state->pipeline];
    [encoder setBuffer:state->source offset:0 atIndex:0];
    [encoder setBuffer:state->output offset:0 atIndex:1];
    [encoder setBuffer:state->gamma_to_linear offset:0 atIndex:2];
    [encoder setBuffer:state->linear_to_gamma offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    const NSUInteger threads = std::min(
        kPreferredThreads, state->pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(pixel_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    const CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;

    const uint8_t* output = static_cast<const uint8_t*>(state->output.contents);
    for (int y = 0; y < picture->height; ++y) {
      std::memcpy(picture->y + static_cast<size_t>(y) * picture->y_stride,
                  output + static_cast<size_t>(y) * picture->width,
                  picture->width);
    }
    const uint8_t* u = output + y_size;
    const uint8_t* v = u + uv_size;
    for (uint32_t y = 0; y < uv_height; ++y) {
      std::memcpy(picture->u + static_cast<size_t>(y) * picture->uv_stride,
                  u + static_cast<size_t>(y) * uv_width, uv_width);
      std::memcpy(picture->v + static_cast<size_t>(y) * picture->uv_stride,
                  v + static_cast<size_t>(y) * uv_width, uv_width);
    }
    if (EnvironmentFlag("WEBP_METAL_VERBOSE", false)) {
      std::fprintf(stderr,
                   "WebP-Metal: lossy RGB->YUV %dx%d in %.3f ms\n",
                   picture->width, picture->height,
                   (CFAbsoluteTimeGetCurrent() - start) * 1000.0);
    }
  }
  return 1;
}
