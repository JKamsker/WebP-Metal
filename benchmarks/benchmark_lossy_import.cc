// Benchmark WebPPictureImportRGB independently from image decoding and VP8
// compression. Run in separate processes with WEBP_METAL_LOSSY=0 and =1.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "webp/encode.h"

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 4000;
  const int height = argc > 2 ? std::atoi(argv[2]) : 3000;
  const int iterations = argc > 3 ? std::atoi(argv[3]) : 20;
  const int warmups = argc > 4 ? std::atoi(argv[4]) : 1;
  if (width <= 0 || height <= 0 || iterations <= 0 || warmups < 0) return 1;

  std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3u);
  uint32_t state = 0x12345678u;
  for (uint8_t& value : rgb) {
    state = state * 1664525u + 1013904223u;
    value = static_cast<uint8_t>(state >> 24);
  }

  WebPPicture picture;
  // Warm up allocation, dispatch, and lazily initialized Metal state unless a
  // cold first-call measurement was requested.
  for (int i = 0; i < warmups; ++i) {
    if (!WebPPictureInit(&picture)) return 1;
    picture.width = width;
    picture.height = height;
    if (!WebPPictureImportRGB(&picture, rgb.data(), width * 3)) return 1;
    WebPPictureFree(&picture);
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    if (!WebPPictureInit(&picture)) return 1;
    picture.width = width;
    picture.height = height;
    if (!WebPPictureImportRGB(&picture, rgb.data(), width * 3)) return 1;
    WebPPictureFree(&picture);
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  std::printf("%dx%d iterations=%d warmups=%d total=%.6f average_ms=%.3f\n",
              width, height, iterations, warmups, seconds,
              seconds * 1000.0 / iterations);
  return 0;
}
