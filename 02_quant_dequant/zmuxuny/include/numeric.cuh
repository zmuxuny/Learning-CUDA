#pragma once
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lp {
enum Dtype { FP32 = 0, FP16 = 1, BF16 = 2 };
enum Format { MXFP8 = 0, NVFP4 = 1 };

// BF16 conversion is implemented with integer operations, including on sm_75.
__host__ __device__ inline uint16_t bf16(float x) {
  union {
    float f;
    uint32_t u;
  } v{x};
  v.u += 0x7fffU + ((v.u >> 16) & 1U);
  return static_cast<uint16_t>(v.u >> 16);
}
__host__ __device__ inline float unbf16(uint16_t x) {
  union {
    uint32_t u;
    float f;
  } v{uint32_t(x) << 16};
  return v.f;
}
__host__ __device__ inline float load(const void *p, size_t i, int dtype) {
  if (dtype == FP32)
    return static_cast<const float *>(p)[i];
  if (dtype == FP16)
    return __half2float(static_cast<const __half *>(p)[i]);
  return unbf16(static_cast<const uint16_t *>(p)[i]);
}
__host__ __device__ inline void save(void *p, size_t i, float x, int dtype) {
  if (dtype == FP32)
    static_cast<float *>(p)[i] = x;
  else if (dtype == FP16)
    static_cast<__half *>(p)[i] = __float2half_rn(x);
  else
    static_cast<uint16_t *>(p)[i] = bf16(x);
}
__host__ __device__ inline float cast(float x, int dtype) {
  if (dtype == FP16)
    return __half2float(__float2half_rn(x));
  return dtype == BF16 ? unbf16(bf16(x)) : x;
}
__host__ __device__ inline float fp8_value(unsigned c) {
  unsigned a = c & 127, e = a >> 3, m = a & 7;
  float v = e == 0 ? ldexpf(float(m), -9) : ldexpf(1.0f + float(m) / 8.0f, int(e) - 7);
  return (c & 128) ? -v : v;
}
__host__ __device__ inline float fp4_value(unsigned c) {
  unsigned a = c & 7;
  float v =
      a < 2 ? float(a) * 0.5f : ldexpf(1.0f + float(a & 1) * 0.5f, int(a >> 1) - 1);
  return (c & 8) ? -v : v;
}
// Counter-based rounding: independent of launch geometry and repeatable by seed.
__host__ __device__ inline float uniform(size_t i, uint32_t seed) {
  uint32_t x = uint32_t(i) ^ (uint32_t(i >> 32) * 0x9e3779b9U) ^ seed;
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return float(x >> 8) * (1.0f / 16777216.0f);
}
__host__ __device__ inline int round_code(float pos, bool stochastic, float u) {
  int lo = int(floorf(pos));
  float frac = pos - float(lo);
  return lo + (stochastic ? u < frac : (frac > 0.5f || (frac == 0.5f && (lo & 1))));
}
__host__ __device__ inline uint8_t encode8(float x, bool stochastic = false,
                                           float u = 0) {
  unsigned sign = signbit(x) ? 128 : 0;
  float a = fminf(fabsf(x), 448.0f);
  if (a < 0.015625f)
    return uint8_t(sign | round_code(a * 512.0f, stochastic, u));
  int e;
  frexpf(a, &e);
  // Within an exponent bin codes are consecutive, including its upper boundary.
  // Round the mantissa before adding the exponent bits. Adding the integer
  // code offset in float first can erase an ULP just above a midpoint.
  int code = (e + 5) * 8 + round_code(ldexpf(a, 4 - e), stochastic, u);
  return uint8_t(sign | min(126, code));
}
__host__ __device__ inline uint8_t encode4(float x, bool stochastic = false,
                                           float u = 0) {
  float a = fminf(fabsf(x), 6.0f);
  int lo = a < 2 ? int(a * 2) : (a < 4 ? int(a) + 2 : int(a * 0.5f) + 4);
  lo = min(lo, 7);
  float l = fp4_value(lo), h = fp4_value(min(lo + 1, 7));
  int c = lo;
  if (lo < 7) {
    float p = (a - l) / (h - l);
    c += stochastic ? u < p : (p > 0.5f || (p == 0.5f && (lo & 1)));
  }
  return uint8_t(c | (signbit(x) ? 8 : 0));
}
// Round the required scale upward to a power of two: no finite-input clipping.
__host__ __device__ inline uint8_t mx_scale(float amax) {
  if (amax == 0)
    return 127;
  int e;
  float m = frexpf(amax, &e);
  e -= m <= 0.875f ? 9 : 8;
  return uint8_t(max(0, min(254, e + 127)));
}
__host__ __device__ inline float scale_value(uint8_t s, int fmt) {
  return fmt == MXFP8 ? ldexpf(1.0f, int(s) - 127) : fp8_value(s);
}
__host__ __device__ inline float global_scale(float amax) {
  // Preserve a nonzero scale for subnormal FP32 inputs as well.
  return amax == 0 ? 1.0f : fmaxf(amax / 2688.0f, 0x1p-126f);
}
} // namespace lp
