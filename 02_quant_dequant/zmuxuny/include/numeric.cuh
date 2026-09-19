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
  union {
    uint32_t u;
    float f;
  } bits{((e + 120) << 23) | (m << 20)};
  float v = e == 0 ? float(m) * 0x1p-9f : bits.f;
  return (c & 128) ? -v : v;
}
__host__ __device__ inline float fp4_value(unsigned c) {
  unsigned a = c & 7;
  union {
    uint32_t u;
    float f;
  } bits{((126 + (a >> 1)) << 23) | ((a & 1) << 22)};
  float v = a < 2 ? float(a) * 0.5f : bits.f;
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
  union {
    float f;
    uint32_t u;
  } bits{a};
  int exponent = int(bits.u >> 23) - 120;
  uint32_t mantissa = bits.u & 0x7fffffU;
  int lo = int(mantissa >> 20);
  uint32_t remainder = mantissa & 0xfffffU;
  bool up = stochastic ? u < float(remainder) * 0x1p-20f
                       : (remainder > 0x80000U || (remainder == 0x80000U && (lo & 1)));
  return uint8_t(sign | min(126, exponent * 8 + lo + int(up)));
}
__host__ __device__ inline uint8_t encode4(float x, bool stochastic = false,
                                           float u = 0) {
  float a = fminf(fabsf(x), 6.0f);
  int lo = a < 2 ? int(a * 2) : (a < 4 ? int(a) + 2 : int(a * 0.5f) + 4);
  lo = min(lo, 7);
  float l = fp4_value(lo);
  int c = lo;
  if (lo < 7) {
    float p = (a - l) * (a < 2 ? 2.0f : (a < 4 ? 1.0f : 0.5f));
    c += stochastic ? u < p : (p > 0.5f || (p == 0.5f && (lo & 1)));
  }
  return uint8_t(c | (signbit(x) ? 8 : 0));
}
// E4M3 scales times the E2M1 decision midpoints are exactly representable
// in FP32. Comparing in the scaled domain removes one division for RNE.
// Stochastic rounding still uses the rounded quotient and its original RNG.
__host__ __device__ inline uint8_t encode4_scaled(float x, float scale, bool stochastic,
                                                  float u) {
  if (scale == 0)
    return 0;
  if (stochastic)
    return encode4(x / scale, true, u);
  float a = fabsf(x);
  unsigned c = (a > scale * 0.25f) + (a >= scale * 0.75f) + (a > scale * 1.25f) +
               (a >= scale * 1.75f) + (a > scale * 2.5f) + (a >= scale * 3.5f) +
               (a > scale * 5.0f);
  return uint8_t(c | (signbit(x) ? 8 : 0));
}
// Round the required scale upward to a power of two: no finite-input clipping.
__host__ __device__ inline uint8_t mx_scale(float amax) {
  if (amax == 0)
    return 127;
  union {
    float f;
    uint32_t u;
  } bits{amax};
  int exponent = int(bits.u >> 23);
  if (exponent != 0)
    return uint8_t(
        max(0, min(254, exponent - 8 + int((bits.u & 0x7fffffU) > 0x600000U))));
  // Subnormal FP32 values need the full normalization path.
  int e;
  float m = frexpf(amax, &e);
  e -= m <= 0.875f ? 9 : 8;
  return uint8_t(max(0, min(254, e + 127)));
}
__host__ __device__ inline float scale_value(uint8_t s, int fmt) {
  if (fmt != MXFP8)
    return fp8_value(s);
  union {
    uint32_t u;
    float f;
  } bits{s == 0 ? 0x00400000U : uint32_t(s) << 23};
  return bits.f;
}
// Dividing by an E8M0 scale equals multiplying by its exact power-of-two
// reciprocal. Scale code zero has no finite FP32 reciprocal and uses division.
__host__ __device__ inline float mx_scaled(float x, uint8_t s) {
  if (s == 0)
    return x / 0x1p-127f;
  union {
    uint32_t u;
    float f;
  } reciprocal{s == 254 ? 0x00400000U : uint32_t(254 - s) << 23};
  return x * reciprocal.f;
}
__host__ __device__ inline float global_scale(float amax) {
  // Preserve a nonzero scale for subnormal FP32 inputs as well.
  return amax == 0 ? 1.0f : fmaxf(amax / 2688.0f, 0x1p-126f);
}
} // namespace lp
