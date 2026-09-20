#pragma once
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef LP_NATIVE_FP8
#define LP_NATIVE_FP8 0
#endif
#if LP_NATIVE_FP8 && (defined(__MACACC__) || defined(__ILUVATAR__))
#error "Native NVIDIA FP8 encoding is unavailable on this platform; use the software target"
#endif
#if LP_NATIVE_FP8 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
#error "The optional native FP8 comparison requires ARCH=89 or newer"
#endif

namespace lp {
#if defined(__MACACC__)
constexpr unsigned long long FULL_WARP_MASK = ~0ULL;
#else
// CoreX also accepts a 32-bit mask: SDK 4.4 implements sync shuffles with
// native lockstep shuffles and uses their width argument to define subgroups.
constexpr unsigned FULL_WARP_MASK = 0xffffffffU;
#endif
inline const char *compute_platform() {
#if defined(__MACACC__)
  return "metax_maca";
#elif defined(__ILUVATAR__)
  return "iluvatar_corex";
#else
  return "nvidia_cuda";
#endif
}
inline const char *fp8_encoding_backend() {
  return LP_NATIVE_FP8 ? "native_rne_software_sr" : "software";
}
enum Dtype { FP32 = 0, FP16 = 1, BF16 = 2 };
enum Format { MXFP8 = 0, NVFP4 = 1 };

// Use native round-to-nearest BF16 conversion on Ampere+, with the original
// bit-exact integer implementation for the CPU and older architectures.
__host__ __device__ inline uint16_t bf16(float x) {
#if !defined(__MACACC__) && !defined(__ILUVATAR__) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
  uint16_t result;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(x));
  return result;
#else
  union {
    float f;
    uint32_t u;
  } v{x};
  v.u += 0x7fffU + ((v.u >> 16) & 1U);
  return static_cast<uint16_t>(v.u >> 16);
#endif
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
#if defined(__ILUVATAR__) && defined(__CUDA_ARCH__)
  // CoreX 4.4 can incorrectly broadcast shifts of a lane-derived hash at -O3.
  // Volatile keeps each hash step lane-local; tests/validate_numeric.cu checks
  // every lane against the host, including indices above 2^32. This affects
  // random signs/stochastic rounding only, not nearest-rounding benchmarks.
  volatile uint32_t x = uint32_t(i) ^ (uint32_t(i >> 32) * 0x9e3779b9U) ^ seed;
#else
  uint32_t x = uint32_t(i) ^ (uint32_t(i >> 32) * 0x9e3779b9U) ^ seed;
#endif
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
// Inspect the sign bit on both host and device, including negative zero.
// CoreX exposes an unqualified signbit overload only on the device.
__host__ __device__ inline bool negative(float x) {
  union {
    float f;
    uint32_t u;
  } bits{x};
  return (bits.u >> 31) != 0;
}
__host__ __device__ inline uint8_t encode8(float x, bool stochastic = false,
                                           float u = 0) {
#if LP_NATIVE_FP8 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  if (!stochastic) {
    uint16_t result;
    // PTX packs source a into the upper byte and source b into the lower byte.
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(result) : "f"(0.0f), "f"(x));
    return uint8_t(result);
  }
#endif
  unsigned sign = negative(x) ? 128 : 0;
  float a = fminf(fabsf(x), 448.0f);
  if (a < 0.015625f) {
#if defined(__CUDA_ARCH__)
    if (!stochastic)
      return uint8_t(sign | unsigned(__float2int_rn(a * 512.0f)));
#endif
    return uint8_t(sign | round_code(a * 512.0f, stochastic, u));
  }
  union {
    float f;
    uint32_t u;
  } bits{a};
  if (!stochastic) {
    // RNE at the retained mantissa LSB; carry naturally increments exponent.
    // Clamping the magnitude to 448 above bounds the resulting code at 126.
    uint32_t rounded = bits.u + 0x7ffffU + ((bits.u >> 20) & 1U);
    return uint8_t(sign | ((rounded >> 20) - 960U));
  }
  int exponent = int(bits.u >> 23) - 120;
  uint32_t mantissa = bits.u & 0x7fffffU;
  int lo = int(mantissa >> 20);
  uint32_t remainder = mantissa & 0xfffffU;
  bool up = u < float(remainder) * 0x1p-20f;
  int code = exponent * 8 + lo + int(up);
  return uint8_t(sign | (code < 126 ? code : 126));
}
// Two adjacent values share one native conversion instruction. Software/SR
// retain exactly the scalar encoder's element-index-dependent rounding.
__host__ __device__ inline uint16_t encode8_pair(float lo, float hi, bool stochastic,
                                                 float u0, float u1) {
#if LP_NATIVE_FP8 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  if (!stochastic) {
    uint16_t result;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(result) : "f"(hi), "f"(lo));
    return result;
  }
#endif
  return uint16_t(encode8(lo, stochastic, u0)) |
         (uint16_t(encode8(hi, stochastic, u1)) << 8);
}
__host__ __device__ inline uint8_t encode4(float x, bool stochastic = false,
                                           float u = 0) {
  float a = fminf(fabsf(x), 6.0f);
  int lo = a < 2 ? int(a * 2) : (a < 4 ? int(a) + 2 : int(a * 0.5f) + 4);
  lo = lo < 7 ? lo : 7;
  float l = fp4_value(lo);
  int c = lo;
  if (lo < 7) {
    float p = (a - l) * (a < 2 ? 2.0f : (a < 4 ? 1.0f : 0.5f));
    c += stochastic ? u < p : (p > 0.5f || (p == 0.5f && (lo & 1)));
  }
  return uint8_t(c | (negative(x) ? 8 : 0));
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
  return uint8_t(c | (negative(x) ? 8 : 0));
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
  if (exponent != 0) {
    int code = exponent - 8 + int((bits.u & 0x7fffffU) > 0x600000U);
    return uint8_t(code < 0 ? 0 : (code > 254 ? 254 : code));
  }
  // Subnormal FP32 values need the full normalization path.
  int e;
  float m = frexpf(amax, &e);
  e -= m <= 0.875f ? 9 : 8;
  int code = e + 127;
  return uint8_t(code < 0 ? 0 : (code > 254 ? 254 : code));
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
