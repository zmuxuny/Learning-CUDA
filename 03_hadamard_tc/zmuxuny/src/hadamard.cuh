#pragma once
#include "../include/gpu.cuh"
#include <mma.h>

namespace lp {
// One warp transforms one row. XOR butterflies stay in registers: the first
// five stages exchange lanes; remaining stages exchange each lane's registers.
template <int D, int MODE>
__global__ void hadamard_kernel(const void *x, void *y, size_t rows, int dtype,
                                bool normalize, bool random_sign, uint32_t sign_seed,
                                uint8_t *data, uint8_t *scales, float *amax, Layout p) {
  constexpr int V = (D + 31) / 32;
  int lane = threadIdx.x & 31;
  size_t row = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
  if (row >= rows)
    return;
  float v[V];
#pragma unroll
  for (int j = 0; j < V; ++j) {
    int c = j * 32 + lane;
    v[j] = c < D ? load(x, row * D + c, dtype) : 0;
    if (random_sign && uniform(c, sign_seed) < 0.5f)
      v[j] = -v[j];
  }
#pragma unroll
  for (int step = 1; step < 32 && step < D; step *= 2) {
#pragma unroll
    for (int j = 0; j < V; ++j) {
      float other = __shfl_xor_sync(0xffffffff, v[j], step);
      v[j] = (lane & step) ? other - v[j] : v[j] + other;
    }
  }
#pragma unroll
  for (int step = 1; step < V; step *= 2) {
#pragma unroll
    for (int j = 0; j < V; ++j)
      if (!(j & step)) {
        float a = v[j], b = v[j + step];
        v[j] = a + b;
        v[j + step] = a - b;
      }
  }
#pragma unroll
  for (int j = 0; j < V; ++j)
    v[j] = cast(v[j] * (normalize ? 1.0f / sqrtf(float(D)) : 1.0f), dtype);
  if (MODE == 0) {
#pragma unroll
    for (int j = 0; j < V; ++j)
      if (j * 32 + lane < D)
        save(y, row * D + j * 32 + lane, v[j], dtype);
  } else if (MODE == 1) {
    float m = 0;
#pragma unroll
    for (int j = 0; j < V; ++j)
      if (j * 32 + lane < D)
        m = fmaxf(m, fabsf(v[j]));
    m = warp_max(m);
    if (lane == 0)
      atomicMax(reinterpret_cast<unsigned *>(amax), __float_as_uint(m));
  } else {
    // Fused path uses canonical 32-value MXFP8 / 16-value NVFP4 blocks.
    // Rounding to the transform output dtype before quantizing makes the
    // packed result exactly equal to the unfused, materialized pipeline.
    int b = p.fmt == MXFP8 ? 32 : 16;
    float global = p.fmt == NVFP4 ? global_scale(*amax) : 1;
#pragma unroll
    for (int j = 0; j < V; ++j) {
      int c = j * 32 + lane;
      float m = warp_max(c < D ? fabsf(v[j]) : 0, b);
      uint8_t s = p.fmt == MXFP8 ? mx_scale(m) : encode8((m / global) / 6.0f);
      float scale = scale_value(s, p.fmt);
      float z = p.fmt == NVFP4 ? v[j] / global : v[j];
      z = scale == 0 ? 0 : z / scale;
      size_t i = row * D + c;
      uint8_t code = p.fmt == MXFP8 ? encode8(z, p.stochastic, uniform(i, p.seed))
                                    : encode4(z, p.stochastic, uniform(i, p.seed));
      if (c < D && lane % b == 0)
        scales[row * ((D + b - 1) / b) + c / b] = s;
      if (p.fmt == MXFP8) {
        if (c < D)
          data[i] = code;
      } else {
        unsigned hi = __shfl_xor_sync(0xffffffff, unsigned(code), 1);
        if (c < D && !(lane & 1))
          data[row * ((D + 1) / 2) + c / 2] = code | ((c + 1 < D ? hi : 0) << 4);
      }
    }
  }
}
template <int MODE>
inline void launch_had(const void *x, void *y, size_t rows, int d, int dtype, bool norm,
                       bool signs, uint32_t seed, uint8_t *data, uint8_t *scales,
                       float *amax, Layout p) {
#define HAD_CASE(D)                                                                    \
  case D:                                                                              \
    hadamard_kernel<D, MODE><<<(rows + 3) / 4, 128>>>(x, y, rows, dtype, norm, signs,  \
                                                      seed, data, scales, amax, p);    \
    break
  switch (d) {
    HAD_CASE(1);
    HAD_CASE(2);
    HAD_CASE(4);
    HAD_CASE(8);
    HAD_CASE(16);
    HAD_CASE(32);
    HAD_CASE(64);
    HAD_CASE(128);
    HAD_CASE(256);
    HAD_CASE(512);
    HAD_CASE(1024);
  }
#undef HAD_CASE
}
inline void fused_had(const void *x, size_t rows, int d, int dtype, bool norm,
                      bool signs, uint32_t seed, uint8_t *data, uint8_t *scales,
                      float *amax, Layout p) {
  if (p.fmt == NVFP4) {
    cudaMemsetAsync(amax, 0, sizeof(float));
    launch_had<1>(x, nullptr, rows, d, dtype, norm, signs, seed, nullptr, nullptr, amax,
                  p);
  }
  launch_had<2>(x, nullptr, rows, d, dtype, norm, signs, seed, data, scales, amax, p);
}

// Explicit FP16 Tensor Core comparison: dense X*H, 16x16 WMMA tiles.
// Its O(D^2) arithmetic differs from the O(D log D) butterfly algorithm.
__global__ void hadamard_tc(const __half *x, const __half *h, __half *y, size_t rows,
                            int d, bool norm, bool signs, uint32_t seed) {
  using namespace nvcuda;
  __shared__ __align__(32) __half a[4][256];
  __shared__ __align__(32) float result[4][256];
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  int col = (blockIdx.y * 4 + warp) * 16;
  size_t row = blockIdx.x * 16;
  if (col >= d)
    return;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int k = 0; k < d; k += 16) {
    for (int i = lane; i < 256; i += 32) {
      size_t r = row + i / 16;
      int c = k + i % 16;
      float v = r < rows ? __half2float(x[r * d + c]) : 0;
      if (signs && uniform(c, seed) < 0.5f)
        v = -v;
      a[warp][i] = __float2half_rn(v);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
    wmma::load_matrix_sync(af, a[warp], 16);
    wmma::load_matrix_sync(bf, h + k * d + col, d);
    wmma::mma_sync(acc, af, bf, acc);
    __syncwarp();
  }
  wmma::store_matrix_sync(result[warp], acc, 16, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 256; i += 32) {
    size_t r = row + i / 16;
    if (r < rows)
      y[r * d + col + i % 16] =
          __float2half_rn(result[warp][i] * (norm ? 1.0f / sqrtf(float(d)) : 1.0f));
  }
}
} // namespace lp
