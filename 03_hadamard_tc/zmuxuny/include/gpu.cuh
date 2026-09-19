#pragma once
#include "numeric.cuh"

namespace lp {
struct Layout {
  size_t rows, cols;
  int dtype, fmt, block;
  bool tensor, stochastic;
  uint32_t seed;
  size_t count() const { return rows * cols; }
  size_t groups() const { return tensor ? 1 : rows * ((cols + block - 1) / block); }
  size_t bytes() const { return fmt == MXFP8 ? count() : rows * ((cols + 1) / 2); }
};
__device__ inline float warp_max(float x, int width = 32) {
  for (int s = width / 2; s; s /= 2)
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, s, width));
  return x;
}
__global__ void maximum(const void *x, size_t n, int dtype, float *result) {
  float a = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x)
    a = fmaxf(a, fabsf(load(x, i, dtype)));
  a = warp_max(a);
  if ((threadIdx.x & 31) == 0)
    atomicMax(reinterpret_cast<unsigned *>(result), __float_as_uint(a));
}
__global__ void scales_kernel(const void *x, uint8_t *scales, const float *amax,
                              Layout p) {
  size_t groups_per_row = (p.cols + p.block - 1) / p.block;
  size_t g = size_t(blockIdx.x) * 8 + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  size_t ng = p.tensor ? 1 : p.rows * groups_per_row;
  if (g >= ng)
    return;
  float a = 0;
  if (p.tensor)
    a = *amax;
  else {
    size_t row = g / groups_per_row, start = (g % groups_per_row) * p.block;
    for (int j = lane; j < p.block && start + j < p.cols; j += 32)
      a = fmaxf(a, fabsf(load(x, row * p.cols + start + j, p.dtype)));
    a = warp_max(a);
  }
  if (lane == 0)
    scales[g] =
        p.fmt == MXFP8 ? mx_scale(a) : encode8((a / global_scale(*amax)) / 6.0f);
}
__device__ inline size_t group_index(size_t row, size_t col, Layout p) {
  return p.tensor ? 0 : row * ((p.cols + p.block - 1) / p.block) + col / p.block;
}
__device__ inline uint8_t quant_element(const void *x, size_t row, size_t col,
                                        const uint8_t *scales, const float *amax,
                                        Layout p) {
  size_t i = row * p.cols + col;
  float s = scale_value(scales[group_index(row, col, p)], p.fmt);
  float value = load(x, i, p.dtype);
  if (p.fmt == NVFP4)
    value /= global_scale(*amax);
  float y = s == 0 ? 0 : value / s;
  float u = uniform(i, p.seed);
  return p.fmt == MXFP8 ? encode8(y, p.stochastic, u) : encode4(y, p.stochastic, u);
}
__global__ void quant_kernel(const void *x, uint8_t *data, const uint8_t *scales,
                             const float *amax, Layout p) {
  size_t stride = p.fmt == MXFP8 ? p.cols : (p.cols + 1) / 2;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.rows * stride)
    return;
  size_t row = i / stride, col = i % stride;
  if (p.fmt == MXFP8)
    data[i] = quant_element(x, row, col, scales, amax, p);
  else {
    col *= 2;
    uint8_t lo = quant_element(x, row, col, scales, amax, p);
    uint8_t hi = col + 1 < p.cols ? quant_element(x, row, col + 1, scales, amax, p) : 0;
    data[i] = lo | (hi << 4);
  }
}
__global__ void dequant_kernel(const uint8_t *data, const uint8_t *scales, float global,
                               void *out, int out_type, Layout p) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.rows * p.cols)
    return;
  size_t row = i / p.cols, col = i % p.cols;
  uint8_t c = p.fmt == MXFP8
                  ? data[i]
                  : (data[row * ((p.cols + 1) / 2) + col / 2] >> ((col & 1) * 4)) & 15;
  float v = p.fmt == MXFP8 ? fp8_value(c) : fp4_value(c);
  float s = scale_value(scales[group_index(row, col, p)], p.fmt);
  // Same multiplication order as the independent file decoder.
  save(out, i, (v * s) * global, out_type);
}
inline void launch_max(const void *x, size_t n, int dtype, float *amax) {
  cudaMemsetAsync(amax, 0, sizeof(float));
  maximum<<<min(size_t(256), (n + 255) / 256), 256>>>(x, n, dtype, amax);
}
inline void quantize(const void *x, uint8_t *data, uint8_t *scales, float *amax,
                     Layout p) {
  if (p.fmt == NVFP4 || p.tensor)
    launch_max(x, p.count(), p.dtype, amax);
  scales_kernel<<<(p.groups() + 7) / 8, 256>>>(x, scales, amax, p);
  quant_kernel<<<(p.bytes() + 255) / 256, 256>>>(x, data, scales, amax, p);
}
inline void dequantize(const uint8_t *data, const uint8_t *scales, float global,
                       void *out, int out_type, Layout p) {
  dequant_kernel<<<(p.count() + 255) / 256, 256>>>(data, scales, global, out, out_type,
                                                   p);
}
} // namespace lp
