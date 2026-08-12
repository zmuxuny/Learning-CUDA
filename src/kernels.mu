// Moore Threads MUSA implementation. The same reference algorithms are
// compiled by mcc and use musa* runtime calls on the MTT accelerator.
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <vector>
#include <musa_fp16.h>

#include "../tester/utils.h"

// The MUSA implementation keeps the same numerically stable algorithms as
// the CUDA version while using the MUSA runtime and half-precision helpers.
// Keeping the conversion logic in one place also makes float/half behavior
// consistent across RMSNorm and attention.
// Keep accumulation in float for both float and half inputs. The specialized
// half conversions avoid relying on implicit device-side casts.
template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<half>(half value) {
  return __half2float(value);
}

// Convert the float accumulator back to MUSA's requested output type.
template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half from_float<half>(float value) {
  return __float2half(value);
}

// One block handles one token row. Shared-memory reduction computes the mean
// square before every thread writes its normalized elements.
template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t rows, size_t hidden_dim, float eps) {
  const size_t row = static_cast<size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  extern __shared__ float partial_sums[];
  float sum = 0.0f;
  const size_t row_offset = row * hidden_dim;
  for (size_t column = threadIdx.x; column < hidden_dim;
       column += blockDim.x) {
    const float value = to_float(input[row_offset + column]);
    sum += value * value;
  }
  partial_sums[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale =
      rsqrtf(partial_sums[0] / static_cast<float>(hidden_dim) + eps);
  for (size_t column = threadIdx.x; column < hidden_dim;
       column += blockDim.x) {
    const float value = to_float(input[row_offset + column]);
    const float factor = to_float(weight[column]);
    output[row_offset + column] = from_float<T>(value * scale * factor);
  }
}

// Allocate MUSA buffers, transfer the host tensors, execute RMSNorm, and copy
// the synchronized result back to the caller-owned host vector.
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  const size_t input_bytes = rows * hidden_dim * sizeof(T);
  const size_t weight_bytes = hidden_dim * sizeof(T);
  h_output.resize(rows * hidden_dim);

  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_input), input_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_weight), weight_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_output), input_bytes));

  RUNTIME_CHECK(musaMemcpy(d_input, h_input.data(), input_bytes,
                           musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           musaMemcpyHostToDevice));

  constexpr unsigned int threads = 256;
  rmsNormKernel<T><<<static_cast<unsigned int>(rows), threads,
                     threads * sizeof(float)>>>(
      d_input, d_weight, d_output, rows, hidden_dim, eps);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaDeviceSynchronize());
  RUNTIME_CHECK(musaMemcpy(h_output.data(), d_output, input_bytes,
                           musaMemcpyDeviceToHost));

  RUNTIME_CHECK(musaFree(d_output));
  RUNTIME_CHECK(musaFree(d_weight));
  RUNTIME_CHECK(musaFree(d_input));
}

template <typename T>
// This straightforward one-thread-per-output implementation supports causal
// masking and grouped-query attention, and uses a max-shifted softmax to avoid
// overflow in expf for long or high-magnitude sequences.
__global__ void flashAttentionKernel(
    const T* q, const T* k, const T* v, T* o, int batch_size,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    int head_dim, bool is_causal) {
  const size_t output_size = static_cast<size_t>(batch_size) * target_seq_len *
                             query_heads * head_dim;
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
  if (output_index >= output_size) {
    return;
  }

  const int dimension = output_index % head_dim;
  const size_t query_index = output_index / head_dim;
  const int query_head = query_index % query_heads;
  const int target_index = query_index / query_heads;
  const int target_position = target_index % target_seq_len;
  const int batch = target_index / target_seq_len;
  const int kv_head = query_head / (query_heads / kv_heads);
  const float score_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const size_t q_offset =
      (static_cast<size_t>(batch) * target_seq_len + target_position) *
          query_heads * head_dim +
      static_cast<size_t>(query_head) * head_dim;

  float max_score = -FLT_MAX;
  for (int source_position = 0; source_position < src_seq_len;
       ++source_position) {
    if (is_causal && source_position > target_position) {
      continue;
    }
    const size_t k_offset =
        (static_cast<size_t>(batch) * src_seq_len + source_position) *
            kv_heads * head_dim +
        static_cast<size_t>(kv_head) * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      score += to_float(q[q_offset + d]) * to_float(k[k_offset + d]);
    }
    max_score = fmaxf(max_score, score * score_scale);
  }

  float denominator = 0.0f;
  float result = 0.0f;
  for (int source_position = 0; source_position < src_seq_len;
       ++source_position) {
    if (is_causal && source_position > target_position) {
      continue;
    }
    const size_t k_offset =
        (static_cast<size_t>(batch) * src_seq_len + source_position) *
            kv_heads * head_dim +
        static_cast<size_t>(kv_head) * head_dim;
    const size_t v_offset = k_offset;
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      score += to_float(q[q_offset + d]) * to_float(k[k_offset + d]);
    }
    const float weight = expf(score * score_scale - max_score);
    denominator += weight;
    result += weight * to_float(v[v_offset + dimension]);
  }
  o[output_index] = from_float<T>(result / denominator);
}

// Compute scaled dot-product attention with optional causal masking. The
// layout follows the tester: Q is [B,T,H,D], while K/V are [B,S,Hkv,D].
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  const size_t q_size = static_cast<size_t>(batch_size) * target_seq_len *
                        query_heads * head_dim;
  const size_t kv_size = static_cast<size_t>(batch_size) * src_seq_len *
                         kv_heads * head_dim;
  const size_t q_bytes = q_size * sizeof(T);
  const size_t kv_bytes = kv_size * sizeof(T);
  h_o.resize(q_size);

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_q), q_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_k), kv_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_v), kv_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_o), q_bytes));
  RUNTIME_CHECK(musaMemcpy(d_q, h_q.data(), q_bytes, musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_k, h_k.data(), kv_bytes, musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_v, h_v.data(), kv_bytes, musaMemcpyHostToDevice));

  constexpr unsigned int threads = 256;
  const size_t blocks = (q_size + threads - 1) / threads;
  flashAttentionKernel<T><<<static_cast<unsigned int>(blocks), threads>>>(
      d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaDeviceSynchronize());
  RUNTIME_CHECK(musaMemcpy(h_o.data(), d_o, q_bytes, musaMemcpyDeviceToHost));

  RUNTIME_CHECK(musaFree(d_o));
  RUNTIME_CHECK(musaFree(d_v));
  RUNTIME_CHECK(musaFree(d_k));
  RUNTIME_CHECK(musaFree(d_q));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
