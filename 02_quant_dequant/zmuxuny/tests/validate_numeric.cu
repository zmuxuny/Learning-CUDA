// Regression for lane-dependent hashing under optimized device compilation.
// Exercise native-wave boundaries, multiple CTAs and both halves of size_t.
#include "../include/numeric.cuh"
#include <cstdio>
#include <vector>

constexpr int COUNT = 4096;
__global__ void hashes(float *out, uint64_t base, uint32_t seed) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < COUNT)
    out[i] = lp::uniform(base + i, seed);
}

int main() {
  float *device = nullptr;
  auto check = [](cudaError_t e) {
    if (e == cudaSuccess)
      return true;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(e));
    return false;
  };
  if (!check(cudaMalloc(&device, COUNT * sizeof(float))))
    return 1;
  std::vector<float> result(COUNT);
  size_t checked = 0;
  for (uint64_t base : {0ULL, (1ULL << 32) - 65, 1ULL << 40}) {
    for (uint32_t seed : {0U, 7U, 42U, 0xffffffffU}) {
      hashes<<<COUNT / 256, 256>>>(device, base, seed);
      if (!check(cudaGetLastError()) ||
          !check(cudaMemcpy(result.data(), device, COUNT * sizeof(float),
                            cudaMemcpyDeviceToHost)))
        return 1;
      for (unsigned i = 0; i < COUNT; ++i) {
        float expected = lp::uniform(base + i, seed);
        if (result[i] != expected) {
          std::fprintf(
              stderr,
              "hash mismatch: base=%llu index=%u seed=%u expected=%.9g actual=%.9g\n",
              static_cast<unsigned long long>(base), i, seed, expected, result[i]);
          return 1;
        }
        ++checked;
      }
    }
  }
  if (!check(cudaFree(device)))
    return 1;
  std::printf("{\"status\":\"PASS\",\"exact_hash_checks\":%zu}\n", checked);
}
