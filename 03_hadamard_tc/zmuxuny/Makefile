NVCC ?= nvcc
HOSTCXX ?= $(shell command -v g++-10 2>/dev/null || command -v g++)
ARCH ?= 75
CXXFLAGS := -DLP_CUDA_ARCH=$(ARCH) -ccbin $(HOSTCXX) -std=c++17 -O3 -lineinfo --fmad=false -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
.PHONY: all test clean native
all: build/hadamard
build/hadamard: src/main.cu src/hadamard.cuh src/tensor_core.cuh $(wildcard include/*)
	mkdir -p build
	$(NVCC) $(CXXFLAGS) $< -o $@
test: all
	python3 tests/validate.py
clean:
	rm -rf build

# Explicit bonus target; the default binary always uses software FP8 encoding.
native: build/hadamard_native
build/hadamard_native: src/main.cu src/hadamard.cuh src/tensor_core.cuh $(wildcard include/*)
	mkdir -p build
	$(NVCC) $(CXXFLAGS) -DLP_NATIVE_FP8=1 $< -o $@
