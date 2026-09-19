NVCC ?= nvcc
HOSTCXX ?= $(shell command -v g++-10 2>/dev/null || command -v g++)
ARCH ?= 75
CXXFLAGS := -ccbin $(HOSTCXX) -std=c++17 -O3 -lineinfo --fmad=false -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
.PHONY: all test clean native
all: build/quantize
build/quantize: src/main.cu $(wildcard include/*)
	mkdir -p build
	$(NVCC) $(CXXFLAGS) $< -o $@
test: all
	python3 tests/validate.py --quick
clean:
	rm -rf build

# Explicit bonus target; the default binary always uses software FP8 encoding.
native: build/quantize_native
build/quantize_native: src/main.cu $(wildcard include/*)
	mkdir -p build
	$(NVCC) $(CXXFLAGS) -DLP_NATIVE_FP8=1 $< -o $@
