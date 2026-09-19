NVCC ?= nvcc
HOSTCXX ?= $(shell command -v g++-10 2>/dev/null || command -v g++)
ARCH ?= 75
CXXFLAGS := -ccbin $(HOSTCXX) -std=c++17 -O3 -lineinfo --fmad=false -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
.PHONY: all test clean
all: build/quantize
build/quantize: src/main.cu $(wildcard include/*)
	mkdir -p build
	$(NVCC) $(CXXFLAGS) $< -o $@
test: all
	python3 tests/validate.py --quick
clean:
	rm -rf build
