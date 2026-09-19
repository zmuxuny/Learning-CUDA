PLATFORM ?= nvidia
MACA_PATH ?= /opt/maca
NVCC ?= nvcc
HOSTCXX ?= $(shell command -v g++-10 2>/dev/null || command -v g++)
ARCH ?= 75
CXXFLAGS := -ccbin $(HOSTCXX) -std=c++17 -O3 -lineinfo --fmad=false -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
ifeq ($(PLATFORM),metax)
export MACA_PATH
export CUDA_PATH := $(MACA_PATH)/tools/cu-bridge
NVCC := $(MACA_PATH)/tools/cu-bridge/bin/cucc
CXXFLAGS := -std=c++17 -O3 --fmad=false -I$(MACA_PATH)/tools/cu-bridge/include
BUILD_DIR := build/metax
RUN_ENV := LD_LIBRARY_PATH=$(MACA_PATH)/lib:$${LD_LIBRARY_PATH:-}
TEST_OUTPUT := results/c500/correctness_quick.json
else ifeq ($(PLATFORM),nvidia)
BUILD_DIR := build
TEST_OUTPUT := results/correctness_quick.json
else
$(error Unsupported PLATFORM '$(PLATFORM)' (expected nvidia or metax))
endif

.PHONY: all test clean native
all: $(BUILD_DIR)/quantize
$(BUILD_DIR)/quantize: src/main.cu $(wildcard include/*) Makefile
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $< -o $@
test: all
	mkdir -p $(dir $(TEST_OUTPUT))
	$(RUN_ENV) python3 tests/validate.py --binary $(BUILD_DIR)/quantize --output $(TEST_OUTPUT) --quick
clean:
	rm -rf $(BUILD_DIR)

# Explicit bonus target; the default binary always uses software FP8 encoding.
ifeq ($(PLATFORM),metax)
native:
	$(error The native target uses NVIDIA FP8 instructions; run make PLATFORM=metax instead)
else
native: $(BUILD_DIR)/quantize_native
$(BUILD_DIR)/quantize_native: src/main.cu $(wildcard include/*) Makefile
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) -DLP_NATIVE_FP8=1 $< -o $@
endif
