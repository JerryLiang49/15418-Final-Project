CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra
LDFLAGS  :=

# Detect OpenMP support: use -fopenmp on Linux (GHC/PSC), Xcode-openmp on macOS with libomp
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  # Check if libomp is actually installed (header exists) via Homebrew
  LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
  LIBOMP_HEADER := $(wildcard $(LIBOMP_PREFIX)/include/omp.h)
  ifneq ($(LIBOMP_HEADER),)
    CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    LDFLAGS  += -L$(LIBOMP_PREFIX)/lib -lomp
  else
    $(info [WARNING] libomp not found. Building WITHOUT OpenMP. Install with: brew install libomp)
    CXXFLAGS += -Wno-unknown-pragmas
  endif
else
  # Linux (GHC / PSC machines)
  CXXFLAGS += -fopenmp
  LDFLAGS  += -fopenmp
endif

# Detect CUDA support
# GHC machines: nvcc at /usr/local/cuda-11.7/bin/nvcc, RTX 2080 B (sm_75)
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
  # Try GHC default CUDA path
  NVCC := $(wildcard /usr/local/cuda-11.7/bin/nvcc)
endif
ifneq ($(NVCC),)
  GPU_ARCH   ?= sm_75
  # CUDA 11.7 requires GCC <= 11 as host compiler.
  # GHC has g++-11 at /usr/bin/g++-11. Use -ccbin to select it.
  CUDA_HOST_CXX := $(wildcard /usr/bin/g++-11)
  ifneq ($(CUDA_HOST_CXX),)
    NVCC_FLAGS := -O3 -arch=$(GPU_ARCH) -DCUDA_ENABLED -ccbin $(CUDA_HOST_CXX)
  else
    NVCC_FLAGS := -O3 -arch=$(GPU_ARCH) -DCUDA_ENABLED --allow-unsupported-compiler
  endif
  CXXFLAGS   += -DCUDA_ENABLED
  CUDA_DIR   := $(shell dirname $(NVCC))/..
  LDFLAGS    += -L$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib -lcudart -Wl,-rpath,$(CUDA_DIR)/lib64
  $(info [INFO] CUDA detected: $(NVCC), targeting $(GPU_ARCH))
else
  $(info [INFO] CUDA not found. Building without GPU support.)
endif

SRC_DIR     := src
INCLUDE_DIR := include
BUILD_DIR   := build
TEST_DIR    := tests

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# CUDA sources (empty if nvcc not found)
ifneq ($(NVCC),)
  CUDA_SRCS := $(wildcard $(SRC_DIR)/*.cu)
  CUDA_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SRCS))
else
  CUDA_OBJS :=
endif

ALL_OBJS := $(OBJS) $(CUDA_OBJS)

# Main binary
TARGET := graphcolor

# Test binary
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_TARGET := test_runner

.PHONY: all clean test

all: $(TARGET)

ifneq ($(NVCC),)
# When CUDA is enabled, link with nvcc so it can find cudart automatically.
# Pass g++ flags via -Xlinker and -Xcompiler.
$(TARGET): $(ALL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp -o $@ $^
else
$(TARGET): $(ALL_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^
endif

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

# CUDA compile rule
ifneq ($(NVCC),)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<
endif

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

ifneq ($(NVCC),)
$(TEST_TARGET): $(TEST_SRCS) $(filter-out $(BUILD_DIR)/main.o,$(ALL_OBJS)) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -I$(INCLUDE_DIR) -o $@ $^
else
$(TEST_TARGET): $(TEST_SRCS) $(filter-out $(BUILD_DIR)/main.o,$(ALL_OBJS)) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(LDFLAGS) -o $@ $^
endif

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_TARGET)
