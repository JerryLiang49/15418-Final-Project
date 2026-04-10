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

SRC_DIR     := src
INCLUDE_DIR := include
BUILD_DIR   := build
TEST_DIR    := tests

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Main binary
TARGET := graphcolor

# Test binary
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_TARGET := test_runner

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRCS) $(filter-out $(BUILD_DIR)/main.o,$(OBJS)) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(LDFLAGS) -o $@ $^

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_TARGET)
