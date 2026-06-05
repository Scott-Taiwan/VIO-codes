# ─────────────────────────────────────────────────────────────────────────────
# Makefile for drone_localize (shot_estimate/)
#
# Build variants:
#   make             — CPU SIFT + BFMatcher  (no CUDA required)
#   make USE_GPU=1   — PopSIFT + cuBLAS      (requires CUDA + PopSIFT)
#
# Prerequisite — MAVLink C headers (header-only, download once):
#   make get-mavlink
# ─────────────────────────────────────────────────────────────────────────────

# ── paths — adjust if your layout differs ─────────────────────────────────────
MAVLINK_DIR   ?= $(HOME)/mavlink/c_library_v2
POPSIFT_DIR   ?= $(HOME)/claude-project/popsift/install

PARENT_DIR    := ..
PHOTO_OUTDIR  := photo_obtained

# ── compiler ──────────────────────────────────────────────────────────────────
CXX    := g++
NVCC   := nvcc

CXXSTD := -std=c++17
OPTFLG := -O2
WARNFL := -Wall -Wextra -Wno-unused-parameter

# pkg-config on this Jetson reports wrong paths; use known-good system paths
OPENCV_CFLAGS := -I/usr/include/opencv4
OPENCV_LIBS   := -L/usr/lib \
                 -lopencv_features2d -lopencv_flann -lopencv_calib3d \
                 -lopencv_imgcodecs  -lopencv_videoio \
                 -lopencv_imgproc    -lopencv_core

# ── source files ──────────────────────────────────────────────────────────────
TARGET  := drone_localize
SRC     := drone_localize.cpp

# ── CPU-only build (default) ──────────────────────────────────────────────────

INCDIRS_CPU := -I$(PARENT_DIR) -I$(MAVLINK_DIR)
CXXFLAGS_CPU := $(CXXSTD) $(OPTFLG) $(WARNFL) $(INCDIRS_CPU) $(OPENCV_CFLAGS)
LDFLAGS_CPU  := $(OPENCV_LIBS) -lpthread

# ── GPU build (USE_GPU=1) ─────────────────────────────────────────────────────

CUDA_ARCH     := sm_87        # Jetson Orin Nano (Ampere)
CUDA_INC      := $(shell find /usr/local/cuda/include -maxdepth 0 2>/dev/null)

INCDIRS_GPU   := -I$(PARENT_DIR) -I$(MAVLINK_DIR) \
                 -I$(POPSIFT_DIR)/include \
                 $(if $(CUDA_INC),-I$(CUDA_INC))

CXXFLAGS_GPU  := $(CXXSTD) $(OPTFLG) $(WARNFL) \
                 -DUSE_GPU \
                 $(INCDIRS_GPU) \
                 $(OPENCV_CFLAGS)

NVCC_FLAGS    := -std=c++17 -O2 -arch=$(CUDA_ARCH) \
                 -DUSE_GPU \
                 $(INCDIRS_GPU) \
                 $(OPENCV_CFLAGS) \
                 --compiler-options "$(WARNFL)"

LDFLAGS_GPU   := $(OPENCV_LIBS) -lpthread \
                 -L$(POPSIFT_DIR)/lib \
                 -Wl,-rpath,$(POPSIFT_DIR)/lib \
                 -lPopSift \
                 -L/usr/local/cuda/lib64 \
                 -lcudart -lcublas

# ── rules ─────────────────────────────────────────────────────────────────────

.PHONY: all gpu clean get-mavlink check-mavlink $(PHOTO_OUTDIR)

# Default: CPU build
all: check-mavlink $(PHOTO_OUTDIR) $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS_CPU) -o $@ $< $(LDFLAGS_CPU)
	@echo ""
	@echo "Built (CPU): ./$(TARGET)"
	@echo "Run  : ./$(TARGET) [--port /dev/ttyTHS1] [--baud 57600] [--zoom 18]"

# GPU build: compiles CUDA objects then links
gpu: check-mavlink $(PHOTO_OUTDIR)
	$(NVCC) $(NVCC_FLAGS) -c -o cuda_bf_matcher.o $(PARENT_DIR)/cuda_bf_matcher.cu
	$(NVCC) $(NVCC_FLAGS) -c -o popsift_sift.o    $(PARENT_DIR)/popsift_sift.cpp
	$(CXX)  $(CXXFLAGS_GPU) -c -o drone_localize.o $(SRC)
	$(CXX)  -o $(TARGET) drone_localize.o cuda_bf_matcher.o popsift_sift.o \
	        $(LDFLAGS_GPU)
	@echo ""
	@echo "Built (GPU): ./$(TARGET)"
	@echo "Run  : ./$(TARGET) [--port /dev/ttyTHS1] [--baud 57600] [--zoom 18]"

# Create photo output directory
$(PHOTO_OUTDIR):
	mkdir -p $(PHOTO_OUTDIR)

# Download MAVLink C headers (one-time)
get-mavlink:
	@if [ -d "$(MAVLINK_DIR)" ]; then \
	    echo "MAVLink headers already at $(MAVLINK_DIR)"; \
	else \
	    echo "Cloning MAVLink C library to $(MAVLINK_DIR) …"; \
	    git clone --depth 1 https://github.com/mavlink/c_library_v2.git $(MAVLINK_DIR); \
	    echo "Done."; \
	fi

check-mavlink:
	@if [ ! -d "$(MAVLINK_DIR)/common" ]; then \
	    echo ""; \
	    echo "ERROR: MAVLink headers not found at: $(MAVLINK_DIR)"; \
	    echo "Run:  make get-mavlink"; \
	    echo ""; \
	    exit 1; \
	fi

# Simulation build (no MAVLink, no camera hardware needed)
sim: $(PHOTO_OUTDIR)
	$(CXX) $(CXXSTD) $(OPTFLG) $(WARNFL) \
	    -I$(PARENT_DIR) $(OPENCV_CFLAGS) \
	    -o drone_sim drone_sim.cpp \
	    $(OPENCV_LIBS) -lpthread
	@echo ""
	@echo "Built: ./drone_sim"
	@echo "Run : ./drone_sim"

clean:
	rm -f $(TARGET) drone_sim *.o

# ── quick help ────────────────────────────────────────────────────────────────
help:
	@echo "Targets:"
	@echo "  make              — CPU build (default)"
	@echo "  make gpu          — GPU build (PopSIFT + cuBLAS)"
	@echo "  make get-mavlink  — download MAVLink headers (run once)"
	@echo "  make clean        — remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  MAVLINK_DIR=$(MAVLINK_DIR)"
	@echo "  POPSIFT_DIR=$(POPSIFT_DIR)"
