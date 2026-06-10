# ─────────────────────────────────────────────────────────────────────────────
# Makefile for best_altitude — autonomous altitude-optimization survey
#
# Build:
#   make             — CPU SIFT (default)
#
# Prerequisite — MAVLink C headers (download once):
#   make get-mavlink
# ─────────────────────────────────────────────────────────────────────────────

MAVLINK_DIR  ?= $(HOME)/mavlink/c_library_v2
PARENT_DIR   := ..

CXX    := g++
CXXSTD := -std=c++17
OPTFLG := -O2
WARNFL := -Wall -Wextra -Wno-unused-parameter

OPENCV_CFLAGS := -I/usr/include/opencv4
OPENCV_LIBS   := -L/usr/lib \
                 -lopencv_features2d -lopencv_flann -lopencv_calib3d \
                 -lopencv_imgcodecs  -lopencv_videoio \
                 -lopencv_imgproc    -lopencv_core

INCDIRS  := -I$(PARENT_DIR) -I$(MAVLINK_DIR)
CXXFLAGS := $(CXXSTD) $(OPTFLG) $(WARNFL) $(INCDIRS) $(OPENCV_CFLAGS)
LDFLAGS  := $(OPENCV_LIBS) -lpthread

TARGET := best_altitude
SRC    := best_altitude.cpp

.PHONY: all clean get-mavlink check-mavlink result

all: check-mavlink result $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo ""
	@echo "Built: ./$(TARGET)"
	@echo "Run : ./$(TARGET) [--port /dev/ttyTHS1] [--baud 57600] [--zoom 19]"

result:
	mkdir -p result

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

clean:
	rm -f $(TARGET)
