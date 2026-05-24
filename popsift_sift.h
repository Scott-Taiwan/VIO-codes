#pragma once
/*
 * popsift_sift.h — thin wrapper around PopSift for GPU-accelerated SIFT.
 *
 * Drop-in replacement for cv::SIFT::create()->detectAndCompute() in Phase 2.
 * Descriptors are scaled identically to OpenCV SIFT (multiplier=9 → ×512),
 * so they are cross-compatible with the pre-built FLANN index.
 *
 * Lifetime: call init_popsift() once at program start (after CUDA init),
 *           call shutdown_popsift() before exit.
 */

#include <opencv2/core.hpp>
#include <vector>

// Called once at startup — allocates GPU resources.
void init_popsift();

// Called at program exit — frees GPU resources.
void shutdown_popsift();

// GPU SIFT extraction.  gray_u8 must be CV_8UC1.
// max_kp=0 means no limit.  Output matches cv::SIFT detectAndCompute:
//   kps  — std::vector<cv::KeyPoint>
//   descs — (n × 128) CV_32F, values in [0 … ~500] (same scale as OpenCV SIFT)
void popsift_detect_compute(const cv::Mat& gray_u8,
                             std::vector<cv::KeyPoint>& kps,
                             cv::Mat& descs,
                             int max_kp = 0);
