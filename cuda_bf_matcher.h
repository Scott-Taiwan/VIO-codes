#pragma once
/*
 * cuda_bf_matcher.h — GPU-accelerated brute-force L2 matcher via cuBLAS.
 *
 * Replaces cv::BFMatcher(cv::NORM_L2).knnMatch(..., k=2) in Phase 2.
 * Requires: CUDA 12, cuBLAS.  Falls back to cv::BFMatcher if unavailable.
 */

#include <opencv2/features2d.hpp>
#include <vector>

/*
 * Performs knnMatch(k=2) on CPU descriptors using cuBLAS.
 * descs_q : (n_q × 128) CV_32F  — query descriptors
 * descs_t : (n_t × 128) CV_32F  — train descriptors
 * matches : output, same format as cv::BFMatcher::knnMatch(k=2)
 */
void cuda_knn2_match(const cv::Mat& descs_q,
                     const cv::Mat& descs_t,
                     std::vector<std::vector<cv::DMatch>>& matches);
