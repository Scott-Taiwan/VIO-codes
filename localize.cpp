/*
 * localize.cpp — GPS-free visual localization (C++ port of localize.py)
 *
 * Reads index/sift_index_z18.bin (produced by export_index.py).
 *
 * Build:
 *   mkdir build && cd build && cmake .. && make -j4
 *
 * Usage:
 *   ./localize photo.jpg
 *   ./localize photo.jpg --show
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "cuda_bf_matcher.h"
#include "popsift_sift.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>

namespace fs = std::filesystem;
#include <string>
#include <vector>

// ── config (mirrors config.py) ────────────────────────────────────────────────

static bool        g_gpu_sift    = false;   // set true if PopSIFT init succeeds
static const int   TILE_SIZE     = 256;
static const float MATCH_RATIO   = 0.75f;
static const int   MIN_INLIERS   = 15;
static const int   TOP_CANDIDATES = 5;
static const char* TILE_DIR      = "tiles";
static const char* INDEX_DIR     = "index";

// ── coordinate math (mirrors tile_utils.py) ───────────────────────────────────

static std::pair<double,double> pixel_to_latlon(
    double px, double py, int tile_x, int tile_y, int zoom)
{
    int    n      = 1 << zoom;
    double wx     = tile_x + px / TILE_SIZE;
    double wy     = tile_y + py / TILE_SIZE;
    double lon    = wx / n * 360.0 - 180.0;
    double lat    = std::atan(std::sinh(M_PI * (1.0 - 2.0 * wy / n))) * 180.0 / M_PI;
    return {lat, lon};
}

// ── index ─────────────────────────────────────────────────────────────────────

struct TileEntry {
    int    tile_x, tile_y, zoom;
    cv::Mat descs;          // rows×128, CV_32F
};

static std::vector<TileEntry> load_index(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open index: " + path);

    int32_t num_tiles = 0;
    f.read(reinterpret_cast<char*>(&num_tiles), sizeof(int32_t));

    std::vector<TileEntry> idx;
    idx.reserve(num_tiles);

    for (int i = 0; i < num_tiles; ++i) {
        TileEntry e;
        int32_t n_descs = 0;
        f.read(reinterpret_cast<char*>(&e.tile_x),  sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&e.tile_y),  sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&e.zoom),    sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&n_descs),   sizeof(int32_t));
        e.descs = cv::Mat(n_descs, 128, CV_32F);
        f.read(reinterpret_cast<char*>(e.descs.data),
               (std::streamsize)n_descs * 128 * sizeof(float));
        idx.push_back(std::move(e));
    }
    return idx;
}

// ── Phase 1: FLANN — build once, query per frame ─────────────────────────────

struct FlannIndex {
    cv::Ptr<cv::FlannBasedMatcher> matcher;
    std::vector<int>               offsets;   // tile boundary offsets into all_descs
    int                            total_kp;
};

// Called once at startup — expensive (KD-tree build over all tile descriptors).
static FlannIndex build_flann(const std::vector<TileEntry>& index)
{
    FlannIndex fi;
    fi.offsets.resize(index.size() + 1, 0);
    for (size_t i = 0; i < index.size(); ++i)
        fi.offsets[i + 1] = fi.offsets[i] + index[i].descs.rows;

    fi.total_kp = fi.offsets.back();
    cv::Mat all_descs(fi.total_kp, 128, CV_32F);
    for (size_t i = 0; i < index.size(); ++i)
        index[i].descs.copyTo(all_descs.rowRange(fi.offsets[i], fi.offsets[i+1]));

    std::cout << "Total train kp: " << fi.total_kp << "  — building FLANN …" << std::flush;
    fi.matcher = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(5),
        cv::makePtr<cv::flann::SearchParams>(50));
    fi.matcher->add(all_descs);
    fi.matcher->train();
    std::cout << " done.\n";
    return fi;
}

// Called per frame — fast (KD-tree already built).
static std::vector<std::pair<int,int>> flann_vote(
    const cv::Mat& descs_q,
    const FlannIndex& fi,
    size_t n_tiles)
{
    std::vector<std::vector<cv::DMatch>> raw;
    fi.matcher->knnMatch(descs_q, raw, 2);

    std::vector<int> votes(n_tiles, 0);
    int good_count = 0;
    for (auto& pair : raw) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < MATCH_RATIO * pair[1].distance) {
            ++good_count;
            int train_idx = pair[0].trainIdx;
            int tile_i = (int)(std::upper_bound(
                fi.offsets.begin(), fi.offsets.end(), train_idx)
                - fi.offsets.begin()) - 1;
            if (tile_i >= 0 && tile_i < (int)n_tiles)
                votes[tile_i]++;
        }
    }
    std::cout << "Good matches  : " << good_count << " (after ratio test)\n";

    std::vector<std::pair<int,int>> ranked;
    for (size_t i = 0; i < n_tiles; ++i)
        if (votes[i] > 0) ranked.push_back({(int)i, (int)votes[i]});
    std::sort(ranked.begin(), ranked.end(),
              [](auto& a, auto& b){ return a.second > b.second; });
    return ranked;
}

// ── SIFT wrapper: GPU (PopSIFT) with CPU (cv::SIFT) fallback ─────────────────

static void detect_compute(const cv::Mat& gray, std::vector<cv::KeyPoint>& kps,
                            cv::Mat& descs, int max_kp)
{
    if (g_gpu_sift) {
        popsift_detect_compute(gray, kps, descs, max_kp);
    } else {
        auto sift = cv::SIFT::create(max_kp > 0 ? max_kp : 2000);
        sift->detectAndCompute(gray, cv::noArray(), kps, descs);
    }
}

// ── Phase 2: stitched direct match ────────────────────────────────────────────

static cv::Mat stitch_tiles(int cx, int cy, int zoom, const char* tile_dir, int radius = 1)
{
    int size = 2 * radius + 1;
    cv::Mat canvas(size * TILE_SIZE, size * TILE_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int dx = 0; dx < size; ++dx) {
        for (int dy = 0; dy < size; ++dy) {
            int tx = cx - radius + dx;
            int ty = cy - radius + dy;
            std::string path = std::string(tile_dir) + "/" +
                               std::to_string(zoom) + "/" +
                               std::to_string(tx) + "/" +
                               std::to_string(ty) + ".png";
            cv::Mat tile = cv::imread(path);
            if (!tile.empty())
                tile.copyTo(canvas(cv::Rect(dx*TILE_SIZE, dy*TILE_SIZE, TILE_SIZE, TILE_SIZE)));
        }
    }
    return canvas;
}

struct MatchResult {
    double lat, lon;
    int    inliers;
    cv::Mat H, composite;
};

static std::optional<MatchResult> direct_match(
    const cv::Mat& query_gray,
    const std::vector<cv::KeyPoint>& kps_q,
    const cv::Mat& descs_q,
    int cx, int cy, int zoom,
    const char* tile_dir,
    int radius = 1, int min_inliers = 4)
{
    using clk = std::chrono::high_resolution_clock;
    auto t_sift0 = clk::now();

    cv::Mat composite = stitch_tiles(cx, cy, zoom, tile_dir, radius);
    cv::Mat comp_gray;
    cv::cvtColor(composite, comp_gray, cv::COLOR_BGR2GRAY);

    // Extract features — GPU (PopSIFT) if available, else CPU (cv::SIFT).
    // Both composite and query use the same extractor for descriptor compatibility.
    std::vector<cv::KeyPoint> kps_c, kps_q2;
    cv::Mat descs_c, descs_q2;
    detect_compute(comp_gray,  kps_c,  descs_c,  3000);
    detect_compute(query_gray, kps_q2, descs_q2, 2000);
    long sift_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now()-t_sift0).count();

    if (descs_c.empty()  || (int)kps_c.size()  < min_inliers)
        return std::nullopt;
    if (descs_q2.empty() || (int)kps_q2.size() < min_inliers)
        return std::nullopt;

    // Match descriptors — CUDA BFMatcher if GPU available, else OpenCV BFMatcher
    auto t_match0 = clk::now();
    std::vector<std::vector<cv::DMatch>> raw;
    if (g_gpu_sift) {
        cuda_knn2_match(descs_q2, descs_c, raw);
    } else {
        auto bf = cv::BFMatcher::create(cv::NORM_L2);
        bf->knnMatch(descs_q2, descs_c, raw, 2);
    }
    long match_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now()-t_match0).count();
    std::cout << "  [Phase2 SIFT=" << sift_ms << "ms  Match=" << match_ms << "ms";

    std::vector<cv::DMatch> good;
    for (auto& pair : raw)
        if (pair.size() >= 2 && pair[0].distance < MATCH_RATIO * pair[1].distance)
            good.push_back(pair[0]);

    if ((int)good.size() < min_inliers) {
        std::cout << "  ratio=" << good.size() << "/" << raw.size() << "]\n";
        return std::nullopt;
    }

    auto t_ransac0 = clk::now();
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (auto& m : good) {
        src_pts.push_back(kps_q2[m.queryIdx].pt);
        dst_pts.push_back(kps_c[m.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0, mask);
    long ransac_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now()-t_ransac0).count();
    std::cout << "  RANSAC=" << ransac_ms << "ms]\n";
    if (H.empty()) return std::nullopt;

    int inliers = cv::countNonZero(mask);
    if (inliers < min_inliers) {
        std::cout << "  [inliers=" << inliers << " < " << min_inliers << ", rejected]\n";
        return std::nullopt;
    }

    // Project query centre into composite pixel space
    float qw = (float)query_gray.cols, qh = (float)query_gray.rows;
    std::vector<cv::Point2f> ctr_in  = {{qw / 2, qh / 2}};
    std::vector<cv::Point2f> ctr_out;
    cv::perspectiveTransform(ctr_in, ctr_out, H);
    float cpx = ctr_out[0].x, cpy = ctr_out[0].y;

    int comp_size = (2 * radius + 1) * TILE_SIZE;
    int margin    = TILE_SIZE;
    if (cpx < -margin || cpx > comp_size + margin ||
        cpy < -margin || cpy > comp_size + margin)
        return std::nullopt;

    int   tile_dx = (int)(cpx / TILE_SIZE);
    int   tile_dy = (int)(cpy / TILE_SIZE);
    float local_px = cpx - tile_dx * TILE_SIZE;
    float local_py = cpy - tile_dy * TILE_SIZE;

    int x_orig = cx - radius;
    int y_orig = cy - radius;
    auto [lat, lon] = pixel_to_latlon(local_px, local_py,
                                      x_orig + tile_dx, y_orig + tile_dy, zoom);
    return MatchResult{lat, lon, inliers, H, composite};
}

// ── per-image localization ────────────────────────────────────────────────────

static void localize_one(const std::string& image_path, bool show,
                         const std::vector<TileEntry>& index,
                         const FlannIndex& fi,
                         int zoom_level)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = [&t0]() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
    };

    std::cout << "\n" << std::string(55, '-') << "\n";
    std::cout << "Image: " << image_path << "\n";

    cv::Mat query = cv::imread(image_path);
    if (query.empty()) {
        std::cerr << "  ERROR: cannot read image\n";
        return;
    }
    cv::Mat query_gray;
    cv::cvtColor(query, query_gray, cv::COLOR_BGR2GRAY);
    std::cout << "  Size    : " << query.cols << "×" << query.rows << " px\n";

    auto sift_q = cv::SIFT::create(2000);
    std::vector<cv::KeyPoint> kps_q;
    cv::Mat descs_q;
    sift_q->detectAndCompute(query_gray, cv::noArray(), kps_q, descs_q);
    std::cout << "  Features: " << kps_q.size() << " keypoints\n";

    // Phase 1 — coarse vote (FLANN already built)
    auto ranked = flann_vote(descs_q, fi, index.size());

    std::cout << "  Top candidates:\n";
    for (int i = 0; i < std::min((int)ranked.size(), TOP_CANDIDATES); ++i) {
        auto [ti, votes] = ranked[i];
        std::cout << "    tile (" << index[ti].tile_x << "," << index[ti].tile_y
                  << ")  votes=" << votes << "\n";
    }

    // Phase 2 — stitched direct match
    std::optional<MatchResult> result;
    int result_cx = -1, result_cy = -1;

    for (int i = 0; i < std::min((int)ranked.size(), TOP_CANDIDATES); ++i) {
        int ti = ranked[i].first;
        int cx = index[ti].tile_x;
        int cy = index[ti].tile_y;
        result = direct_match(query_gray, kps_q, descs_q,
                              cx, cy, zoom_level, TILE_DIR, 1, MIN_INLIERS);
        if (result) { result_cx = cx; result_cy = cy; break; }
    }

    long total_ms = elapsed_ms();

    if (!result) {
        std::cout << "  RESULT  : Could not determine location.\n";
        std::cout << "  Time    : " << std::fixed << std::setprecision(2)
                  << total_ms / 1000.0 << " s\n";
        return;
    }

    std::cout << std::fixed << std::setprecision(7);
    std::cout << "  GPS     : " << result->lat << ",  " << result->lon << "\n";
    std::cout << "  Inliers : " << result->inliers
              << "  |  tile (" << result_cx << "," << result_cy << ") z" << zoom_level << "\n";
    std::cout << "  Maps    : https://maps.google.com/?q="
              << result->lat << "," << result->lon << "\n";
    std::cout << std::setprecision(2) << "  Time    : " << total_ms / 1000.0 << " s\n";

    if (show) {
        float qw = (float)query_gray.cols, qh = (float)query_gray.rows;
        std::vector<cv::Point2f> corners_in = {{0,0},{qw,0},{qw,qh},{0,qh}};
        std::vector<cv::Point2f> corners_out;
        cv::perspectiveTransform(corners_in, corners_out, result->H);

        cv::Mat vis = result->composite.clone();
        std::vector<cv::Point> pts;
        for (auto& p : corners_out) pts.push_back({(int)p.x, (int)p.y});
        cv::polylines(vis, pts, true, {0, 255, 0}, 2);

        std::vector<cv::Point2f> ctr_in  = {{qw / 2, qh / 2}};
        std::vector<cv::Point2f> ctr_out;
        cv::perspectiveTransform(ctr_in, ctr_out, result->H);
        cv::drawMarker(vis, {(int)ctr_out[0].x, (int)ctr_out[0].y},
                       {0, 0, 255}, cv::MARKER_CROSS, 20, 2);

        // Save per-image result files
        fs::path p(image_path);
        std::string stem = p.stem().string();
        std::string tile_out  = "result_tile_"  + stem + ".png";
        std::string query_out = "result_query_" + stem + ".png";
        cv::imwrite(tile_out,  vis);
        cv::imwrite(query_out, query);
        std::cout << "  Saved   : " << tile_out << "  " << query_out << "\n";
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " image1 [image2 ...] [--show] [--zoom N]\n";
        return 1;
    }

    // Split args into image paths and flags
    std::vector<std::string> images;
    bool show = false;
    int zoom_level = 18;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--show") show = true;
        else if (std::string(argv[i]) == "--zoom" && i + 1 < argc)
            zoom_level = std::stoi(argv[++i]);
        else images.push_back(argv[i]);
    }
    if (images.empty()) {
        std::cerr << "No image files specified.\n";
        return 1;
    }

    // ── one-time startup (done before takeoff) ───────────────────────────────
    std::string index_path = std::string(INDEX_DIR) + "/sift_index_z" +
                             std::to_string(zoom_level) + ".bin";
    std::cout << "Loading index from " << index_path << " …\n";

    auto t_startup = std::chrono::high_resolution_clock::now();

    // Check available RAM before attempting GPU init.
    // PopSIFT calls abort() on CUDA OOM (not catchable), so we pre-screen.
    auto avail_mb = []() -> long {
        std::ifstream f("/proc/meminfo");
        std::string key; long val; std::string unit;
        while (f >> key >> val >> unit) {
            if (key == "MemAvailable:") return val / 1024;
        }
        return 0;
    };

    long mem_mb = avail_mb();
    std::cout << "Initialising GPU (PopSift + cuBLAS) … (avail=" << mem_mb << " MB) " << std::flush;
    if (mem_mb >= 512) {
        try {
            init_popsift();
            { cv::Mat tiny(4,128,CV_32F,cv::Scalar(0));
              std::vector<std::vector<cv::DMatch>> dummy;
              cuda_knn2_match(tiny, tiny, dummy); }
            g_gpu_sift = true;
            std::cout << "done (GPU).\n";
        } catch (const std::exception& e) {
            std::cout << "FAILED (" << e.what() << ")\n";
            std::cout << "Falling back to CPU SIFT.\n";
            g_gpu_sift = false;
        }
    } else {
        std::cout << "skipped — low RAM, using CPU SIFT.\n";
        g_gpu_sift = false;
    }

    auto index = load_index(index_path);
    long load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_startup).count();
    std::cout << "Tile index    : " << index.size()
              << " tiles  (loaded in " << load_ms << " ms)\n";

    auto fi = build_flann(index);
    long startup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_startup).count();
    std::cout << "Startup total : " << std::fixed << std::setprecision(2)
              << startup_ms / 1000.0 << " s  (index load + FLANN build)\n";

    // ── per-frame loop (each call is one drone camera frame) ─────────────────
    auto t_all = std::chrono::high_resolution_clock::now();
    for (auto& img_path : images)
        localize_one(img_path, show, index, fi, zoom_level);

    long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_all).count();
    std::cout << "\n" << std::string(55, '=') << "\n";
    std::cout << "  " << images.size() << " images processed in "
              << std::fixed << std::setprecision(2) << total_ms / 1000.0 << " s"
              << "  (index load excluded)\n";
    std::cout << std::string(55, '=') << "\n";
    shutdown_popsift();
    return 0;
}
