/*
 * drone_localize.cpp — GPS-free visual localization during drone flight
 *
 * Connects to Pixhawk via MAVLink serial (default /dev/ttyTHS1).
 * Monitors relative altitude; when the drone takes off and reaches
 * TAKEOFF_ALT (30 m), captures a CSI camera frame every interval
 * seconds, runs two-phase SIFT localization against the tile index, and
 * saves each photo as:
 *   photo_obtained/{lat_real}_{lon_real}-{lat_est}_{lon_est}.png
 *
 * Build:
 *   cd shot_estimate
 *   make          # CPU SIFT
 *   make USE_GPU=1  # PopSIFT + cuBLAS (faster)
 *
 * Run:
 *   ./drone_localize
 *   ./drone_localize --port /dev/ttyUSB0 --baud 115200 --zoom 18
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// MAVLink C library (header-only, cloned via `make get-mavlink`)
// Include path: -I$(MAVLINK_DIR) where MAVLINK_DIR = .../c_library_v2
#include <common/mavlink.h>

// GPU SIFT + CUDA BF-matcher (parent dir, optional — compiled with USE_GPU=1)
#ifdef USE_GPU
#  include "../popsift_sift.h"
#  include "../cuda_bf_matcher.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

// ── config ────────────────────────────────────────────────────────────────────

static const char*  DEFAULT_PORT      = "/dev/ttyTHS1";
static const int    DEFAULT_BAUD      = 57600;
static const int    DEFAULT_ZOOM      = 18;
static const char*  INDEX_DIR         = "../index";
static const char*  TILE_DIR          = "../tiles";
static const char*  PHOTO_DIR         = "photo_obtained";

static const double TAKEOFF_ALT       = 30.0;   // m  — start capturing above this
static const double MIN_CAPTURE_ALT   = 10.0;   // m  — pause below this (landing)
// interval is now a CLI parameter (--interval), default 5 s

static const float  MAX_TILT_DEG      = 15.0f;  // skip capture if roll or pitch exceeds this
                                                 // (pitch/roll up to 15° is corrected by
                                                 //  build_correction_H before localization)

// ── drift-prevention parameters ───────────────────────────────────────────────
static const int    MIN_INLIERS_SEND      = 10;    // below this RANSAC geometry is useless
static const double MIN_WAYPOINT_SPACING  = 300.0; // m — warn if waypoints closer (#4)

static const int    TILE_SIZE         = 256;
static const float  MATCH_RATIO       = 0.75f;
static const int    MIN_INLIERS       = 6;
static const int    TOP_CANDIDATES    = 20;

// CSI camera
static const int    CSI_SENSOR_ID     = 0;
static const int    CSI_WIDTH         = 1280;
static const int    CSI_HEIGHT        = 720;
static const int    CSI_FPS           = 30;
static const int    CSI_FLIP          = 0;    // 0 = none, 2 = 180°

static bool g_gpu_sift = false;
static std::atomic<bool> g_shutdown{false};

// ── dual-output logger (stdout + timestamped log file) ────────────────────────

struct Tee {
    std::ofstream file;

    void open(const std::string& path) {
        file.open(path, std::ios::out | std::ios::trunc);
    }
    void close() { if (file.is_open()) { file.flush(); file.close(); } }
    bool is_open() const { return file.is_open(); }

    template<typename T>
    Tee& operator<<(const T& v) {
        std::cout << v;
        if (file.is_open()) { file << v; file.flush(); }
        return *this;
    }
    // std::endl, std::flush, etc.
    Tee& operator<<(std::ostream& (*f)(std::ostream&)) {
        f(std::cout);
        if (file.is_open()) { f(file); file.flush(); }
        return *this;
    }
    // std::fixed, std::left, etc.
    Tee& operator<<(std::ios_base& (*f)(std::ios_base&)) {
        f(std::cout);
        if (file.is_open()) f(file);
        return *this;
    }
};
static Tee g_log;

// Wall-clock timestamp string for log lines
static std::string ts()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf {}; localtime_r(&t, &tm_buf);
    char buf[16]; strftime(buf, sizeof(buf), "[%H:%M:%S] ", &tm_buf);
    return buf;
}

// yyyymmdd_HHmmss string (used for log filename)
static std::string timestamp_tag()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf {}; localtime_r(&t, &tm_buf);
    char buf[20]; strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return buf;
}

// ── signal handler ────────────────────────────────────────────────────────────

static void handle_signal(int) { g_shutdown = true; }

// ── coordinate math ───────────────────────────────────────────────────────────

static std::pair<double,double> pixel_to_latlon(
    double px, double py, int tile_x, int tile_y, int zoom)
{
    int n    = 1 << zoom;
    double wx = tile_x + px / TILE_SIZE;
    double wy = tile_y + py / TILE_SIZE;
    double lon = wx / n * 360.0 - 180.0;
    double lat = std::atan(std::sinh(M_PI * (1.0 - 2.0 * wy / n))) * 180.0 / M_PI;
    return {lat, lon};
}

static double haversine_m(double lat1, double lon1, double lat2, double lon2)
{
    constexpr double R = 6'371'000.0;
    double phi1 = lat1 * M_PI / 180.0, phi2 = lat2 * M_PI / 180.0;
    double dphi = (lat2 - lat1) * M_PI / 180.0;
    double dlam = (lon2 - lon1) * M_PI / 180.0;
    double a = std::sin(dphi/2)*std::sin(dphi/2)
             + std::cos(phi1)*std::cos(phi2)*std::sin(dlam/2)*std::sin(dlam/2);
    return 2.0 * R * std::asin(std::sqrt(a));
}

// ── drift-prevention helpers ──────────────────────────────────────────────────

// Map RANSAC inlier count → horizontal accuracy (metres) for GPS_INPUT.
// Pixhawk EKF blends this fix with IMU proportional to hacc —
// high hacc = IMU does most of the work; low hacc = fix is trusted strongly.
static float hacc_from_inliers(int inliers)
{
    if (inliers > 80) return  10.0f;  // strong match  — trust fix
    if (inliers >= 40) return 30.0f;  // good match
    if (inliers >= 20) return 60.0f;  // weak match    — IMU does most work
    return                  100.0f;   // 10–19 inliers — IMU almost entirely
}

// Send GPS_INPUT MAVLink message to Pixhawk so it can use SIFT position.
// ignore_flags = 191: ignore alt(1)|hdop(2)|vdop(4)|vel_h(8)|vel_v(16)|
//                          speed_acc(32)|vert_acc(128)
// We DO send horiz_accuracy (bit 64 NOT set in ignore mask).
// gps_id = 1  →  GPS2.  GPS1 remains the real GNSS module.
// ArduPilot GPS_AUTO_SWITCH=4 picks whichever has lower hacc:
//   good SIFT fix (hacc 10–30 m) vs blocked GNSS (hacc 100–500 m) → SIFT wins.
//   clear sky GNSS (hacc 1–3 m)  vs SIFT              → real GPS wins.
static void send_gps_input(int fd, double lat, double lon, float hacc)
{
    mavlink_message_t msg;
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];

    mavlink_msg_gps_input_pack(
        1, 200, &msg,
        0,                          // time_usec (0 = autopilot uses own clock)
        1,                          // gps_id = 1  →  GPS2 (GPS1 = real GNSS)
        191,                        // ignore_flags (see above)
        0, 0,                       // time_week_ms, time_week (ignored)
        3,                          // fix_type: 3D fix
        (int32_t)(lat * 1e7),       // lat degE7
        (int32_t)(lon * 1e7),       // lon degE7
        0.0f,                       // alt (ignored)
        1.0f, 1.0f,                 // hdop, vdop (ignored)
        0.0f, 0.0f, 0.0f,          // vn, ve, vd (ignored)
        0.0f,                       // speed_accuracy (ignored)
        hacc,                       // horiz_accuracy — key trust field
        5.0f,                       // vert_accuracy (ignored)
        10,                         // satellites_visible (plausible value)
        0                           // yaw: 0 = unknown
    );

    uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
    ::write(fd, buf, len);
}

// ── #4: waypoint spacing validation ──────────────────────────────────────────

struct Waypoint { double lat, lon; };

static std::vector<Waypoint> parse_waypoints(const std::string& s)
{
    // Format: "lat1,lon1 lat2,lon2 lat3,lon3"
    std::vector<Waypoint> wps;
    std::istringstream iss(s);
    std::string token;
    while (iss >> token) {
        auto comma = token.find(',');
        if (comma == std::string::npos)
            throw std::runtime_error(
                "Waypoint format error: expected lat,lon  e.g. 25.061,121.471");
        wps.push_back({std::stod(token.substr(0, comma)),
                       std::stod(token.substr(comma + 1))});
    }
    return wps;
}

static void validate_waypoints(const std::vector<Waypoint>& wps)
{
    if (wps.empty()) return;
    std::cout << "\n[#4] Waypoint spacing check (" << wps.size() << " waypoints):\n";
    bool all_ok = true;
    for (size_t i = 1; i < wps.size(); ++i) {
        double d = haversine_m(wps[i-1].lat, wps[i-1].lon, wps[i].lat, wps[i].lon);
        bool ok  = d >= MIN_WAYPOINT_SPACING;
        if (!ok) all_ok = false;
        std::cout << "  WP" << i << " → WP" << (i+1) << " : "
                  << std::fixed << std::setprecision(1) << d << " m  "
                  << (ok ? "OK" : "WARNING — too close, bearing error risk") << "\n";
    }
    if (!all_ok)
        std::cout << "  *** Keep waypoints >= " << MIN_WAYPOINT_SPACING
                  << " m apart to avoid large bearing errors near target. ***\n";
    std::cout << "  Final target (WP" << wps.size() << "): "
              << std::setprecision(7) << wps.back().lat
              << ", " << wps.back().lon << "\n\n";
}


// ── mission fetch from Pixhawk ────────────────────────────────────────────────

// Ask ArduPilot to start sending GLOBAL_POSITION_INT at 4 Hz on this port.
// Without this, stream rates on TELEM3 default to 0 and the GPS-fix wait loops forever.
static void request_streams(int fd)
{
    mavlink_message_t msg;
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    mavlink_msg_request_data_stream_pack(
        1, 200, &msg,
        1, 1,                      // target sysid / compid
        MAV_DATA_STREAM_POSITION,  // GLOBAL_POSITION_INT
        4,                         // 4 Hz
        1);                        // start
    uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
    ::write(fd, buf, len);
    std::cout << "[stream] Requested POSITION stream at 4 Hz.\n";

    // Also request ATTITUDE stream so we can guard against oblique photos
    mavlink_msg_request_data_stream_pack(
        1, 200, &msg,
        1, 1,
        MAV_DATA_STREAM_EXTRA1,    // ATTITUDE
        10,                        // 10 Hz — attitude changes fast
        1);
    len = mavlink_msg_to_send_buffer(buf, &msg);
    ::write(fd, buf, len);
    std::cout << "[stream] Requested ATTITUDE stream at 10 Hz.\n";
}

// Block until one HEARTBEAT is received from Pixhawk.
static void wait_for_heartbeat(int fd)
{
    std::cout << "Waiting for Pixhawk heartbeat …" << std::flush;
    mavlink_message_t msg;
    mavlink_status_t  status;
    uint8_t byte;
    while (!g_shutdown) {
        ssize_t n = ::read(fd, &byte, 1);
        if (n > 0 && mavlink_parse_char(MAVLINK_COMM_0, byte, &msg, &status))
            if (msg.msgid == MAVLINK_MSG_ID_HEARTBEAT) { std::cout << " ok.\n"; return; }
    }
}

// Download the active mission from Pixhawk using the MAVLink mission protocol.
// Returns MAV_CMD_NAV_WAYPOINT items only; item 0 (home) is always skipped.
// Returns an empty vector on timeout or if no mission is loaded.
static std::vector<Waypoint> fetch_mission_from_pixhawk(int fd)
{
    auto send_raw = [&](mavlink_message_t& m) {
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];
        uint16_t len = mavlink_msg_to_send_buffer(buf, &m);
        ::write(fd, buf, len);
    };

    mavlink_message_t msg, req;
    mavlink_status_t  status;
    uint8_t byte;

    // Step 1 — request mission list
    mavlink_msg_mission_request_list_pack(1, 200, &req, 1, 1,
                                          MAV_MISSION_TYPE_MISSION);
    send_raw(req);
    std::cout << "[mission] Requesting mission …" << std::flush;

    // Step 2 — wait for MISSION_COUNT
    int count = 0;
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        bool got = false;
        while (!g_shutdown && std::chrono::steady_clock::now() < deadline) {
            ssize_t n = ::read(fd, &byte, 1);
            if (n > 0 && mavlink_parse_char(MAVLINK_COMM_0, byte, &msg, &status)) {
                if (msg.msgid == MAVLINK_MSG_ID_MISSION_COUNT) {
                    mavlink_mission_count_t mc;
                    mavlink_msg_mission_count_decode(&msg, &mc);
                    count = mc.count;
                    got = true;
                    break;
                }
            }
        }
        if (!got) { std::cout << " timeout (no mission loaded).\n"; return {}; }
    }
    std::cout << " " << count << " item(s).\n";
    if (count == 0) return {};

    // Step 3 — request each item individually
    std::vector<Waypoint> wps;
    for (int i = 0; i < count && !g_shutdown; ++i) {
        mavlink_msg_mission_request_int_pack(1, 200, &req, 1, 1,
                                             (uint16_t)i, MAV_MISSION_TYPE_MISSION);
        send_raw(req);

        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        bool got = false;
        while (!g_shutdown && std::chrono::steady_clock::now() < deadline) {
            ssize_t n = ::read(fd, &byte, 1);
            if (n > 0 && mavlink_parse_char(MAVLINK_COMM_0, byte, &msg, &status)) {
                if (msg.msgid == MAVLINK_MSG_ID_MISSION_ITEM_INT) {
                    mavlink_mission_item_int_t item;
                    mavlink_msg_mission_item_int_decode(&msg, &item);
                    if (item.seq == (uint16_t)i) {
                        // item 0 is the home position — skip it
                        if (i > 0 && item.command == MAV_CMD_NAV_WAYPOINT
                                   && item.x != 0 && item.y != 0) {
                            double lat = item.x / 1e7;
                            double lon = item.y / 1e7;
                            std::cout << "  WP" << i << ": ("
                                      << std::fixed << std::setprecision(7)
                                      << lat << ", " << lon << ")\n";
                            wps.push_back({lat, lon});
                        } else {
                            std::cout << "  item" << i << ": cmd=" << item.command
                                      << " (skipped)\n";
                        }
                        got = true;
                        break;
                    }
                }
            }
        }
        if (!got) {
            std::cerr << "  Timeout waiting for item " << i << " — aborting.\n";
            return {};
        }
    }

    // Step 4 — acknowledge receipt
    mavlink_msg_mission_ack_pack(1, 200, &req, 1, 1,
                                  MAV_MISSION_ACCEPTED, MAV_MISSION_TYPE_MISSION, 0);
    send_raw(req);

    std::cout << "[mission] " << wps.size() << " navigation waypoint(s) loaded.\n";
    return wps;
}

// ── tile index ────────────────────────────────────────────────────────────────

struct TileEntry {
    int    tile_x, tile_y, zoom;
    cv::Mat descs;   // (N × 128) CV_32F
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

// ── FLANN index (built once) ──────────────────────────────────────────────────

struct FlannIndex {
    cv::Ptr<cv::FlannBasedMatcher> matcher;
    std::vector<int>               offsets;
    int                            n_tiles;
};

static FlannIndex build_flann(const std::vector<TileEntry>& index)
{
    FlannIndex fi;
    fi.n_tiles = (int)index.size();
    fi.offsets.resize(fi.n_tiles + 1, 0);
    for (int i = 0; i < fi.n_tiles; ++i)
        fi.offsets[i+1] = fi.offsets[i] + index[i].descs.rows;

    int total = fi.offsets.back();
    cv::Mat all(total, 128, CV_32F);
    for (int i = 0; i < fi.n_tiles; ++i)
        index[i].descs.copyTo(all.rowRange(fi.offsets[i], fi.offsets[i+1]));

    std::cout << "Total train kp : " << total << "  — building FLANN …" << std::flush;
    fi.matcher = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(5),
        cv::makePtr<cv::flann::SearchParams>(50));
    fi.matcher->add(all);
    fi.matcher->train();
    std::cout << " done.\n";
    return fi;
}

static std::vector<std::pair<int,int>> flann_vote(
    const cv::Mat& descs_q, const FlannIndex& fi)
{
    std::vector<std::vector<cv::DMatch>> raw;
    fi.matcher->knnMatch(descs_q, raw, 2);

    std::vector<int> votes(fi.n_tiles, 0);
    for (auto& pr : raw) {
        if (pr.size() < 2) continue;
        if (pr[0].distance < MATCH_RATIO * pr[1].distance) {
            int ti = (int)(std::upper_bound(fi.offsets.begin(), fi.offsets.end(),
                                            pr[0].trainIdx) - fi.offsets.begin()) - 1;
            if (ti >= 0 && ti < fi.n_tiles) votes[ti]++;
        }
    }
    std::vector<std::pair<int,int>> ranked;
    for (int i = 0; i < fi.n_tiles; ++i)
        if (votes[i] > 0) ranked.push_back({i, votes[i]});
    std::sort(ranked.begin(), ranked.end(),
              [](auto& a, auto& b){ return a.second > b.second; });
    return ranked;
}

// ── SIFT wrapper (GPU with CPU fallback) ──────────────────────────────────────

static void detect_compute(const cv::Mat& gray,
                            std::vector<cv::KeyPoint>& kps, cv::Mat& descs, int max_kp)
{
#ifdef USE_GPU
    if (g_gpu_sift) {
        popsift_detect_compute(gray, kps, descs, max_kp);
        return;
    }
#endif
    auto sift = cv::SIFT::create(max_kp > 0 ? max_kp : 2000);
    sift->detectAndCompute(gray, cv::noArray(), kps, descs);
}

// ── tile stitching + direct match ─────────────────────────────────────────────

static cv::Mat stitch_tiles(int cx, int cy, int zoom, const char* tile_dir, int r = 1)
{
    int side = 2*r+1;
    cv::Mat canvas(side*TILE_SIZE, side*TILE_SIZE, CV_8UC3, cv::Scalar(0,0,0));
    for (int dx = 0; dx < side; ++dx)
        for (int dy = 0; dy < side; ++dy) {
            int tx = cx - r + dx, ty = cy - r + dy;
            std::string path = std::string(tile_dir) + "/" +
                               std::to_string(zoom) + "/" +
                               std::to_string(tx)   + "/" +
                               std::to_string(ty)   + ".png";
            cv::Mat tile = cv::imread(path);
            if (!tile.empty())
                tile.copyTo(canvas(cv::Rect(dx*TILE_SIZE, dy*TILE_SIZE, TILE_SIZE, TILE_SIZE)));
        }
    return canvas;
}

struct MatchResult { double lat, lon; int inliers; };

static std::optional<MatchResult> direct_match(
    const cv::Mat& query_gray,
    int cx, int cy, int zoom, const char* tile_dir,
    int radius, int min_inliers, std::string& reason)
{
    cv::Mat composite = stitch_tiles(cx, cy, zoom, tile_dir, radius);
    cv::Mat comp_gray;
    cv::cvtColor(composite, comp_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> kps_c, kps_q;
    cv::Mat descs_c, descs_q;
    detect_compute(comp_gray,  kps_c,  descs_c,  3000);
    detect_compute(query_gray, kps_q,  descs_q,  2000);

    if (descs_c.empty() || (int)kps_c.size() < min_inliers) {
        reason = "composite too few kp (" + std::to_string(kps_c.size()) + ")";
        return std::nullopt;
    }
    if (descs_q.empty() || (int)kps_q.size() < min_inliers) {
        reason = "query too few kp (" + std::to_string(kps_q.size()) + ")";
        return std::nullopt;
    }

    std::vector<std::vector<cv::DMatch>> raw;
#if defined(USE_GPU)
    if (g_gpu_sift) {
        cuda_knn2_match(descs_q, descs_c, raw);
    } else
#endif
    {
        auto bf = cv::BFMatcher::create(cv::NORM_L2);
        bf->knnMatch(descs_q, descs_c, raw, 2);
    }

    std::vector<cv::DMatch> good;
    for (auto& pr : raw)
        if (pr.size() >= 2 && pr[0].distance < MATCH_RATIO * pr[1].distance)
            good.push_back(pr[0]);
    if ((int)good.size() < min_inliers) {
        reason = "ratio-test matches " + std::to_string(good.size())
                 + " < " + std::to_string(min_inliers);
        return std::nullopt;
    }

    std::vector<cv::Point2f> src, dst;
    for (auto& m : good) {
        src.push_back(kps_q[m.queryIdx].pt);
        dst.push_back(kps_c[m.trainIdx].pt);
    }
    cv::Mat mask;
    cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 5.0, mask);
    if (H.empty()) {
        reason = "homography failed (good=" + std::to_string(good.size()) + ")";
        return std::nullopt;
    }

    int inliers = cv::countNonZero(mask);
    if (inliers < min_inliers) {
        reason = "inliers " + std::to_string(inliers)
                 + " < " + std::to_string(min_inliers);
        return std::nullopt;
    }

    // Check convexity (rejects degenerate homographies)
    float qw = (float)query_gray.cols, qh = (float)query_gray.rows;
    std::vector<cv::Point2f> corners_in = {{0,0},{qw,0},{qw,qh},{0,qh}};
    std::vector<cv::Point2f> corners_out;
    cv::perspectiveTransform(corners_in, corners_out, H);
    float area = cv::contourArea(corners_out);
    if (area < qw*qh*0.05f || area > qw*qh*50.0f) {
        reason = "area check failed (area=" + std::to_string((int)area)
                 + " expected " + std::to_string((int)(qw*qh*0.05f))
                 + ".." + std::to_string((int)(qw*qh*50.0f)) + ")";
        return std::nullopt;
    }
    if (!cv::isContourConvex(std::vector<cv::Point2f>(corners_out))) {
        reason = "non-convex homography (inliers=" + std::to_string(inliers) + ")";
        return std::nullopt;
    }

    // Project query centre → composite pixel → GPS
    std::vector<cv::Point2f> ctr_in  = {{qw/2, qh/2}};
    std::vector<cv::Point2f> ctr_out;
    cv::perspectiveTransform(ctr_in, ctr_out, H);
    float cpx = ctr_out[0].x, cpy = ctr_out[0].y;

    int comp_size = (2*radius+1) * TILE_SIZE;
    int margin    = TILE_SIZE;
    if (cpx < -margin || cpx > comp_size+margin ||
        cpy < -margin || cpy > comp_size+margin) {
        reason = "centre out of bounds (cpx=" + std::to_string((int)cpx)
                 + " cpy=" + std::to_string((int)cpy) + ")";
        return std::nullopt;
    }

    int x_orig = cx - radius, y_orig = cy - radius;
    int tile_dx = (int)(cpx / TILE_SIZE);
    int tile_dy = (int)(cpy / TILE_SIZE);
    float lpx = cpx - tile_dx * TILE_SIZE;
    float lpy = cpy - tile_dy * TILE_SIZE;
    auto [lat, lon] = pixel_to_latlon(lpx, lpy, x_orig+tile_dx, y_orig+tile_dy, zoom);
    reason = "OK inliers=" + std::to_string(inliers);
    return MatchResult{lat, lon, inliers};
}

// ── vehicle state (MAVLink telemetry, thread-safe) ────────────────────────────

struct VehicleState {
    std::mutex mtx;
    double  alt_m    = 0.0;
    double  lat      = 0.0;
    double  lon      = 0.0;
    bool    gps_ok   = false;
    float   roll_rad  = 0.0f;
    float   pitch_rad = 0.0f;
    float   yaw_rad   = 0.0f;

    void update(int32_t lat_1e7, int32_t lon_1e7, int32_t rel_alt_mm) {
        std::lock_guard<std::mutex> lk(mtx);
        lat    = lat_1e7  / 1e7;
        lon    = lon_1e7  / 1e7;
        alt_m  = rel_alt_mm / 1000.0;
        gps_ok = true;
    }

    void update_attitude(float roll, float pitch, float yaw) {
        std::lock_guard<std::mutex> lk(mtx);
        roll_rad  = roll;
        pitch_rad = pitch;
        yaw_rad   = yaw;
    }

    struct Snap { double alt_m, lat, lon; bool gps_ok; float roll_rad, pitch_rad, yaw_rad; };
    Snap snapshot() {
        std::lock_guard<std::mutex> lk(mtx);
        return {alt_m, lat, lon, gps_ok, roll_rad, pitch_rad, yaw_rad};
    }
};

// ── serial port (POSIX termios) ───────────────────────────────────────────────

static int open_serial(const char* port, int baud)
{
    int fd = ::open(port, O_RDWR | O_NOCTTY);
    if (fd < 0) throw std::runtime_error(
        std::string("Cannot open serial port: ") + port + " — " + strerror(errno));

    struct termios tty {};
    tcgetattr(fd, &tty);

    speed_t speed = B57600;
    if      (baud == 9600)   speed = B9600;
    else if (baud == 19200)  speed = B19200;
    else if (baud == 38400)  speed = B38400;
    else if (baud == 57600)  speed = B57600;
    else if (baud == 115200) speed = B115200;
    else if (baud == 230400) speed = B230400;
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |=  CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |=  CREAD | CLOCAL;
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~OPOST;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 10;   // 100 ms read timeout

    tcsetattr(fd, TCSANOW, &tty);
    return fd;
}

// ── MAVLink reader thread ─────────────────────────────────────────────────────

static void mavlink_reader(int fd, VehicleState& state)
{
    mavlink_message_t msg;
    mavlink_status_t  status;
    uint8_t byte;

    std::cout << "[mav] reader thread started.\n";
    while (!g_shutdown) {
        ssize_t n = ::read(fd, &byte, 1);
        if (n <= 0) continue;
        if (mavlink_parse_char(MAVLINK_COMM_0, byte, &msg, &status)) {
            if (msg.msgid == MAVLINK_MSG_ID_GLOBAL_POSITION_INT) {
                mavlink_global_position_int_t pos;
                mavlink_msg_global_position_int_decode(&msg, &pos);
                state.update(pos.lat, pos.lon, pos.relative_alt);
            } else if (msg.msgid == MAVLINK_MSG_ID_ATTITUDE) {
                mavlink_attitude_t att;
                mavlink_msg_attitude_decode(&msg, &att);
                state.update_attitude(att.roll, att.pitch, att.yaw);
            }
        }
    }
    std::cout << "[mav] reader thread stopped.\n";
}

// ── CSI camera (GStreamer) ────────────────────────────────────────────────────

static std::string gst_pipeline(int sensor_id, int w, int h, int fps, int flip)
{
    std::ostringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
       << "video/x-raw(memory:NVMM), width=(int)" << w
       << ", height=(int)" << h
       << ", framerate=(fraction)" << fps << "/1 ! "
       << "nvvidconv flip-method=" << flip << " ! "
       << "video/x-raw, width=(int)" << w << ", height=(int)" << h
       << ", format=(string)BGRx ! "
       << "videoconvert ! "
       << "video/x-raw, format=(string)BGR ! appsink sync=false max-buffers=1 drop=true";
    return ss.str();
}

// ── photo filename ────────────────────────────────────────────────────────────

static std::string build_filename(double real_lat, double real_lon,
                                   double est_lat,  double est_lon,
                                   bool has_fix, double dist_m,
                                   double real_alt)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(7);
    ss << real_lat << "_" << real_lon << "-";
    if (has_fix) {
        ss << est_lat << "_" << est_lon;
        ss << "__" << std::setprecision(1) << dist_m << "m";
    } else {
        ss << "NOFIX_NOFIX__NAm";
    }
    // append real altitude (e.g. __32.4m)
    ss << "__" << std::setprecision(1) << real_alt << "m";
    ss << ".png";
    return ss.str();
}

// ── perspective correction helper ────────────────────────────────────────────

static cv::Mat build_correction_H(cv::Size img_size,
                                   float pitch_deg, float roll_deg, float yaw_deg,
                                   float fov_h_deg = 80.0f)
{
    double f = img_size.width / (2.0 * std::tan(fov_h_deg * M_PI / 360.0));
    cv::Mat K = (cv::Mat_<double>(3,3) << f, 0, img_size.width  / 2.0,
                                          0, f, img_size.height / 2.0,
                                          0, 0, 1);
    double p = -pitch_deg * M_PI / 180.0;
    double r = -roll_deg  * M_PI / 180.0;
    double y = -yaw_deg   * M_PI / 180.0;
    cv::Mat Rx = (cv::Mat_<double>(3,3) <<
        1, 0,            0,
        0, std::cos(p), -std::sin(p),
        0, std::sin(p),  std::cos(p));
    cv::Mat Ry = (cv::Mat_<double>(3,3) <<
        std::cos(r),  0, std::sin(r),
        0,            1, 0,
        -std::sin(r), 0, std::cos(r));
    cv::Mat Rz = (cv::Mat_<double>(3,3) <<
        std::cos(y), -std::sin(y), 0,
        std::sin(y),  std::cos(y), 0,
        0,            0,           1);
    cv::Mat R = Rx * Ry * Rz;
    return K * R * K.inv();
}

// ── localize one frame ────────────────────────────────────────────────────────

static std::optional<MatchResult> localize_frame(
    const cv::Mat& frame,
    const std::vector<TileEntry>& index,
    const FlannIndex& fi,
    int zoom,
    float pitch_deg = 0.0f,
    float roll_deg  = 0.0f,
    float yaw_deg   = 0.0f)
{
    // Apply perspective correction for pitch / roll / yaw before feature extraction
    cv::Mat corrected;
    if (std::fabs(pitch_deg) > 0.5f || std::fabs(roll_deg) > 0.5f || std::fabs(yaw_deg) > 0.5f) {
        cv::Mat H = build_correction_H(frame.size(), pitch_deg, roll_deg, yaw_deg);
        cv::warpPerspective(frame, corrected, H, frame.size(),
                            cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        g_log << ts() << "  Tilt correction applied"
              << " pitch=" << std::setprecision(1) << pitch_deg
              << "° roll="  << roll_deg
              << "° yaw="   << yaw_deg << "°\n";
    } else {
        corrected = frame;
    }

    cv::Mat gray;
    cv::cvtColor(corrected, gray, cv::COLOR_BGR2GRAY);

    // Phase 1 — coarse FLANN vote (CPU SIFT to match pre-built index)
    auto sift = cv::SIFT::create(2000);
    std::vector<cv::KeyPoint> kps_q;
    cv::Mat descs_q;
    sift->detectAndCompute(gray, cv::noArray(), kps_q, descs_q);

    g_log << ts() << "  [P1] extracted " << kps_q.size() << " keypoints\n";
    if (descs_q.empty() || (int)kps_q.size() < 5) {
        g_log << ts() << "  [P1] FAIL: too few features ("
              << kps_q.size() << " < 5) — skip frame\n";
        return std::nullopt;
    }

    auto ranked = flann_vote(descs_q, fi);
    g_log << ts() << "  [P1] FLANN voted on " << ranked.size() << " tile(s)\n";
    if (ranked.empty()) {
        g_log << ts() << "  [P1] FAIL: no tile candidates from FLANN\n";
        return std::nullopt;
    }

    // Log top-5 candidates
    int show = std::min((int)ranked.size(), 5);
    g_log << ts() << "  [P1] top " << show << " candidates:\n";
    for (int i = 0; i < show; ++i) {
        int ti = ranked[i].first;
        g_log << ts() << "       #" << (i+1) << " tile("
              << index[ti].tile_x << "," << index[ti].tile_y
              << ") votes=" << ranked[i].second << "\n";
    }

    // Phase 2 — direct match on top candidates
    int n_try = std::min((int)ranked.size(), TOP_CANDIDATES);
    for (int i = 0; i < n_try; ++i) {
        int ti = ranked[i].first;
        int cx = index[ti].tile_x, cy = index[ti].tile_y;
        std::string reason;
        auto res = direct_match(gray, cx, cy, zoom, TILE_DIR, 1, MIN_INLIERS, reason);
        if (res) {
            g_log << ts() << "  [P2] MATCH tile(" << cx << "," << cy
                  << ") votes=" << ranked[i].second << " — " << reason << "\n";
            return res;
        }
        g_log << ts() << "  [P2] FAIL  tile(" << cx << "," << cy
              << ") votes=" << ranked[i].second << " — " << reason << "\n";
    }
    g_log << ts() << "  [P2] all " << n_try << " candidate(s) rejected — NOFIX\n";
    return std::nullopt;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    // ── parse args ────────────────────────────────────────────────────────────
    const char* port    = DEFAULT_PORT;
    int         baud    = DEFAULT_BAUD;
    int         zoom    = DEFAULT_ZOOM;
    double      interval = 5.0;            // seconds between photos (--interval)
    std::string waypoints_str;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--port")      && i+1 < argc) port     = argv[++i];
        else if (!strcmp(argv[i], "--baud")      && i+1 < argc) baud     = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--zoom")      && i+1 < argc) zoom     = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--interval")  && i+1 < argc) interval = std::stod(argv[++i]);
        else if (!strcmp(argv[i], "--waypoints") && i+1 < argc) waypoints_str = argv[++i];
        else {
            std::cerr << "Usage: " << argv[0]
                      << " [--port /dev/ttyTHS1] [--baud 57600] [--zoom 18]\n"
                      << "       [--interval 5]  (seconds between photos, default 5)\n"
                      << "       [--waypoints \"lat1,lon1 lat2,lon2 ...\"]\n"
                      << "  Mission waypoints are auto-fetched from Pixhawk at startup.\n"
                      << "  --waypoints overrides the auto-fetch.\n";
            return 1;
        }
    }

    // ── signal handlers ───────────────────────────────────────────────────────
    signal(SIGINT,  handle_signal);
    signal(SIGTERM, handle_signal);

    // ── open log file (same dir as executable, named by start timestamp) ─────
    {
        char exe_path[4096] = {};
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
        std::string exe_dir = (len > 0)
            ? fs::path(exe_path).parent_path().string() : ".";
        std::string log_path = exe_dir + "/" + timestamp_tag() + ".log";
        g_log.open(log_path);
        g_log << "=== drone_localize " << timestamp_tag() << " ===\n";
        g_log << "port=" << port << " baud=" << baud
              << " zoom=" << zoom << " interval=" << interval << "s\n\n";
        std::cout << "Log file       : " << log_path << "\n";
    }

    // ── create photo output directory ─────────────────────────────────────────
    fs::create_directories(PHOTO_DIR);
    std::cout << "Photos → " << fs::absolute(PHOTO_DIR) << "\n";

    // ── #4: parse + validate waypoints ────────────────────────────────────────
    std::vector<Waypoint> waypoints;
    if (!waypoints_str.empty()) {
        try {
            waypoints = parse_waypoints(waypoints_str);
            validate_waypoints(waypoints);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return 1;
        }
    }

    // ── GPU init (optional) ───────────────────────────────────────────────────
#ifdef USE_GPU
    {
        auto avail_mb = []() -> long {
            std::ifstream f("/proc/meminfo");
            std::string key; long val; std::string unit;
            while (f >> key >> val >> unit)
                if (key == "MemAvailable:") return val / 1024;
            return 0;
        };
        long mem = avail_mb();
        std::cout << "GPU init (avail=" << mem << " MB) … " << std::flush;
        if (mem >= 512) {
            try {
                init_popsift();
                { cv::Mat t(4,128,CV_32F,cv::Scalar(0));
                  std::vector<std::vector<cv::DMatch>> d;
                  cuda_knn2_match(t,t,d); }
                g_gpu_sift = true;
                std::cout << "done (GPU).\n";
            } catch (const std::exception& e) {
                std::cout << "failed (" << e.what() << ") — CPU fallback.\n";
            }
        } else {
            std::cout << "skipped (low RAM) — CPU fallback.\n";
        }
    }
#else
    std::cout << "SIFT backend   : CPU (build with USE_GPU=1 for PopSIFT)\n";
#endif

    // ── load tile index + build FLANN ─────────────────────────────────────────
    std::string index_path = std::string(INDEX_DIR) + "/sift_index_z"
                             + std::to_string(zoom) + ".bin";
    std::cout << "Loading index  : " << index_path << " … " << std::flush;
    std::vector<TileEntry> index;
    try {
        index = load_index(index_path);
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        std::cerr << "Run export_index.py first to generate the .bin index.\n";
        return 1;
    }
    std::cout << index.size() << " tiles.\n";
    FlannIndex fi = build_flann(index);

    // ── open serial ───────────────────────────────────────────────────────────
    std::cout << "MAVLink serial : " << port << " @ " << baud << " baud … " << std::flush;
    int serial_fd;
    try {
        serial_fd = open_serial(port, baud);
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
    std::cout << "ok.\n";

    // ── heartbeat + auto-fetch mission (before starting reader thread) ────────
    wait_for_heartbeat(serial_fd);
    if (waypoints.empty()) {
        // No --waypoints override: read the active mission from Pixhawk
        waypoints = fetch_mission_from_pixhawk(serial_fd);
        if (!waypoints.empty())
            validate_waypoints(waypoints);
    } else {
        std::cout << "[mission] Using --waypoints override ("
                  << waypoints.size() << " point(s)).\n";
    }

    // Ask Pixhawk to start streaming position data on this port
    request_streams(serial_fd);

    // ── start MAVLink reader thread ───────────────────────────────────────────
    VehicleState state;
    std::thread mav_thread(mavlink_reader, serial_fd, std::ref(state));

    // Wait for first GPS fix
    std::cout << "Waiting for Pixhawk GPS fix …" << std::flush;
    while (!g_shutdown) {
        auto s = state.snapshot();
        if (s.gps_ok) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "." << std::flush;
    }
    std::cout << " ok.\n";
    {
        auto s = state.snapshot();
        g_log << ts() << "[startup] GPS fix: lat=" << std::fixed << std::setprecision(7)
              << s.lat << " lon=" << s.lon
              << " alt=" << std::setprecision(1) << s.alt_m << " m\n";
    }

    // ── open CSI camera ───────────────────────────────────────────────────────
    std::string pipeline = gst_pipeline(CSI_SENSOR_ID, CSI_WIDTH, CSI_HEIGHT,
                                        CSI_FPS, CSI_FLIP);
    std::cout << "Camera pipeline: " << pipeline << "\n";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open CSI camera via GStreamer.\n";
        g_shutdown = true;
        mav_thread.join();
        close(serial_fd);
        return 1;
    }
    std::cout << "CSI camera     : opened.\n";

    // ── flight loop ───────────────────────────────────────────────────────────
    bool   was_on_ground    = true;
    bool   capturing_active = false;
    auto   last_capture     = std::chrono::steady_clock::now()
                              - std::chrono::seconds(100);  // allow immediate first shot

    std::cout << std::string(55, '-') << "\n";
    std::cout << "Waiting for takeoff (altitude ≥ " << TAKEOFF_ALT << " m) …\n";

    while (!g_shutdown) {
        auto [alt, lat, lon, gps_ok, roll_unused, pitch_unused] = state.snapshot();

        // takeoff detection
        if (was_on_ground && alt >= TAKEOFF_ALT) {
            g_log << ts() << "Takeoff confirmed — alt=" << std::fixed
                  << std::setprecision(1) << alt << " m  lat=" << std::setprecision(7)
                  << lat << " lon=" << lon
                  << "  capture every " << interval << " s\n";
            capturing_active = true;
            was_on_ground    = false;
        }

        // landing detection
        if (alt < MIN_CAPTURE_ALT) {
            if (capturing_active) {
                g_log << ts() << "Landing detected — alt=" << std::fixed
                      << std::setprecision(1) << alt << " m < " << MIN_CAPTURE_ALT
                      << " m — capture paused, closing log.\n";
                g_log.close();
            }
            capturing_active = false;
            was_on_ground    = true;
        }

        // periodic capture
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - last_capture).count();

        if (capturing_active && elapsed_s >= interval) {
            last_capture = now;
            g_log << "\n" << ts() << std::string(50, '-') << "\n";

            // fresh GPS + attitude snapshot right before the shot
            auto [snap_alt, real_lat, real_lon, snap_ok, roll_r, pitch_r, yaw_r] = state.snapshot();
            g_log << ts() << "Capture — alt=" << std::fixed << std::setprecision(1)
                  << snap_alt << " m  GPS=" << std::setprecision(7)
                  << real_lat << "," << real_lon;
            if (!snap_ok) {
                g_log << "  [NO GPS FIX — skip]\n";
                continue;
            }

            // Attitude guard — skip oblique photos (matching only works nadir)
            float roll_deg  = roll_r  * 57.2958f;
            float pitch_deg = pitch_r * 57.2958f;
            float yaw_deg   = yaw_r   * 57.2958f;
            g_log << "  roll=" << std::setprecision(1) << roll_deg
                  << "° pitch=" << pitch_deg << "°";
            if (std::fabs(roll_deg) > MAX_TILT_DEG || std::fabs(pitch_deg) > MAX_TILT_DEG) {
                g_log << "  [TILTED — skip]\n";
                last_capture -= std::chrono::milliseconds(
                    static_cast<int>(interval * 800));  // retry sooner (80% of interval)
                continue;
            }
            g_log << "\n";

            // grab frame
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                g_log << ts() << "  [CAMERA READ FAILED — skip]\n";
                continue;
            }
            g_log << ts() << "  Frame: " << frame.cols << "x" << frame.rows << "\n";

            // SIFT localization (attitude-corrected)
            g_log << ts() << "  Starting SIFT localization …\n";
            auto result = localize_frame(frame, index, fi, zoom,
                                         pitch_deg, roll_deg, yaw_deg);

            bool has_fix = result.has_value();
            double est_lat = 0, est_lon = 0, dist_m = -1.0;
            if (has_fix) {
                est_lat = result->lat;
                est_lon = result->lon;
                dist_m  = haversine_m(real_lat, real_lon, est_lat, est_lon);
                g_log << ts() << "  SIFT result: lat=" << std::setprecision(7)
                      << est_lat << " lon=" << est_lon
                      << " inliers=" << result->inliers
                      << " dist=" << std::setprecision(1) << dist_m << " m\n";
            } else {
                g_log << ts() << "  SIFT result: NOFIX\n";
            }

            // save photo (filename includes real alt at end)
            std::string fname = build_filename(real_lat, real_lon,
                                               est_lat, est_lon, has_fix,
                                               dist_m, snap_alt);
            std::string fpath = std::string(PHOTO_DIR) + "/" + fname;
            cv::imwrite(fpath, frame);
            g_log << ts() << "  Saved: " << fpath << "\n";

            // companion attitude JSON — used by correct_tilt.py for perspective correction
            {
                std::string jpath = fpath.substr(0, fpath.rfind('.')) + ".attitude.json";
                std::ofstream jf(jpath);
                jf << std::fixed;
                jf << "{\n"
                   << "  \"roll_deg\":  " << std::setprecision(4) << roll_deg  << ",\n"
                   << "  \"pitch_deg\": " << std::setprecision(4) << pitch_deg << ",\n"
                   << "  \"yaw_deg\":   " << std::setprecision(4) << yaw_deg   << ",\n"
                   << "  \"alt_m\":     " << std::setprecision(1) << snap_alt  << ",\n"
                   << "  \"lat\":       " << std::setprecision(7) << real_lat  << ",\n"
                   << "  \"lon\":       " << std::setprecision(7) << real_lon  << "\n"
                   << "}\n";
                g_log << ts() << "  Attitude JSON: " << jpath << "\n";
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── cleanup ───────────────────────────────────────────────────────────────
    g_log << "\n" << ts() << "=== shutdown ===\n";
    g_log.close();
    cap.release();
    mav_thread.join();
    close(serial_fd);
#ifdef USE_GPU
    shutdown_popsift();
#endif
    std::cout << "Done.\n";
    return 0;
}
