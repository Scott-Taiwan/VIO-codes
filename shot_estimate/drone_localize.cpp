/*
 * drone_localize.cpp — GPS-free visual localization during drone flight
 *
 * Connects to Pixhawk via MAVLink serial (default /dev/ttyTHS1).
 * Monitors relative altitude; when the drone takes off and reaches
 * TAKEOFF_ALT (50 m), captures a CSI camera frame every CAPTURE_INTERVAL
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

static const double TAKEOFF_ALT       = 50.0;   // m  — start capturing above this
static const double MIN_CAPTURE_ALT   = 10.0;   // m  — pause below this (landing)
static const double CAPTURE_INTERVAL  = 5.0;    // s  — between frames

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
static void send_gps_input(int fd, double lat, double lon, float hacc)
{
    mavlink_message_t msg;
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];

    mavlink_msg_gps_input_pack(
        1, 200, &msg,
        0,                          // time_usec (0 = autopilot uses own clock)
        0,                          // gps_id
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
                                  MAV_MISSION_ACCEPTED, MAV_MISSION_TYPE_MISSION);
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
    int radius = 1, int min_inliers = MIN_INLIERS)
{
    cv::Mat composite = stitch_tiles(cx, cy, zoom, tile_dir, radius);
    cv::Mat comp_gray;
    cv::cvtColor(composite, comp_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> kps_c, kps_q;
    cv::Mat descs_c, descs_q;
    detect_compute(comp_gray,  kps_c,  descs_c,  3000);
    detect_compute(query_gray, kps_q,  descs_q,  2000);

    if (descs_c.empty() || (int)kps_c.size() < min_inliers) return std::nullopt;
    if (descs_q.empty() || (int)kps_q.size() < min_inliers) return std::nullopt;

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
    if ((int)good.size() < min_inliers) return std::nullopt;

    std::vector<cv::Point2f> src, dst;
    for (auto& m : good) {
        src.push_back(kps_q[m.queryIdx].pt);
        dst.push_back(kps_c[m.trainIdx].pt);
    }
    cv::Mat mask;
    cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 5.0, mask);
    if (H.empty()) return std::nullopt;

    int inliers = cv::countNonZero(mask);
    if (inliers < min_inliers) return std::nullopt;

    // Check convexity (rejects degenerate homographies)
    float qw = (float)query_gray.cols, qh = (float)query_gray.rows;
    std::vector<cv::Point2f> corners_in = {{0,0},{qw,0},{qw,qh},{0,qh}};
    std::vector<cv::Point2f> corners_out;
    cv::perspectiveTransform(corners_in, corners_out, H);
    float area = cv::contourArea(corners_out);
    if (area < qw*qh*0.05f || area > qw*qh*50.0f) return std::nullopt;
    if (!cv::isContourConvex(std::vector<cv::Point2f>(corners_out))) return std::nullopt;

    // Project query centre → composite pixel → GPS
    std::vector<cv::Point2f> ctr_in  = {{qw/2, qh/2}};
    std::vector<cv::Point2f> ctr_out;
    cv::perspectiveTransform(ctr_in, ctr_out, H);
    float cpx = ctr_out[0].x, cpy = ctr_out[0].y;

    int comp_size = (2*radius+1) * TILE_SIZE;
    int margin    = TILE_SIZE;
    if (cpx < -margin || cpx > comp_size+margin ||
        cpy < -margin || cpy > comp_size+margin) return std::nullopt;

    int x_orig = cx - radius, y_orig = cy - radius;
    int tile_dx = (int)(cpx / TILE_SIZE);
    int tile_dy = (int)(cpy / TILE_SIZE);
    float lpx = cpx - tile_dx * TILE_SIZE;
    float lpy = cpy - tile_dy * TILE_SIZE;
    auto [lat, lon] = pixel_to_latlon(lpx, lpy, x_orig+tile_dx, y_orig+tile_dy, zoom);
    return MatchResult{lat, lon, inliers};
}

// ── vehicle state (MAVLink telemetry, thread-safe) ────────────────────────────

struct VehicleState {
    std::mutex mtx;
    double  alt_m  = 0.0;
    double  lat    = 0.0;
    double  lon    = 0.0;
    bool    gps_ok = false;

    void update(int32_t lat_1e7, int32_t lon_1e7, int32_t rel_alt_mm) {
        std::lock_guard<std::mutex> lk(mtx);
        lat    = lat_1e7  / 1e7;
        lon    = lon_1e7  / 1e7;
        alt_m  = rel_alt_mm / 1000.0;
        gps_ok = true;
    }

    struct Snap { double alt_m, lat, lon; bool gps_ok; };
    Snap snapshot() {
        std::lock_guard<std::mutex> lk(mtx);
        return {alt_m, lat, lon, gps_ok};
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
       << "video/x-raw, format=(string)BGR ! appsink";
    return ss.str();
}

// ── photo filename ────────────────────────────────────────────────────────────

static std::string build_filename(double real_lat, double real_lon,
                                   double est_lat,  double est_lon,
                                   bool has_fix, double dist_m = -1.0)
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
    ss << ".png";
    return ss.str();
}

// ── localize one frame ────────────────────────────────────────────────────────

static std::optional<MatchResult> localize_frame(
    const cv::Mat& frame,
    const std::vector<TileEntry>& index,
    const FlannIndex& fi,
    int zoom)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Phase 1 — coarse FLANN vote
    // (use CPU SIFT for the query to match the pre-built index)
    auto sift = cv::SIFT::create(2000);
    std::vector<cv::KeyPoint> kps_q;
    cv::Mat descs_q;
    sift->detectAndCompute(gray, cv::noArray(), kps_q, descs_q);
    if (descs_q.empty() || (int)kps_q.size() < 5) {
        std::cout << "  Too few features.\n";
        return std::nullopt;
    }
    std::cout << "  Query features: " << kps_q.size() << " kp\n";

    auto ranked = flann_vote(descs_q, fi);
    if (ranked.empty()) {
        std::cout << "  No FLANN votes.\n";
        return std::nullopt;
    }

    // Phase 2 — stitched direct match on top candidates
    int n_try = std::min((int)ranked.size(), TOP_CANDIDATES);
    for (int i = 0; i < n_try; ++i) {
        int ti = ranked[i].first;
        int cx = index[ti].tile_x, cy = index[ti].tile_y;
        std::cout << "  Phase2: tile (" << cx << "," << cy
                  << ") votes=" << ranked[i].second << " …\n";
        auto res = direct_match(gray, cx, cy, zoom, TILE_DIR, 1, MIN_INLIERS);
        if (res) return res;
    }
    return std::nullopt;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    // ── parse args ────────────────────────────────────────────────────────────
    const char* port = DEFAULT_PORT;
    int         baud = DEFAULT_BAUD;
    int         zoom = DEFAULT_ZOOM;
    std::string waypoints_str;   // #4 & #5

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--port")      && i+1 < argc) port = argv[++i];
        else if (!strcmp(argv[i], "--baud")      && i+1 < argc) baud = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--zoom")      && i+1 < argc) zoom = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--waypoints") && i+1 < argc) waypoints_str = argv[++i];
        else {
            std::cerr << "Usage: " << argv[0]
                      << " [--port /dev/ttyTHS1] [--baud 57600] [--zoom 18]\n"
                      << "       [--waypoints \"lat1,lon1 lat2,lon2 ...\"]\n"
                      << "  Mission waypoints are auto-fetched from Pixhawk at startup.\n"
                      << "  --waypoints overrides the auto-fetch.\n";
            return 1;
        }
    }

    // ── signal handlers ───────────────────────────────────────────────────────
    signal(SIGINT,  handle_signal);
    signal(SIGTERM, handle_signal);

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
        auto [alt, lat, lon, gps_ok] = state.snapshot();

        // takeoff detection
        if (was_on_ground && alt >= TAKEOFF_ALT) {
            std::cout << "\nTakeoff confirmed — altitude " << std::fixed
                      << std::setprecision(1) << alt << " m.  "
                      << "Starting capture every " << CAPTURE_INTERVAL << " s.\n";
            capturing_active = true;
            was_on_ground    = false;
        }

        // landing detection
        if (alt < MIN_CAPTURE_ALT) {
            if (capturing_active)
                std::cout << "\nAltitude " << alt << " m < " << MIN_CAPTURE_ALT
                          << " m — capture paused (landing).\n";
            capturing_active = false;
            was_on_ground    = true;
        }

        // periodic capture
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - last_capture).count();

        if (capturing_active && elapsed_s >= CAPTURE_INTERVAL) {
            last_capture = now;
            std::cout << "\n" << std::string(55, '-') << "\n";
            std::cout << "Capture — alt=" << std::fixed << std::setprecision(1)
                      << alt << " m\n";

            // get fresh GPS snapshot right before the shot
            auto [snap_alt, real_lat, real_lon, snap_ok] = state.snapshot();
            if (!snap_ok) {
                std::cout << "  No GPS fix — skipping.\n";
                continue;
            }
            std::cout << "  Real GPS : " << std::setprecision(7)
                      << real_lat << ", " << real_lon << "\n";

            // grab frame
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "  Camera read failed — skipping.\n";
                continue;
            }

            // SIFT localization
            auto result = localize_frame(frame, index, fi, zoom);

            bool has_fix = result.has_value();
            double est_lat = 0, est_lon = 0, dist_m = -1.0;
            if (has_fix) {
                est_lat = result->lat;
                est_lon = result->lon;
                dist_m  = haversine_m(real_lat, real_lon, est_lat, est_lon);
                std::cout << "  Est. GPS : " << std::setprecision(7)
                          << est_lat << ", " << est_lon << "\n";
                std::cout << "  Inliers  : " << result->inliers
                          << "   Distance: " << std::setprecision(1) << dist_m << " m\n";
            } else {
                std::cout << "  Est. GPS : NOFIX\n";
            }

            // ── send fix to Pixhawk (hacc expresses confidence to EKF) ──────
            if (has_fix && result->inliers >= MIN_INLIERS_SEND) {
                float hacc = hacc_from_inliers(result->inliers);
                std::cout << "  → GPS_INPUT: ("
                          << std::setprecision(7) << est_lat << ", " << est_lon << ")"
                          << "  inliers=" << result->inliers
                          << "  hacc=" << std::setprecision(1) << hacc << " m\n";
                send_gps_input(serial_fd, est_lat, est_lon, hacc);
            } else if (has_fix) {
                std::cout << "  → inliers=" << result->inliers
                          << " < " << MIN_INLIERS_SEND
                          << " — geometry unreliable, not sent; Pixhawk uses IMU\n";
            } else {
                std::cout << "  → NOFIX — not sent; Pixhawk uses IMU\n";
            }

            // save photo
            std::string fname = build_filename(real_lat, real_lon,
                                               est_lat, est_lon, has_fix, dist_m);
            std::string fpath = std::string(PHOTO_DIR) + "/" + fname;
            cv::imwrite(fpath, frame);
            std::cout << "  Saved    : " << fpath << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── cleanup ───────────────────────────────────────────────────────────────
    std::cout << "\nShutting down …\n";
    cap.release();
    mav_thread.join();
    close(serial_fd);
#ifdef USE_GPU
    shutdown_popsift();
#endif
    std::cout << "Done.\n";
    return 0;
}
