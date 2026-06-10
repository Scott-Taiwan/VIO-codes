/*
 * best_altitude.cpp — Autonomous altitude-optimization survey
 *
 * Commands Pixhawk (ArduCopter GUIDED mode) to:
 *   1. Arm and take off
 *   2. Visit 5 test points at increasing altitudes (20–60 m),
 *      each at a different position within 50 m of home
 *   3. At each point: capture photo, run SIFT, record GPS error
 *   4. RTL at 110 s (leaves 10 s headroom before the 2-minute deadline)
 *   5. Print best altitude (lowest mean SIFT error)
 *
 * Test pattern:
 *   Point 0:  20 m alt,   0 m N,   0 m E  (above home)
 *   Point 1:  30 m alt,  20 m N,   0 m E
 *   Point 2:  40 m alt,   0 m N,  20 m E
 *   Point 3:  50 m alt,  20 m S,   0 m E
 *   Point 4:  60 m alt,   0 m N,  20 m W
 *
 * Photos saved to result/ as:
 *   {lat_real}_{lon_real}-{lat_est}_{lon_est}__{dist}m_alt{alt}m.png
 *   {lat_real}_{lon_real}-NOFIX_NOFIX__NAm_alt{alt}m.png
 *
 * Build:
 *   cd best_altitude && make
 * Run:
 *   ./best_altitude [--port /dev/ttyTHS1] [--baud 57600] [--zoom 19]
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <common/mavlink.h>

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
using Clock  = std::chrono::steady_clock;

// ── config ────────────────────────────────────────────────────────────────────

static const char*  DEFAULT_PORT       = "/dev/ttyTHS1";
static const int    DEFAULT_BAUD       = 57600;
static const int    DEFAULT_ZOOM       = 19;
static const char*  INDEX_DIR          = "../index";
static const char*  TILE_DIR           = "../tiles";
static const char*  RESULT_DIR         = "result";

static const int    MAX_FLIGHT_SECS    = 110;   // force RTL this many seconds after takeoff
static const float  TAKEOFF_ALT_M     = 20.0f;  // initial takeoff altitude
static const float  REACH_HORIZ_M     =  3.0f;  // "arrived" horizontal threshold
static const float  REACH_ALT_M       =  2.5f;  // "arrived" altitude threshold
static const int    WAIT_REACHED_S    = 25;      // max wait to reach each test point
static const int    WAIT_ARMED_S      = 12;      // max wait after arm command
static const int    WAIT_TAKEOFF_S    = 35;      // max wait for takeoff altitude
static const int    STAB_MS           = 1500;    // stabilise (ms) before photo

static const int    TILE_SIZE         = 256;
static const float  MATCH_RATIO       = 0.75f;
static const int    MIN_INLIERS       = 6;
static const int    TOP_CANDIDATES    = 20;

// ArduCopter custom modes
static const uint32_t COPTER_MODE_GUIDED = 4;

// CSI camera
static const int    CSI_SENSOR_ID     = 0;
static const int    CSI_WIDTH         = 1280;
static const int    CSI_HEIGHT        = 720;
static const int    CSI_FPS           = 30;
static const int    CSI_FLIP          = 0;   // 0 = none, 2 = 180°

static std::atomic<bool> g_shutdown{false};

// ── signal handler ────────────────────────────────────────────────────────────

static void handle_signal(int) { g_shutdown = true; }

// ── test pattern ─────────────────────────────────────────────────────────────

struct TestPoint { double north_m, east_m, alt_m; };

static const TestPoint TEST_POINTS[] = {
    {   0.0,   0.0, 20.0 },   // directly above home
    {  20.0,   0.0, 30.0 },   // 20 m north
    {   0.0,  20.0, 40.0 },   // 20 m east
    { -20.0,   0.0, 50.0 },   // 20 m south
    {   0.0, -20.0, 60.0 },   // 20 m west
};
static const int N_POINTS = (int)(sizeof(TEST_POINTS)/sizeof(TEST_POINTS[0]));

// ── coordinate math ───────────────────────────────────────────────────────────

static std::pair<double,double> pixel_to_latlon(
    double px, double py, int tile_x, int tile_y, int zoom)
{
    int n     = 1 << zoom;
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

// Offset a GPS coordinate by metres (flat-earth approximation, good < 1 km)
static std::pair<double,double> offset_latlon(
    double lat, double lon, double north_m, double east_m)
{
    double dlat = north_m / 111111.0;
    double dlon = east_m  / (111111.0 * std::cos(lat * M_PI / 180.0));
    return {lat + dlat, lon + dlon};
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

// ── FLANN index ───────────────────────────────────────────────────────────────

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

// ── tile stitching + direct match ─────────────────────────────────────────────

static cv::Mat stitch_tiles(int cx, int cy, int zoom, const char* tile_dir, int r = 1)
{
    int side = 2*r+1;
    cv::Mat canvas(side*TILE_SIZE, side*TILE_SIZE, CV_8UC3, cv::Scalar(0,0,0));
    for (int dx = 0; dx < side; ++dx)
        for (int dy = 0; dy < side; ++dy) {
            int tx = cx - r + dx, ty = cy - r + dy;
            std::string path = std::string(tile_dir) + "/" +
                               std::to_string(zoom)  + "/" +
                               std::to_string(tx)    + "/" +
                               std::to_string(ty)    + ".png";
            cv::Mat tile = cv::imread(path);
            if (!tile.empty())
                tile.copyTo(canvas(cv::Rect(dx*TILE_SIZE, dy*TILE_SIZE,
                                            TILE_SIZE, TILE_SIZE)));
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

    auto sift = cv::SIFT::create(3000);
    std::vector<cv::KeyPoint> kps_c, kps_q;
    cv::Mat descs_c, descs_q_local;
    sift->detectAndCompute(comp_gray,  cv::noArray(), kps_c,       descs_c);
    sift->detectAndCompute(query_gray, cv::noArray(), kps_q, descs_q_local);

    if (descs_c.empty() || (int)kps_c.size() < min_inliers) return std::nullopt;
    if (descs_q_local.empty() || (int)kps_q.size() < min_inliers) return std::nullopt;

    auto bf = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> raw;
    bf->knnMatch(descs_q_local, descs_c, raw, 2);

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

    float qw = (float)query_gray.cols, qh = (float)query_gray.rows;
    std::vector<cv::Point2f> corners_in = {{0,0},{qw,0},{qw,qh},{0,qh}};
    std::vector<cv::Point2f> corners_out;
    cv::perspectiveTransform(corners_in, corners_out, H);
    float area = cv::contourArea(corners_out);
    if (area < qw*qh*0.05f || area > qw*qh*50.0f) return std::nullopt;
    if (!cv::isContourConvex(std::vector<cv::Point2f>(corners_out))) return std::nullopt;

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

// ── localize one frame ────────────────────────────────────────────────────────

static std::optional<MatchResult> localize_frame(
    const cv::Mat& frame,
    const std::vector<TileEntry>& index,
    const FlannIndex& fi,
    int zoom)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    auto sift = cv::SIFT::create(2000);
    std::vector<cv::KeyPoint> kps_q;
    cv::Mat descs_q;
    sift->detectAndCompute(gray, cv::noArray(), kps_q, descs_q);
    if (descs_q.empty() || (int)kps_q.size() < 5) {
        std::cout << "  Too few features.\n";
        return std::nullopt;
    }
    std::cout << "  Query features : " << kps_q.size() << " kp\n";

    auto ranked = flann_vote(descs_q, fi);
    if (ranked.empty()) {
        std::cout << "  No FLANN votes.\n";
        return std::nullopt;
    }

    int n_try = std::min((int)ranked.size(), TOP_CANDIDATES);
    for (int i = 0; i < n_try; ++i) {
        int ti = ranked[i].first;
        int cx = index[ti].tile_x, cy = index[ti].tile_y;
        std::cout << "  Phase2 tile (" << cx << "," << cy
                  << ") votes=" << ranked[i].second << " …\n";
        auto res = direct_match(gray, cx, cy, zoom, TILE_DIR, 1, MIN_INLIERS);
        if (res) return res;
    }
    return std::nullopt;
}

// ── vehicle state (thread-safe) ───────────────────────────────────────────────

struct VehicleState {
    std::mutex mtx;
    double   alt_m       = 0.0;
    double   lat         = 0.0;
    double   lon         = 0.0;
    bool     gps_ok      = false;
    bool     armed       = false;
    uint32_t custom_mode = 0;

    void update_pos(int32_t lat_1e7, int32_t lon_1e7, int32_t rel_alt_mm) {
        std::lock_guard<std::mutex> lk(mtx);
        lat    = lat_1e7  / 1e7;
        lon    = lon_1e7  / 1e7;
        alt_m  = rel_alt_mm / 1000.0;
        gps_ok = true;
    }

    void update_hb(uint8_t base_mode, uint32_t cust_mode) {
        std::lock_guard<std::mutex> lk(mtx);
        armed       = (base_mode & MAV_MODE_FLAG_SAFETY_ARMED) != 0;
        custom_mode = cust_mode;
    }

    struct Snap { double alt_m, lat, lon; bool gps_ok, armed; uint32_t custom_mode; };
    Snap snapshot() {
        std::lock_guard<std::mutex> lk(mtx);
        return {alt_m, lat, lon, gps_ok, armed, custom_mode};
    }
};

// ── serial port ───────────────────────────────────────────────────────────────

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

    while (!g_shutdown) {
        ssize_t n = ::read(fd, &byte, 1);
        if (n <= 0) continue;
        if (mavlink_parse_char(MAVLINK_COMM_0, byte, &msg, &status)) {
            switch (msg.msgid) {
            case MAVLINK_MSG_ID_GLOBAL_POSITION_INT: {
                mavlink_global_position_int_t pos;
                mavlink_msg_global_position_int_decode(&msg, &pos);
                state.update_pos(pos.lat, pos.lon, pos.relative_alt);
                break;
            }
            case MAVLINK_MSG_ID_HEARTBEAT: {
                mavlink_heartbeat_t hb;
                mavlink_msg_heartbeat_decode(&msg, &hb);
                // ignore GCS heartbeats (MAV_TYPE_GCS = 6)
                if (hb.type != MAV_TYPE_GCS)
                    state.update_hb(hb.base_mode, hb.custom_mode);
                break;
            }
            }
        }
    }
}

// ── MAVLink write helpers (main thread only) ──────────────────────────────────

static void mav_write(int fd, const mavlink_message_t& msg)
{
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
    ::write(fd, buf, len);
}

static void send_heartbeat(int fd)
{
    mavlink_message_t msg;
    mavlink_msg_heartbeat_pack(255, 190, &msg,
        MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID, 0, 0, MAV_STATE_ACTIVE);
    mav_write(fd, msg);
}

static void send_command_long(int fd, uint16_t cmd,
    float p1=0, float p2=0, float p3=0, float p4=0,
    float p5=0, float p6=0, float p7=0)
{
    mavlink_message_t msg;
    mavlink_msg_command_long_pack(1, 200, &msg,
        1, 1,   // target system, component
        cmd, 0, p1, p2, p3, p4, p5, p6, p7);
    mav_write(fd, msg);
}

// ── flight control ────────────────────────────────────────────────────────────

static void set_guided_mode(int fd)
{
    send_command_long(fd, MAV_CMD_DO_SET_MODE,
        MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        (float)COPTER_MODE_GUIDED);
}

static void cmd_arm(int fd)
{
    send_command_long(fd, MAV_CMD_COMPONENT_ARM_DISARM, 1.0f);
}

static void cmd_takeoff(int fd, float alt_m)
{
    // param7 = altitude (m above home)
    send_command_long(fd, MAV_CMD_NAV_TAKEOFF, 0,0,0,0,0,0, alt_m);
}

// SET_POSITION_TARGET_GLOBAL_INT — fly to GPS position in GUIDED mode
// type_mask 0x0FF8: use position only, ignore velocity/accel/yaw
static void fly_to(int fd, double lat, double lon, float alt_m)
{
    mavlink_message_t msg;
    mavlink_msg_set_position_target_global_int_pack(
        1, 200, &msg,
        0,      // time_boot_ms (ignored)
        1, 1,   // target system, component
        MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0x0FF8, // position only
        (int32_t)(lat * 1e7),
        (int32_t)(lon * 1e7),
        alt_m,
        0.0f, 0.0f, 0.0f,   // velocity (ignored)
        0.0f, 0.0f, 0.0f,   // acceleration (ignored)
        0.0f, 0.0f);         // yaw, yaw_rate (ignored)
    mav_write(fd, msg);
}

static void cmd_rtl(int fd)
{
    send_command_long(fd, MAV_CMD_NAV_RETURN_TO_LAUNCH);
}

// ── wait helpers ──────────────────────────────────────────────────────────────

static bool wait_gps(VehicleState& state, int timeout_s = 30)
{
    auto deadline = Clock::now() + std::chrono::seconds(timeout_s);
    while (!g_shutdown && Clock::now() < deadline) {
        if (state.snapshot().gps_ok) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

static bool wait_armed(VehicleState& state, int timeout_s = WAIT_ARMED_S)
{
    auto deadline = Clock::now() + std::chrono::seconds(timeout_s);
    while (!g_shutdown && Clock::now() < deadline) {
        if (state.snapshot().armed) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

// Wait until relative altitude is within tolerance of target
static bool wait_altitude(VehicleState& state, float target_m,
                           float tol = REACH_ALT_M, int timeout_s = WAIT_TAKEOFF_S)
{
    auto deadline = Clock::now() + std::chrono::seconds(timeout_s);
    while (!g_shutdown && Clock::now() < deadline) {
        float alt = (float)state.snapshot().alt_m;
        if (std::abs(alt - target_m) <= tol) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

// Wait until drone is within REACH_HORIZ_M horizontal and REACH_ALT_M vertical of target
static bool wait_reached(VehicleState& state,
                          double tgt_lat, double tgt_lon, float tgt_alt,
                          int timeout_s = WAIT_REACHED_S)
{
    auto deadline = Clock::now() + std::chrono::seconds(timeout_s);
    while (!g_shutdown && Clock::now() < deadline) {
        auto s = state.snapshot();
        if (s.gps_ok) {
            float hdist = (float)haversine_m(s.lat, s.lon, tgt_lat, tgt_lon);
            float vdist = std::abs((float)s.alt_m - tgt_alt);
            if (hdist <= REACH_HORIZ_M && vdist <= REACH_ALT_M) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

// ── CSI camera pipeline ───────────────────────────────────────────────────────

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
                                   bool has_fix,    double dist_m,
                                   double alt_m)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(7);
    ss << real_lat << "_" << real_lon << "-";
    if (has_fix)
        ss << est_lat << "_" << est_lon
           << "__" << std::setprecision(1) << dist_m << "m";
    else
        ss << "NOFIX_NOFIX__NAm";
    ss << "_alt" << std::setprecision(0) << alt_m << "m.png";
    return ss.str();
}

// ── heartbeat sender (keeps GCS failsafe from triggering) ─────────────────────

static void heartbeat_sender(int fd)
{
    while (!g_shutdown) {
        send_heartbeat(fd);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    const char* port = DEFAULT_PORT;
    int         baud = DEFAULT_BAUD;
    int         zoom = DEFAULT_ZOOM;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--port") && i+1 < argc) port = argv[++i];
        else if (!strcmp(argv[i], "--baud") && i+1 < argc) baud = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--zoom") && i+1 < argc) zoom = std::stoi(argv[++i]);
        else {
            std::cerr << "Usage: " << argv[0]
                      << " [--port /dev/ttyTHS1] [--baud 57600] [--zoom 19]\n";
            return 1;
        }
    }

    signal(SIGINT,  handle_signal);
    signal(SIGTERM, handle_signal);

    fs::create_directories(RESULT_DIR);
    std::cout << "Result photos  → " << fs::absolute(RESULT_DIR) << "\n";
    std::cout << "Max flight     : " << MAX_FLIGHT_SECS << " s (RTL at timeout)\n";
    std::cout << "Test altitudes : 20 / 30 / 40 / 50 / 60 m\n";
    std::cout << "Max radius     : 50 m from home\n\n";

    // ── load tile index + build FLANN ─────────────────────────────────────────
    std::string index_path = std::string(INDEX_DIR) + "/sift_index_z"
                             + std::to_string(zoom) + ".bin";
    std::cout << "Loading index  : " << index_path << " … " << std::flush;
    std::vector<TileEntry> index;
    try {
        index = load_index(index_path);
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n"
                  << "Run export_index.py first to generate the .bin index.\n";
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

    // ── start background threads ──────────────────────────────────────────────
    VehicleState state;
    std::thread mav_th(mavlink_reader, serial_fd, std::ref(state));
    std::thread hb_th(heartbeat_sender, serial_fd);

    // ── open CSI camera ───────────────────────────────────────────────────────
    std::string pipeline = gst_pipeline(CSI_SENSOR_ID, CSI_WIDTH, CSI_HEIGHT,
                                        CSI_FPS, CSI_FLIP);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open CSI camera.\n";
        g_shutdown = true;
        mav_th.join(); hb_th.join();
        close(serial_fd);
        return 1;
    }
    std::cout << "CSI camera     : opened.\n";

    // ── wait for GPS ──────────────────────────────────────────────────────────
    std::cout << "Waiting for Pixhawk GPS …" << std::flush;
    if (!wait_gps(state, 60)) {
        std::cerr << "\nERROR: No GPS within 60 s — aborting.\n";
        g_shutdown = true;
        mav_th.join(); hb_th.join();
        close(serial_fd);
        return 1;
    }
    std::cout << " ok.\n";

    // ── record home position ──────────────────────────────────────────────────
    double home_lat, home_lon;
    {
        auto s  = state.snapshot();
        home_lat = s.lat;
        home_lon = s.lon;
    }
    std::cout << "Home position  : " << std::fixed << std::setprecision(7)
              << home_lat << ", " << home_lon << "\n\n";

    // ── flight sequence ───────────────────────────────────────────────────────
    bool airborne = false;

    auto safe_abort = [&](const char* reason) {
        std::cerr << "ABORT: " << reason << "\n";
        if (airborne) {
            std::cout << "Sending RTL for safety.\n";
            cmd_rtl(serial_fd);
        }
        g_shutdown = true;
    };

    // Step 1: GUIDED mode
    std::cout << std::string(60,'-') << "\n";
    std::cout << "Setting GUIDED mode …\n";
    set_guided_mode(serial_fd);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    if (g_shutdown) goto cleanup;

    // Step 2: Arm
    std::cout << "Arming …\n";
    cmd_arm(serial_fd);
    if (!wait_armed(state, WAIT_ARMED_S)) {
        safe_abort("arm timeout — check pre-arm conditions");
        goto cleanup;
    }
    std::cout << "Armed.\n";

    // Step 3: Takeoff
    {
        float first_alt = TAKEOFF_ALT_M;
        std::cout << "Takeoff to " << std::fixed << std::setprecision(0)
                  << first_alt << " m …\n";
        cmd_takeoff(serial_fd, first_alt);
        if (!wait_altitude(state, first_alt, REACH_ALT_M, WAIT_TAKEOFF_S)) {
            safe_abort("takeoff altitude not reached");
            goto cleanup;
        }
        airborne = true;
        std::cout << "Airborne at " << std::setprecision(1)
                  << state.snapshot().alt_m << " m.\n\n";
    }

    // Step 4: Test loop
    {
        auto flight_start = Clock::now();
        struct TestResult { double alt_m; double dist_m; bool has_fix; };
        std::vector<TestResult> results;

        for (int i = 0; i < N_POINTS && !g_shutdown; ++i) {

            // Check time budget
            double elapsed = std::chrono::duration<double>(
                Clock::now() - flight_start).count();
            if (elapsed >= (double)MAX_FLIGHT_SECS) {
                std::cout << "\nTime budget reached (" << (int)elapsed
                          << " s) — skipping remaining points.\n";
                break;
            }

            const TestPoint& tp = TEST_POINTS[i];
            auto [tgt_lat, tgt_lon] = offset_latlon(home_lat, home_lon,
                                                     tp.north_m, tp.east_m);
            float tgt_alt = (float)tp.alt_m;

            std::cout << std::string(60,'-') << "\n";
            std::cout << "Test " << (i+1) << "/" << N_POINTS
                      << "  alt=" << std::setprecision(0) << tgt_alt << " m"
                      << "  offset=(" << tp.north_m << " N, "
                      << tp.east_m << " E)\n";
            std::cout << "  Budget remaining : "
                      << std::setprecision(0) << (MAX_FLIGHT_SECS - elapsed) << " s\n";

            // Fly to test position
            fly_to(serial_fd, tgt_lat, tgt_lon, tgt_alt);

            if (!wait_reached(state, tgt_lat, tgt_lon, tgt_alt, WAIT_REACHED_S)) {
                std::cout << "  Position not reached in " << WAIT_REACHED_S
                          << " s — skipping.\n";
                results.push_back({tgt_alt, -1.0, false});
                continue;
            }

            // Check budget again after flying
            elapsed = std::chrono::duration<double>(
                Clock::now() - flight_start).count();
            if (elapsed >= (double)MAX_FLIGHT_SECS) {
                std::cout << "  Time budget exceeded after flight — skipping capture.\n";
                break;
            }

            // Stabilise
            std::this_thread::sleep_for(std::chrono::milliseconds(STAB_MS));

            // Read true GPS
            auto s = state.snapshot();
            std::cout << "  Position reached  alt=" << std::setprecision(1)
                      << s.alt_m << " m\n";
            std::cout << "  Real GPS : " << std::setprecision(7)
                      << s.lat << ", " << s.lon << "\n";

            // Capture frame
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "  Camera read failed — skipping.\n";
                results.push_back({tgt_alt, -1.0, false});
                continue;
            }

            // SIFT localization
            auto res = localize_frame(frame, index, fi, zoom);
            bool   has_fix = res.has_value();
            double est_lat = 0, est_lon = 0, dist_m = -1.0;

            if (has_fix) {
                est_lat = res->lat;
                est_lon = res->lon;
                dist_m  = haversine_m(s.lat, s.lon, est_lat, est_lon);
                std::cout << "  Est. GPS : " << std::setprecision(7)
                          << est_lat << ", " << est_lon << "\n";
                std::cout << "  Inliers  : " << res->inliers
                          << "   Error: " << std::setprecision(1) << dist_m << " m\n";
            } else {
                std::cout << "  Est. GPS : NOFIX\n";
            }

            results.push_back({tgt_alt, dist_m, has_fix});

            // Save photo — filename encodes all measurement data
            std::string fname = build_filename(s.lat, s.lon,
                                               est_lat, est_lon, has_fix,
                                               dist_m, tgt_alt);
            cv::imwrite(std::string(RESULT_DIR) + "/" + fname, frame);
            std::cout << "  Saved    : " << fname << "\n";
        }

        // Step 5: RTL
        std::cout << "\n" << std::string(60,'-') << "\n";
        std::cout << "RTL …\n";
        cmd_rtl(serial_fd);

        // Step 6: Print summary
        std::cout << "\n" << std::string(50,'=') << "\n";
        std::cout << "  BEST ALTITUDE SUMMARY\n";
        std::cout << std::string(50,'=') << "\n";
        std::cout << std::fixed;

        double best_dist = 1e9;
        double best_alt  = -1.0;

        for (auto& r : results) {
            std::cout << "  Alt " << std::setw(4) << std::setprecision(0)
                      << r.alt_m << " m : ";
            if (r.has_fix) {
                std::cout << std::setprecision(1) << r.dist_m << " m error";
                if (r.dist_m < best_dist) {
                    best_dist = r.dist_m;
                    best_alt  = r.alt_m;
                }
            } else {
                std::cout << "NOFIX (not reached or no match)";
            }
            std::cout << "\n";
        }

        if (best_alt > 0.0) {
            std::cout << "\n  BEST : " << std::setprecision(0) << best_alt
                      << " m  (SIFT error = "
                      << std::setprecision(1) << best_dist << " m)\n";
        } else {
            std::cout << "\n  No valid SIFT fix obtained.\n";
        }
        std::cout << std::string(50,'=') << "\n";
    }

cleanup:
    g_shutdown = true;
    mav_th.join();
    hb_th.join();
    cap.release();
    close(serial_fd);
    return 0;
}
