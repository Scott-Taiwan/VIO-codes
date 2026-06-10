/*
 * best_altitude.cpp — Autonomous (position × altitude) SIFT accuracy survey
 *
 * For each of 5 horizontal positions within 30 m of home, climbs through
 * 5 altitude levels (20→30→40→50→60 m), taking a photo and running SIFT
 * at each step.  Up to 25 test shots total; time budget checked at every
 * step — any tests not reached are marked "--" in the result table.
 *
 * Flight path:
 *   ① Arm + takeoff → home @ 20 m
 *   ② At home: climb 20→60 m (5 shots)
 *   ③ fly_to(next_pos, 20 m) — descend + fly simultaneously
 *   ④ At next pos: climb 20→60 m (5 shots)
 *   ⑤ Repeat for all 5 positions
 *   ⑥ RTL at 110 s (hard deadline before 2-min limit)
 *   ⑦ Print 2-D error table  +  best (altitude, position) combo
 *
 * Test positions (all ≤ 30 m from home):
 *   home, 30 m N, 30 m E, 30 m S, 30 m W
 *
 * Photos: result/{lat_r}_{lon_r}-{lat_e}_{lon_e}__{dist}m_pos{name}_alt{alt}m.png
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
#include <limits>
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

static const char*  DEFAULT_PORT      = "/dev/ttyTHS1";
static const int    DEFAULT_BAUD      = 57600;
static const int    DEFAULT_ZOOM      = 19;
static const char*  INDEX_DIR         = "../index";
static const char*  TILE_DIR          = "../tiles";
static const char*  RESULT_DIR        = "result";

static const int    MAX_FLIGHT_SECS   = 110;   // RTL at this elapsed time
static const float  REACH_HORIZ_M    =  3.0f;  // horizontal arrival threshold
static const float  REACH_ALT_M      =  2.5f;  // altitude arrival threshold
static const int    WAIT_TAKEOFF_S   = 35;      // max wait for takeoff altitude
static const int    WAIT_REACHED_S   = 40;      // max wait per fly-to command
static const int    WAIT_ARMED_S     = 12;
static const int    STAB_MS          = 1500;    // stabilise before each photo

static const int    TILE_SIZE        = 256;
static const float  MATCH_RATIO      = 0.75f;
static const int    MIN_INLIERS      = 6;
static const int    TOP_CANDIDATES   = 20;

// ArduCopter custom modes
static const uint32_t COPTER_MODE_GUIDED = 4;

// CSI camera
static const int    CSI_SENSOR_ID    = 0;
static const int    CSI_WIDTH        = 1280;
static const int    CSI_HEIGHT       = 720;
static const int    CSI_FPS          = 30;
static const int    CSI_FLIP         = 0;

static std::atomic<bool> g_shutdown{false};
static void handle_signal(int) { g_shutdown = true; }

// ── test matrix ───────────────────────────────────────────────────────────────

struct Position { double north_m, east_m; const char* name; };

static const Position POSITIONS[] = {
    {   0.0,   0.0, "home" },
    {  30.0,   0.0, "30N"  },
    {   0.0,  30.0, "30E"  },
    { -30.0,   0.0, "30S"  },
    {   0.0, -30.0, "30W"  },
};
static const float ALTITUDES[]  = { 20.0f, 30.0f, 40.0f, 50.0f, 60.0f };
static const int   N_POSITIONS  = (int)(sizeof(POSITIONS) / sizeof(POSITIONS[0]));
static const int   N_ALTITUDES  = (int)(sizeof(ALTITUDES) / sizeof(ALTITUDES[0]));

// ── result table ──────────────────────────────────────────────────────────────

static const double NOT_TESTED = -2.0;   // placeholder: drone did not reach it
static const double NOFIX      = -1.0;   // reached but SIFT returned no match

static double g_results[N_POSITIONS][N_ALTITUDES];   // [pos][alt] = error in m

static void init_results()
{
    for (int pi = 0; pi < N_POSITIONS; ++pi)
        for (int ai = 0; ai < N_ALTITUDES; ++ai)
            g_results[pi][ai] = NOT_TESTED;
}

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
    cv::Mat descs;
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
    if (ranked.empty()) { std::cout << "  No FLANN votes.\n"; return std::nullopt; }

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

// ── vehicle state ─────────────────────────────────────────────────────────────

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

    struct Snap { double alt_m, lat, lon; bool gps_ok, armed; };
    Snap snapshot() {
        std::lock_guard<std::mutex> lk(mtx);
        return {alt_m, lat, lon, gps_ok, armed};
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
    cfsetispeed(&tty, speed); cfsetospeed(&tty, speed);
    tty.c_cflag &= ~PARENB; tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;  tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS; tty.c_cflag |= CREAD | CLOCAL;
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~OPOST;
    tty.c_cc[VMIN] = 0; tty.c_cc[VTIME] = 10;
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
        if (::read(fd, &byte, 1) <= 0) continue;
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
                if (hb.type != MAV_TYPE_GCS)
                    state.update_hb(hb.base_mode, hb.custom_mode);
                break;
            }
            }
        }
    }
}

// ── MAVLink send helpers ──────────────────────────────────────────────────────

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
    mavlink_msg_command_long_pack(1, 200, &msg, 1, 1, cmd, 0,
        p1, p2, p3, p4, p5, p6, p7);
    mav_write(fd, msg);
}

static void heartbeat_sender(int fd)
{
    while (!g_shutdown) {
        send_heartbeat(fd);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// ── flight control ────────────────────────────────────────────────────────────

static void set_guided_mode(int fd)
{
    send_command_long(fd, MAV_CMD_DO_SET_MODE,
        MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, (float)COPTER_MODE_GUIDED);
}

static void cmd_arm(int fd) {
    send_command_long(fd, MAV_CMD_COMPONENT_ARM_DISARM, 1.0f);
}

static void cmd_takeoff(int fd, float alt_m) {
    send_command_long(fd, MAV_CMD_NAV_TAKEOFF, 0,0,0,0,0,0, alt_m);
}

// SET_POSITION_TARGET_GLOBAL_INT — position only (type_mask 0x0FF8)
static void fly_to(int fd, double lat, double lon, float alt_m)
{
    mavlink_message_t msg;
    mavlink_msg_set_position_target_global_int_pack(
        1, 200, &msg, 0, 1, 1,
        MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0x0FF8,
        (int32_t)(lat * 1e7), (int32_t)(lon * 1e7), alt_m,
        0, 0, 0, 0, 0, 0, 0, 0);
    mav_write(fd, msg);
}

static void cmd_rtl(int fd) {
    send_command_long(fd, MAV_CMD_NAV_RETURN_TO_LAUNCH);
}

// ── wait helpers ──────────────────────────────────────────────────────────────

static bool wait_gps(VehicleState& state, int timeout_s = 30)
{
    auto dl = Clock::now() + std::chrono::seconds(timeout_s);
    while (!g_shutdown && Clock::now() < dl) {
        if (state.snapshot().gps_ok) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

static bool wait_armed(VehicleState& state, int timeout_s = WAIT_ARMED_S)
{
    auto dl = Clock::now() + std::chrono::seconds(timeout_s);
    while (!g_shutdown && Clock::now() < dl) {
        if (state.snapshot().armed) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

static bool wait_altitude(VehicleState& state, float target_m,
                           float tol = REACH_ALT_M, int timeout_s = WAIT_TAKEOFF_S)
{
    auto dl = Clock::now() + std::chrono::seconds(timeout_s);
    while (!g_shutdown && Clock::now() < dl) {
        if (std::abs((float)state.snapshot().alt_m - target_m) <= tol) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

static bool wait_reached(VehicleState& state,
                          double tgt_lat, double tgt_lon, float tgt_alt,
                          int timeout_s = WAIT_REACHED_S)
{
    auto dl = Clock::now() + std::chrono::seconds(timeout_s);
    while (!g_shutdown && Clock::now() < dl) {
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

// ── CSI camera ────────────────────────────────────────────────────────────────

static std::string gst_pipeline(int sensor_id, int w, int h, int fps, int flip)
{
    std::ostringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
       << "video/x-raw(memory:NVMM), width=(int)" << w
       << ", height=(int)" << h << ", framerate=(fraction)" << fps << "/1 ! "
       << "nvvidconv flip-method=" << flip << " ! "
       << "video/x-raw, width=(int)" << w << ", height=(int)" << h
       << ", format=(string)BGRx ! videoconvert ! "
       << "video/x-raw, format=(string)BGR ! appsink";
    return ss.str();
}

// ── photo filename ────────────────────────────────────────────────────────────

static std::string build_filename(double real_lat, double real_lon,
                                   double est_lat,  double est_lon,
                                   bool has_fix,    double dist_m,
                                   const char* pos_name, float alt_m)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(7);
    ss << real_lat << "_" << real_lon << "-";
    if (has_fix)
        ss << est_lat << "_" << est_lon
           << "__" << std::setprecision(1) << dist_m << "m";
    else
        ss << "NOFIX_NOFIX__NAm";
    ss << "_pos" << pos_name
       << "_alt" << std::setprecision(0) << alt_m << "m.png";
    return ss.str();
}

// ── summary printer ───────────────────────────────────────────────────────────

static void print_summary()
{
    std::cout << "\n" << std::string(60,'=') << "\n";
    std::cout << "  SURVEY RESULTS  (SIFT error in metres)\n";
    std::cout << std::string(60,'=') << "\n";

    // Header row
    std::cout << std::left << std::setw(8) << "Alt(m)";
    for (int pi = 0; pi < N_POSITIONS; ++pi)
        std::cout << std::setw(10) << POSITIONS[pi].name;
    std::cout << "\n" << std::string(60,'-') << "\n";

    double best_err = std::numeric_limits<double>::max();
    int    best_ai  = -1, best_pi = -1;

    for (int ai = 0; ai < N_ALTITUDES; ++ai) {
        std::cout << std::left << std::setw(8) << (int)ALTITUDES[ai];
        for (int pi = 0; pi < N_POSITIONS; ++pi) {
            double v = g_results[pi][ai];
            std::ostringstream cell;
            if      (v == NOT_TESTED) cell << "--";
            else if (v == NOFIX)      cell << "NOFIX";
            else {
                cell << std::fixed << std::setprecision(1) << v;
                if (v < best_err) { best_err = v; best_ai = ai; best_pi = pi; }
            }
            std::cout << std::setw(10) << cell.str();
        }
        std::cout << "\n";
    }

    std::cout << std::string(60,'=') << "\n";

    // Per-altitude average (across positions that were tested)
    std::cout << "\n  Per-altitude mean error (tested positions only):\n";
    for (int ai = 0; ai < N_ALTITUDES; ++ai) {
        double sum = 0; int cnt = 0;
        for (int pi = 0; pi < N_POSITIONS; ++pi) {
            double v = g_results[pi][ai];
            if (v >= 0.0) { sum += v; cnt++; }
        }
        std::cout << "    " << std::setw(3) << (int)ALTITUDES[ai] << " m : ";
        if (cnt > 0)
            std::cout << std::fixed << std::setprecision(1) << sum/cnt
                      << " m  (n=" << cnt << ")";
        else
            std::cout << "no valid data";
        std::cout << "\n";
    }

    if (best_ai >= 0 && best_pi >= 0) {
        std::cout << "\n  BEST : " << (int)ALTITUDES[best_ai] << " m"
                  << " at " << POSITIONS[best_pi].name
                  << "  (error = " << std::setprecision(1) << best_err << " m)\n";
    } else {
        std::cout << "\n  No valid SIFT fix obtained in this survey.\n";
    }
    std::cout << std::string(60,'=') << "\n";
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

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    init_results();
    fs::create_directories(RESULT_DIR);

    std::cout << "Result photos  -> " << fs::absolute(RESULT_DIR) << "\n";
    std::cout << "Max flight     : " << MAX_FLIGHT_SECS << " s\n";
    std::cout << "Test positions : home / 30N / 30E / 30S / 30W\n";
    std::cout << "Test altitudes : 20 / 30 / 40 / 50 / 60 m\n\n";

    // ── load tile index + FLANN ───────────────────────────────────────────────
    std::string index_path = std::string(INDEX_DIR) + "/sift_index_z"
                             + std::to_string(zoom) + ".bin";
    std::cout << "Loading index  : " << index_path << " … " << std::flush;
    std::vector<TileEntry> index;
    try {
        index = load_index(index_path);
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n"
                  << "Run export_index.py first.\n";
        return 1;
    }
    std::cout << index.size() << " tiles.\n";
    FlannIndex fi = build_flann(index);

    // ── serial + threads ──────────────────────────────────────────────────────
    std::cout << "MAVLink serial : " << port << " @ " << baud << " … " << std::flush;
    int serial_fd;
    try {
        serial_fd = open_serial(port, baud);
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n"; return 1;
    }
    std::cout << "ok.\n";

    VehicleState state;
    std::thread mav_th(mavlink_reader, serial_fd, std::ref(state));
    std::thread hb_th(heartbeat_sender, serial_fd);

    // ── camera ────────────────────────────────────────────────────────────────
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
        std::cerr << "\nERROR: No GPS within 60 s.\n";
        g_shutdown = true;
        mav_th.join(); hb_th.join();
        close(serial_fd);
        return 1;
    }
    std::cout << " ok.\n";

    double home_lat, home_lon;
    {
        auto s = state.snapshot();
        home_lat = s.lat;
        home_lon = s.lon;
    }
    std::cout << "Home position  : " << std::fixed << std::setprecision(7)
              << home_lat << ", " << home_lon << "\n\n";

    // ── flight ────────────────────────────────────────────────────────────────
    bool airborne = false;

    auto abort_flight = [&](const char* reason) {
        std::cerr << "ABORT: " << reason << "\n";
        if (airborne) { std::cout << "RTL for safety.\n"; cmd_rtl(serial_fd); }
        g_shutdown = true;
    };

    // GUIDED mode
    std::cout << std::string(60,'-') << "\n";
    std::cout << "Setting GUIDED mode …\n";
    set_guided_mode(serial_fd);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    if (g_shutdown) goto cleanup;

    // Arm
    std::cout << "Arming …\n";
    cmd_arm(serial_fd);
    if (!wait_armed(state)) { abort_flight("arm timeout"); goto cleanup; }
    std::cout << "Armed.\n";

    // Takeoff to first altitude (20 m)
    {
        float first_alt = ALTITUDES[0];
        std::cout << "Takeoff to " << std::fixed << std::setprecision(0)
                  << first_alt << " m …\n";
        cmd_takeoff(serial_fd, first_alt);
        if (!wait_altitude(state, first_alt)) {
            abort_flight("takeoff altitude not reached");
            goto cleanup;
        }
        airborne = true;
        std::cout << "Airborne. alt=" << std::setprecision(1)
                  << state.snapshot().alt_m << " m\n\n";
    }

    // ── main survey loop ──────────────────────────────────────────────────────
    {
        auto flight_start = Clock::now();

        // Helper lambda: seconds elapsed since takeoff
        auto elapsed_s = [&]() {
            return std::chrono::duration<double>(Clock::now() - flight_start).count();
        };

        for (int pi = 0; pi < N_POSITIONS && !g_shutdown; ++pi) {

            // Compute absolute GPS target for this position
            auto [pos_lat, pos_lon] = offset_latlon(home_lat, home_lon,
                POSITIONS[pi].north_m, POSITIONS[pi].east_m);

            // Travel to this position at 20 m altitude.
            // For home (pi==0) we are already there after takeoff.
            // For subsequent positions: issue fly_to at 20 m so the drone descends
            // and moves horizontally at the same time.
            if (pi > 0) {
                std::cout << std::string(60,'-') << "\n";
                std::cout << "Flying to " << POSITIONS[pi].name
                          << " @ 20 m  (elapsed " << std::setprecision(0)
                          << elapsed_s() << " s / " << MAX_FLIGHT_SECS << " s)\n";

                if (elapsed_s() >= MAX_FLIGHT_SECS) {
                    std::cout << "Time budget reached — ending survey.\n";
                    break;
                }

                fly_to(serial_fd, pos_lat, pos_lon, ALTITUDES[0]);
                if (!wait_reached(state, pos_lat, pos_lon, ALTITUDES[0])) {
                    std::cout << "  Could not reach " << POSITIONS[pi].name
                              << " in time — skipping.\n";
                    continue;
                }
            }

            std::cout << std::string(60,'=') << "\n";
            std::cout << "Position : " << POSITIONS[pi].name
                      << "  (offset " << POSITIONS[pi].north_m << " N, "
                      << POSITIONS[pi].east_m << " E)\n";
            std::cout << std::string(60,'=') << "\n";

            // Sweep altitudes 20 → 60 m (ascending — faster than descending)
            for (int ai = 0; ai < N_ALTITUDES && !g_shutdown; ++ai) {

                float tgt_alt = ALTITUDES[ai];

                if (elapsed_s() >= MAX_FLIGHT_SECS) {
                    std::cout << "Time budget reached — ending survey.\n";
                    goto rtl;
                }

                std::cout << "  Climbing to " << std::setprecision(0) << tgt_alt << " m"
                          << "  (budget left: " << std::setprecision(0)
                          << (MAX_FLIGHT_SECS - elapsed_s()) << " s)\n";

                // Fly to same horizontal position at new altitude
                fly_to(serial_fd, pos_lat, pos_lon, tgt_alt);
                if (!wait_reached(state, pos_lat, pos_lon, tgt_alt)) {
                    std::cout << "  Altitude " << (int)tgt_alt
                              << " m not reached — skipping.\n";
                    continue;
                }

                // Stabilise
                std::this_thread::sleep_for(std::chrono::milliseconds(STAB_MS));

                // Read true GPS
                auto s = state.snapshot();
                std::cout << "  alt=" << std::setprecision(1) << s.alt_m
                          << " m   Real GPS: " << std::setprecision(7)
                          << s.lat << ", " << s.lon << "\n";

                // Capture
                cv::Mat frame;
                if (!cap.read(frame) || frame.empty()) {
                    std::cerr << "  Camera read failed — skipping.\n";
                    g_results[pi][ai] = NOFIX;
                    continue;
                }

                // SIFT
                auto res = localize_frame(frame, index, fi, zoom);
                bool   has_fix = res.has_value();
                double est_lat = 0, est_lon = 0, dist_m = -1.0;

                if (has_fix) {
                    est_lat = res->lat;
                    est_lon = res->lon;
                    dist_m  = haversine_m(s.lat, s.lon, est_lat, est_lon);
                    std::cout << "  Est. GPS: " << std::setprecision(7)
                              << est_lat << ", " << est_lon
                              << "  error=" << std::setprecision(1) << dist_m << " m"
                              << "  inliers=" << res->inliers << "\n";
                    g_results[pi][ai] = dist_m;
                } else {
                    std::cout << "  NOFIX\n";
                    g_results[pi][ai] = NOFIX;
                }

                // Save photo
                std::string fname = build_filename(s.lat, s.lon,
                    est_lat, est_lon, has_fix, dist_m,
                    POSITIONS[pi].name, tgt_alt);
                cv::imwrite(std::string(RESULT_DIR) + "/" + fname, frame);
                std::cout << "  Saved: " << fname << "\n";
            }
        }
    }

rtl:
    std::cout << "\n" << std::string(60,'-') << "\n";
    std::cout << "Survey complete — RTL.\n";
    cmd_rtl(serial_fd);

    print_summary();

cleanup:
    g_shutdown = true;
    mav_th.join();
    hb_th.join();
    cap.release();
    close(serial_fd);
    return 0;
}
