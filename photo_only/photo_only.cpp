/*
 * photo_only.cpp — Visual localization accuracy measurement tool
 *
 * Connects to Pixhawk via MAVLink serial. Monitors relative altitude.
 * When the drone is above CAPTURE_ALT_MIN (30 m), captures a CSI camera
 * frame every CAPTURE_INTERVAL seconds and runs SIFT localization against
 * the ESRI tile index.
 *
 * For each frame the program records:
 *   - True GPS  : read from Pixhawk GLOBAL_POSITION_INT
 *   - Est. GPS  : computed by SIFT visual localization
 *   - Distance  : haversine metres between the two
 *
 * Everything is stored only as a photo filename under result/:
 *   {lat_real}_{lon_real}-{lat_est}_{lon_est}__{dist}m.png
 *   {lat_real}_{lon_real}-NOFIX_NOFIX__NAm.png
 *
 * No navigation commands and no GPS_INPUT are sent to Pixhawk.
 * This tool measures SIFT accuracy only.
 *
 * Build:
 *   cd photo_only
 *   make
 *
 * Run:
 *   ./photo_only
 *   ./photo_only --port /dev/ttyUSB0 --baud 115200 --zoom 19
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

// ── config ────────────────────────────────────────────────────────────────────

static const char*  DEFAULT_PORT     = "/dev/ttyTHS1";
static const int    DEFAULT_BAUD     = 57600;
static const int    DEFAULT_ZOOM     = 19;
static const char*  INDEX_DIR        = "../index";
static const char*  TILE_DIR         = "../tiles";
static const char*  RESULT_DIR       = "result";

static const double CAPTURE_ALT_MIN  = 30.0;   // m — start capturing above this
static const double CAPTURE_ALT_STOP = 20.0;   // m — stop capturing below this
static const double CAPTURE_INTERVAL = 5.0;    // s — between frames

static const int    TILE_SIZE        = 256;
static const float  MATCH_RATIO      = 0.75f;
static const int    MIN_INLIERS      = 6;
static const int    TOP_CANDIDATES   = 20;

// CSI camera
static const int    CSI_SENSOR_ID    = 0;
static const int    CSI_WIDTH        = 1280;
static const int    CSI_HEIGHT       = 720;
static const int    CSI_FPS          = 30;
static const int    CSI_FLIP         = 0;   // 0 = none, 2 = 180°

static std::atomic<bool> g_shutdown{false};

// ── signal handler ────────────────────────────────────────────────────────────

static void handle_signal(int) { g_shutdown = true; }

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
                                   bool has_fix,    double dist_m = -1.0)
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
                      << " [--port /dev/ttyTHS1] [--baud 57600] [--zoom 18]\n"
                      << "  Photos saved to: " << RESULT_DIR << "/\n";
            return 1;
        }
    }

    signal(SIGINT,  handle_signal);
    signal(SIGTERM, handle_signal);

    fs::create_directories(RESULT_DIR);
    std::cout << "Result photos  → " << fs::absolute(RESULT_DIR) << "\n";
    std::cout << "Capture above  : " << CAPTURE_ALT_MIN  << " m\n";
    std::cout << "Stop below     : " << CAPTURE_ALT_STOP << " m\n";
    std::cout << "Interval       : " << CAPTURE_INTERVAL << " s\n";

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

    // ── open serial + start MAVLink reader ────────────────────────────────────
    std::cout << "MAVLink serial : " << port << " @ " << baud << " baud … " << std::flush;
    int serial_fd;
    try {
        serial_fd = open_serial(port, baud);
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
    std::cout << "ok.\n";

    VehicleState state;
    std::thread mav_thread(mavlink_reader, serial_fd, std::ref(state));

    // Wait for first position message from Pixhawk
    std::cout << "Waiting for Pixhawk GPS …" << std::flush;
    while (!g_shutdown) {
        if (state.snapshot().gps_ok) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "." << std::flush;
    }
    std::cout << " ok.\n";

    // ── open CSI camera ───────────────────────────────────────────────────────
    std::string pipeline = gst_pipeline(CSI_SENSOR_ID, CSI_WIDTH, CSI_HEIGHT,
                                        CSI_FPS, CSI_FLIP);
    std::cout << "Camera         : " << pipeline << "\n";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open CSI camera.\n";
        g_shutdown = true;
        mav_thread.join();
        close(serial_fd);
        return 1;
    }
    std::cout << "Camera         : opened.\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Waiting for altitude >= " << CAPTURE_ALT_MIN << " m …\n";

    // ── main loop ─────────────────────────────────────────────────────────────
    bool capturing_active = false;
    auto last_capture     = std::chrono::steady_clock::now()
                            - std::chrono::seconds(100);

    while (!g_shutdown) {
        auto [alt, lat, lon, gps_ok] = state.snapshot();

        // altitude hysteresis: start at 30 m, stop at 20 m
        if (!capturing_active && alt >= CAPTURE_ALT_MIN) {
            std::cout << "\nAltitude " << std::fixed << std::setprecision(1)
                      << alt << " m — capture started.\n";
            capturing_active = true;
        }
        if (capturing_active && alt < CAPTURE_ALT_STOP) {
            std::cout << "\nAltitude " << std::fixed << std::setprecision(1)
                      << alt << " m < " << CAPTURE_ALT_STOP
                      << " m — capture stopped.\n";
            capturing_active = false;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_capture).count();

        if (capturing_active && elapsed >= CAPTURE_INTERVAL) {
            last_capture = now;

            auto [snap_alt, real_lat, real_lon, snap_ok] = state.snapshot();
            std::cout << "\n" << std::string(60, '-') << "\n";
            std::cout << "Capture  alt=" << std::fixed << std::setprecision(1)
                      << snap_alt << " m\n";

            if (!snap_ok || real_lat == 0.0) {
                std::cout << "  Pixhawk GPS not ready — skipping.\n";
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

            bool   has_fix = result.has_value();
            double est_lat = 0, est_lon = 0, dist_m = -1.0;

            if (has_fix) {
                est_lat = result->lat;
                est_lon = result->lon;
                dist_m  = haversine_m(real_lat, real_lon, est_lat, est_lon);
                std::cout << "  Est. GPS : " << std::setprecision(7)
                          << est_lat << ", " << est_lon << "\n";
                std::cout << "  Inliers  : " << result->inliers
                          << "   Error: " << std::setprecision(1) << dist_m << " m\n";
            } else {
                std::cout << "  Est. GPS : NOFIX\n";
            }

            // save photo — filename carries all the data
            std::string fname = build_filename(real_lat, real_lon,
                                               est_lat, est_lon, has_fix, dist_m);
            std::string fpath = std::string(RESULT_DIR) + "/" + fname;
            cv::imwrite(fpath, frame);
            std::cout << "  Saved    : " << fname << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── cleanup ───────────────────────────────────────────────────────────────
    std::cout << "\nShutting down …\n";
    cap.release();
    mav_thread.join();
    close(serial_fd);
    std::cout << "Done.\n";
    return 0;
}
