/*
 * drone_sim.cpp — Simulates drone_localize without real hardware.
 *
 * Replaces:
 *   • Pixhawk serial   → simulated altitude ramp (0 → 55 m) + fake GPS drift
 *   • CSI camera       → cycles through real test images in ../
 *
 * Uses the real tile index + real SIFT localization pipeline, so the
 * matching results reflect actual performance.
 *
 * Build:
 *   make sim
 *
 * Run:
 *   ./drone_sim
 *   ./drone_sim --zoom 18 --images ../tennis.JPG ../school2.JPG ../circle.JPG
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

// ── config (mirrors drone_localize.cpp) ───────────────────────────────────────

static const char*  INDEX_DIR        = "../index";
static const char*  TILE_DIR         = "../tiles";
static const char*  PHOTO_DIR        = "photo_obtained";
static const int    DEFAULT_ZOOM     = 18;
static const int    TILE_SIZE        = 256;
static const float  MATCH_RATIO      = 0.75f;
static const int    MIN_INLIERS      = 6;
static const int    TOP_CANDIDATES   = 20;
static const double TAKEOFF_ALT      = 50.0;
static const double CAPTURE_INTERVAL = 5.0;   // simulated seconds between shots

// Default test images (relative to gpsless_mapping/)
static const std::vector<std::string> DEFAULT_IMAGES = {
    "../tennis.JPG",
    "../school2.JPG",
    "../tennis2.JPG",
    "../circle.JPG",
};

// Simulated GPS start point (Taipei area — matches tile coverage)
static const double SIM_LAT_START = 25.0610000;
static const double SIM_LON_START = 121.4710000;

// ── helpers: log with elapsed time ───────────────────────────────────────────

static clk::time_point g_t0;

static void log(const std::string& msg, const std::string& tag = "   ")
{
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  clk::now() - g_t0).count();
    int s  = (int)(ms / 1000);
    int ms3 = (int)(ms % 1000);
    std::cout << "[" << std::setw(3) << s << "." << std::setw(3)
              << std::setfill('0') << ms3 << std::setfill(' ') << "s] "
              << tag << " " << msg << "\n" << std::flush;
}

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
    double phi1 = lat1*M_PI/180.0, phi2 = lat2*M_PI/180.0;
    double dphi = (lat2-lat1)*M_PI/180.0, dlam = (lon2-lon1)*M_PI/180.0;
    double a = std::sin(dphi/2)*std::sin(dphi/2)
             + std::cos(phi1)*std::cos(phi2)*std::sin(dlam/2)*std::sin(dlam/2);
    return 2.0*R*std::asin(std::sqrt(a));
}

// ── tile index ────────────────────────────────────────────────────────────────

struct TileEntry { int tile_x, tile_y, zoom; cv::Mat descs; };

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

// ── FLANN ─────────────────────────────────────────────────────────────────────

struct FlannIndex {
    cv::Ptr<cv::FlannBasedMatcher> matcher;
    std::vector<int> offsets;
    int n_tiles;
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
    fi.matcher = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(5),
        cv::makePtr<cv::flann::SearchParams>(50));
    fi.matcher->add(all);
    fi.matcher->train();
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

// ── stitch + direct match ─────────────────────────────────────────────────────

static cv::Mat stitch_tiles(int cx, int cy, int zoom, const char* tile_dir, int r=1)
{
    int side = 2*r+1;
    cv::Mat canvas(side*TILE_SIZE, side*TILE_SIZE, CV_8UC3, cv::Scalar(0,0,0));
    for (int dx = 0; dx < side; ++dx)
        for (int dy = 0; dy < side; ++dy) {
            int tx = cx-r+dx, ty = cy-r+dy;
            std::string p = std::string(tile_dir)+"/"+std::to_string(zoom)
                          + "/"+std::to_string(tx)+"/"+std::to_string(ty)+".png";
            cv::Mat tile = cv::imread(p);
            if (!tile.empty())
                tile.copyTo(canvas(cv::Rect(dx*TILE_SIZE, dy*TILE_SIZE, TILE_SIZE, TILE_SIZE)));
        }
    return canvas;
}

struct MatchResult { double lat, lon; int inliers; };

static std::optional<MatchResult> direct_match(
    const cv::Mat& qgray, int cx, int cy, int zoom,
    const char* tile_dir, int radius=1, int min_inliers=MIN_INLIERS)
{
    cv::Mat composite = stitch_tiles(cx, cy, zoom, tile_dir, radius);
    cv::Mat cgray;
    cv::cvtColor(composite, cgray, cv::COLOR_BGR2GRAY);

    auto sift = cv::SIFT::create(3000);
    std::vector<cv::KeyPoint> kps_c, kps_q;
    cv::Mat descs_c, descs_q;
    sift->detectAndCompute(cgray,  cv::noArray(), kps_c, descs_c);
    sift->detectAndCompute(qgray,  cv::noArray(), kps_q, descs_q);
    if (descs_c.empty() || (int)kps_c.size() < min_inliers) return std::nullopt;
    if (descs_q.empty() || (int)kps_q.size() < min_inliers) return std::nullopt;

    auto bf = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> raw;
    bf->knnMatch(descs_q, descs_c, raw, 2);
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

    float qw=(float)qgray.cols, qh=(float)qgray.rows;
    std::vector<cv::Point2f> corners_in={{0,0},{qw,0},{qw,qh},{0,qh}};
    std::vector<cv::Point2f> corners_out;
    cv::perspectiveTransform(corners_in, corners_out, H);
    float area = cv::contourArea(corners_out);
    if (area < qw*qh*0.05f || area > qw*qh*50.0f) return std::nullopt;
    if (!cv::isContourConvex(std::vector<cv::Point2f>(corners_out))) return std::nullopt;

    std::vector<cv::Point2f> ctr_in={{qw/2,qh/2}}, ctr_out;
    cv::perspectiveTransform(ctr_in, ctr_out, H);
    float cpx=ctr_out[0].x, cpy=ctr_out[0].y;
    int comp_size=(2*radius+1)*TILE_SIZE, margin=TILE_SIZE;
    if (cpx<-margin||cpx>comp_size+margin||cpy<-margin||cpy>comp_size+margin)
        return std::nullopt;

    int tile_dx=(int)(cpx/TILE_SIZE), tile_dy=(int)(cpy/TILE_SIZE);
    float lpx=cpx-tile_dx*TILE_SIZE, lpy=cpy-tile_dy*TILE_SIZE;
    int xo=cx-radius, yo=cy-radius;
    auto [lat,lon]=pixel_to_latlon(lpx,lpy,xo+tile_dx,yo+tile_dy,zoom);
    return MatchResult{lat,lon,inliers};
}

// ── filename ──────────────────────────────────────────────────────────────────

static std::string build_filename(double rlat, double rlon,
                                   double elat, double elon, bool ok)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(7);
    ss << rlat << "_" << rlon << "-";
    if (ok) ss << elat << "_" << elon;
    else    ss << "NOFIX_NOFIX";
    ss << ".png";
    return ss.str();
}

// ── simulated vehicle state ───────────────────────────────────────────────────

struct SimVehicle {
    double alt_m = 0.0;
    double lat   = SIM_LAT_START;
    double lon   = SIM_LON_START;

    // Call every ~0.5 s of simulated time; step is 0–100 (0=ground,100=cruise)
    void tick(int step) {
        // Altitude profile: 0→55 m over 20 steps, then cruise with ±1 m noise
        if (step < 20)
            alt_m = step * 2.75;
        else
            alt_m = 55.0 + (std::sin(step * 0.3) * 1.2);

        // Slight GPS drift (simulate wind / positioning)
        lat += 0.000002 * std::sin(step * 0.7);
        lon += 0.000003 * std::cos(step * 0.5);
    }
};

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    g_t0 = clk::now();

    // parse args
    int zoom = DEFAULT_ZOOM;
    std::vector<std::string> images = DEFAULT_IMAGES;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i],"--zoom") && i+1<argc) zoom=std::stoi(argv[++i]);
        else if (!strcmp(argv[i],"--images")) {
            images.clear();
            while (i+1<argc && argv[i+1][0]!='-') images.push_back(argv[++i]);
        }
    }

    std::cout << "\n";
    std::cout << std::string(60,'=') << "\n";
    std::cout << "  drone_localize  —  SIMULATION MODE\n";
    std::cout << std::string(60,'=') << "\n\n";

    // ── startup ───────────────────────────────────────────────────────────────
    fs::create_directories(PHOTO_DIR);

    log("SIFT backend   : CPU (cv::SIFT)", "INF");
    log("Photo output   : " + std::string(PHOTO_DIR), "INF");
    log("Tile zoom      : " + std::to_string(zoom), "INF");
    log("Test images    : " + std::to_string(images.size()) + " files", "INF");

    std::string index_path = std::string(INDEX_DIR)+"/sift_index_z"
                             +std::to_string(zoom)+".bin";
    log("Loading index  : " + index_path, "INF");
    std::vector<TileEntry> index;
    try {
        index = load_index(index_path);
    } catch (const std::exception& e) {
        log(std::string("ERROR: ") + e.what(), "!!!");
        return 1;
    }
    log("Tile index     : " + std::to_string(index.size()) + " tiles loaded", "INF");

    log("Building FLANN KD-tree (one-time cost) …", "INF");
    auto fi = build_flann(index);
    log("FLANN ready.", "INF");

    // ── simulate serial connect + GPS fix ─────────────────────────────────────
    std::cout << "\n";
    log("Connecting to Pixhawk [SIMULATED]", "SIM");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    log("Heartbeat OK — system=1 component=1 [SIMULATED]", "SIM");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    log("GPS fix obtained: lat=" + [&]{
        std::ostringstream s;
        s << std::fixed << std::setprecision(7) << SIM_LAT_START
          << "  lon=" << SIM_LON_START; return s.str();}() + "  [SIMULATED]", "SIM");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    log("CSI camera opened [SIMULATED — using file images]", "SIM");

    // ── altitude ramp simulation ──────────────────────────────────────────────
    std::cout << "\n";
    std::cout << std::string(60,'-') << "\n";
    log("Waiting for takeoff (altitude >= " +
        [&]{ std::ostringstream s; s<<std::fixed<<std::setprecision(1)<<TAKEOFF_ALT;
             return s.str();}() + " m) …", "INF");
    std::cout << std::string(60,'-') << "\n";

    SimVehicle vehicle;
    int step = 0;
    while (vehicle.alt_m < TAKEOFF_ALT) {
        vehicle.tick(step++);
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(1)
            << "alt = " << std::setw(5) << vehicle.alt_m << " m"
            << "   lat=" << std::setprecision(7) << vehicle.lat
            << "  lon=" << vehicle.lon;
        log(msg.str(), "SIM");
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }

    std::cout << "\n";
    log("*** TAKEOFF CONFIRMED — altitude " +
        [&]{ std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<vehicle.alt_m;
             return s.str();}() + " m ***", ">>>");
    log("Starting photo capture every " +
        [&]{ std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<CAPTURE_INTERVAL;
             return s.str();}() + " s", ">>>");

    // ── capture loop ──────────────────────────────────────────────────────────
    int photo_num = 0;
    int n_images  = (int)images.size();

    for (int cap = 0; cap < n_images; ++cap) {
        // simulate inter-capture interval (shortened for demo)
        std::cout << "\n";
        log("(waiting " + [&]{ std::ostringstream s;
            s<<std::fixed<<std::setprecision(1)<<CAPTURE_INTERVAL;
            return s.str();}() + " s until next capture …)", "   ");
        for (int w = 0; w < 5; ++w) {
            vehicle.tick(step++);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        // ── capture event ──────────────────────────────────────────────────
        vehicle.tick(step++);
        double real_lat = vehicle.lat;
        double real_lon = vehicle.lon;

        std::cout << std::string(60,'-') << "\n";
        std::ostringstream hdr;
        hdr << "=== Capture #" << (cap+1) << "/" << n_images
            << "   alt=" << std::fixed << std::setprecision(1) << vehicle.alt_m << " m ===";
        log(hdr.str(), ">>>");

        std::ostringstream gps_msg;
        gps_msg << std::fixed << std::setprecision(7)
                << "Real GPS  : " << real_lat << ",  " << real_lon
                << "  [from Pixhawk SIM]";
        log(gps_msg.str(), "GPS");

        // load simulated "camera frame" from file
        const std::string& img_path = images[cap % n_images];
        log("Camera frame  : " + img_path + "  [SIMULATED]", "CAM");
        cv::Mat frame = cv::imread(img_path);
        if (frame.empty()) {
            log("Cannot read image — skipping.", "!!!");
            continue;
        }
        std::ostringstream sz_msg;
        sz_msg << "Frame size    : " << frame.cols << "×" << frame.rows << " px";
        log(sz_msg.str(), "CAM");

        // ── SIFT Phase 1 ───────────────────────────────────────────────────
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        auto sift = cv::SIFT::create(2000);
        std::vector<cv::KeyPoint> kps_q;
        cv::Mat descs_q;
        auto t0 = clk::now();
        sift->detectAndCompute(gray, cv::noArray(), kps_q, descs_q);
        long sift_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           clk::now()-t0).count();

        std::ostringstream feat_msg;
        feat_msg << "Query features: " << kps_q.size()
                 << " keypoints  (" << sift_ms << " ms)";
        log(feat_msg.str(), "P1 ");

        if (descs_q.empty() || (int)kps_q.size() < 5) {
            log("Too few features — NOFIX", "!!!");
            std::string fname = build_filename(real_lat, real_lon, 0, 0, false);
            cv::imwrite(std::string(PHOTO_DIR)+"/"+fname, frame);
            log("Saved: " + fname, "SAV");
            continue;
        }

        auto t1 = clk::now();
        auto ranked = flann_vote(descs_q, fi);
        long vote_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           clk::now()-t1).count();

        std::ostringstream vote_msg;
        vote_msg << "FLANN vote    : " << ranked.size()
                 << " tiles matched  (" << vote_ms << " ms)";
        log(vote_msg.str(), "P1 ");

        if (!ranked.empty()) {
            log("Top candidates:", "P1 ");
            for (int k = 0; k < std::min((int)ranked.size(), 5); ++k) {
                auto [ti, votes] = ranked[k];
                std::ostringstream cand;
                cand << "    #" << k+1 << "  tile ("
                     << index[ti].tile_x << "," << index[ti].tile_y
                     << ")  votes=" << votes;
                log(cand.str(), "   ");
            }
        }

        // ── SIFT Phase 2 ───────────────────────────────────────────────────
        std::optional<MatchResult> result;
        int n_try = std::min((int)ranked.size(), TOP_CANDIDATES);
        for (int k = 0; k < n_try && !result; ++k) {
            int ti = ranked[k].first;
            int cx = index[ti].tile_x, cy = index[ti].tile_y;
            std::ostringstream try_msg;
            try_msg << "Trying tile (" << cx << "," << cy
                    << ") votes=" << ranked[k].second << " …";
            log(try_msg.str(), "P2 ");
            auto t2 = clk::now();
            result = direct_match(gray, cx, cy, zoom, TILE_DIR, 1, MIN_INLIERS);
            long dm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             clk::now()-t2).count();
            if (result) {
                std::ostringstream ok_msg;
                ok_msg << "Match! inliers=" << result->inliers
                       << "  (" << dm_ms << " ms)";
                log(ok_msg.str(), "P2 ");
            } else {
                std::ostringstream fail_msg;
                fail_msg << "No match  (" << dm_ms << " ms)";
                log(fail_msg.str(), "P2 ");
            }
        }

        // ── result ────────────────────────────────────────────────────────
        std::cout << "\n";
        bool has_fix = result.has_value();
        double est_lat = 0, est_lon = 0;
        if (has_fix) {
            est_lat = result->lat;
            est_lon = result->lon;
            double err = haversine_m(real_lat, real_lon, est_lat, est_lon);

            std::ostringstream res_msg;
            res_msg << std::fixed << std::setprecision(7)
                    << "Est. GPS  : " << est_lat << ",  " << est_lon;
            log(res_msg.str(), "GPS");
            std::ostringstream err_msg;
            err_msg << "RANSAC inliers: " << result->inliers
                    << "   Error from real GPS: "
                    << std::fixed << std::setprecision(1) << err << " m";
            log(err_msg.str(), "GPS");
        } else {
            log("Est. GPS  : NOFIX (no tile matched)", "!!!");
        }

        std::string fname = build_filename(real_lat, real_lon,
                                            est_lat, est_lon, has_fix);
        std::string fpath = std::string(PHOTO_DIR) + "/" + fname;
        cv::imwrite(fpath, frame);
        log("Saved     : " + fpath, "SAV");
        ++photo_num;
    }

    // ── simulate landing ──────────────────────────────────────────────────────
    std::cout << "\n" << std::string(60,'-') << "\n";
    log("Mission complete — simulating descent …", "SIM");
    for (int d = 0; d < 8; ++d) {
        vehicle.alt_m = std::max(0.0, vehicle.alt_m - 7.0);
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(1)
            << "alt = " << std::setw(5) << vehicle.alt_m << " m";
        log(msg.str(), "SIM");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    log("Altitude < 10 m — capture paused (landing detected).", "INF");
    log("Shutting down … camera released.", "INF");

    std::cout << "\n" << std::string(60,'=') << "\n";
    std::cout << "  Simulation complete — " << photo_num << " photo(s) saved to "
              << PHOTO_DIR << "/\n";
    std::cout << std::string(60,'=') << "\n\n";

    // list saved photos
    std::cout << "Saved files:\n";
    for (auto& entry : fs::directory_iterator(PHOTO_DIR)) {
        if (entry.path().extension() == ".png")
            std::cout << "  " << entry.path().filename().string() << "\n";
    }
    std::cout << "\n";
    return 0;
}
