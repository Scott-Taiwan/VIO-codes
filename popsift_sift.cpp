/*
 * popsift_sift.cpp — GPU SIFT via PopSift.
 */

#include "popsift_sift.h"

#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/features.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

// PopSift and SiftJob live in the global namespace; popsift:: is for Features/Config
static PopSift* g_popsift = nullptr;

void init_popsift()
{
    popsift::Config cfg;
    // Match OpenCV SIFT's Lowe parameters
    cfg.setMode(popsift::Config::PopSift);
    cfg.setOctaves(-1);            // auto-select from image size
    cfg.setLevels(3);
    cfg.setThreshold(0.04f / 3.0f);
    cfg.setEdgeLimit(10.0f);
    // Scale output to the same [0, ~512] range as OpenCV SIFT (multiplier = 2^9 = 512)
    cfg.setNormalizationMultiplier(9);

    g_popsift = new PopSift(cfg, popsift::Config::ExtractingMode);

    // Pre-allocate pyramid memory NOW (before FLANN consumes shared NVMM).
    // Must be >= the largest image we'll process (768×768 = 3×3 tile composite).
    // PopSift only reallocates when a larger image arrives, so this prevents
    // mid-run OOM after FLANN has loaded.
    cv::Mat dummy(768, 768, CV_8UC1, cv::Scalar(128));
    SiftJob* warm_job = g_popsift->enqueue(768, 768, dummy.ptr<unsigned char>());
    popsift::FeaturesHost* warm_f = warm_job->getHost();
    delete warm_f;
    delete warm_job;
}

void shutdown_popsift()
{
    delete g_popsift;
    g_popsift = nullptr;
}

void popsift_detect_compute(const cv::Mat& gray_u8,
                             std::vector<cv::KeyPoint>& kps,
                             cv::Mat& descs,
                             int max_kp)
{
    if (!g_popsift)
        throw std::runtime_error("PopSift not initialized — call init_popsift() first");
    if (gray_u8.type() != CV_8UC1)
        throw std::runtime_error("popsift_detect_compute: input must be CV_8UC1");

    SiftJob* job = g_popsift->enqueue(gray_u8.cols, gray_u8.rows,
                                       gray_u8.ptr<unsigned char>());
    popsift::FeaturesHost* features = job->getHost();
    delete job;

    kps.clear();
    descs = cv::Mat();

    if (!features || features->getDescriptorCount() == 0) {
        delete features;
        return;
    }

    int n_ext  = features->getFeatureCount();
    int n_desc = features->getDescriptorCount();

    kps.reserve(n_desc);
    cv::Mat out(n_desc, 128, CV_32F);
    int row = 0;

    const popsift::Feature* feat = features->getFeatures();
    for (int i = 0; i < n_ext; ++i) {
        const popsift::Feature& f = feat[i];
        for (int o = 0; o < f.num_ori; ++o) {
            float angle = f.orientation[o] * (180.0f / (float)M_PI);
            kps.push_back(cv::KeyPoint(f.xpos, f.ypos, 4.0f * f.sigma, angle));
            std::memcpy(out.ptr<float>(row++), f.desc[o]->features, 128 * sizeof(float));
        }
    }
    delete features;

    descs = out.rowRange(0, row);

    if (max_kp > 0 && (int)kps.size() > max_kp) {
        kps.resize(max_kp);
        descs = descs.rowRange(0, max_kp).clone();
    }
}
