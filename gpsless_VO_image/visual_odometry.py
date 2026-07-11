#!/usr/bin/env python3
"""
GPS-less visual odometry (VO) for a downward-facing drone camera.

Estimates drone displacement between successive photos by matching features,
fitting a 2D similarity transform (rotation + scale + translation) with
RANSAC, and converting the pixel-space translation to meters using the
per-frame altitude (from a rangefinder/barometer) and the camera's known
ground footprint.

This gives *relative* displacement in the camera's own image frame (no
compass/heading fusion), which is exactly what VO/dead-reckoning provides:
chain enough of these together (with the estimated inter-frame rotation)
and you get a trajectory anchored at the first frame, not absolute GPS
coordinates.

Usage:
    python3 visual_odometry.py photo_obtained/
    python3 visual_odometry.py img1.png img2.png img3.png
    python3 visual_odometry.py photo_obtained/ --altitude 60,60,60
"""
import argparse
import math
import os
import re
import sys

import cv2
import numpy as np

import config

FILENAME_ALT_RE = re.compile(r'__[A-Za-z0-9]*__(\d+(?:\.\d+)?)m', re.IGNORECASE)
FILENAME_GPS_RE = re.compile(r'^(-?\d+\.\d+)_(-?\d+\.\d+)-')


# ── Input handling ────────────────────────────────────────────────────────

def list_images(paths):
    """Expand directories, then order frames by capture time (mtime).

    On the real drone, feed this script the photos in capture order (or
    give filenames that sort chronologically); mtime is only a convenience
    for this offline test set, since the GPS coordinates baked into these
    filenames are NOT in flight order.
    """
    files = []
    for p in paths:
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files.append(os.path.join(p, name))
        else:
            files.append(p)
    files.sort(key=os.path.getmtime)
    return files


def parse_altitude_m(path):
    m = FILENAME_ALT_RE.search(os.path.basename(path))
    return float(m.group(1)) if m else None


def parse_gps(path):
    """Ground-truth GPS from the filename, if present. VALIDATION ONLY —
    never fed into the displacement estimate itself."""
    m = FILENAME_GPS_RE.match(os.path.basename(path))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ── Feature matching ──────────────────────────────────────────────────────

def _central_crop(gray):
    h, w = gray.shape[:2]
    f = config.CENTRAL_CROP_FRACTION
    if f >= 1.0:
        return gray, (0, 0)
    cw, ch = int(w * f), int(h * f)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return gray[y0:y0 + ch, x0:x0 + cw], (x0, y0)


def _matcher_norm():
    return cv2.NORM_HAMMING if config.DETECTOR == 'orb' else cv2.NORM_L2


def _make_detector():
    if config.DETECTOR == 'orb':
        return cv2.ORB_create(nfeatures=config.ORB_N_FEATURES)
    return cv2.SIFT_create(nfeatures=config.SIFT_N_FEATURES)


def detect_features(gray):
    """Detect keypoints/descriptors on a central crop of a single frame.

    Returns (keypoints, descriptors, crop_offset). Split out from
    detect_and_match() so a caller stepping through a video stream (e.g.
    live_vo_gps.py) can cache one frame's features and reuse them as frame 1
    of the next pair, instead of re-running detection on the same frame
    twice.
    """
    crop, off = _central_crop(gray)
    detector = _make_detector()
    kp, des = detector.detectAndCompute(crop, None)
    return kp, des, off


def match_features(feat1, feat2):
    """Match two detect_features() results. Returns matched point pairs
    (Nx2 arrays, in ORIGINAL full-frame pixel coordinates) after Lowe's
    ratio test."""
    kp1, des1, off1 = feat1
    kp2, des2, off2 = feat2
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    bf = cv2.BFMatcher(_matcher_norm())
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if n is not None and m.distance < config.MATCH_RATIO * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    if len(pts1):
        pts1 += np.array(off1)
        pts2 += np.array(off2)
    return pts1, pts2


def detect_and_match(gray1, gray2):
    """Returns matched point pairs (Nx2 arrays, in ORIGINAL full-frame pixel
    coordinates) after Lowe's ratio test. Convenience wrapper around
    detect_features() + match_features() for callers that don't need to
    cache features across calls."""
    return match_features(detect_features(gray1), detect_features(gray2))


def estimate_similarity(pts1, pts2):
    """Fit pts2 ~= s*R*pts1 + t via RANSAC. Returns (M 2x3, inlier_mask)."""
    if len(pts1) < 4:
        return None, None
    M, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.RANSAC_REPROJ_THRESHOLD_PX,
        confidence=0.99,
    )
    return M, inliers


# ── Pixel motion -> metric displacement ──────────────────────────────────

def displacement_from_transform(M, altitude_m):
    """Convert a pts1->pts2 similarity transform into the camera's own
    metric displacement, expressed in image axes (x=image-right, y=image-down)
    as they appeared in frame 1.

    Reasoning: a stationary ground feature's apparent position shifts
    OPPOSITE to the camera's own motion (walk forward, the ground appears to
    flow backward under you). So we transform the image center through M to
    see how a center-fixed feature appears to move, then negate that to get
    the camera's own translation.
    """
    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = math.sqrt(a * a + c * c)
    theta_rad = math.atan2(c, a)  # apparent rotation of the world in-image

    cx, cy = config.IMAGE_WIDTH / 2.0, config.IMAGE_HEIGHT / 2.0
    center2 = np.array([a * cx + b * cy + tx, c * cx + d * cy + ty])
    apparent_flow = center2 - np.array([cx, cy])
    cam_shift_px = -apparent_flow  # camera's own displacement, in pixels

    gsd_x = altitude_m / config.FX_PIXELS
    gsd_y = altitude_m / config.FY_PIXELS
    dx_m = cam_shift_px[0] * gsd_x
    dy_m = cam_shift_px[1] * gsd_y

    return {
        'dx_m': dx_m,           # + = camera moved toward image-right
        'dy_m': dy_m,           # + = camera moved toward image-down
        'magnitude_m': math.hypot(dx_m, dy_m),
        'yaw_change_deg': -math.degrees(theta_rad),  # camera's own yaw change
        'scale': scale,         # >1: frame2 zoomed in => flew lower
        'altitude_change_m': altitude_m * (1.0 / scale - 1.0) if scale > 0 else float('nan'),
    }


# ── Pipeline over a sequence ──────────────────────────────────────────────

def process_pair(path1, path2, alt1, alt2):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"could not read {path1} or {path2}")
    result = process_pair_arrays(img1, img2, alt1, alt2)
    result['from'], result['to'] = path1, path2
    return result


def process_pair_arrays(gray1, gray2, alt1, alt2):
    """Same as process_pair(), but on already-decoded grayscale arrays —
    used by the live capture loop, which never round-trips frames to disk."""
    return process_pair_features(detect_features(gray1), detect_features(gray2), alt1, alt2)


def process_pair_features(feat1, feat2, alt1, alt2):
    """Same as process_pair_arrays(), but takes precomputed detect_features()
    results instead of raw grayscale arrays, so a caller stepping through a
    video stream can cache the previous frame's features and reuse them as
    frame 1 of the next pair instead of re-running detection on it twice."""
    pts1, pts2 = match_features(feat1, feat2)
    M, inliers = estimate_similarity(pts1, pts2)

    n_matches = len(pts1)
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    result = {
        'n_matches': n_matches,
        'n_inliers': n_inliers,
        'ok': False,
        'confident': False,
    }
    if M is None or inliers is None or n_inliers < config.MIN_INLIERS:
        result['reason'] = 'too few inliers'
        return result

    altitude_m = (alt1 + alt2) / 2.0 if (alt1 and alt2) else (alt1 or alt2 or config.REFERENCE_ALTITUDE_M)
    disp = displacement_from_transform(M, altitude_m)
    result.update(disp)
    result['ok'] = True
    result['altitude_used_m'] = altitude_m
    result['confident'] = (n_inliers / n_matches) >= config.MIN_INLIER_RATIO
    if not result['confident']:
        result['reason'] = 'low inlier ratio (likely low image overlap or repetitive texture) -- displacement below is UNTRUSTED'
    return result


def run(image_paths, altitudes=None):
    frames = list_images(image_paths)
    if len(frames) < 2:
        raise SystemExit("need at least 2 images")

    if altitudes is None:
        altitudes = [parse_altitude_m(f) for f in frames]

    results = []
    traj = [(0.0, 0.0)]
    heading_deg = 0.0  # cumulative estimated yaw relative to frame 0

    for i in range(len(frames) - 1):
        r = process_pair(frames[i], frames[i + 1], altitudes[i], altitudes[i + 1])
        results.append(r)
        if r['ok'] and r['confident']:
            heading_deg += r['yaw_change_deg']
            th = math.radians(heading_deg)
            # rotate this leg's image-frame displacement into frame-0's reference frame
            dx, dy = r['dx_m'], r['dy_m']
            wx = dx * math.cos(th) - dy * math.sin(th)
            wy = dx * math.sin(th) + dy * math.cos(th)
            px, py = traj[-1]
            traj.append((px + wx, py + wy))
        else:
            traj.append(traj[-1])

    return frames, results, traj


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='image files and/or a directory of images')
    ap.add_argument('--altitude', help='comma-separated altitude (m) per frame, overrides filename parsing')
    args = ap.parse_args()

    altitudes = None
    if args.altitude:
        altitudes = [float(x) for x in args.altitude.split(',')]

    frames, results, traj = run(args.paths, altitudes)

    print(f"Sequence ({len(frames)} frames, ordered by capture time):")
    for f in frames:
        print(f"  {f}")
    print()

    for r in results:
        name1, name2 = os.path.basename(r['from']), os.path.basename(r['to'])
        if not r['ok']:
            print(f"[FAIL] {name1} -> {name2}: {r['n_inliers']}/{r['n_matches']} inliers "
                  f"-- {r.get('reason', 'unknown')}")
            continue
        tag = "[OK]  " if r['confident'] else "[LOW-CONFIDENCE]"
        print(f"{tag} {name1} -> {name2}")
        if not r['confident']:
            print(f"       WARNING: {r['reason']}")
        print(f"       matches={r['n_matches']} inliers={r['n_inliers']} altitude_used={r['altitude_used_m']:.1f}m")
        print(f"       displacement: dx={r['dx_m']:+.2f}m dy={r['dy_m']:+.2f}m "
              f"magnitude={r['magnitude_m']:.2f}m")
        print(f"       yaw_change={r['yaw_change_deg']:+.2f}deg altitude_change={r['altitude_change_m']:+.2f}m")

        gps1, gps2 = parse_gps(r['from']), parse_gps(r['to'])
        if gps1 and gps2:
            truth_m = haversine_m(gps1[0], gps1[1], gps2[0], gps2[1])
            err = r['magnitude_m'] - truth_m
            print(f"       ground truth (GPS in filename, validation only): {truth_m:.2f}m "
                  f"(error {err:+.2f}m, {abs(err)/truth_m*100:.0f}%)")
        print()

    print("Estimated trajectory (meters, frame-0 reference, x=image-right/y=image-down axes):")
    for f, (x, y) in zip(frames, traj):
        print(f"  {os.path.basename(f):55s} ({x:+7.2f}, {y:+7.2f})")


if __name__ == '__main__':
    main()
