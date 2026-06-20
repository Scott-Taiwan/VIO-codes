#!/usr/bin/env python3
"""
simulate_localize.py — Test SIFT localization on existing photos without flying.

Loads the tile index and builds the FLANN KD-tree ONCE at startup, then
processes each input photo through the two-phase SIFT pipeline and saves
the result to photo_obtained/ with the standard filename format.

"Real GPS" comes from last_home.json (saved by gps_check.py) or --lat/--lon.
This lets you verify the full localization pipeline on the ground.

Usage:
    cd /home/scott/claude-project/gpsless_mapping
    python3 shot_estimate/simulate_localize.py test_photo.jpg
    python3 shot_estimate/simulate_localize.py *.jpg --zoom 19
    python3 shot_estimate/simulate_localize.py test_photo.jpg --lat 25.0868 --lon 121.6003
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
TILE_DIR  = ROOT / "tiles"
PHOTO_DIR = Path(__file__).resolve().parent / "photo_obtained"
HOME_CACHE = ROOT / "last_home.json"

# ── constants ─────────────────────────────────────────────────────────────────
TILE_SIZE     = 256
MATCH_RATIO   = 0.75
MIN_INLIERS   = 6
TOP_CANDIDATES = 20


# ── index loading ─────────────────────────────────────────────────────────────

def load_index(zoom):
    path = INDEX_DIR / f"sift_index_z{zoom}.bin"
    if not path.exists():
        print(f"ERROR: index not found: {path}")
        print("Run:  python3 convert_index.py --zoom", zoom)
        sys.exit(1)

    import struct
    entries = []
    with open(path, "rb") as f:
        (num_tiles,) = struct.unpack("<i", f.read(4))
        print(f"  Loading {num_tiles} tiles … ", end="", flush=True)
        for _ in range(num_tiles):
            tx, ty, z, n = struct.unpack("<iiii", f.read(16))
            raw = f.read(n * 128 * 4)
            descs = np.frombuffer(raw, dtype=np.float32).reshape(n, 128).copy()
            entries.append({"tile_x": tx, "tile_y": ty, "zoom": z, "descs": descs})
    print("done.")
    return entries


# ── FLANN (built once) ────────────────────────────────────────────────────────

def build_flann(index):
    all_descs = np.vstack([e["descs"] for e in index]).astype(np.float32)
    offsets = [0]
    for e in index:
        offsets.append(offsets[-1] + len(e["descs"]))

    print(f"  Building FLANN on {len(all_descs)} descriptors … ", end="", flush=True)
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )
    flann.add([all_descs])
    flann.train()
    print("done.")
    return flann, offsets


# ── coordinate math ───────────────────────────────────────────────────────────

def pixel_to_latlon(px, py, tile_x, tile_y, zoom):
    n   = 1 << zoom
    wx  = tile_x + px / TILE_SIZE
    wy  = tile_y + py / TILE_SIZE
    lon = wx / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * wy / n))))
    return lat, lon


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── tile stitching ────────────────────────────────────────────────────────────

def stitch_tiles(cx, cy, zoom, radius=1):
    side   = 2 * radius + 1
    canvas = np.zeros((side * TILE_SIZE, side * TILE_SIZE, 3), dtype=np.uint8)
    for dx in range(side):
        for dy in range(side):
            tx, ty = cx - radius + dx, cy - radius + dy
            path = TILE_DIR / str(zoom) / str(tx) / f"{ty}.png"
            tile = cv2.imread(str(path))
            if tile is not None:
                canvas[dy*TILE_SIZE:(dy+1)*TILE_SIZE,
                       dx*TILE_SIZE:(dx+1)*TILE_SIZE] = tile
    return canvas


# ── two-phase localization ────────────────────────────────────────────────────

def localize(gray, index, flann, offsets, zoom):
    sift = cv2.SIFT_create(nfeatures=2000)
    kps_q, descs_q = sift.detectAndCompute(gray, None)

    if descs_q is None or len(kps_q) < 5:
        print(f"  Too few features ({len(kps_q) if kps_q else 0} kp) — no match.")
        return None

    print(f"  Query features : {len(kps_q)} keypoints")

    # Phase 1 — FLANN vote
    raw  = flann.knnMatch(descs_q.astype(np.float32), k=2)
    good = [m for m, n in raw if len((m,n)) == 2 and m.distance < MATCH_RATIO * n.distance]
    print(f"  FLANN good matches: {len(good)}")

    import bisect
    from collections import defaultdict
    votes = defaultdict(int)
    for m in good:
        ti = bisect.bisect_right(offsets, m.trainIdx) - 1
        votes[ti] += 1

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        print("  No FLANN votes — no match.")
        return None

    print(f"  Top candidates:")
    for ti, v in ranked[:5]:
        e = index[ti]
        print(f"    tile ({e['tile_x']},{e['tile_y']})  votes={v}")

    # Phase 2 — direct match on top candidates
    sift_c = cv2.SIFT_create(nfeatures=3000)
    for ti, _ in ranked[:TOP_CANDIDATES]:
        e  = index[ti]
        cx, cy = e["tile_x"], e["tile_y"]

        composite = stitch_tiles(cx, cy, zoom)
        comp_gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)

        kps_c, descs_c = sift_c.detectAndCompute(comp_gray, None)
        if descs_c is None or len(kps_c) < MIN_INLIERS:
            continue

        bf  = cv2.BFMatcher(cv2.NORM_L2)
        raw2 = bf.knnMatch(descs_q, descs_c, k=2)
        good2 = [m for m, n in raw2
                 if len((m,n)) == 2 and m.distance < MATCH_RATIO * n.distance]
        if len(good2) < MIN_INLIERS:
            continue

        src = np.float32([kps_q[m.queryIdx].pt for m in good2])
        dst = np.float32([kps_c[m.trainIdx].pt for m in good2])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            continue

        inliers = int(mask.sum())
        if inliers < MIN_INLIERS:
            continue

        qh, qw = gray.shape
        corners_in  = np.float32([[0,0],[qw,0],[qw,qh],[0,qh]]).reshape(-1,1,2)
        corners_out = cv2.perspectiveTransform(corners_in, H).reshape(-1, 2)
        area = cv2.contourArea(corners_out)
        if area < qw*qh*0.05 or area > qw*qh*50:
            continue
        if not cv2.isContourConvex(corners_out.astype(np.float32)):
            continue

        ctr = cv2.perspectiveTransform(
            np.float32([[qw/2, qh/2]]).reshape(-1,1,2), H).reshape(2)
        comp_size = 3 * TILE_SIZE
        if not (-TILE_SIZE < ctr[0] < comp_size+TILE_SIZE and
                -TILE_SIZE < ctr[1] < comp_size+TILE_SIZE):
            continue

        tile_dx = int(ctr[0] / TILE_SIZE)
        tile_dy = int(ctr[1] / TILE_SIZE)
        lpx = ctr[0] - tile_dx * TILE_SIZE
        lpy = ctr[1] - tile_dy * TILE_SIZE
        lat, lon = pixel_to_latlon(lpx, lpy, cx - 1 + tile_dx, cy - 1 + tile_dy, zoom)

        print(f"  Match: tile ({cx},{cy})  inliers={inliers}")
        return lat, lon, inliers

    print("  No match found in any candidate tile.")
    return None


# ── filename builder ──────────────────────────────────────────────────────────

def build_filename(real_lat, real_lon, est_lat, est_lon, dist_m):
    real = f"{real_lat:.7f}_{real_lon:.7f}"
    if est_lat is not None:
        return f"{real}-{est_lat:.7f}_{est_lon:.7f}__{dist_m:.1f}m.png"
    return f"{real}-NOFIX_NOFIX__NAm.png"


# ── main ─────────────────────────────────────────────────────────────────────

def load_home():
    if HOME_CACHE.exists():
        with open(HOME_CACHE) as f:
            d = json.load(f)
        return d["lat"], d["lon"]
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Simulate drone_localize: localize photos and save to photo_obtained/")
    parser.add_argument("photos", nargs="+", help="Input photo files")
    parser.add_argument("--zoom",    type=int,   default=19)
    parser.add_argument("--lat",     type=float, default=None,
                        help="Real GPS latitude  (default: from last_home.json)")
    parser.add_argument("--lon",     type=float, default=None,
                        help="Real GPS longitude (default: from last_home.json)")
    args = parser.parse_args()

    PHOTO_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve real GPS
    if args.lat is not None and args.lon is not None:
        real_lat, real_lon = args.lat, args.lon
    else:
        real_lat, real_lon = load_home()
        if real_lat is None:
            print("ERROR: no home GPS found. Run gps_check.py first, or use --lat/--lon")
            sys.exit(1)
    print(f"Real GPS (ground truth): {real_lat:.7f}, {real_lon:.7f}")

    # Load index and build FLANN — done once for all photos
    print(f"\nLoading index (zoom {args.zoom}) …")
    t0 = time.time()
    index = load_index(args.zoom)
    print(f"Building FLANN …")
    flann, offsets = build_flann(index)
    print(f"Startup done in {time.time()-t0:.1f} s  ({len(index)} tiles)\n")

    # Process each photo
    for photo_path in args.photos:
        print("=" * 60)
        print(f"Photo: {photo_path}")

        frame = cv2.imread(photo_path)
        if frame is None:
            print(f"  ERROR: cannot read {photo_path}")
            continue

        h, w = frame.shape[:2]
        print(f"  Size: {w}×{h} px")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t1   = time.time()
        result = localize(gray, index, flann, offsets, args.zoom)
        elapsed = time.time() - t1

        if result is not None:
            est_lat, est_lon, inliers = result
            dist_m = haversine_m(real_lat, real_lon, est_lat, est_lon)
            print(f"  Est. GPS : {est_lat:.7f}, {est_lon:.7f}")
            print(f"  Inliers  : {inliers}")
            print(f"  Distance : {dist_m:.1f} m from real GPS")
            fname = build_filename(real_lat, real_lon, est_lat, est_lon, dist_m)
        else:
            fname = build_filename(real_lat, real_lon, None, None, None)

        out = PHOTO_DIR / fname
        cv2.imwrite(str(out), frame)
        print(f"  Time     : {elapsed:.2f} s")
        print(f"  Saved    : {out}")

    print("\n" + "=" * 60)
    print(f"Done. Results in: {PHOTO_DIR}")


if __name__ == "__main__":
    main()
