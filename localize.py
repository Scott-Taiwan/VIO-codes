"""
Identify the GPS location of a query photo by matching it against the tile index.

Algorithm
---------
Phase 1 — coarse (FLANN vote):
  Extract SIFT from query; match against the full tile descriptor pool;
  vote to find which tile cluster wins.

Phase 2 — fine (stitched direct match):
  Stitch the winning tile + its 8 neighbours into a 3×3 composite;
  run SIFT+BFMatcher directly on the composite — more matching surface,
  no tile-boundary splits, far more inliers than Phase 1 alone.

Usage:
    python localize.py photo.jpg
    python localize.py photo.jpg --zoom 17 --show
"""

import argparse
import bisect
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import (ZOOM_LEVEL, INDEX_DIR, TILE_DIR,
                    MATCH_RATIO, MIN_INLIERS, TOP_CANDIDATES, TILE_SIZE)
from tile_utils import pixel_to_latlon, tile_bounds


# ── index helpers ─────────────────────────────────────────────────────────────

def load_index(index_dir, zoom):
    path = Path(index_dir) / f'sift_index_z{zoom}.pkl'
    if not path.exists():
        raise FileNotFoundError(
            f'Index not found: {path}\nRun build_index.py first.')
    print(f'Loading index from {path} …')
    with open(path, 'rb') as f:
        return pickle.load(f)


def kp_from_tuple(t):
    return cv2.KeyPoint(x=t[0][0], y=t[0][1], size=t[1], angle=t[2],
                        response=t[3], octave=t[4], class_id=int(t[5]))


# ── Phase 1: FLANN vote ───────────────────────────────────────────────────────

def flann_vote(descs_q, index):
    """
    Match query descriptors against all tile descriptors via FLANN.
    Returns ranked list of (tile_idx, [matches]) sorted by vote count.
    """
    tile_offsets = [0]
    for entry in index:
        tile_offsets.append(tile_offsets[-1] + len(entry['descs']))
    all_descs = np.vstack([e['descs'] for e in index]).astype(np.float32)

    print(f'Total train kp: {len(all_descs):,}  — building FLANN …')
    index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann.add([all_descs])
    flann.train()

    raw = flann.knnMatch(descs_q.astype(np.float32), k=2)
    good = [m for m, n in raw if len((m, n)) == 2 and
            m.distance < MATCH_RATIO * n.distance]

    print(f'Good matches  : {len(good)} (after ratio test)')

    votes = defaultdict(list)
    for m in good:
        tile_idx = bisect.bisect_right(tile_offsets, m.trainIdx) - 1
        votes[tile_idx].append(m)

    return sorted(votes.items(), key=lambda kv: len(kv[1]), reverse=True)


# ── Phase 2: stitched direct match ────────────────────────────────────────────

def stitch_tiles(cx, cy, zoom, tile_dir, radius=1):
    """
    Load a (2r+1)×(2r+1) grid of tiles centred on (cx, cy) and stitch
    them into one BGR image.  Returns (composite_bgr, x_origin, y_origin).
    """
    r = radius
    size = 2 * r + 1
    canvas = np.zeros((size * TILE_SIZE, size * TILE_SIZE, 3), dtype=np.uint8)
    for dx in range(size):
        for dy in range(size):
            tx, ty = cx - r + dx, cy - r + dy
            path = tile_dir / str(zoom) / str(tx) / f'{ty}.png'
            tile = cv2.imread(str(path))
            if tile is not None:
                canvas[dy * TILE_SIZE:(dy + 1) * TILE_SIZE,
                       dx * TILE_SIZE:(dx + 1) * TILE_SIZE] = tile
    return canvas, cx - r, cy - r


def direct_match(query_gray, kps_q, descs_q, cx, cy, zoom, tile_dir,
                 radius=1, min_inliers=4):
    """
    Stitch a (2*radius+1)² tile grid around (cx, cy), run SIFT+BFMatcher,
    RANSAC homography, and return (lat, lon, inliers) or None.
    """
    composite, x_orig, y_orig = stitch_tiles(cx, cy, zoom, tile_dir, radius)
    comp_gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=3000)
    kps_c, descs_c = sift.detectAndCompute(comp_gray, None)
    if descs_c is None or len(kps_c) < min_inliers:
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(descs_q, descs_c, k=2)
    good = [m for m, n in raw if m.distance < MATCH_RATIO * n.distance]

    if len(good) < min_inliers:
        return None

    src_pts = np.float32([kps_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return None

    # Reject degenerate homographies: project the four query corners and check
    # that the resulting quadrilateral has a reasonable area and aspect ratio.
    h, w = query_gray.shape
    corners = cv2.perspectiveTransform(
        np.float32([[[0,0],[w,0],[w,h],[0,h]]]), H)[0]
    area = cv2.contourArea(corners)
    if area < (w * h * 0.05) or area > (w * h * 50):
        return None
    # Check convexity (degenerate homographies produce concave quads)
    if not cv2.isContourConvex(np.int32(corners)):
        return None

    # Project query centre into composite pixel space
    h, w = query_gray.shape
    centre = cv2.perspectiveTransform(
        np.float32([[[w / 2, h / 2]]]), H)[0][0]
    cpx, cpy = float(centre[0]), float(centre[1])

    # Composite pixel → individual tile + local pixel → GPS
    comp_size = (2 * radius + 1) * TILE_SIZE
    margin = TILE_SIZE
    if not (-margin <= cpx <= comp_size + margin and
            -margin <= cpy <= comp_size + margin):
        return None

    tile_dx = int(cpx / TILE_SIZE)
    tile_dy = int(cpy / TILE_SIZE)
    local_px = cpx - tile_dx * TILE_SIZE
    local_py = cpy - tile_dy * TILE_SIZE
    lat, lon = pixel_to_latlon(local_px, local_py,
                               x_orig + tile_dx, y_orig + tile_dy, zoom)
    return lat, lon, inliers, H, composite


# ── main ─────────────────────────────────────────────────────────────────────

def localize(image_path, zoom=None, index_dir=None, tile_dir=None, show=False):
    zoom = zoom or ZOOM_LEVEL
    index_dir = index_dir or INDEX_DIR
    tile_dir = Path(tile_dir or TILE_DIR)

    # Load query
    query = cv2.imread(str(image_path))
    if query is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=2000)
    kps_q, descs_q = sift.detectAndCompute(query_gray, None)
    if descs_q is None or len(kps_q) < 5:
        print('Too few features in query image.')
        return None
    print(f'Query image   : {image_path}  ({query.shape[1]}×{query.shape[0]} px)')
    print(f'Query features: {len(kps_q)} keypoints')

    # Phase 1 — coarse vote
    index = load_index(index_dir, zoom)
    print(f'Tile index    : {len(index)} tiles')
    ranked = flann_vote(descs_q, index)

    print('\nTop candidate tiles:')
    for idx, ms in ranked[:TOP_CANDIDATES]:
        e = index[idx]
        print(f'  tile ({e["tile_x"]},{e["tile_y"]})  votes={len(ms)}')

    # Phase 2 — stitched direct match on top candidates
    result = None
    for tile_idx, ms in ranked[:TOP_CANDIDATES]:
        entry = index[tile_idx]
        cx, cy = entry['tile_x'], entry['tile_y']
        print(f'\nDirect match on 3×3 stitched grid centred ({cx},{cy}) …')

        out = direct_match(query_gray, kps_q, descs_q,
                           cx, cy, zoom, tile_dir,
                           radius=1, min_inliers=MIN_INLIERS)
        if out is not None:
            lat, lon, inliers, H_comp, composite = out
            result = (lat, lon, inliers, cx, cy, H_comp, composite)
            break

    # Report
    print()
    if result is None:
        print('Could not determine location.')
        print('Suggestions:')
        print('  • Check that the photo location is within the downloaded tiles')
        print('  • Try a nadir (straight-down) photo with clear ground features')
        print('  • Increase SIFT_N_FEATURES in config.py and rebuild the index')
        return None

    lat, lon, inliers, cx, cy, H_comp, composite = result
    print('=' * 55)
    print(f'  GPS Location: {lat:.7f},  {lon:.7f}')
    print(f'  RANSAC inliers: {inliers}  |  centre tile ({cx},{cy}) z{zoom}')
    print(f'  Google Maps: https://maps.google.com/?q={lat:.7f},{lon:.7f}')
    print('=' * 55)

    if show:
        _save_result(query, query_gray, composite, H_comp, lat, lon)

    return lat, lon


# ── visualisation (saves to file, no Qt) ─────────────────────────────────────

def _save_result(query_img, query_gray, composite, H, lat, lon):
    h, w = query_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)
    comp_vis = cv2.polylines(composite.copy(), [np.int32(projected)],
                             True, (0, 255, 0), 2)

    centre = cv2.perspectiveTransform(
        np.float32([[[w / 2, h / 2]]]), H)[0][0]
    cv2.drawMarker(comp_vis, (int(centre[0]), int(centre[1])),
                   (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.imwrite('result_tile.png', comp_vis)
    cv2.imwrite('result_query.png', query_img)
    print(f'Match saved → result_tile.png  (3×3 composite with footprint)')
    print(f'Query saved → result_query.png')


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Identify GPS location of a photo using satellite map tiles')
    parser.add_argument('image', help='Path to the query photo')
    parser.add_argument('--zoom', type=int, default=None)
    parser.add_argument('--index-dir', default=None)
    parser.add_argument('--tile-dir', default=None)
    parser.add_argument('--show', action='store_true',
                        help='Save match visualisation to result_tile.png')
    args = parser.parse_args()

    localize(args.image, zoom=args.zoom, index_dir=args.index_dir,
             tile_dir=args.tile_dir, show=args.show)
