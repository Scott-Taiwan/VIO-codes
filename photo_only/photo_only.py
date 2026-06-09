#!/usr/bin/env python3
"""
photo_only.py — Visual localization accuracy measurement tool

Connects to Pixhawk via MAVLink serial. Monitors relative altitude.
When the drone is above CAPTURE_ALT_MIN (30 m), captures a CSI camera
frame every CAPTURE_INTERVAL seconds and runs SIFT localization against
the ESRI tile index.

For each frame the program records:
  - True GPS  : read from Pixhawk GLOBAL_POSITION_INT
  - Est. GPS  : computed by SIFT visual localization
  - Distance  : haversine metres between the two

Everything is stored only as a photo filename under result/:
  {lat_real}_{lon_real}-{lat_est}_{lon_est}__{dist}m.png
  {lat_real}_{lon_real}-NOFIX_NOFIX__NAm.png

No navigation commands and no GPS_INPUT are sent to Pixhawk.
This tool measures SIFT accuracy only.

Usage:
  cd /home/scott/claude-project/gpsless_mapping
  /usr/bin/python3 photo_only/photo_only.py
  /usr/bin/python3 photo_only/photo_only.py --port /dev/ttyUSB0 --baud 115200 --zoom 19
"""

import argparse
import bisect
import logging
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localize import load_index, direct_match
from config import (ZOOM_LEVEL, INDEX_DIR, TILE_DIR,
                    MATCH_RATIO, MIN_INLIERS, TOP_CANDIDATES)

# ── constants ─────────────────────────────────────────────────────────────────
CAPTURE_ALT_MIN  = 30.0   # m — start capturing above this
CAPTURE_ALT_STOP = 20.0   # m — stop capturing below this
CAPTURE_INTERVAL =  5.0   # s — between frames

RESULT_DIR = Path(__file__).parent / 'result'

# CSI camera
CSI_SENSOR_ID      = 0
CSI_CAPTURE_WIDTH  = 1280
CSI_CAPTURE_HEIGHT = 720
CSI_FRAMERATE      = 30
CSI_FLIP_METHOD    = 0    # 0 = no flip; 2 = 180° (camera mounted inverted)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('photo_only')


# ── MAVLink state ─────────────────────────────────────────────────────────────

class VehicleState:
    def __init__(self):
        self._lock  = threading.Lock()
        self.alt_m  = 0.0
        self.lat    = None
        self.lon    = None
        self.gps_ok = False

    def update(self, lat_1e7, lon_1e7, relative_alt_mm):
        with self._lock:
            self.alt_m  = relative_alt_mm / 1000.0
            self.lat    = lat_1e7 / 1e7
            self.lon    = lon_1e7 / 1e7
            self.gps_ok = True

    def snapshot(self):
        with self._lock:
            return self.alt_m, self.lat, self.lon, self.gps_ok


def _mavlink_reader(conn, state: VehicleState, stop: threading.Event):
    log.info('MAVLink reader thread started.')
    while not stop.is_set():
        msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        if msg:
            state.update(msg.lat, msg.lon, msg.relative_alt)
    log.info('MAVLink reader thread stopped.')


# ── CSI camera ────────────────────────────────────────────────────────────────

def _gst_pipeline(sensor_id, width, height, framerate, flip_method):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


def open_camera():
    pipeline = _gst_pipeline(CSI_SENSOR_ID, CSI_CAPTURE_WIDTH, CSI_CAPTURE_HEIGHT,
                              CSI_FRAMERATE, CSI_FLIP_METHOD)
    log.info(f'GStreamer pipeline: {pipeline}')
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError('Failed to open CSI camera — check GStreamer pipeline.')
    return cap


def grab_frame(cap):
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError('Camera read() returned no frame.')
    return frame


# ── SIFT localizer ────────────────────────────────────────────────────────────

class SiftLocalizer:
    def __init__(self, zoom=ZOOM_LEVEL, index_dir=INDEX_DIR, tile_dir=TILE_DIR):
        self.zoom     = zoom
        self.tile_dir = Path(tile_dir)

        log.info('Loading SIFT tile index …')
        self._index = load_index(index_dir, zoom)
        log.info(f'  {len(self._index)} tiles loaded.')

        log.info('Building FLANN KD-tree …')
        self._build_flann()
        log.info('  FLANN ready.')

    def _build_flann(self):
        offsets = [0]
        for e in self._index:
            offsets.append(offsets[-1] + len(e['descs']))
        all_descs = np.vstack([e['descs'] for e in self._index]).astype(np.float32)

        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        flann.add([all_descs])
        flann.train()
        self._flann   = flann
        self._offsets = offsets

    def _vote(self, descs_q):
        raw  = self._flann.knnMatch(descs_q.astype(np.float32), k=2)
        good = [m for m, n in raw
                if len((m, n)) == 2 and m.distance < MATCH_RATIO * n.distance]
        log.info(f'  FLANN good matches: {len(good)}')

        votes = defaultdict(list)
        for m in good:
            ti = bisect.bisect_right(self._offsets, m.trainIdx) - 1
            votes[ti].append(m)
        return sorted(votes.items(), key=lambda kv: len(kv[1]), reverse=True)

    def localize(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=2000)
        kps_q, descs_q = sift.detectAndCompute(gray, None)

        if descs_q is None or len(kps_q) < 5:
            log.warning('  Too few features — skipping.')
            return None
        log.info(f'  Query features: {len(kps_q)} keypoints')

        ranked = self._vote(descs_q)
        if not ranked:
            log.warning('  No FLANN votes.')
            return None

        for tile_idx, _ in ranked[:TOP_CANDIDATES]:
            e  = self._index[tile_idx]
            cx, cy = e['tile_x'], e['tile_y']
            log.info(f'  Phase 2: tile ({cx},{cy}) …')
            out = direct_match(gray, kps_q, descs_q,
                               cx, cy, self.zoom, self.tile_dir,
                               radius=1, min_inliers=MIN_INLIERS)
            if out is not None:
                lat, lon, inliers, *_ = out
                log.info(f'  Match! inliers={inliers}  est=({lat:.7f}, {lon:.7f})')
                return lat, lon, inliers

        log.warning('  No match in any candidate tile.')
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_filename(real_lat, real_lon, est_lat, est_lon, dist_m=None):
    real_part = f'{real_lat:.7f}_{real_lon:.7f}'
    if est_lat is not None:
        dist_str = f'{dist_m:.1f}' if dist_m is not None else 'NA'
        return f'{real_part}-{est_lat:.7f}_{est_lon:.7f}__{dist_str}m.png'
    return f'{real_part}-NOFIX_NOFIX__NAm.png'


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SIFT accuracy measurement tool')
    parser.add_argument('--port', default='/dev/ttyTHS1')
    parser.add_argument('--baud', type=int, default=57600)
    parser.add_argument('--zoom', type=int, default=ZOOM_LEVEL)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f'Result photos  → {RESULT_DIR.resolve()}')
    log.info(f'Capture above  : {CAPTURE_ALT_MIN} m')
    log.info(f'Stop below     : {CAPTURE_ALT_STOP} m')
    log.info(f'Interval       : {CAPTURE_INTERVAL} s')

    # connect to Pixhawk
    log.info(f'Connecting to Pixhawk on {args.port} @ {args.baud} baud …')
    conn = mavutil.mavlink_connection(args.port, baud=args.baud)
    conn.wait_heartbeat()
    log.info(f'Heartbeat OK — system={conn.target_system} component={conn.target_component}')

    state   = VehicleState()
    stop_ev = threading.Event()
    mav_th  = threading.Thread(target=_mavlink_reader, args=(conn, state, stop_ev),
                               daemon=True, name='mav-reader')
    mav_th.start()

    # one-time setup
    localizer = SiftLocalizer(zoom=args.zoom)
    cap       = open_camera()
    log.info('CSI camera opened.')
    log.info(f'Waiting for altitude >= {CAPTURE_ALT_MIN} m …')

    # main loop
    capturing_active = False
    last_capture_t   = 0.0

    try:
        while True:
            alt, lat, lon, gps_ok = state.snapshot()

            # altitude hysteresis: start at 30 m, stop at 20 m
            if not capturing_active and alt >= CAPTURE_ALT_MIN:
                log.info(f'Altitude {alt:.1f} m — capture started.')
                capturing_active = True
            if capturing_active and alt < CAPTURE_ALT_STOP:
                log.info(f'Altitude {alt:.1f} m < {CAPTURE_ALT_STOP} m — capture stopped.')
                capturing_active = False

            now = time.time()
            if capturing_active and (now - last_capture_t) >= CAPTURE_INTERVAL:
                last_capture_t = now

                _, real_lat, real_lon, gps_ok = state.snapshot()
                log.info('=' * 55)
                log.info(f'Capture  alt={alt:.1f} m')

                if not gps_ok or real_lat is None:
                    log.warning('  Pixhawk GPS not ready — skipping.')
                    continue
                log.info(f'  Real GPS : {real_lat:.7f}, {real_lon:.7f}')

                try:
                    frame = grab_frame(cap)
                except RuntimeError as exc:
                    log.error(f'  Camera error: {exc}')
                    continue

                est = localizer.localize(frame)
                if est is not None:
                    est_lat, est_lon, inliers = est
                    dist_m = _haversine_m(real_lat, real_lon, est_lat, est_lon)
                    log.info(f'  Est. GPS : {est_lat:.7f}, {est_lon:.7f}')
                    log.info(f'  Inliers  : {inliers}   Error: {dist_m:.1f} m')
                else:
                    est_lat = est_lon = dist_m = None
                    log.warning('  Est. GPS : NOFIX')

                fname   = build_filename(real_lat, real_lon, est_lat, est_lon, dist_m)
                outpath = RESULT_DIR / fname
                cv2.imwrite(str(outpath), frame)
                log.info(f'  Saved    : {fname}')

            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info('Stopped by user (Ctrl-C).')
    finally:
        stop_ev.set()
        cap.release()
        log.info('Camera released. Done.')


if __name__ == '__main__':
    main()
