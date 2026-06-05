#!/usr/bin/env python3
"""
drone_localize.py — Autonomous GPS estimation during drone flight.

Connects to Pixhawk via MAVLink serial.  Monitors relative altitude.
When the drone takes off and reaches TAKEOFF_ALT (50 m), captures a
CSI camera frame every CAPTURE_INTERVAL seconds.

Before each shot the real GPS is read from Pixhawk.
After each shot the frame is run through the SIFT tile index to
estimate the GPS position.

Photo is saved to photo_obtained/ as:
    {lat_real}_{lon_real}-{lat_est}_{lon_est}.png
    e.g.  25.0523400_121.4623100-25.0523012_121.4622875.png
    If SIFT gives no fix: 25.0523400_121.4623100-NOFIX_NOFIX.png

Usage:
    cd /home/scott/claude-project/gpsless_mapping
    /usr/bin/python3 shot_estimate/drone_localize.py
    /usr/bin/python3 shot_estimate/drone_localize.py --port /dev/ttyUSB0 --baud 115200
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

# ── import shared helpers from parent project ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localize import load_index, direct_match
from config import (ZOOM_LEVEL, INDEX_DIR, TILE_DIR,
                    MATCH_RATIO, MIN_INLIERS, TOP_CANDIDATES)

# ── tuneable constants ────────────────────────────────────────────────────────
TAKEOFF_ALT      = 50.0   # m  — begin capturing once above this
MIN_CAPTURE_ALT  = 10.0   # m  — pause if drone descends below this (landing)
CAPTURE_INTERVAL = 5.0    # s  — time between photos

PHOTO_DIR = Path(__file__).parent / 'photo_obtained'

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
log = logging.getLogger('drone_localize')


# ── MAVLink state ─────────────────────────────────────────────────────────────

class VehicleState:
    """Thread-safe container for the latest Pixhawk telemetry."""

    def __init__(self):
        self._lock   = threading.Lock()
        self.alt_m   = 0.0
        self.lat     = None   # float degrees
        self.lon     = None
        self.gps_ok  = False

    def update(self, lat_1e7, lon_1e7, relative_alt_mm):
        with self._lock:
            self.alt_m  = relative_alt_mm / 1000.0
            self.lat    = lat_1e7  / 1e7
            self.lon    = lon_1e7  / 1e7
            self.gps_ok = True

    def snapshot(self):
        """Return (alt_m, lat, lon, gps_ok) atomically."""
        with self._lock:
            return self.alt_m, self.lat, self.lon, self.gps_ok


def _mavlink_reader(conn, state: VehicleState, stop: threading.Event):
    """Background thread: pump GLOBAL_POSITION_INT into VehicleState."""
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


# ── SIFT localizer (FLANN built once) ────────────────────────────────────────

class SiftLocalizer:
    """
    Wraps the two-phase SIFT localization pipeline from localize.py.
    Loads the index and builds the FLANN KD-tree once at construction
    so every subsequent call to localize() reuses the same structure.
    """

    def __init__(self, zoom=ZOOM_LEVEL, index_dir=INDEX_DIR, tile_dir=TILE_DIR):
        self.zoom     = zoom
        self.tile_dir = Path(tile_dir)

        log.info('Loading SIFT tile index …')
        self._index = load_index(index_dir, zoom)
        log.info(f'  {len(self._index)} tiles loaded.')

        log.info('Building FLANN KD-tree (one-time startup cost) …')
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
        """Phase 1: FLANN vote — returns ranked [(tile_idx, [matches]), ...]."""
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
        """
        Run two-phase SIFT localization on frame_bgr.
        Returns (lat, lon) on success, or None if no match found.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=2000)
        kps_q, descs_q = sift.detectAndCompute(gray, None)

        if descs_q is None or len(kps_q) < 5:
            log.warning('  Too few features in frame — skipping localization.')
            return None
        log.info(f'  Query features: {len(kps_q)} keypoints')

        ranked = self._vote(descs_q)
        if not ranked:
            log.warning('  No FLANN votes — frame has no match in tile index.')
            return None

        for tile_idx, _ in ranked[:TOP_CANDIDATES]:
            e  = self._index[tile_idx]
            cx, cy = e['tile_x'], e['tile_y']
            log.info(f'  Phase 2: trying tile ({cx},{cy}) …')
            out = direct_match(gray, kps_q, descs_q,
                               cx, cy, self.zoom, self.tile_dir,
                               radius=1, min_inliers=MIN_INLIERS)
            if out is not None:
                lat, lon, inliers, *_ = out
                log.info(f'  Match! inliers={inliers}  est=({lat:.7f}, {lon:.7f})')
                return lat, lon

        log.warning('  No match found in any candidate tile.')
        return None


# ── filename helper ───────────────────────────────────────────────────────────

def build_filename(real_lat, real_lon, est_lat, est_lon):
    """
    Format: {lat_real}_{lon_real}-{lat_est}_{lon_est}.png
    Example: 25.0523400_121.4623100-25.0523012_121.4622875.png
    """
    real_part = f'{real_lat:.7f}_{real_lon:.7f}'
    if est_lat is not None and est_lon is not None:
        est_part = f'{est_lat:.7f}_{est_lon:.7f}'
    else:
        est_part = 'NOFIX_NOFIX'
    return f'{real_part}-{est_part}.png'


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Drone GPS-free visual localization')
    parser.add_argument('--port',  default='/dev/ttyTHS1',
                        help='Serial port to Pixhawk (default: /dev/ttyTHS1)')
    parser.add_argument('--baud',  type=int, default=57600,
                        help='Baud rate (default: 57600)')
    parser.add_argument('--zoom',  type=int, default=ZOOM_LEVEL,
                        help=f'Map tile zoom level (default: {ZOOM_LEVEL})')
    args = parser.parse_args()

    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f'Photos will be saved to: {PHOTO_DIR.resolve()}')

    # ── connect to Pixhawk ────────────────────────────────────────────────────
    log.info(f'Connecting to Pixhawk on {args.port} @ {args.baud} baud …')
    conn = mavutil.mavlink_connection(args.port, baud=args.baud)
    conn.wait_heartbeat()
    log.info(f'Heartbeat OK — system={conn.target_system} component={conn.target_component}')

    state   = VehicleState()
    stop_ev = threading.Event()
    mav_th  = threading.Thread(target=_mavlink_reader, args=(conn, state, stop_ev),
                               daemon=True, name='mav-reader')
    mav_th.start()

    # ── one-time setup (before takeoff) ───────────────────────────────────────
    localizer = SiftLocalizer(zoom=args.zoom)
    cap       = open_camera()
    log.info('CSI camera opened.')

    # ── main flight loop ──────────────────────────────────────────────────────
    was_on_ground    = True
    capturing_active = False
    last_capture_t   = 0.0

    log.info(f'Waiting for takeoff … (trigger altitude ≥ {TAKEOFF_ALT} m)')

    try:
        while True:
            alt, lat, lon, gps_ok = state.snapshot()

            # ── takeoff detection ─────────────────────────────────────────────
            if was_on_ground and alt >= TAKEOFF_ALT:
                log.info(f'Takeoff confirmed — altitude {alt:.1f} m. '
                         f'Starting photo capture every {CAPTURE_INTERVAL} s.')
                capturing_active = True
                was_on_ground    = False

            # ── landing / descent detection ───────────────────────────────────
            if alt < MIN_CAPTURE_ALT:
                if capturing_active:
                    log.info(f'Altitude {alt:.1f} m < {MIN_CAPTURE_ALT} m — '
                             f'capture paused (possible landing).')
                capturing_active = False
                was_on_ground    = True

            # ── periodic capture ──────────────────────────────────────────────
            now = time.time()
            if capturing_active and (now - last_capture_t) >= CAPTURE_INTERVAL:
                last_capture_t = now
                log.info(f'=== Capture triggered  alt={alt:.1f} m ===')

                # Obtain real GPS from Pixhawk right before the shot
                _, real_lat, real_lon, gps_ok = state.snapshot()
                if not gps_ok or real_lat is None:
                    log.warning('  Pixhawk GPS not ready — skipping capture.')
                    continue
                log.info(f'  Real GPS : {real_lat:.7f}, {real_lon:.7f}')

                # Capture frame from CSI camera
                try:
                    frame = grab_frame(cap)
                except RuntimeError as exc:
                    log.error(f'  Camera error: {exc}')
                    continue

                # Estimate location via SIFT tile matching
                est = localizer.localize(frame)
                if est is not None:
                    est_lat, est_lon = est
                    log.info(f'  Est. GPS : {est_lat:.7f}, {est_lon:.7f}')
                    err_m = _haversine_m(real_lat, real_lon, est_lat, est_lon)
                    log.info(f'  Error    : {err_m:.1f} m')
                else:
                    est_lat = est_lon = None

                # Save photo with GPS info in filename
                fname   = build_filename(real_lat, real_lon, est_lat, est_lon)
                outpath = PHOTO_DIR / fname
                cv2.imwrite(str(outpath), frame)
                log.info(f'  Saved    : {fname}')

            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info('Stopped by user (Ctrl-C).')
    finally:
        stop_ev.set()
        cap.release()
        log.info('Camera released. Done.')


# ── utility ───────────────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2):
    """Approximate great-circle distance in metres between two GPS points."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


if __name__ == '__main__':
    main()
