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

# ── drift-prevention parameters ───────────────────────────────────────────────
MIN_INLIERS_SEND     = 10     # below this RANSAC geometry is useless
MIN_WAYPOINT_SPACING = 300.0  # m — warn if waypoints closer than this (#4)

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
                return lat, lon, inliers

        log.warning('  No match found in any candidate tile.')
        return None


# ── drift-prevention helpers ──────────────────────────────────────────────────

def hacc_from_inliers(inliers):
    """
    Map RANSAC inlier count to horiz_accuracy (metres) for GPS_INPUT.
    Pixhawk EKF blends fix with IMU proportional to hacc —
    high hacc = IMU does most work; low hacc = fix is trusted strongly.
    """
    if inliers > 80:  return  10.0   # strong match  — trust fix
    if inliers >= 40: return  30.0   # good match
    if inliers >= 20: return  60.0   # weak match    — IMU does most work
    return                   100.0   # 10–19 inliers — IMU almost entirely


def send_gps_input(conn, lat, lon, hacc):
    """
    Send GPS_INPUT MAVLink message to Pixhawk.
    ignore_flags = 191: ignore alt|hdop|vdop|vel_h|vel_v|speed_acc|vert_acc.
    horiz_accuracy IS sent — this is the key trust field Pixhawk uses.
    """
    # 1+2+4+8+16+32+128 = 191 (ignore everything except horiz_accuracy)
    ignore_flags = 191
    conn.mav.gps_input_send(
        0,                    # time_usec (0 = autopilot uses own clock)
        0,                    # gps_id
        ignore_flags,
        0,                    # time_week_ms (ignored)
        0,                    # time_week   (ignored)
        3,                    # fix_type: 3D fix
        int(lat * 1e7),       # lat degE7
        int(lon * 1e7),       # lon degE7
        0.0,                  # alt (ignored)
        1.0,                  # hdop (ignored)
        1.0,                  # vdop (ignored)
        0.0, 0.0, 0.0,        # vn, ve, vd (ignored)
        0.0,                  # speed_accuracy (ignored)
        hacc,                 # horiz_accuracy — key field
        5.0,                  # vert_accuracy  (ignored)
        10                    # satellites_visible
    )


# ── mission fetch from Pixhawk ────────────────────────────────────────────────

def fetch_mission_from_pixhawk(conn):
    """
    Download the active mission from Pixhawk using MAVLink mission protocol.
    Returns a list of Waypoint for MAV_CMD_NAV_WAYPOINT (cmd=16) items.
    Item 0 (home position) is always skipped.
    Returns [] on timeout or if no mission is loaded.
    """
    log.info('[mission] Requesting mission from Pixhawk …')

    # Step 1: request list
    conn.mav.mission_request_list_send(
        conn.target_system, conn.target_component,
        0)  # mission_type = MAV_MISSION_TYPE_MISSION

    # Step 2: wait for MISSION_COUNT
    msg = conn.recv_match(type='MISSION_COUNT', blocking=True, timeout=10)
    if msg is None:
        log.warning('[mission] Timeout — no mission loaded in Pixhawk.')
        return []
    count = msg.count
    log.info(f'[mission] {count} item(s).')
    if count == 0:
        return []

    # Step 3: request each item
    wps = []
    for i in range(count):
        conn.mav.mission_request_int_send(
            conn.target_system, conn.target_component,
            i, 0)  # seq, mission_type

        item = conn.recv_match(type='MISSION_ITEM_INT', blocking=True, timeout=5)
        if item is None:
            log.error(f'[mission] Timeout waiting for item {i} — aborting.')
            return []
        if item.seq != i:
            log.error(f'[mission] Got item seq={item.seq}, expected {i} — aborting.')
            return []

        # item 0 is the home position; only keep NAV_WAYPOINT (cmd=16)
        if i > 0 and item.command == 16 and item.x != 0 and item.y != 0:
            lat = item.x / 1e7
            lon = item.y / 1e7
            log.info(f'  WP{i}: ({lat:.7f}, {lon:.7f})')
            wps.append(Waypoint(lat, lon))
        else:
            log.info(f'  item{i}: cmd={item.command} (skipped)')

    # Step 4: acknowledge
    conn.mav.mission_ack_send(
        conn.target_system, conn.target_component,
        0, 0)  # type=MAV_MISSION_ACCEPTED, mission_type=MAV_MISSION_TYPE_MISSION

    log.info(f'[mission] {len(wps)} navigation waypoint(s) loaded.')
    return wps


# ── waypoint helpers (#4 and #5) ─────────────────────────────────────────────

class Waypoint:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon


def parse_waypoints(s):
    """Parse 'lat1,lon1 lat2,lon2 ...' string into list of Waypoint."""
    wps = []
    for token in s.strip().split():
        parts = token.split(',')
        if len(parts) != 2:
            raise ValueError(f"Bad waypoint token '{token}' — expected lat,lon")
        wps.append(Waypoint(float(parts[0]), float(parts[1])))
    return wps


def validate_waypoints(wps):
    """Warn (#4) if consecutive waypoints are closer than MIN_WAYPOINT_SPACING."""
    if len(wps) < 2:
        return
    for i in range(1, len(wps)):
        d = _haversine_m(wps[i-1].lat, wps[i-1].lon, wps[i].lat, wps[i].lon)
        if d < MIN_WAYPOINT_SPACING:
            log.warning(
                f'Waypoint spacing warning (#4): WP{i-1}→WP{i} is only {d:.0f} m '
                f'(minimum recommended: {MIN_WAYPOINT_SPACING:.0f} m). '
                f'SIFT accuracy (~50 m) may be insufficient for close waypoints.'
            )



# ── filename helper ───────────────────────────────────────────────────────────

def build_filename(real_lat, real_lon, est_lat, est_lon, dist_m=None):
    """
    Format: {lat_real}_{lon_real}-{lat_est}_{lon_est}__{meters}m.png
    Example: 25.0523400_121.4623100-25.0523012_121.4622875__28.3m.png
    No fix:  25.0523400_121.4623100-NOFIX_NOFIX__NAm.png
    """
    real_part = f'{real_lat:.7f}_{real_lon:.7f}'
    if est_lat is not None and est_lon is not None:
        dist_str = f'{dist_m:.1f}' if dist_m is not None else 'NA'
        est_part = f'{est_lat:.7f}_{est_lon:.7f}__{dist_str}m'
    else:
        est_part = 'NOFIX_NOFIX__NAm'
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
    parser.add_argument('--waypoints', default='',
                        help='Override: space-separated "lat,lon" waypoints (last = target). '
                             'If omitted, waypoints are auto-fetched from the Pixhawk mission.')
    args = parser.parse_args()

    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f'Photos will be saved to: {PHOTO_DIR.resolve()}')

    # ── parse and validate waypoints (#4) ────────────────────────────────────
    waypoints = []
    if args.waypoints:
        waypoints = parse_waypoints(args.waypoints)
        log.info(f'Waypoints loaded: {len(waypoints)} point(s)')
        validate_waypoints(waypoints)   # logs warnings for spacing < 300 m
        log.info(f'Target waypoint: ({waypoints[-1].lat}, {waypoints[-1].lon})')

    # ── connect to Pixhawk ────────────────────────────────────────────────────
    log.info(f'Connecting to Pixhawk on {args.port} @ {args.baud} baud …')
    conn = mavutil.mavlink_connection(args.port, baud=args.baud)
    conn.wait_heartbeat()
    log.info(f'Heartbeat OK — system={conn.target_system} component={conn.target_component}')

    # ── auto-fetch mission from Pixhawk (before starting reader thread) ───────
    if not waypoints:
        waypoints = fetch_mission_from_pixhawk(conn)
        if waypoints:
            validate_waypoints(waypoints)
    else:
        log.info(f'[mission] Using --waypoints override ({len(waypoints)} point(s)).')

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
                    est_lat, est_lon, inliers = est
                    dist_m = _haversine_m(real_lat, real_lon, est_lat, est_lon)
                    log.info(f'  Est. GPS : {est_lat:.7f}, {est_lon:.7f}')
                    log.info(f'  Inliers  : {inliers}   Distance: {dist_m:.1f} m')
                else:
                    est_lat = est_lon = dist_m = None
                    inliers = 0

                # ── send fix to Pixhawk (hacc expresses confidence to EKF) ───
                if est is not None and inliers >= MIN_INLIERS_SEND:
                    hacc = hacc_from_inliers(inliers)
                    log.info(f'  → GPS_INPUT: ({est_lat:.7f}, {est_lon:.7f})'
                             f'  inliers={inliers}  hacc={hacc:.0f} m')
                    send_gps_input(conn, est_lat, est_lon, hacc)
                elif est is not None:
                    log.warning(f'  → inliers={inliers} < {MIN_INLIERS_SEND}'
                                f' — geometry unreliable, not sent; Pixhawk uses IMU')
                else:
                    log.warning('  → NOFIX — not sent; Pixhawk uses IMU')

                # Save photo with GPS info in filename
                fname   = build_filename(real_lat, real_lon, est_lat, est_lon, dist_m)
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
