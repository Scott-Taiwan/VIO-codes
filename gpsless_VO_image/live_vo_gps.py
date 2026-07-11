#!/usr/bin/env python3
"""
live_vo_gps.py — capture photos from the Jetson CSI camera in a loop,
run visual_odometry between each new frame and the previous one, and feed
the result through dead_reckoning.PositionTracker into Pixhawk as GPS2
(same GPS_INPUT mechanism as gpsless_mapping/gps_sim.py).

IMPORTANT — environment quirk found while building this: a pip-installed
`opencv-python` under ~/.local/lib/python3.10/site-packages shadows the
JetPack system OpenCV and silently breaks camera access (no GStreamer
support in the pip wheel — this is also why gpsless_mapping/capture_sample.py
currently fails with "Could not open CSI camera"). This script works
around it by putting the system dist-packages path first, but the
underlying shadowing should get fixed properly at some point (see the
chat writeup) so every script doesn't need this workaround.

Capture cadence is a fixed interval (--interval), not distance-adaptive —
pick it so that even at your max cruise speed, consecutive frames keep
plenty of overlap (see the MIN_INLIER_RATIO discussion in config.py: sparse,
low-overlap frame pairs produce untrustworthy displacement). At 60 m
altitude with the ~70x55 m footprint in config.py, keeping displacement per
leg under roughly 20 m is a reasonable starting target.

Usage:
    # Handheld ground test, fully offline (no Pixhawk at all).
    # REMOVE PROPELLERS FIRST.
    python3 live_vo_gps.py --handheld --dry-run --interval 1 --save-dir ./handheld_test

    # Handheld ground test with the F450 powered on and wired to the Jetson,
    # so the real compass heading (and GPS fix, if any) gets used as the
    # anchor instead of a dummy one. Connecting and sending GPS_INPUT never
    # arms the vehicle or moves the motors either way. REMOVE PROPELLERS FIRST.
    python3 live_vo_gps.py --handheld --port /dev/ttyTHS1 --interval 1 --save-dir ./handheld_test

    # Camera + VO only, no Pixhawk needed at all:
    python3 live_vo_gps.py --dry-run --interval 2 --altitude 60

    # Live, anchored from Pixhawk's own current GPS + heading:
    python3 live_vo_gps.py --port /dev/ttyTHS1 --interval 2

    # Live, with a manual anchor (e.g. GPS1 has no fix yet):
    python3 live_vo_gps.py --port /dev/ttyTHS1 --interval 2 \\
        --anchor-lat 25.0432642 --anchor-lon 121.4743882 \\
        --anchor-alt 60.0 --anchor-heading 90
"""
import argparse
import os
import sys
import time
from datetime import datetime

# The JetPack system OpenCV (has GStreamer + SIFT); put it ahead of any
# pip-installed opencv-python that may be shadowing it in site-packages.
sys.path.insert(0, '/usr/lib/python3/dist-packages')
import cv2  # noqa: E402

import config  # noqa: E402
import visual_odometry as vo  # noqa: E402
from dead_reckoning import PositionTracker  # noqa: E402


def gstreamer_pipeline(sensor_id=0, width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate=30/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=True"
    )


def open_camera(sensor_id=0):
    cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=sensor_id), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Could not open CSI camera — check sensor connection / nvargus-daemon.")
    for _ in range(10):  # let exposure settle, same as capture_sample.py
        cap.read()
    return cap


def capture_gray(cap):
    ret, frame = cap.read()
    if not ret or frame is None:
        return None, None
    return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ── Pixhawk telemetry (lazy pymavlink import — --dry-run needs none of this) ──

def connect_pixhawk(port, baud):
    from pymavlink import mavutil
    print(f"Connecting to {port} @ {baud} baud ...", end=' ', flush=True)
    conn = mavutil.mavlink_connection(port, baud=baud)
    if not conn.wait_heartbeat(timeout=10):
        raise SystemExit("FAILED — no heartbeat.")
    print(f"OK  sys={conn.target_system} comp={conn.target_component}")
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)
    return conn


def read_global_position(conn, timeout=3.0):
    """Latest (lat, lon, alt_msl_m, relative_alt_m), or None if nothing arrived."""
    msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=timeout)
    if msg is None or (msg.lat == 0 and msg.lon == 0):
        return None
    return msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1000.0, msg.relative_alt / 1000.0


def read_heading(conn, timeout=3.0):
    """Latest true compass heading in degrees, or None."""
    msg = conn.recv_match(type='VFR_HUD', blocking=True, timeout=timeout)
    return msg.heading if msg is not None else None


def resolve_anchor(conn, args):
    """Priority: CLI overrides -> live Pixhawk telemetry -> gps_sim's cached
    last_home.json (lat/lon/alt only — heading still required some other way
    in that case)."""
    lat, lon, alt_msl, alt_agl, heading = (
        args.anchor_lat, args.anchor_lon, args.anchor_alt, args.altitude, args.anchor_heading,
    )

    if conn is not None and (lat is None or lon is None or alt_msl is None):
        pos = read_global_position(conn, timeout=10)
        if pos:
            lat = lat if lat is not None else pos[0]
            lon = lon if lon is not None else pos[1]
            alt_msl = alt_msl if alt_msl is not None else pos[2]
            if alt_agl is None:
                alt_agl = pos[3]

    if conn is not None and heading is None:
        heading = read_heading(conn, timeout=10)

    if lat is None or lon is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gpsless_mapping'))
        from gps_sim import load_cached_home
        cached = load_cached_home()
        if cached:
            lat, lon = lat if lat is not None else cached[0], lon if lon is not None else cached[1]
            alt_msl = alt_msl if alt_msl is not None else cached[2]

    if lat is None or lon is None or alt_msl is None or heading is None:
        if args.handheld:
            print("WARNING: --handheld could not resolve a real anchor from Pixhawk/cache -- "
                  "falling back to a dummy anchor (missing fields default to 0). Ignore the "
                  "printed lat/lon; only displacement/yaw/confidence matter for this test.")
            lat = lat if lat is not None else 0.0
            lon = lon if lon is not None else 0.0
            alt_msl = alt_msl if alt_msl is not None else 0.0
            heading = heading if heading is not None else 0.0
        else:
            raise SystemExit(
                "Could not resolve an anchor position/heading. Provide --anchor-lat/--anchor-lon/"
                "--anchor-alt/--anchor-heading explicitly, or connect to a Pixhawk that has a fix."
            )
    if alt_agl is None:
        alt_agl = config.REFERENCE_ALTITUDE_M
        print(f"WARNING: no AGL altitude available — defaulting to {alt_agl}m for GSD scaling. "
              f"Pass --altitude to override.")

    return lat, lon, alt_msl, alt_agl, heading


# ── main loop ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--interval', type=float, default=2.0, help='seconds between captures (default 2)')
    ap.add_argument('--gps-hz', type=float, default=5.0, help='GPS_INPUT resend rate between captures (default 5)')
    ap.add_argument('--altitude', type=float, default=None,
                     help='fixed AGL altitude (m) for GSD, if no Pixhawk relative_alt is available')
    ap.add_argument('--anchor-lat', type=float, default=None)
    ap.add_argument('--anchor-lon', type=float, default=None)
    ap.add_argument('--anchor-alt', type=float, default=None, help='anchor altitude, m MSL')
    ap.add_argument('--anchor-heading', type=float, default=None, help='TRUE compass heading, deg from North')
    ap.add_argument('--port', default='/dev/ttyTHS1')
    ap.add_argument('--baud', type=int, default=57600)
    ap.add_argument('--sensor-id', type=int, default=0)
    ap.add_argument('--save-dir', default=None, help='optionally save every captured frame here')
    ap.add_argument('--dry-run', action='store_true', help="don't connect to Pixhawk, just print")
    ap.add_argument('--handheld', action='store_true',
                     help='ground/bench test: hand-carry the frame to exercise capture+VO without '
                          'flying. Defaults --altitude to 1.5m (hand-held height, not the 60m '
                          'flight default -- the IMX219 is fixed-focus and may blur much closer '
                          'than that). Anchor: uses real Pixhawk telemetry if connected (pass '
                          '--port, omit --dry-run) so the true compass heading gets used; falls '
                          'back to a dummy anchor (0,0,0,0) instead of erroring out if none is '
                          'available. REMOVE PROPELLERS BEFORE HANDLING THE FRAME, whether or not '
                          'the flight controller is powered.')
    args = ap.parse_args()

    if args.handheld:
        if args.altitude is None:
            args.altitude = 1.5
        print("=" * 70)
        print("HANDHELD TEST MODE -- ground test, not a real flight.")
        print("SAFETY: propellers must be removed before handling the frame,")
        print("whether or not the flight controller is powered on.")
        if args.dry_run:
            print("No Pixhawk connection (--dry-run) -- using a dummy anchor.")
        else:
            print("Connecting to Pixhawk for a real compass heading/GPS fix if available")
            print("(connecting and sending GPS_INPUT never arms the vehicle or moves the")
            print("motors) -- falls back to a dummy anchor if no fix is available.")
        print("=" * 70)
        print()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    conn = None
    send_gps_input = None
    if not args.dry_run:
        conn = connect_pixhawk(args.port, args.baud)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gpsless_mapping'))
        from gps_sim import send_gps_input, maps_url
    else:
        def maps_url(lat, lon):
            return f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}"

    anchor_lat, anchor_lon, anchor_alt_msl, alt_agl, anchor_heading = resolve_anchor(conn, args)
    tracker = PositionTracker(lat=anchor_lat, lon=anchor_lon, alt_m=anchor_alt_msl, heading_deg=anchor_heading)
    print(f"Anchor: {anchor_lat:.7f}, {anchor_lon:.7f}  alt_msl={anchor_alt_msl:.1f}m  "
          f"agl={alt_agl:.1f}m  heading={anchor_heading:.0f}deg")
    print(f"Capture interval: {args.interval}s   GPS_INPUT resend rate: {args.gps_hz} Hz")
    print()

    cap = open_camera(args.sensor_id)
    print("Camera open. Ctrl+C to stop.\n")

    prev_feat, prev_time, prev_alt_agl = None, None, alt_agl
    vn, ve = 0.0, 0.0
    n_frame = 0

    try:
        while True:
            frame, gray = capture_gray(cap)
            now = time.time()
            n_frame += 1
            if frame is None:
                print("WARNING: frame grab failed, skipping")
            else:
                cur_alt_agl = alt_agl
                if conn is not None:
                    pos = read_global_position(conn, timeout=0.2)
                    if pos:
                        cur_alt_agl = pos[3]

                if args.save_dir:
                    # Double-underscore "__live__NNm" tag matches
                    # visual_odometry.FILENAME_ALT_RE, so replaying this
                    # folder through visual_odometry.py auto-parses the
                    # altitude instead of silently falling back to
                    # config.REFERENCE_ALTITUDE_M (60m).
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(os.path.join(args.save_dir, f"{ts}__live__{cur_alt_agl:.1f}m.jpg"), frame)

                # Detect this frame's features once and cache them — the next
                # loop iteration reuses them as "frame 1" instead of paying
                # for detection on the same frame twice.
                t_detect = time.time()
                feat = vo.detect_features(gray)
                detect_ms = (time.time() - t_detect) * 1000.0

                if prev_feat is not None:
                    t_match = time.time()
                    r = vo.process_pair_features(prev_feat, feat, prev_alt_agl, cur_alt_agl)
                    match_ms = (time.time() - t_match) * 1000.0
                    dt_s = now - prev_time
                    state = tracker.apply_leg(r, dt_s=dt_s)
                    vn, ve = state['vn'], state['ve']

                    tag = "OK  " if r.get('confident') else ("FAIL" if not r['ok'] else "LOW-CONF")
                    print(f"[frame {n_frame}] [{tag}] dt={dt_s:.2f}s "
                          f"detect={detect_ms:.0f}ms match={match_ms:.0f}ms", end='  ')
                    if r['ok']:
                        print(f"disp={r['magnitude_m']:.2f}m yaw={r['yaw_change_deg']:+.2f}deg", end='  ')
                    else:
                        print(f"({r.get('reason', 'no match')})", end='  ')
                    print(f"-> {state['lat']:.7f},{state['lon']:.7f} alt={state['alt_m']:.1f}m "
                          f"hacc={state['hacc_m']:.2f}m")
                    print(f"          {maps_url(state['lat'], state['lon'])}")

                prev_feat, prev_time, prev_alt_agl = feat, now, cur_alt_agl

            # Resend the current best estimate at gps_hz until the next capture,
            # so Pixhawk doesn't consider GPS2 stale between photos.
            next_capture = now + args.interval
            gps_dt = 1.0 / args.gps_hz
            while time.time() < next_capture:
                if conn is not None:
                    send_gps_input(conn, tracker.lat, tracker.lon, tracker.alt_m, vn, ve, hacc=tracker.hacc_m)
                time.sleep(min(gps_dt, max(0.0, next_capture - time.time())))

    except KeyboardInterrupt:
        print(f"\nStopped after {n_frame} frames, {tracker.legs_since_anchor} leg(s). "
              f"Final: {tracker.lat:.7f},{tracker.lon:.7f} hacc={tracker.hacc_m:.2f}m")
    finally:
        cap.release()


if __name__ == '__main__':
    main()
