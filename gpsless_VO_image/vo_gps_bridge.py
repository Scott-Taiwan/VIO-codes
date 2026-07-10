#!/usr/bin/env python3
"""
Bridge: visual_odometry.py -> dead_reckoning.PositionTracker -> Pixhawk GPS_INPUT.

Reference driver showing the intended integration: starting from one
absolute anchor fix (lat/lon/alt/heading — normally from the SIFT/
SuperPoint tile-matcher in gpsless_mapping / gpsless_superpoint, or a real
GPS fix before GPS1 is switched off), walk forward through a sequence of
photos, and for each consecutive pair inject the dead-reckoned position
into Pixhawk as GPS2 via the same GPS_INPUT mechanism gps_sim.py uses.

This script processes a fixed batch of already-captured photos (matching
visual_odometry.py's interface) for testing/replay. Wiring this into an
actual flight would mean calling PositionTracker.apply_leg() from inside
the photo-capture loop as each new frame lands, instead of iterating a
pre-existing directory — the tracker and the MAVLink call are the same
either way.

Usage:
    # Dry run against the test photos — no Pixhawk needed, just prints
    # what would be sent:
    python3 vo_gps_bridge.py photo_obtained/ \\
        --anchor-lat 25.0434821 --anchor-lon 121.4747908 \\
        --anchor-alt 60.1 --anchor-heading 0 --dry-run

    # Live, against a real Pixhawk:
    python3 vo_gps_bridge.py photo_obtained/ \\
        --anchor-lat 25.0434821 --anchor-lon 121.4747908 \\
        --anchor-alt 60.1 --anchor-heading 90 \\
        --port /dev/ttyTHS1 --baud 57600
"""
import argparse
import os
import sys

import config
import visual_odometry as vo
from dead_reckoning import PositionTracker


def maps_url(lat, lon):
    return f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='image files and/or a directory of images')
    ap.add_argument('--altitude', help='comma-separated altitude (m) per frame, overrides filename parsing')
    ap.add_argument('--anchor-lat', type=float, required=True)
    ap.add_argument('--anchor-lon', type=float, required=True)
    ap.add_argument('--anchor-alt', type=float, required=True, help='anchor altitude, m MSL')
    ap.add_argument('--anchor-heading', type=float, required=True,
                     help='TRUE compass heading (deg from North) of the airframe at the anchor photo')
    ap.add_argument('--port', default='/dev/ttyTHS1')
    ap.add_argument('--baud', type=int, default=57600)
    ap.add_argument('--dry-run', action='store_true', help="don't connect to Pixhawk, just print")
    args = ap.parse_args()

    altitudes = [float(x) for x in args.altitude.split(',')] if args.altitude else None
    frames = vo.list_images(args.paths)
    if altitudes is None:
        altitudes = [vo.parse_altitude_m(f) for f in frames]
    if len(frames) < 2:
        raise SystemExit("need at least 2 images")

    conn = None
    send_gps_input = None
    if not args.dry_run:
        from pymavlink import mavutil
        # gps_sim.py lives in the sibling gpsless_mapping project — reuse
        # its MAVLink GPS_INPUT sender rather than duplicating it. Imported
        # lazily so --dry-run works even without pymavlink installed.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gpsless_mapping'))
        from gps_sim import send_gps_input

        print(f"Connecting to {args.port} @ {args.baud} baud ...", end=' ', flush=True)
        conn = mavutil.mavlink_connection(args.port, baud=args.baud)
        if not conn.wait_heartbeat(timeout=10):
            raise SystemExit("FAILED — no heartbeat.")
        print(f"OK  sys={conn.target_system} comp={conn.target_component}")

    tracker = PositionTracker(
        lat=args.anchor_lat, lon=args.anchor_lon, alt_m=args.anchor_alt,
        heading_deg=args.anchor_heading,
    )
    print(f"Anchor: {args.anchor_lat:.7f}, {args.anchor_lon:.7f}  "
          f"alt={args.anchor_alt:.1f}m  heading={args.anchor_heading:.0f}deg  "
          f"hacc={tracker.hacc_m:.2f}m")
    print()

    for i in range(len(frames) - 1):
        path1, path2 = frames[i], frames[i + 1]
        r = vo.process_pair(path1, path2, altitudes[i], altitudes[i + 1])

        dt_s = None
        try:
            dt_s = os.path.getmtime(path2) - os.path.getmtime(path1)
            dt_s = dt_s if dt_s > 0 else None
        except OSError:
            pass

        state = tracker.apply_leg(r, dt_s=dt_s)

        tag = "OK  " if r.get('confident') else ("FAIL" if not r['ok'] else "LOW-CONF")
        print(f"[{tag}] {os.path.basename(path1)} -> {os.path.basename(path2)}")
        if r['ok']:
            print(f"         displacement magnitude={r['magnitude_m']:.2f}m "
                  f"yaw_change={r['yaw_change_deg']:+.2f}deg")
        else:
            print(f"         {r.get('reason', 'no displacement available')}")
        print(f"         -> lat={state['lat']:.7f} lon={state['lon']:.7f} "
              f"alt={state['alt_m']:.1f}m hacc={state['hacc_m']:.2f}m "
              f"vn={state['vn']:+.2f} ve={state['ve']:+.2f} m/s")
        print(f"         {maps_url(state['lat'], state['lon'])}")

        if conn is not None:
            send_gps_input(conn, state['lat'], state['lon'], state['alt_m'],
                            state['vn'], state['ve'], hacc=state['hacc_m'])
        print()

    print(f"Final hacc after {tracker.legs_since_anchor} leg(s): {tracker.hacc_m:.2f}m "
          f"({'still trustworthy' if tracker.hacc_m < config.HACC_MAX_M * 0.5 else 'getting stale — get a fresh absolute fix'})")


if __name__ == '__main__':
    main()
