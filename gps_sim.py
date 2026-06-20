#!/usr/bin/env python3
"""
gps_sim.py  —  Simulate drone GPS movement via MAVLink GPS_INPUT injection.

Pixhawk MUST have this parameter set first (via QGroundControl):
    GPS1_TYPE = 14   (MAVLink — replaces real GPS with injected GPS)

Simulation path:
    Read real GPS from Pixhawk  →  set as HOME
    Steps  1–10 :  HOME + (step × 20 m) north   (moves to +200 m north)
    Steps 11–20 :  return south 20 m per step    (back to HOME)

GPS_INPUT is sent at 5 Hz throughout so Pixhawk never loses lock.
Mission Planner will show the drone moving north then south every 5 s.

Usage:
    python3 gps_sim.py
    python3 gps_sim.py --port /dev/ttyTHS1 --baud 57600
    python3 gps_sim.py --step 30 --interval 3   # 30 m step, 3 s interval
"""

import argparse
import json
import math
import os
import sys
import time

HOME_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_home.json")

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not installed.  Run: pip install pymavlink pyserial")
    sys.exit(1)

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PORT     = "/dev/ttyTHS1"
DEFAULT_BAUD     = 57600
STEP_METRES      = 20.0   # metres north per step
STEPS_OUT        = 10     # steps going north
STEPS_BACK       = 10     # steps going south (same count → returns to home)
INTERVAL_S       = 5.0    # seconds at each waypoint
GPS_HZ           = 5      # how often we send GPS_INPUT (Hz) — keeps lock alive

GPS_EPOCH_UNIX   = 315964800   # Jan 6 1980 00:00:00 UTC in Unix seconds

# ── helpers ───────────────────────────────────────────────────────────────────

def offset_latlon(lat, lon, north_m, east_m=0.0):
    """Offset a GPS coordinate by metres (flat-earth, good < 1 km)."""
    dlat = north_m / 111111.0
    dlon = east_m  / (111111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def gps_week_ms():
    """Return (GPS_week, ms_into_week) from current system time."""
    t    = time.time() - GPS_EPOCH_UNIX
    week = int(t / 604800)
    ms   = int((t % 604800) * 1000)
    return week, ms


def maps_url(lat, lon):
    return f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}"

# ── MAVLink helpers ───────────────────────────────────────────────────────────

def request_streams(conn):
    """Ask ArduPilot to stream GLOBAL_POSITION_INT on this port."""
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION, 4, 1)
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 2, 1)


def send_gps_input(conn, lat, lon, alt_msl, vn=0.0, ve=0.0):
    """
    Inject a GPS fix into Pixhawk via GPS_INPUT (MAVLink id 232).
    vn/ve: velocity north/east in m/s — sending these helps the EKF accept
    position changes immediately instead of waiting for large accumulated offsets.
    """
    week, week_ms = gps_week_ms()

    # Send position + velocity + hacc; ignore only vertical velocity and accuracy
    ignore = (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT       |
              mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY |
              mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY)

    try:
        conn.mav.gps_input_send(
            int(time.time() * 1e6),  # time_usec
            1,                        # gps_id: 1 = GPS2  (GPS1 = real GNSS module)
            ignore,
            week_ms,                  # time_week_ms
            week,                     # time_week
            3,                        # fix_type: 3D fix
            int(lat * 1e7),           # lat  (degE7)
            int(lon * 1e7),           # lon  (degE7)
            alt_msl,                  # alt  (m, MSL)
            1.0,                      # hdop
            1.5,                      # vdop
            vn, ve, 0.0,             # vn, ve, vd  (m/s) — EKF uses these immediately
            0.3,                      # speed_accuracy (m/s)
            0.5,                      # horiz_accuracy 0.5 m — beats real GPS, GPS2 wins
            0.0,                      # vert_accuracy   (ignored)
            10,                       # satellites_visible
            0,                        # yaw (0 = unknown)
        )
    except TypeError:
        # Older pymavlink builds lack the yaw field
        conn.mav.gps_input_send(
            int(time.time() * 1e6),
            1, ignore, week_ms, week,
            3,
            int(lat * 1e7), int(lon * 1e7), alt_msl,
            1.0, 1.5,
            vn, ve, 0.0,
            0.3, 0.5, 0.0,
            10,
        )


def get_real_gps(conn, timeout=10):
    """Try to read GPS from Pixhawk (works when GPS1_TYPE != 14)."""
    request_streams(conn)
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        if msg is None:
            request_streams(conn)
            continue
        if msg.lat == 0 and msg.lon == 0:
            continue
        return (msg.lat / 1e7,
                msg.lon / 1e7,
                msg.alt / 1000.0)
    return None


def load_cached_home():
    """Read the GPS saved by gps_check.py (last_home.json)."""
    if not os.path.exists(HOME_CACHE):
        return None
    try:
        with open(HOME_CACHE) as f:
            d = json.load(f)
        return d["lat"], d["lon"], d["alt_msl"]
    except Exception:
        return None

# ── waypoint stream ───────────────────────────────────────────────────────────

def move_to(conn, lat_from, lon_from, lat_to, lon_to, alt, seconds, label, step, total):
    """
    Smoothly interpolate position from (lat_from, lon_from) to (lat_to, lon_to)
    over `seconds` at GPS_HZ. Position and velocity are always consistent so the
    EKF tracks every step immediately without accumulation delays.
    """
    dt      = 1.0 / GPS_HZ
    n_ticks = max(1, int(seconds * GPS_HZ))

    # Constant velocity throughout the move (m/s)
    north_m = (lat_to - lat_from) * 111111.0
    east_m  = (lon_to - lon_from) * (111111.0 * math.cos(math.radians(lat_from)))
    vn = north_m / seconds
    ve = east_m  / seconds

    for i in range(n_ticks):
        frac = i / n_ticks
        lat  = lat_from + (lat_to - lat_from) * frac
        lon  = lon_from + (lon_to - lon_from) * frac
        send_gps_input(conn, lat, lon, alt, vn, ve)
        time.sleep(dt)

    # Arrive: send final position with zero velocity
    send_gps_input(conn, lat_to, lon_to, alt, 0.0, 0.0)
    print(f"  {step:>3}/{total}  {label:<26}  {lat_to:.7f}  {lon_to:.7f}"
          f"  {maps_url(lat_to, lon_to)}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate GPS movement via MAVLink GPS_INPUT injection")
    parser.add_argument("--port",      default=DEFAULT_PORT)
    parser.add_argument("--baud",      type=int,   default=DEFAULT_BAUD)
    parser.add_argument("--step",      type=float, default=STEP_METRES,
                        help=f"Metres per step (default {STEP_METRES})")
    parser.add_argument("--interval",  type=float, default=INTERVAL_S,
                        help=f"Seconds at each waypoint (default {INTERVAL_S})")
    parser.add_argument("--steps",     type=int,   default=STEPS_OUT,
                        help=f"Steps north (same count south, default {STEPS_OUT})")
    parser.add_argument("--home-lat",  type=float, default=None,
                        help="Home latitude  (override auto-detect)")
    parser.add_argument("--home-lon",  type=float, default=None,
                        help="Home longitude (override auto-detect)")
    parser.add_argument("--home-alt",  type=float, default=None,
                        help="Home altitude in metres MSL (override)")
    args = parser.parse_args()

    # ── connect ───────────────────────────────────────────────────────────────
    print(f"Connecting to {args.port} @ {args.baud} baud …", end=" ", flush=True)
    conn = mavutil.mavlink_connection(args.port, baud=args.baud)
    hb   = conn.wait_heartbeat(timeout=10)
    if not hb:
        print("FAILED — no heartbeat.")
        sys.exit(1)
    print(f"OK  sys={conn.target_system} comp={conn.target_component}")

    # ── resolve home GPS ──────────────────────────────────────────────────────
    # Priority: CLI args → Pixhawk live GPS → last_home.json saved by gps_check.py
    if args.home_lat is not None and args.home_lon is not None:
        home_lat = args.home_lat
        home_lon = args.home_lon
        home_alt = args.home_alt if args.home_alt is not None else 0.0
        print(f"Home GPS       :  {home_lat:.7f}, {home_lon:.7f}  "
              f"alt={home_alt:.1f} m  (from --home-lat/lon)\n")

    else:
        print("Reading home GPS from Pixhawk …", end=" ", flush=True)
        home = get_real_gps(conn, timeout=10)

        if home:
            home_lat, home_lon, home_alt = home
            print(f"OK  ({home_lat:.7f}, {home_lon:.7f})\n")
        else:
            print("no fix (GPS1_TYPE=14 — trying last_home.json …)", end=" ", flush=True)
            home = load_cached_home()
            if home:
                home_lat, home_lon, home_alt = home
                print(f"OK\n  loaded: {home_lat:.7f}, {home_lon:.7f}  "
                      f"alt={home_alt:.1f} m MSL\n")
            else:
                print("NOT FOUND\n")
                print("ERROR: Cannot determine home position.")
                print()
                print("Fix — choose one:")
                print("  1. Run gps_check.py first (while GPS1_TYPE=1) to save home:")
                print("       python3 gps_check.py --port /dev/ttyTHS1")
                print()
                print("  2. Pass coordinates directly:")
                print("       python3 gps_sim.py --home-lat 25.0868206 --home-lon 121.6002830")
                sys.exit(1)

    print("=" * 70)
    print("  GPS SIMULATION")
    print("=" * 70)
    print(f"  Home       :  {home_lat:.7f}, {home_lon:.7f}  alt={home_alt:.1f} m MSL")
    print(f"  Step size  :  {args.step:.0f} m north per step")
    print(f"  Interval   :  {args.interval:.0f} s per step")
    print(f"  Path       :  {args.steps} steps north → {args.steps} steps south")
    print(f"  Duration   :  ~{int(args.steps * 2 * args.interval)} s total")
    print(f"  GPS rate   :  {GPS_HZ} Hz (continuous injection)")
    print()
    print("  NOTE: Required Pixhawk parameters (set via QGroundControl):")
    print("          GPS1_TYPE     = 1  (Auto — keep real GPS module)")
    print("          GPS2_TYPE     = 14 (MAVLink — Jetson GPS_INPUT)")
    print("          GPS_AUTO_SWITCH = 4 (use best/lowest hacc)")
    print("        Reboot Pixhawk after changing parameters.")
    print("=" * 70)
    print()
    print(f"  {'Step':>5}  {'Label':<26}  {'Latitude':>13}  {'Longitude':>14}")
    print("  " + "-" * 66)

    # ── build waypoint list ────────────────────────────────────────────────────
    waypoints = []

    # Phase 1: go north
    for i in range(1, args.steps + 1):
        north = i * args.step
        lat, lon = offset_latlon(home_lat, home_lon, north)
        waypoints.append((lat, lon, home_alt, f"+{north:.0f} m north"))

    # Phase 2: return south
    for i in range(args.steps - 1, -1, -1):
        north = i * args.step
        lat, lon = offset_latlon(home_lat, home_lon, north)
        label = f"+{north:.0f} m north (return)" if north > 0 else "HOME"
        waypoints.append((lat, lon, home_alt, label))

    total = len(waypoints)

    # ── run simulation ────────────────────────────────────────────────────────
    prev_lat, prev_lon = home_lat, home_lon
    try:
        for idx, (lat, lon, alt, label) in enumerate(waypoints, 1):
            move_to(conn, prev_lat, prev_lon, lat, lon, alt,
                    args.interval, label, idx, total)
            prev_lat, prev_lon = lat, lon

    except KeyboardInterrupt:
        print("\n\nStopped by user — restoring real GPS position …")
        # Send home position a few times so Mission Planner snaps back
        for _ in range(10):
            send_gps_input(conn, home_lat, home_lon, home_alt)
            time.sleep(0.1)
        print("Done.")
        return

    # ── done ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Simulation complete — drone returned to HOME.")
    print(f"  {maps_url(home_lat, home_lon)}")
    print("=" * 70)
    print()
    print("  IMPORTANT: Set GPS1_TYPE back to 1 (auto) in QGroundControl")
    print("  if you want to restore the real GPS for actual flight.")


if __name__ == "__main__":
    main()
