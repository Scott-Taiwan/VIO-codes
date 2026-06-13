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
import math
import sys
import time

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


def send_gps_input(conn, lat, lon, alt_msl):
    """
    Inject a GPS fix into Pixhawk via GPS_INPUT (MAVLink id 232).
    Pixhawk must have GPS1_TYPE = 14 to accept and use this message.
    Sending at 5 Hz keeps ArduPilot's GPS healthy without timing out.
    """
    week, week_ms = gps_week_ms()

    # Ignore velocity and accuracy — only provide position + DOP
    ignore = (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ        |
              mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT         |
              mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY   |
              mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY |
              mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY)

    try:
        conn.mav.gps_input_send(
            int(time.time() * 1e6),  # time_usec
            0,                        # gps_id: 0 = GPS1
            ignore,
            week_ms,                  # time_week_ms
            week,                     # time_week
            3,                        # fix_type: 3D fix
            int(lat * 1e7),           # lat  (degE7)
            int(lon * 1e7),           # lon  (degE7)
            alt_msl,                  # alt  (m, MSL)
            1.0,                      # hdop
            1.5,                      # vdop
            0.0, 0.0, 0.0,           # vn, ve, vd  (ignored)
            0.0,                      # speed_accuracy  (ignored)
            0.0,                      # horiz_accuracy  (ignored)
            0.0,                      # vert_accuracy   (ignored)
            10,                       # satellites_visible
            0,                        # yaw (0 = unknown)
        )
    except TypeError:
        # Older pymavlink builds lack the yaw field
        conn.mav.gps_input_send(
            int(time.time() * 1e6),
            0, ignore, week_ms, week,
            3,
            int(lat * 1e7), int(lon * 1e7), alt_msl,
            1.0, 1.5,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            10,
        )


def get_real_gps(conn, timeout=30):
    """Read the real GPS from Pixhawk before we start spoofing."""
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
                msg.alt / 1000.0)   # alt MSL in metres
    return None

# ── waypoint stream ───────────────────────────────────────────────────────────

def hold_position(conn, lat, lon, alt, seconds, label, step, total):
    """
    Send GPS_INPUT at GPS_HZ for `seconds`, then print a status line.
    """
    dt       = 1.0 / GPS_HZ
    end_time = time.time() + seconds
    while time.time() < end_time:
        send_gps_input(conn, lat, lon, alt)
        time.sleep(dt)
    print(f"  {step:>3}/{total}  {label:<26}  {lat:.7f}  {lon:.7f}"
          f"  {maps_url(lat, lon)}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate GPS movement via MAVLink GPS_INPUT injection")
    parser.add_argument("--port",     default=DEFAULT_PORT)
    parser.add_argument("--baud",     type=int,   default=DEFAULT_BAUD)
    parser.add_argument("--step",     type=float, default=STEP_METRES,
                        help=f"Metres per step (default {STEP_METRES})")
    parser.add_argument("--interval", type=float, default=INTERVAL_S,
                        help=f"Seconds at each waypoint (default {INTERVAL_S})")
    parser.add_argument("--steps",    type=int,   default=STEPS_OUT,
                        help=f"Steps north (same count south, default {STEPS_OUT})")
    args = parser.parse_args()

    # ── connect ───────────────────────────────────────────────────────────────
    print(f"Connecting to {args.port} @ {args.baud} baud …", end=" ", flush=True)
    conn = mavutil.mavlink_connection(args.port, baud=args.baud)
    hb   = conn.wait_heartbeat(timeout=10)
    if not hb:
        print("FAILED — no heartbeat.")
        sys.exit(1)
    print(f"OK  sys={conn.target_system} comp={conn.target_component}")

    # ── read home GPS (real, before spoofing starts) ──────────────────────────
    print("Reading real GPS home position …", end=" ", flush=True)
    home = get_real_gps(conn, timeout=30)
    if home is None:
        print("ERROR: no GPS fix.  Ensure Pixhawk GPS has sky view.")
        sys.exit(1)
    home_lat, home_lon, home_alt = home
    print(f"OK\n")

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
    print("  NOTE: Pixhawk must have GPS1_TYPE = 14 (MAVLink)")
    print("        Set via QGroundControl → Parameters → GPS1_TYPE = 14")
    print("        Then reboot Pixhawk before running this script.")
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
    try:
        for idx, (lat, lon, alt, label) in enumerate(waypoints, 1):
            hold_position(conn, lat, lon, alt, args.interval, label, idx, total)

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
