#!/usr/bin/env python3
"""
gps_check.py — Read GPS location from Pixhawk and print a Google Maps URL.

Auto-tries every common connection type: USB, UART, and UDP.
Prints latitude, longitude, altitude, and two Google Maps links.

Usage:
  python3 gps_check.py                          # auto-detect
  python3 gps_check.py --port /dev/ttyACM0      # USB (Linux)
  python3 gps_check.py --port /dev/ttyTHS1      # Jetson UART
  python3 gps_check.py --port udp:0.0.0.0:14550 # MAVLink over UDP
  python3 gps_check.py --port tcp:192.168.1.1:5760
  python3 gps_check.py --loop                   # keep printing every 2 s
"""

import argparse
import json
import os
import sys
import time

HOME_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_home.json")

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not installed.  Run:  pip install pymavlink pyserial")
    sys.exit(1)

# ── auto-probe list ───────────────────────────────────────────────────────────
# Each entry: (connection_string_for_mavutil, label)
AUTO_CANDIDATES = [
    # (port_or_url, baud_or_None, label)
    # USB (PX4 / ArduPilot via CDC-ACM)
    ("/dev/ttyACM0",         115200, "/dev/ttyACM0 @ 115200"),
    ("/dev/ttyACM0",          57600, "/dev/ttyACM0 @ 57600"),
    ("/dev/ttyACM1",         115200, "/dev/ttyACM1 @ 115200"),
    # USB-UART adapter
    ("/dev/ttyUSB0",          57600, "/dev/ttyUSB0 @ 57600"),
    ("/dev/ttyUSB0",         115200, "/dev/ttyUSB0 @ 115200"),
    # Jetson Orin UART (TELEM port)
    ("/dev/ttyTHS0",          57600, "/dev/ttyTHS0 @ 57600"),
    ("/dev/ttyTHS0",         115200, "/dev/ttyTHS0 @ 115200"),
    ("/dev/ttyTHS1",          57600, "/dev/ttyTHS1 @ 57600"),
    ("/dev/ttyTHS1",         115200, "/dev/ttyTHS1 @ 115200"),
    ("/dev/ttyTHS2",          57600, "/dev/ttyTHS2 @ 57600"),
    ("/dev/ttyTHS2",         115200, "/dev/ttyTHS2 @ 115200"),
    # UDP — MAVLink broadcast (QGroundControl default ports)
    ("udpin:0.0.0.0:14550",   None,  "UDP port 14550"),
    ("udpin:0.0.0.0:14540",   None,  "UDP port 14540"),
    ("udpin:0.0.0.0:14560",   None,  "UDP port 14560"),
    # TCP — SITL / companion-computer relay
    ("tcp:127.0.0.1:5760",    None,  "TCP 127.0.0.1:5760"),
]


def try_connect(port, baud, timeout=4):
    """Return a mavutil connection if a heartbeat arrives within timeout."""
    try:
        if baud:
            conn = mavutil.mavlink_connection(port, baud=baud)
        else:
            conn = mavutil.mavlink_connection(port)
        hb = conn.wait_heartbeat(timeout=timeout)
        if hb:
            return conn
    except Exception:
        pass
    return None


def request_streams(conn):
    """
    Ask ArduPilot to start streaming position data on this port.
    By default TELEM3 stream rates are 0 — nothing is sent until requested.
    """
    conn.mav.request_data_stream_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        4,   # 4 Hz
        1    # start
    )
    # Also request ALL streams at a low rate as fallback
    conn.mav.request_data_stream_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        2,   # 2 Hz
        1
    )


def get_gps(conn, timeout=30):
    """
    Request position stream then block until GLOBAL_POSITION_INT arrives.
    Returns (lat_deg, lon_deg, rel_alt_m, alt_msl_m) or None on timeout.
    """
    request_streams(conn)

    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        if msg is None:
            # Re-request every 5 s in case the first request was missed
            request_streams(conn)
            continue
        lat     = msg.lat          / 1e7
        lon     = msg.lon          / 1e7
        rel_alt = msg.relative_alt / 1000.0
        alt_msl = msg.alt          / 1000.0
        if lat == 0.0 and lon == 0.0:
            continue   # no fix yet
        return lat, lon, rel_alt, alt_msl
    return None


def maps_pin(lat, lon):
    """Plain pin URL — works in any browser."""
    return f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}"


def maps_satellite(lat, lon):
    """Satellite view zoomed in."""
    return (f"https://www.google.com/maps/@{lat:.7f},{lon:.7f},100m"
            f"/data=!3m1!1e3")


def print_fix(lat, lon, rel_alt, alt_msl):
    sep = "=" * 62
    print()
    print(sep)
    print("  Pixhawk GPS Fix")
    print(sep)
    print(f"  Latitude    :  {lat:+.7f} deg")
    print(f"  Longitude   :  {lon:+.7f} deg")
    print(f"  Altitude    :  {rel_alt:.1f} m  (above home)")
    print(f"                 {alt_msl:.1f} m  (above sea level)")
    print()
    print("  Google Maps — pin:")
    print(f"    {maps_pin(lat, lon)}")
    print()
    print("  Google Maps — satellite view:")
    print(f"    {maps_satellite(lat, lon)}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Read Pixhawk GPS and print Google Maps links")
    parser.add_argument("--port", default=None,
        help="Connection string: /dev/ttyACM0, udp:0.0.0.0:14550, etc.")
    parser.add_argument("--baud", type=int, default=57600,
        help="Baud rate when --port is a serial device (default 57600)")
    parser.add_argument("--timeout", type=int, default=30,
        help="Seconds to wait for GPS fix (default 30)")
    parser.add_argument("--loop", action="store_true",
        help="Keep printing updates every 2 s")
    args = parser.parse_args()

    # ── connect ───────────────────────────────────────────────────────────────
    conn = None

    if args.port:
        p = args.port
        # UDP/TCP connections pass None as baud; serial ports use args.baud
        baud = None if p.startswith(("udp", "tcp")) else args.baud
        label = f"{p} @ {baud}" if baud else p
        print(f"Connecting: {label} …", end=" ", flush=True)
        conn = try_connect(p, baud, timeout=10)
        if conn is None:
            print("FAILED — no heartbeat received.")
            print("Check that the cable is connected and the Pixhawk is powered.")
            sys.exit(1)
        print("OK")
    else:
        print("Auto-detecting Pixhawk connection …\n")
        for port, baud, label in AUTO_CANDIDATES:
            print(f"  {label:<35} … ", end="", flush=True)
            conn = try_connect(port, baud, timeout=4)
            if conn:
                print("FOUND")
                break
            print("no response")

        if conn is None:
            print()
            print("ERROR: No Pixhawk heartbeat found on any port.")
            print()
            print("Troubleshooting:")
            print("  USB connected?   Check:  ls /dev/ttyACM*")
            print("  UART connected?  Try:    python3 gps_check.py --port /dev/ttyTHS1 --baud 57600")
            print("  UDP (WiFi/Eth)?  Try:    python3 gps_check.py --port udp:0.0.0.0:14550")
            print("  Powered?         Pixhawk 6C Pro needs either USB-C power or")
            print("                   a separate battery — check the PWR LED.")
            sys.exit(1)

    print(f"Heartbeat  sys={conn.target_system} comp={conn.target_component}\n")

    # ── read + print GPS ──────────────────────────────────────────────────────
    if args.loop:
        print("Streaming GPS — press Ctrl-C to stop.\n")
        try:
            while True:
                fix = get_gps(conn, timeout=5)
                ts  = time.strftime("%H:%M:%S")
                if fix:
                    lat, lon, rel_alt, alt_msl = fix
                    print(f"[{ts}]  {lat:+.7f}  {lon:+.7f}  "
                          f"alt={rel_alt:.1f} m    {maps_pin(lat, lon)}")
                else:
                    print(f"[{ts}]  waiting for GPS fix …")
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print(f"Waiting for GPS fix (up to {args.timeout} s) …")
        fix = get_gps(conn, timeout=args.timeout)

        if fix is None:
            print("ERROR: GPS fix not received.")
            print("  • Is the GPS antenna outdoors with clear sky view?")
            print("  • Run with --loop to wait longer.")
            sys.exit(1)

        lat, lon, rel_alt, alt_msl = fix
        # Save to file so gps_sim.py can use it even after GPS1_TYPE=14
        try:
            with open(HOME_CACHE, "w") as f:
                json.dump({"lat": lat, "lon": lon,
                           "alt_msl": alt_msl, "rel_alt": rel_alt,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
            print(f"  (saved to {HOME_CACHE})")
        except Exception:
            pass
        print_fix(lat, lon, rel_alt, alt_msl)


if __name__ == "__main__":
    main()
