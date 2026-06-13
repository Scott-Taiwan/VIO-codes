# Session Memo — 2026-06-13 Afternoon

## Goal

Test dual-GPS setup: Jetson Orin Nano sends visual SIFT position estimates to Pixhawk 6C Pro
as GPS2, while real GNSS module stays on GPS1. ArduPilot automatically picks whichever source
has better accuracy (lower hacc).

---

## What Was Done

### 1. Pixhawk Connection — `gps_check.py`
- Connects to Pixhawk via UART (TELEM3 → Jetson ttyTHS1)
- Reads `GLOBAL_POSITION_INT` and prints latitude, longitude, altitude
- Outputs two Google Maps URLs: pin view and satellite view
- Saves last fix to `last_home.json` for use by other scripts
- Tested successfully: `/dev/ttyTHS1 @ 57600` → lat=25.0868109, lon=121.6002978, alt=33.8m MSL

### 2. GPS Simulation — `gps_sim.py`
- Reads home GPS from Pixhawk (or falls back to `last_home.json`)
- Injects simulated GPS positions into Pixhawk via `GPS_INPUT` MAVLink message
- Sends to **GPS2** (`gps_id=1`) with `hacc=0.5m` so ArduPilot prefers GPS2 over GPS1
- Simulates: 10 steps north (20m each, 5s apart) → 10 steps south back to home
- GPS sent at 5 Hz continuously to keep ArduPilot lock alive
- Verified: GPS2_RAW message visible in Mission Planner MAVLink Inspector

### 3. `drone_localize.cpp` — GPS2 Update + Build Fix
- Changed `gps_id` from `0` (GPS1) to `1` (GPS2) in `send_gps_input()`
- Fixed build error: added missing `opaque_id=0` argument to `mavlink_msg_mission_ack_pack()`
- hacc sent based on SIFT inlier count:
  - >80 inliers → 10m hacc
  - 40–79 → 30m hacc
  - 20–39 → 60m hacc
  - 10–19 → 100m hacc
  - <10 → not sent

### 4. Hardware Connection (TELEM3 → Jetson)
- Pixhawk 6C Pro TELEM3 JST-GH 6-pin → Jetson Orin Nano 40-pin header
- TX (pin 3 on TELEM3) → Jetson pin 10 (RX, ttyTHS1)
- RX (pin 2 on TELEM3) → Jetson pin 8  (TX, ttyTHS1)
- GND (pin 6 on TELEM3) → Jetson pin 6  (GND)
- Baud rate: 57600

---

## Pixhawk Parameter Changes

| Parameter | Original Value | New Value | Reason |
|---|---|---|---|
| `SERIAL3_PROTOCOL` | `-1` (disabled) | `2` (MAVLink2) | Enable TELEM3 for Jetson connection |
| `SERIAL3_BAUD` | `57` (57600) | `57` (57600) | Already correct, confirmed |
| `GPS1_TYPE` | `1` (Auto) | `1` (Auto) | Keep real GNSS module on GPS1 — **restored** |
| `GPS2_TYPE` | `0` (None) | `14` (MAVLink) | Accept Jetson GPS_INPUT on GPS2 |
| `GPS_AUTO_SWITCH` | `0` (disabled) | `4` (Best hacc) | Auto-pick GPS with lowest hacc |
| `GPS_PRIMARY` | `0` (GPS1) | `1` (GPS2) | Force GPS2 as primary for simulation testing |

> **Note on GPS1_TYPE**: During earlier testing, GPS1_TYPE was temporarily set to `14`
> to test gps_sim.py. It was changed back to `1` (Auto) for the dual-GPS setup.

---

## How to Restore Original Settings

If you want to go back to the original single-GPS setup (real GPS only, no Jetson injection):

```
SERIAL3_PROTOCOL  =  -1   (disable TELEM3)
GPS1_TYPE         =   1   (Auto — no change needed)
GPS2_TYPE         =   0   (None — disable GPS2)
GPS_AUTO_SWITCH   =   0   (disabled)
GPS_PRIMARY       =   0   (GPS1 as primary)
```
Reboot Pixhawk after changing parameters.

---

## Current Operational Setup (Dual GPS)

```
SERIAL3_PROTOCOL  =   2   (MAVLink2)
SERIAL3_BAUD      =  57   (57600)
GPS1_TYPE         =   1   (Auto — real GNSS module)
GPS2_TYPE         =  14   (MAVLink — Jetson GPS_INPUT)
GPS_AUTO_SWITCH   =   4   (Best hacc wins)
GPS_PRIMARY       =   1   (GPS2 as default — set for simulation testing)
```

> **For real SIFT flight (not simulation)**: change `GPS_PRIMARY` back to `0`.
> GPS_AUTO_SWITCH=4 will automatically pick SIFT (GPS2) when real GPS is blocked
> and real GPS (GPS1) when outdoors with clear sky.

---

## Workflow for Simulation Test

```bash
# Step 1 — save home GPS (GPS1 must have sky view)
python3 gps_check.py --port /dev/ttyTHS1

# Step 2 — run simulation (watch drone move in Mission Planner)
python3 gps_sim.py --port /dev/ttyTHS1 --baud 57600

# Quick test (3 steps, 3 seconds each)
python3 gps_sim.py --port /dev/ttyTHS1 --steps 3 --step 20 --interval 3
```

---

## Files Changed This Session

| File | Change |
|---|---|
| `gps_check.py` | New — reads Pixhawk GPS, saves last_home.json, prints Google Maps URLs |
| `gps_sim.py` | New — simulates GPS movement via GPS_INPUT to GPS2 |
| `shot_estimate/drone_localize.cpp` | gps_id 0→1 (GPS2); fixed mavlink_msg_mission_ack_pack build error |
| `best_altitude/best_altitude.cpp` | New — autonomous altitude survey (5 positions × 5 altitudes) |
| `best_altitude/best_altitude_flow.md` | New — process flow diagram |
