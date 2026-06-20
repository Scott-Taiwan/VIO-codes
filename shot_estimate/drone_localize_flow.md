# drone_localize.cpp — Process Flow

## Overview

C++ program that connects to Pixhawk via MAVLink serial, waits for the
drone to reach altitude, then repeatedly captures CSI camera frames and
estimates GPS position via two-phase SIFT matching against pre-downloaded
map tiles. Sends position estimates back to Pixhawk as GPS2.

Build:
```bash
cd /home/scott/claude-project/gpsless_mapping/shot_estimate
make              # CPU SIFT
make gpu          # GPU (PopSIFT + cuBLAS) — current binary
```

Run:
```bash
cd /home/scott/claude-project/gpsless_mapping/shot_estimate

# Standard (5 s interval, zoom 19)
./drone_localize --port /dev/ttyTHS1 --baud 57600 --zoom 19

# Faster photos (3 s interval)
./drone_localize --port /dev/ttyTHS1 --baud 57600 --zoom 19 --interval 3

# Custom photo interval (4 s)
./drone_localize --port /dev/ttyTHS1 --baud 57600 --zoom 19 --interval 4

# Override waypoints manually (skip auto-fetch from Pixhawk)
./drone_localize --port /dev/ttyTHS1 --baud 57600 --zoom 19 --waypoints "25.061,121.471 25.065,121.475"
```

Parameters:
| Parameter | Default | Description |
|---|---|---|
| `--port` | `/dev/ttyTHS1` | Pixhawk serial port (TELEM3) |
| `--baud` | `57600` | Serial baud rate |
| `--zoom` | `18` | Map tile zoom level (use 19) |
| `--interval` | `5` | Seconds between photos |
| `--waypoints` | (from Pixhawk) | Override mission waypoints |

---

## Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `TAKEOFF_ALT` | 30 m | Minimum altitude before capture starts |
| `MIN_CAPTURE_ALT` | 10 m | Below this → assume landing, pause capture |
| `--interval` | 5 s (default) | Time between photos (CLI parameter) |
| `MIN_INLIERS_SEND` | 10 | Minimum RANSAC inliers to send GPS_INPUT |
| `MIN_INLIERS` | 6 | Minimum inliers to accept a homography |
| `TOP_CANDIDATES` | 20 | How many tiles Phase 2 tries per frame |
| `MATCH_RATIO` | 0.75 | Lowe ratio test threshold |
| `CSI_SENSOR_ID` | 0 | Camera connector (CAM0) |
| `DEFAULT_PORT` | /dev/ttyTHS1 | Pixhawk TELEM3 serial port |
| `DEFAULT_ZOOM` | 18 | Map tile zoom (override with --zoom 19) |

---

## Startup (runs once)

```
main()
  │
  ├─ Parse CLI args
  │    --port      /dev/ttyTHS1
  │    --baud      57600
  │    --zoom      19
  │    --waypoints "lat1,lon1 lat2,lon2 ..."  (optional)
  │
  ├─ signal(SIGINT/SIGTERM)  →  g_shutdown = true
  │
  ├─ fs::create_directories("photo_obtained/")
  │
  ├─ [optional] parse + validate --waypoints
  │    warn if consecutive waypoints < 300 m apart
  │
  ├─ [USE_GPU only] GPU init
  │    check MemAvailable ≥ 512 MB
  │    init_popsift()  +  cuda_knn2_match() warm-up
  │    on failure → CPU fallback (g_gpu_sift = false)
  │
  ├─ load_index("../index/sift_index_z19.bin")
  │    read int32: num_tiles
  │    for each tile:
  │      read tile_x, tile_y, zoom, n_descs   (4 × int32)
  │      read descs[n_descs × 128]            (float32)
  │    → vector<TileEntry>  (547 tiles)
  │
  ├─ build_flann(index)
  │    stack all descriptors into one matrix  (N × 128 float32)
  │    cv::FlannBasedMatcher  KDTree(5 trees, 50 checks)
  │    matcher.train()  — done once, reused every frame
  │    → FlannIndex { matcher, offsets[], n_tiles }
  │
  ├─ open_serial(port, baud)
  │    POSIX termios  8N1  100 ms read timeout
  │
  ├─ wait_for_heartbeat(serial_fd)
  │    reads bytes until MAVLink HEARTBEAT from Pixhawk
  │
  ├─ fetch_mission_from_pixhawk(serial_fd)
  │    (skipped if --waypoints override given)
  │    MISSION_REQUEST_LIST → MISSION_COUNT → MISSION_ITEM_INT × N
  │    → vector<Waypoint>  (lat, lon per NAV_WAYPOINT item)
  │    MISSION_ACK sent to Pixhawk
  │    validate_waypoints()  — warns if spacing < 300 m
  │
  ├─ Start MAVLink reader thread  ────────────────────────────────────┐
  │                                                                   │
  ├─ Wait for first GPS fix from Pixhawk                             │
  │    polls VehicleState.gps_ok every 500 ms                        │
  │                                                                   │
  ├─ Open CSI camera (GStreamer)                                     │
  │    nvarguscamerasrc sensor-id=0                                  │
  │    → nvvidconv → BGRx → videoconvert → BGR → appsink            │
  │    1280 × 720 @ 30 fps                                           │
  │                                                                   │
  └─ Enter flight loop  ←───────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │  mavlink_reader thread (background)                                 │
  │                                                                     │
  │  loop until g_shutdown:                                             │
  │    read 1 byte from serial fd                                       │
  │    mavlink_parse_char()                                             │
  │    GLOBAL_POSITION_INT → VehicleState.update(lat, lon, rel_alt)   │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## When Does It Take a Photo?

Photo capture is **time-based (every 5 s)** but **gated by altitude**:

```
Every 100 ms — main loop polls VehicleState:

  auto [alt, lat, lon, gps_ok] = state.snapshot()

  ┌─ Takeoff detection ─────────────────────────────────────────────┐
  │  if was_on_ground AND alt >= 30 m:                              │
  │    capturing_active = true                                       │
  │    was_on_ground    = false                                      │
  │    print "Takeoff confirmed"                                     │
  └─────────────────────────────────────────────────────────────────┘

  ┌─ Landing detection ─────────────────────────────────────────────┐
  │  if alt < 10 m:                                                  │
  │    capturing_active = false                                      │
  │    was_on_ground    = true                                       │
  │    print "capture paused (landing)"                              │
  └─────────────────────────────────────────────────────────────────┘

  ┌─ Capture trigger ───────────────────────────────────────────────┐
  │  if capturing_active AND elapsed >= 5.0 s:                       │
  │    last_capture = now                                            │
  │    → TAKE PHOTO  (see Per-Photo Sequence below)                  │
  └─────────────────────────────────────────────────────────────────┘
```

**Photo is taken every 5 seconds, only while altitude ≥ 30 m.**
No photos during takeoff climb or landing descent.

---

## Altitude State Machine

```
                    alt >= 30 m
  ON GROUND ─────────────────────────► CAPTURING
  (no photo)    takeoff confirmed        (photo every 5 s)
      ▲                                       │
      │               alt < 10 m             │
      └───────────────────────────────────────┘
                   landing detected
```

---

## Per-Photo Sequence

```
Capture triggered (every 5 s while alt ≥ 30 m)
  │
  ├─ 1. Snapshot real GPS
  │       real_lat, real_lon ← VehicleState (from GLOBAL_POSITION_INT)
  │       if gps_ok == false → skip, wait for next interval
  │
  ├─ 2. Grab CSI camera frame
  │       cap.read(frame)  →  1280×720 BGR
  │       if failed → skip
  │
  ├─ 3. localize_frame(frame, index, fi, zoom)
  │   │
  │   ├─ cvtColor(frame, GRAY)
  │   │
  │   ├─ Phase 1 — FLANN coarse vote
  │   │    cv::SIFT::create(2000).detectAndCompute(gray)
  │   │    → kps_q, descs_q
  │   │    if < 5 keypoints → return nullopt
  │   │
  │   │    FlannIndex.knnMatch(descs_q, k=2)
  │   │    Lowe ratio filter: keep if dist[0] < 0.75 × dist[1]
  │   │    vote count per tile → ranked[] sorted by votes
  │   │
  │   └─ Phase 2 — direct match on top 20 candidates
  │        for each candidate tile (cx, cy):
  │          │
  │          ├─ stitch_tiles(cx, cy, zoom, radius=1)
  │          │    load 3×3 grid  tiles/19/{cx±1}/{cy±1}.png
  │          │    → 768×768 BGR composite
  │          │    (missing tiles filled with black — no error)
  │          │
  │          ├─ SIFT on composite  (3000 features)
  │          │
  │          ├─ BFMatcher knnMatch query ↔ composite (k=2)
  │          │    Lowe ratio filter → good matches
  │          │
  │          ├─ findHomography(RANSAC, 5 px threshold)
  │          │    → H + inlier mask
  │          │
  │          ├─ Sanity checks:
  │          │    inliers ≥ 6
  │          │    projected corners form convex shape
  │          │    projected area in [5% … 5000%] of query area
  │          │    projected centre inside composite ± 256 px
  │          │
  │          ├─ [pass] project query centre through H
  │          │          pixel_to_latlon() → (est_lat, est_lon)
  │          │          return MatchResult{lat, lon, inliers}
  │          │
  │          └─ [fail] try next candidate
  │                    all 20 failed → return nullopt  (NOFIX)
  │
  ├─ 4. Send GPS_INPUT to Pixhawk  (if inliers ≥ 10)
  │
  │       hacc = hacc_from_inliers(inliers):
  │         > 80  → hacc =  10 m  (strong — EKF trusts heavily)
  │         40–79 → hacc =  30 m  (good)
  │         20–39 → hacc =  60 m  (weak — EKF mostly uses IMU)
  │         10–19 → hacc = 100 m  (IMU almost entirely)
  │         < 10  → NOT sent
  │
  │       mavlink_msg_gps_input_pack(gps_id=1, lat, lon, hacc)
  │         → GPS2 on Pixhawk
  │         → ArduPilot GPS_AUTO_SWITCH=4 picks GPS1 or GPS2
  │            based on whichever has lower hacc
  │
  └─ 5. Save photo
          filename = {real_lat}_{real_lon}-{est_lat}_{est_lon}__{dist}m.png
          or        {real_lat}_{real_lon}-NOFIX_NOFIX__NAm.png
          saved to  shot_estimate/photo_obtained/
```

---

## GPS Trust Logic

```
GPS_AUTO_SWITCH = 4  (best hacc wins)

Scenario A — GPS-blocked environment (urban canyon, tunnel):
  GPS1 (real GNSS) hacc = 100–500 m  (degraded)
  GPS2 (Jetson SIFT) hacc = 10–30 m  (good match)
  → Pixhawk uses GPS2

Scenario B — Clear sky outdoors:
  GPS1 (real GNSS) hacc = 1–3 m      (excellent)
  GPS2 (Jetson SIFT) hacc = 10–100 m
  → Pixhawk uses GPS1

Scenario C — SIFT no fix:
  GPS_INPUT not sent
  → Pixhawk uses GPS1 + IMU (EKF dead-reckoning)
```

---

## Output Files

### Photos (`photo_obtained/`)

```
{real_lat}_{real_lon}-{est_lat}_{est_lon}__{dist}m__{alt}m.png
e.g.  25.0868109_121.6002978-25.0867500_121.6003100__8.2m__34.5m.png

{real_lat}_{real_lon}-NOFIX_NOFIX__NAm__{alt}m.png
e.g.  25.0868109_121.6002978-NOFIX_NOFIX__NAm__34.5m.png
```
The final `__{alt}m` segment is the real barometric altitude at the moment of capture.

### Log file (same folder as executable)

Named by start timestamp: `20260620_102031.log`

```
=== drone_localize 20260620_102031 ===
port=/dev/ttyTHS1 baud=57600 zoom=19 interval=5s

[10:20:31] [startup] GPS fix: lat=25.0868109 lon=121.6002978 alt=1.2 m
[10:21:45] Takeoff confirmed — alt=31.4 m  lat=25.0868109 lon=121.6002978  capture every 5 s
[10:21:50] --------------------------------------------------
[10:21:50] Capture — alt=33.1 m  GPS=25.0868109,121.6002978
[10:21:50]   Frame: 1280x720
[10:21:50]   Starting SIFT localization …
[10:21:50]   [P1] extracted 1423 keypoints
[10:21:50]   [P1] FLANN voted on 87 tile(s)
[10:21:50]   [P1] top 5 candidates:
[10:21:50]        #1 tile(54871,27442) votes=34
[10:21:50]        #2 tile(54872,27442) votes=18
[10:21:50]   [P2] FAIL  tile(54871,27442) votes=34 — inliers 4 < 6
[10:21:50]   [P2] MATCH tile(54872,27442) votes=18 — OK inliers=21
[10:21:50]   SIFT result: lat=25.0867500 lon=121.6003100 inliers=21 dist=8.2 m
[10:21:50]   Saved: photo_obtained/25.0868109_121.6002978-25.0867500_121.6003100__8.2m__33.1m.png
[10:26:12] Landing detected — alt=8.3 m < 10.0 m — capture paused, closing log.
```

Log is closed automatically on landing (alt < 10 m) or on Ctrl-C / SIGTERM.

---

## Comparison: C++ vs Python version

| Feature | drone_localize.cpp | drone_localize.py |
|---|:---:|:---:|
| SIFT backend | CPU or GPU (PopSIFT) | CPU only |
| MAVLink | C library (header-only) | pymavlink |
| Speed | Faster | Slower |
| Trigger altitude | ≥ 30 m | ≥ 30 m |
| Capture interval | 5 s | 5 s |
| Sends GPS_INPUT | ✓ (GPS2, gps_id=1) | ✓ (GPS1, gps_id=0) |
| Mission auto-fetch | ✓ | ✓ |
| Waypoint spacing check | ✓ | ✓ |
