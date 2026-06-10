# best_altitude.cpp — Process Flow Diagram

## Purpose

Autonomously fly the drone through a 5×5 grid of (position, altitude) combinations,
measure SIFT visual localization error at each point against real Pixhawk GPS,
and report which altitude gives the best accuracy.

**Sends flight commands to Pixhawk** (unlike `photo_only`, this program controls the drone).

---

## Test Matrix

```
Positions (all within 30 m of home):          Altitudes at each position:
  home  (0 m N,  0 m E)                         20 m
  30N   (30 m N, 0 m E)                          30 m
  30E   (0 m N, 30 m E)                          40 m
  30S   (30 m S, 0 m E)                          50 m
  30W   (0 m N, 30 m W)                          60 m

Total: 5 positions × 5 altitudes = up to 25 test shots
Time budget: 110 s from takeoff → any unvisited cell is marked "--"
```

---

## Startup (runs once)

```
main()
  │
  ├─ Parse CLI args
  │    --port  /dev/ttyTHS1   (MAVLink serial to Pixhawk)
  │    --baud  57600
  │    --zoom  19             (ESRI tile zoom level)
  │
  ├─ Create result/ output directory
  │
  ├─ Load tile index from disk
  │    ../index/sift_index_z19.bin
  │    → vector<TileEntry>  (tile_x, tile_y, zoom, descriptors)
  │
  ├─ build_flann()
  │    Stack all tile descriptors into one matrix
  │    Build cv::FlannBasedMatcher  KD-tree (5 trees, 50 checks)
  │    train() — done once, reused for every frame
  │
  ├─ open_serial()
  │    POSIX termios  8N1  100 ms read timeout
  │
  ├─ Start mavlink_reader thread  ──────────────────────────────────────────┐
  │                                                                         │
  ├─ Start heartbeat_sender thread  ───────────────────────────────────────┤
  │                                                                         │
  ├─ Open CSI camera                                                        │
  │    GStreamer: nvarguscamerasrc → nvvidconv → appsink                   │
  │                                                                         │
  ├─ Wait for Pixhawk GPS (GLOBAL_POSITION_INT)  timeout 60 s              │
  │                                                                         │
  └─ Save home_lat, home_lon from first GPS fix                            │
                                                                            │
  ┌─────────────────────────────────────────────────────────────────────────┴──┐
  │  mavlink_reader  (background thread)                                        │
  │                                                                             │
  │  loop:                                                                      │
  │    read 1 byte from serial fd                                               │
  │    mavlink_parse_char()                                                     │
  │    GLOBAL_POSITION_INT  →  VehicleState.update_pos(lat, lon, rel_alt)      │
  │    HEARTBEAT (not GCS)  →  VehicleState.update_hb(base_mode, custom_mode)  │
  │      armed = base_mode & MAV_MODE_FLAG_SAFETY_ARMED                        │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  heartbeat_sender  (background thread)                                       │
  │                                                                             │
  │  every 1 s:  send MAVLink HEARTBEAT  (type=GCS, sys=255, comp=190)         │
  │  Purpose: prevents ArduCopter GCS-failsafe from triggering                 │
  │           (default timeout = 5 s without GCS heartbeat)                    │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Flight Sequence

```
── Arm & Takeoff ─────────────────────────────────────────────────────────────

  send MAV_CMD_DO_SET_MODE  (GUIDED = custom mode 4)
  wait 2 s for mode to take effect

  send MAV_CMD_COMPONENT_ARM_DISARM  (param1 = 1)
  wait_armed()  timeout 12 s
    → polls VehicleState.armed  every 200 ms

  send MAV_CMD_NAV_TAKEOFF  (alt = 20 m)
  wait_altitude(20 m)  timeout 35 s
    → polls VehicleState.alt_m  every 200 ms
    → success when |alt - 20| ≤ 2.5 m

  record flight_start = now()

── Survey Loop ────────────────────────────────────────────────────────────────

  for each position  pi = 0 … 4   (home, 30N, 30E, 30S, 30W)
  │
  │  if pi > 0:
  │    compute (pos_lat, pos_lon) = offset_latlon(home, north_m, east_m)
  │    send SET_POSITION_TARGET_GLOBAL_INT  to (pos_lat, pos_lon, alt=20 m)
  │      ← drone descends from 60 m AND flies horizontally simultaneously
  │    wait_reached(pos_lat, pos_lon, 20 m)  timeout 40 s
  │      → polls every 200 ms
  │      → success when horiz dist ≤ 3 m  AND  |alt - 20| ≤ 2.5 m
  │    if not reached: skip entire position, continue to next
  │
  │  for each altitude  ai = 0 … 4   (20, 30, 40, 50, 60 m)
  │  │
  │  │  check elapsed time  ──── if ≥ 110 s: goto RTL ────
  │  │
  │  │  send SET_POSITION_TARGET_GLOBAL_INT  to (same pos, new alt)
  │  │  wait_reached(pos_lat, pos_lon, tgt_alt)  timeout 40 s
  │  │    → success when horiz dist ≤ 3 m  AND  |alt - tgt_alt| ≤ 2.5 m
  │  │  if not reached: mark result[pi][ai] = NOFIX, continue
  │  │
  │  │  sleep 1500 ms  (stabilise)
  │  │
  │  │  snapshot real GPS  ←  VehicleState.lat, lon, alt_m
  │  │
  │  │  cap.read(frame)  ←  CSI camera  1280×720 BGR
  │  │
  │  │  localize_frame(frame)  ──────────────────────────────────────────────┐
  │  │    returns MatchResult{lat, lon, inliers}  or  nullopt               │
  │  │                                                                       │
  │  │  if match found:                                                      │
  │  │    dist_m = haversine_m(real_gps, estimated_gps)                     │
  │  │    result[pi][ai] = dist_m                                            │
  │  │  else:                                                                │
  │  │    result[pi][ai] = NOFIX  (= -1.0)                                  │
  │  │                                                                       │
  │  │  save photo to result/                                                │
  │  │    filename = {lat_r}_{lon_r}-{lat_e}_{lon_e}__{dist}m               │
  │  │               _pos{name}_alt{alt}m.png                               │
  │  └──                                                                     │
  └──                                                                        │
                                                                             │
── RTL ────────────────────────────────────────────────────────────────────── │
                                                                             │
  send MAV_CMD_NAV_RETURN_TO_LAUNCH                                          │
  (Pixhawk climbs to RTL altitude, flies home, lands automatically)          │
                                                                             │
── Summary ─────────────────────────────────────────────────────────────────  │
                                                                             │
  print_summary()  ← described in Results section below                     │
                                                                             │
── Cleanup ──────────────────────────────────────────────────────────────────  │
                                                                             │
  g_shutdown = true                                                          │
  join mavlink_reader thread                                                 │
  join heartbeat_sender thread                                               │
  cap.release()                                                              │
  close(serial_fd)                                                           │
                                                                             │
  ┌──────────────────────────────────────────────────────────────────────────┘
  │  localize_frame()
  │
  ├─ cvtColor(frame, GRAY)
  │
  ├─ Phase 1 — FLANN coarse vote
  │    cv::SIFT::create(2000) → detectAndCompute(gray)
  │    → kps_q (keypoints),  descs_q (128-dim descriptors)
  │
  │    flann.knnMatch(descs_q, k=2)
  │    Lowe ratio filter: keep if dist < 0.75 × second-best
  │
  │    vote count per tile  →  sorted ranked[]
  │
  └─ Phase 2 — Direct match on top 20 candidate tiles
       for each candidate tile (cx, cy):
         │
         ├─ stitch_tiles()
         │    load 3×3 grid from ../tiles/19/{cx±1}/{cy±1}.png
         │    → 768×768 composite image
         │
         ├─ SIFT on composite  (3000 features)
         │
         ├─ BFMatcher knnMatch query ↔ composite  (k=2)
         │    Lowe ratio filter → good matches
         │
         ├─ findHomography (RANSAC, 5 px threshold)
         │    → H  +  inlier mask
         │
         ├─ Sanity checks:
         │    inliers ≥ 6
         │    projected corners form convex shape
         │    projected area in range [5% … 5000%] of query area
         │    projected centre inside composite ± 256 px
         │
         ├─ [pass]  project query centre through H
         │           pixel_to_latlon() → (est_lat, est_lon)
         │           RETURN MatchResult{lat, lon, inliers}
         │
         └─ [fail]  try next candidate
                    all 20 failed → return nullopt  (NOFIX)
```

---

## Inter-Position Flight Path

```
Each transition: fly_to(next_pos, alt=20 m)
  → drone descends and moves horizontally at the same time
  → arrival check: horiz dist ≤ 3 m  AND  |alt−20| ≤ 2.5 m

Altitude profile across the survey:

 60 m ─────────────────────────────────────────────────────
      home↑              30N↑              30E↑
 50 m  ↑  ↑               ↑  ↑               ↑  ↑
       │  │               │  │               │  │
 40 m  │  │               │  │               │  │
       │  │               │  │               │  │
 30 m  │  │               │  │               │  │
       │  │               │  │               │  │
 20 m ─┘  └──────────────→┘  └──────────────→┘  └── ...
      ↑   ↑              ↑   ↑
    tkoff │            transit│  (simultaneous descent + horizontal flight)
          │                   │
      5 shots             5 shots
      at home             at 30N


Time budget: 110 s from takeoff.
Typical coverage within 2 minutes:
  • 2 full positions (10 shots) — home + 30N
  • Partial 3rd position if SIFT is fast
```

---

## Results

```
result/ directory — one PNG per (position, altitude) test:

  {lat_real}_{lon_real}-{lat_est}_{lon_est}__{dist}m_pos{name}_alt{alt}m.png

  Examples:
    25.0612340_121.4710050-25.0608120_121.4706330__18.2m_poshome_alt40m.png
    25.0631000_121.4710050-25.0628500_121.4708000__22.3m_pos30N_alt40m.png
    25.0612340_121.4710050-NOFIX_NOFIX__NAm_poshome_alt20m.png


Console summary table:

  ============================================================
    SURVEY RESULTS  (SIFT error in metres)
  ============================================================
  Alt(m)  home      30N       30E       30S       30W
  ------------------------------------------------------------
  20       45.2      38.1      --        --        --
  30       32.8      28.4      --        --        --
  40       18.2      22.3      --        --        --
  50       24.1      20.8      --        --        --
  60       31.5      28.9      --        --        --
  ============================================================

  Per-altitude mean error (tested positions only):
     20 m : 41.7 m  (n=2)
     30 m : 30.6 m  (n=2)
     40 m : 20.3 m  (n=2)   ← best average
     50 m : 22.5 m  (n=2)
     60 m : 30.2 m  (n=2)

  BEST : 40 m at home  (error = 18.2 m)
  ============================================================

  "--"    = drone did not reach this point within time budget
  "NOFIX" = drone reached the point but SIFT found no match
```

---

## Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_FLIGHT_SECS` | 110 s | RTL forced after this time from takeoff |
| `REACH_HORIZ_M` | 3.0 m | Horizontal arrival threshold |
| `REACH_ALT_M` | 2.5 m | Altitude arrival tolerance |
| `WAIT_REACHED_S` | 40 s | Max wait for any single fly-to command |
| `WAIT_TAKEOFF_S` | 35 s | Max wait for initial takeoff altitude |
| `WAIT_ARMED_S` | 12 s | Max wait after arm command |
| `STAB_MS` | 1500 ms | Stabilise pause before each photo |
| `DEFAULT_ZOOM` | 19 | ESRI tile zoom level |
| `MIN_INLIERS` | 6 | Minimum RANSAC inliers to accept a match |
| `TOP_CANDIDATES` | 20 | How many tiles Phase 2 tries per frame |

---

## Comparison: best_altitude vs photo_only vs drone_localize

| Feature | drone_localize | photo_only | best_altitude |
|---|:---:|:---:|:---:|
| Sends GPS_INPUT to Pixhawk | ✓ | ✗ | ✗ |
| Sends flight commands | ✓ | ✗ | ✓ |
| Reads real GPS for comparison | ✗ | ✓ | ✓ |
| Autonomous arm + takeoff | ✗ | ✗ | ✓ |
| Flies to test positions | ✗ | ✗ | ✓ |
| Sends GCS heartbeat | ✗ | ✗ | ✓ |
| Sweeps multiple altitudes | ✗ | ✗ | ✓ |
| Reports best altitude | ✗ | ✗ | ✓ |
| Real-time navigation | ✓ | ✗ | ✗ |
