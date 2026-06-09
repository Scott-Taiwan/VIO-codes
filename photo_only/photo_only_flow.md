# photo_only.cpp — Process Flow Diagram

## Purpose

Measure the accuracy of SIFT visual localization against real Pixhawk GPS.
**No commands are sent to Pixhawk.** Read-only connection.

---

## Startup (runs once)

```
main()
  │
  ├─ Parse CLI args
  │    --port  /dev/ttyTHS1   (MAVLink serial to Pixhawk)
  │    --baud  57600
  │    --zoom  18             (tile zoom level for ESRI index)
  │
  ├─ Create result/ output directory
  │
  ├─ Load tile index from disk
  │    ../index/sift_index_z{zoom}.bin
  │    → vector<TileEntry>  (tile_x, tile_y, zoom, descriptors)
  │    Each entry = one map tile with its SIFT feature descriptors
  │
  ├─ build_flann()
  │    Stack all tile descriptors into one big matrix
  │    Build cv::FlannBasedMatcher  KD-tree (5 trees, 50 checks)
  │    train() — done once, reused every frame
  │
  ├─ open_serial()
  │    POSIX termios  8N1  100 ms read timeout
  │
  ├─ Start mavlink_reader thread  ──────────────────────────────────────┐
  │                                                                      │
  ├─ Wait until first GLOBAL_POSITION_INT received                      │
  │    (blocks until Pixhawk GPS is ready)                              │
  │                                                                      │
  └─ Open CSI camera                                                    │
       GStreamer:                                                        │
       nvarguscamerasrc → nvvidconv → appsink                          │
                                                                        │
                                          ┌─────────────────────────────┴──────┐
                                          │  mavlink_reader  (background thread) │
                                          │                                      │
                                          │  loop forever:                       │
                                          │    read 1 byte from serial fd        │
                                          │    mavlink_parse_char()              │
                                          │    if GLOBAL_POSITION_INT:           │
                                          │      VehicleState.update(            │
                                          │        lat, lon, relative_alt)       │
                                          │                                      │
                                          │  (read-only — never writes to        │
                                          │   Pixhawk)                           │
                                          └──────────────────────────────────────┘
```

---

## Main Loop (100 ms tick)

```
while (!g_shutdown)
  │
  ├─ state.snapshot()  →  alt_m, lat, lon
  │
  ├─ Altitude hysteresis
  │    alt >= 30 m  AND  not yet capturing
  │      → capturing_active = true
  │      → log "capture started"
  │
  │    alt < 20 m   AND  currently capturing
  │      → capturing_active = false
  │      → log "capture stopped"
  │
  │                    ┌─────────────────────────────┐
  │                    │  Altitude state machine      │
  │                    │                             │
  │                    │   ground/low                │
  │                    │   alt < 20 m                │
  │                    │   capturing = OFF  ◄───┐    │
  │                    │        │               │    │
  │                    │        │ rises to 30 m │    │
  │                    │        ▼               │    │
  │                    │   airborne              │    │
  │                    │   alt >= 30 m           │    │
  │                    │   capturing = ON  ──────┘    │
  │                    │   (drops below 20 m)         │
  │                    └─────────────────────────────┘
  │
  ├─ [capturing_active  AND  elapsed >= 5 s]
  │    │
  │    │  ① Read true GPS from Pixhawk
  │    │       state.snapshot() → real_lat, real_lon
  │    │
  │    │  ② Grab frame from CSI camera
  │    │       cap.read(frame)   1280 × 720 BGR
  │    │
  │    │  ③ SIFT localization  ──────────────────────────────────────┐
  │    │       localize_frame(frame, index, fi, zoom)                 │
  │    │       returns MatchResult{lat, lon, inliers}  or  nullopt    │
  │    │                                                              │
  │    │  ④ Compute error distance                                    │
  │    │       dist_m = haversine_m(real_lat, real_lon,               │
  │    │                            est_lat,  est_lon)                │
  │    │                                                              │
  │    │  ⑤ Save photo to result/                                     │
  │    │       filename encodes all measurement data:                 │
  │    │                                                              │
  │    │       SIFT found a match:                                    │
  │    │         {lat_real}_{lon_real}-{lat_est}_{lon_est}__{dist}m.png
  │    │       Example:                                               │
  │    │         25.0612340_121.4710050-25.0608120_121.4706330__48.2m.png
  │    │                                                              │
  │    │       SIFT found no match:                                   │
  │    │         {lat_real}_{lon_real}-NOFIX_NOFIX__NAm.png           │
  │    │                                                              │
  │    └─ (no data sent to Pixhawk)                                  │
  │                                                                   │
  └─ sleep 100 ms                                                     │
                                                                      │
  ┌───────────────────────────────────────────────────────────────────┘
  │  localize_frame()
  │
  ├─ Convert frame to grayscale
  │
  ├─ Phase 1 — FLANN coarse vote
  │    cv::SIFT::create(2000) → detectAndCompute
  │    → kps_q (keypoints), descs_q (128-dim descriptors)
  │
  │    flann.knnMatch(descs_q, k=2)
  │    Lowe ratio filter: keep match if
  │      distance < 0.75 × second-best distance
  │
  │    Count votes per tile index
  │    Sort tiles by vote count  →  ranked[]
  │
  │              query frame descriptors
  │                      │
  │              FLANN KD-tree search
  │                      │
  │         ┌────────────┼────────────┐
  │         ▼            ▼            ▼
  │       tile A       tile B       tile C  …  (all tiles in index)
  │       votes=12     votes=8      votes=3
  │         │
  │         └── top 20 candidates passed to Phase 2
  │
  └─ Phase 2 — Direct match on top 20 candidate tiles
       for each candidate tile (cx, cy):
         │
         ├─ stitch_tiles()
         │    Load 3×3 grid of PNG tiles from ../tiles/{zoom}/{x}/{y}.png
         │    → 768×768 composite image
         │
         │         [tile][tile][tile]
         │         [tile][c,cy][tile]   ← centre tile = best candidate
         │         [tile][tile][tile]
         │
         ├─ SIFT on composite  detectAndCompute (3000 features)
         │
         ├─ cv::BFMatcher  knnMatch query ↔ composite  (k=2)
         │    Lowe ratio filter → good matches
         │
         ├─ cv::findHomography (RANSAC, 5 px threshold)
         │    → H (3×3 transform matrix)
         │    → mask (per-match inlier flags)
         │    → inliers = count of inlier matches
         │
         ├─ Sanity checks:
         │    projected corners must form a convex shape
         │    projected area in range [5% … 5000%] of query area
         │    projected centre must be inside composite ± 256 px
         │
         ├─ [all checks pass]
         │    Project query image centre through H
         │    → composite pixel coordinates (cpx, cpy)
         │    → which sub-tile does (cpx, cpy) fall in?
         │    → pixel_to_latlon(local_px, local_py, tile_x, tile_y, zoom)
         │    → (est_lat, est_lon, inliers)
         │    RETURN — stop trying more candidates
         │
         └─ [checks fail]
              try next candidate tile
              all 20 failed → return NOFIX
```

---

## Output File Naming

```
result/
  │
  ├─ 25.0612340_121.4710050-25.0608120_121.4706330__48.2m.png
  │   │                      │                      │
  │   │                      │                      └─ error distance (metres)
  │   │                      └─ SIFT estimated GPS (lat, lon)
  │   └─ True GPS from Pixhawk (lat, lon)
  │
  └─ 25.0612340_121.4710050-NOFIX_NOFIX__NAm.png
                             └─ SIFT found no match this frame
```

---

## Altitude Thresholds

```
Altitude (m)
    │
 30 ┤ ──────────────── START threshold ──────────────────
    │  drone rises through 30 m → capturing_active = true
    │
    │  [capturing zone]
    │  photo every 5 s
    │  SIFT runs on every frame
    │
 20 ┤ ──────────────── STOP threshold ───────────────────
    │  drone drops below 20 m → capturing_active = false
    │
    │  [no capture zone]
    │
  0 ┤  ground
```

The 10 m hysteresis gap (20–30 m) prevents rapid on/off switching
when the drone hovers near the threshold altitude.

---

## What Is NOT Done (by design)

| Action                        | drone_localize | photo_only |
|-------------------------------|:--------------:|:----------:|
| Send GPS_INPUT to Pixhawk     | ✓              | ✗          |
| Fetch mission from Pixhawk    | ✓              | ✗          |
| Validate waypoint spacing     | ✓              | ✗          |
| Send navigation commands      | ✓              | ✗          |
| Influence Pixhawk in any way  | ✓              | ✗          |

`photo_only` is a passive observer — it reads from Pixhawk but never writes.
