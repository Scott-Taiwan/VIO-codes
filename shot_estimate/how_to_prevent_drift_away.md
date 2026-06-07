# How to Prevent the Drone from Flying in the Wrong Direction Due to SIFT Estimation Error

## Assumption

**GPS signal is blocked at all times.** SIFT visual localization is the *only*
position source for Pixhawk. The system must keep sending GPS_INPUT even when
match quality is low — stopping entirely would cause Pixhawk's EKF to lose its
position source and abort AUTO mode.

---

## The Concern

When SIFT returns a wrong estimated location, the Pixhawk uses that wrong position to
calculate the bearing to the next waypoint. The worry is: does the drone lock onto a
fixed wrong bearing angle (e.g. 60°) and never correct it — flying away indefinitely?

---

## Key Fact: Pixhawk Does NOT Lock the Bearing

Pixhawk recalculates the bearing to the target **every control cycle (~10 Hz)**:

```
bearing = atan2(target_lat - current_lat, target_lon - current_lon)
```

Every 100 ms it asks: *"where am I now, where is the target, what angle?"*
The bearing is **not fixed** — it is continuously updated from the current position.

So the real question is: **how much does a 50 m position error distort the bearing?**

---

## Geometry of the Bearing Error

```
Scenario: target is 500 m north, SIFT has 50 m east error

         T  (target)
         │  ↖
         │    \  ← Pixhawk calculates THIS bearing (slightly wrong)
         │     \
         │      A'  ← what SIFT reports (50 m east of truth)
         │      /
         │     /
         A    (true position)
         ←50m→

  True bearing  = atan2(500,   0) = 0°   (due north)
  Wrong bearing = atan2(500, -50) = 354° (6° west of north)
  Bearing error ≈ arctan(50 / 500) ≈ 6°
```

The bearing error shrinks as the drone gets farther from the error source,
and grows as it gets close to the target:

| Distance to target | 50 m SIFT error | Bearing error | Status        |
|--------------------|----------------|---------------|---------------|
| 500 m              | 50 m           | ~6°           | ✓ Safe        |
| 200 m              | 50 m           | ~14°          | ✓ Acceptable  |
| 100 m              | 50 m           | ~27°          | ⚠ Getting bad |
|  50 m              | 50 m           | ~45°          | ✗ Dangerous   |
|  20 m              | 50 m           | ~68°          | ✗ Very wrong  |

**The danger zone is when SIFT error ≥ distance to waypoint.**

---

## The Real Risk: Sudden Outlier Fixes

The truly dangerous scenario is not steady 50 m error, but a sudden **outlier fix**
(SIFT returns a completely wrong tile, e.g. 300 m error):

```
SIFT returns bad fix (300 m error)
         │
         ▼
Pixhawk thinks it is far from where it actually is
         │
         ▼
Pixhawk commands full throttle in wrong direction
         │
         ▼
Drone flies away fast before SIFT can correct it
         │
         ▼
Next frame: another bad fix from new wrong location
         │
         ▼
Drone is now 500 m off course and accelerating  ← real danger
```

---

## Four Implemented Solutions

### 1. hacc Gradient — Express Confidence, Never Hard-Block

Every fix with ≥ 10 RANSAC inliers is sent to Pixhawk via `GPS_INPUT`.
Confidence is expressed through the `horiz_accuracy` (hacc) field.
Pixhawk's EKF blends the SIFT fix with IMU proportional to hacc:
a large hacc means the IMU carries the frame; a small hacc means the fix
is trusted strongly.

```
RANSAC inliers   hacc sent   Pixhawk EKF behaviour
──────────────────────────────────────────────────────────────
> 80             10 m        Strongly trusts the SIFT fix
40 – 80          30 m        Trusts fix, light IMU blend
20 – 39          60 m        Weak fix — IMU does most work
10 – 19          100 m       Very weak — IMU almost entirely
< 10             not sent    Too few points for geometry
NOFIX            not sent    No match found at all
```

When not sent (< 10 inliers or NOFIX): Pixhawk dead-reckons on
barometer + IMU for that single frame. The next frame supplies a new fix.
Pixhawk stays in AUTO mode throughout — it never loses its position source.

### 2. Raw Fix — No Moving Average

An earlier design averaged the last 3 valid fixes (moving average), but
this introduces **lag error** for a moving drone. If each frame is 25 m
apart (5 m/s × 5 s interval), averaging 3 positions gives a result ~25 m
behind the drone's true current location — comparable to the SIFT error
it was trying to eliminate.

The current implementation sends the raw fix directly to Pixhawk with no
buffering. The hacc gradient is the outlier protection; averaging on top
only adds lag.

### 3. Waypoint Spacing Validation

Keep waypoints at least 300 m apart. At that distance, even 50 m SIFT
error causes only ~10° bearing error — well within acceptable tolerance.

At startup, both programs check every consecutive waypoint pair and log
a warning if any pair is closer than `MIN_WAYPOINT_SPACING` (300 m).
Waypoints are auto-fetched from the Pixhawk mission at startup
(`MISSION_REQUEST_LIST` protocol); `--waypoints` is a manual override.

### 4. Final Approach — IMU Dead-Reckons the Last Segment

LOITER mode was considered but removed: it stops the drone 100 m short
of the target instead of letting it arrive.

When the drone is close to the target, SIFT match quality naturally
degrades (fewer inliers, possible NOFIX). The hacc rises or GPS_INPUT
stops entirely. Pixhawk's EKF then relies on IMU dead-reckoning for
the final segment. IMU drift over a short distance (~50–100 m at low
speed) is small enough to complete the approach. No extra command needed.

---

## What Is Implemented in Code

Both `drone_localize.cpp` and `drone_localize.py`:

1. **hacc gradient** — send every fix ≥ 10 inliers; quality expressed via `horiz_accuracy`
2. **Raw fix** — no moving average; fix sent immediately each frame
3. **Waypoint spacing check** — warn at startup if any pair < 300 m apart
4. **Mission auto-fetch** — reads active mission from Pixhawk at startup via MAVLink

### Constants

| Constant               | Value | Meaning                                      |
|------------------------|-------|----------------------------------------------|
| `MIN_INLIERS_SEND`     | 10    | Below this, geometry is useless — don't send |
| `MIN_WAYPOINT_SPACING` | 300 m | Warn if waypoints closer than this           |

### hacc mapping

| Inliers | hacc sent | EKF behaviour               |
|---------|-----------|-----------------------------|
| > 80    | 10 m      | Fix strongly trusted        |
| 40–80   | 30 m      | Fix trusted, light IMU      |
| 20–39   | 60 m      | IMU does most work          |
| 10–19   | 100 m     | IMU almost entirely         |
| < 10    | not sent  | IMU dead-reckons this frame |
| NOFIX   | not sent  | IMU dead-reckons this frame |

---

## Summary

| Scenario                          | Outcome                                        |
|-----------------------------------|------------------------------------------------|
| Strong SIFT fix (> 80 inliers)    | Pixhawk trusts fix, navigates accurately       |
| Weak SIFT fix (10–39 inliers)     | Sent with high hacc; IMU carries the frame     |
| Single bad frame (NOFIX / < 10)   | Not sent; IMU dead-reckons for one frame       |
| Prolonged NOFIX streak            | Pixhawk dead-reckons on IMU; drifts slowly     |
| Good fix > 300 m from target      | ~6° bearing error — safe navigation            |
| Drone within 50 m of target       | hacc rises / NOFIX; IMU completes the approach |

---

## drone_localize.cpp — Process Flow

### Startup (runs once, single thread)

```
main()
  │
  ├─ Parse CLI args
  │    --port  /dev/ttyTHS1   (MAVLink serial)
  │    --baud  57600
  │    --zoom  18             (tile zoom level)
  │    --waypoints "lat,lon …" (optional override)
  │
  ├─ If --waypoints given → parse_waypoints() → validate_waypoints()
  │    (warn if any pair < 300 m apart)
  │
  ├─ Load tile index from disk
  │    index/sift_index_z{zoom}.bin
  │    → vector<TileEntry>  (tile_x, tile_y, zoom, descriptors)
  │
  ├─ build_flann()
  │    Stack all tile descriptors into one matrix
  │    cv::FlannBasedMatcher  KD-tree (5 trees, 50 checks)
  │    train() — one-time cost, reused every frame
  │
  ├─ open_serial()  POSIX termios, 8N1, 100 ms read timeout
  │
  ├─ wait_for_heartbeat()
  │    Read bytes until MAVLINK_MSG_ID_HEARTBEAT received
  │
  ├─ fetch_mission_from_pixhawk()   (skipped if --waypoints given)
  │    Send MISSION_REQUEST_LIST
  │    Receive MISSION_COUNT  (N items)
  │    For each item i = 0 … N-1:
  │      Send MISSION_REQUEST_INT(i)
  │      Receive MISSION_ITEM_INT
  │      Skip item 0 (home) and non-NAV_WAYPOINT commands
  │    Send MISSION_ACK
  │    → waypoints[]   (last entry = final target)
  │
  ├─ validate_waypoints()   warn if pair < 300 m
  │
  ├─ Start mavlink_reader thread  ──────────────────────────────────────┐
  │                                                                      │
  ├─ Wait for first GLOBAL_POSITION_INT from reader thread              │
  │                                                                      │
  └─ Open CSI camera (GStreamer pipeline)                               │
       nvarguscamerasrc → nvvidconv → appsink                          │
                                                                        │
                                                                        ▼
                                              ┌─────────────────────────────────────┐
                                              │  mavlink_reader thread (background) │
                                              │                                     │
                                              │  loop:                              │
                                              │    read 1 byte from serial_fd       │
                                              │    mavlink_parse_char()             │
                                              │    if GLOBAL_POSITION_INT:          │
                                              │      VehicleState.update(           │
                                              │        lat, lon, relative_alt)      │
                                              └─────────────────────────────────────┘
```

### Main Flight Loop (100 ms tick)

```
while (!g_shutdown)
  │
  ├─ state.snapshot()  →  alt_m, lat, lon
  │
  ├─ Takeoff detection
  │    alt >= 50 m  AND  was_on_ground
  │      → capturing_active = true
  │
  ├─ Landing detection
  │    alt < 10 m
  │      → capturing_active = false
  │
  ├─ [capturing_active AND elapsed >= 5 s]
  │    │
  │    ├─ Snapshot real GPS (lat_real, lon_real) from VehicleState
  │    │
  │    ├─ cap.read(frame)   CSI camera grab
  │    │
  │    ├─ localize_frame()  ──────────────────────────────────────────┐
  │    │    returns MatchResult{lat, lon, inliers}  or  nullopt        │
  │    │                                                               │
  │    ├─ [has_fix AND inliers >= 10]                                  │
  │    │    hacc = hacc_from_inliers(inliers)                         │
  │    │    send_gps_input(fd, est_lat, est_lon, hacc)                │
  │    │    → MAVLink GPS_INPUT message → Pixhawk EKF                 │
  │    │                                                               │
  │    ├─ [has_fix AND inliers < 10]                                   │
  │    │    log "geometry unreliable — Pixhawk uses IMU"              │
  │    │                                                               │
  │    ├─ [NOFIX]                                                      │
  │    │    log "NOFIX — Pixhawk uses IMU"                            │
  │    │                                                               │
  │    └─ cv::imwrite()  save photo as:                               │
  │         {lat_real}_{lon_real}-{lat_est}_{lon_est}__{dist}m.png   │
  │                                                                    │
  └─ sleep 100 ms                                                      │
                                                                       │
  ┌────────────────────────────────────────────────────────────────────┘
  │  localize_frame()
  │
  ├─ Phase 1 — FLANN coarse vote
  │    cv::cvtColor → grayscale
  │    cv::SIFT::create(2000) → detectAndCompute → kps_q, descs_q
  │    flann.knnMatch(descs_q, k=2)
  │    Lowe ratio filter  (distance < 0.75 × second-best)
  │    Count votes per tile  →  ranked[] sorted by vote count
  │
  └─ Phase 2 — Direct match on top 20 candidates
       for each candidate tile (cx, cy):
         │
         ├─ stitch_tiles()
         │    Load 3×3 grid of 256×256 PNG tiles from disk
         │    → 768×768 composite image
         │
         ├─ SIFT on composite  detectAndCompute  (3000 features)
         │
         ├─ cv::BFMatcher knnMatch query ↔ composite  (k=2)
         │    Lowe ratio filter → good[]
         │
         ├─ cv::findHomography(RANSAC, threshold=5 px)
         │    → H (3×3), mask (inlier flags)
         │    inliers = countNonZero(mask)
         │
         ├─ Sanity checks on H:
         │    isContourConvex(projected corners)
         │    area in range [0.05 × query_area … 50 × query_area]
         │    projected centre inside composite ± 256 px margin
         │
         ├─ [pass] Project query centre through H
         │    composite pixel → tile coords → pixel_to_latlon()
         │    → (lat, lon, inliers)   RETURN
         │
         └─ [fail] try next candidate
              all failed → return nullopt  (NOFIX)
```

### hacc → GPS_INPUT → Pixhawk EKF

```
inliers   hacc    What Pixhawk EKF does
────────  ──────  ─────────────────────────────────────────────
> 80      10 m    Strongly trusts the SIFT fix
40–80     30 m    Trusts fix, light IMU blend
20–39     60 m    Weak fix — IMU carries most of the frame
10–19     100 m   Very weak — IMU almost entirely
< 10      —       GPS_INPUT not sent; IMU dead-reckons this frame
NOFIX     —       GPS_INPUT not sent; IMU dead-reckons this frame
```
