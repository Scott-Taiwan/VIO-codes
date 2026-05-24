# Readme.md

This file provides what has been learnt.


## download ESRI to construct index

# Full Taipei bbox from config.py (default)
python3 download_tiles.py

# Custom bbox with default zoom (18)
python3 download_tiles.py --lat-min 25.04 --lat-max 25.06 --lon-min 121.46 --lon-max 121.49

# Custom bbox with specific zoom level
python3 download_tiles.py --lat-min 25.04 --lat-max 25.06 --lon-min 121.46 --lon-max 121.49 --zoom 18


## Build the index

python3 build_index.py

## validate using a png file
C++
./build/localize esri_18_219522_112216.png
or C++
./build/localize ./your_photo.jpg
or python3
python3 localize.py ./your_photo.jpg



## download the ESRI for a specific GPS and zoom level

Download ESRI photo to construct a image index is the most important
command is: python3 download_tile_at_gps.py <lat> <lon> <zoom>

like:
python3 download_tile_at_gps.py 25.05452 121.46801 19


## check with google map


https://www.google.com/maps/@25.0426818,121.4737701,18z





## Key Findings — Zoom Level and Drone Altitude

### Use zoom 19 with ~60 m flight altitude

- **ESRI zoom 19** resolution: ~0.27 m/px at Taipei (25°N)
- At **60 m altitude**, the drone's IMX219 camera captures a ground footprint
  of roughly 70 m × 55 m — close to one zoom-19 tile (69 m × 69 m).
  This scale overlap gives SIFT enough matching keypoints for reliable localization.
- Zoom 18 (0.54 m/px) requires ~540 m altitude to match — too high for a small drone.
- Zoom 19 is the practical sweet spot for low-altitude drone flights (~60 m).

### Image source must match the index source

- The index is built from **ESRI World Imagery** tiles.
- Test images must also come from **ESRI** (not Google Maps, Bing, etc.).
- Google Maps uses different color grading and rendering — SIFT descriptors are
  incompatible, causing false positives (e.g. 6 inliers instead of 100+).
- Real drone camera footage (raw optical) behaves like ESRI and matches well.

### Trust the inlier count

| Inliers | Meaning |
|---------|---------|
| < 10    | False positive — reject |
| 10–50   | Weak match — treat with caution |
| 50+     | Reliable GPS fix |
| 200+    | Excellent match (same-domain imagery) |

Recommended: raise `MIN_INLIERS` to **15–20** for production deployment to avoid
false positives when the image source doesn't perfectly match the index.

---

## Refinement
# Default zoom 18 (unchanged behaviour)
./build/localize photo.jpg

# Explicit zoom level
./build/localize photo.jpg --zoom 19
./build/localize photo.jpg --zoom 18 --show

# Works with multiple images too
./build/localize img1.png img2.png --zoom 18