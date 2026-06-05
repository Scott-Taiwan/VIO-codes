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

## Git operation

git add .
git commit -m "Initial commit"
git push -u origin main

-git account: scott600@mitac.com.tw, using key to access: the key  was generated in the following process:

Step 1: ssh-keygen -t ed25519 -C "scott600@mitac.com.tw"
Step 2: cat ~/.ssh/id_ed25519.pub
Step 3: use the output in step 2, open the githup web, login into it
and in the setting, security page, deploy key page, to click the adding key, type the name and content
Step 4: test is using shell command:ssh -T git@github.com
Step  5: configure the user information:
git config --global user.email "scott600@mitac.com.tw"
git config --global user.name "scott600"
Step 6: adding files into it using command: 
git remote add origin git@github.com:Scott-Taiwan/VIO-codes.git
Step 7:git add localize.cpp popsift_sift.cpp popsift_sift.h
...
Step 8: git commit -m "Add GPS-free visual localization(zoom19, PopSIFT, CUDA)"
Step 9: git push -u origin main
Step 10: doing them all at a time
git add .
git commit -m "Initial commit"
git push -u origin main


# Task 2
running jetson , when altitude higher than 50 m, start to capture (5 seconds per frame) then search db, finally save the result as file name.

cd /home/scott/claude-project/gpsless_mapping/shot_estimate/

./drone_localize                         # default /dev/ttyTHS1 @ 57600
./drone_localize --port /dev/ttyUSB0     # USB serial adapter
./drone_localize --baud 115200 --zoom 18 # custom baud + zoom level