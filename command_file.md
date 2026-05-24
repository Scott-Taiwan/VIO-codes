# Common Commands

---

## C++ Build & Run (faster localization)

### Step 1 — Export index to binary format (once after each index rebuild)
```bash
python export_index.py
```

### Step 2 — Build the C++ binary (once)
```bash
mkdir -p build && cd build && cmake .. && make -j4
```

### Step 3 — Run C++ localizer
```bash
./build/localize ./your_photo.jpg
./build/localize ./your_photo.jpg --show
```

---

## Download a Fresh ESRI Tile at a Specific GPS

Download a single ESRI World Imagery tile at a known GPS coordinate for testing:
```bash
# Default zoom 18
python3 download_tile_at_gps.py <lat> <lon>

# Specify zoom level
python3 download_tile_at_gps.py <lat> <lon> <zoom>

# Example — Screenshot_30 location
python3 download_tile_at_gps.py 25.05452 121.46801
python3 download_tile_at_gps.py 25.05452 121.46801 19
```
The script prints the tile's exact GPS bounds and center so you know the expected answer before running the localizer.

Then test localization against the downloaded tile:
```bash
./build/localize esri_18_219522_112216.png
```

---

## 1. Download Tiles
```bash
# Full Taipei bbox from config.py (default zoom 18)
python3 download_tiles.py

# Custom bounding box (min/max lat and lon)
python3 download_tiles.py --lat-min 25.04 --lat-max 25.06 --lon-min 121.46 --lon-max 121.49

# Custom bounding box with specific zoom level
python3 download_tiles.py --lat-min 25.04 --lat-max 25.06 --lon-min 121.46 --lon-max 121.49 --zoom 18
```
Downloads ESRI satellite tiles into the `tiles/` directory. Skips tiles already on disk.

## 2. Build Index
```bash
python build_index.py
```
Extracts SIFT features from all tiles and saves `index/sift_index_z18.pkl`.

## 3. Localize a Photo
```bash
python localize.py ./your_photo.jpg
```
Add `--show` to also save a match visualisation to `result_tile.png`:
```bash
python localize.py ./your_photo.jpg --show
```
