#!/usr/bin/env python3
"""
download_tile_at_gps.py — Download an ESRI World Imagery tile at a specific GPS coordinate.

Usage:
    python3 download_tile_at_gps.py <lat> <lon> [zoom]

Examples:
    python3 download_tile_at_gps.py 25.05452 121.46801
    python3 download_tile_at_gps.py 25.05452 121.46801 18
    python3 download_tile_at_gps.py 25.05452 121.46801 19

Output:
    - Saves tile as  esri_<z>_<x>_<y>.png
    - Prints tile bounds and center GPS so you know the exact location
"""

import sys
import math
import requests
from pathlib import Path

ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services"
    "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; tile-downloader)"}


def gps_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds(z: int, x: int, y: int) -> dict:
    """Return the NW and SE corners plus the center of a tile."""
    n = 2 ** z

    def _lat(yi):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yi / n))))

    def _lon(xi):
        return xi / n * 360.0 - 180.0

    nw = (_lat(y),     _lon(x))
    se = (_lat(y + 1), _lon(x + 1))
    center = ((_lat(y) + _lat(y + 1)) / 2, (_lon(x) + _lon(x + 1)) / 2)
    return {"nw": nw, "se": se, "center": center}


def download_tile(z: int, x: int, y: int, out_path: Path) -> bool:
    url = ESRI_URL.format(z=z, y=y, x=x)
    print(f"Downloading: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except requests.RequestException as e:
        print(f"ERROR: {e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 download_tile_at_gps.py <lat> <lon> [zoom=18]")
        sys.exit(1)

    lat  = float(sys.argv[1].strip(","))
    lon  = float(sys.argv[2].strip(","))
    zoom = int(sys.argv[3]) if len(sys.argv) >= 4 else 18

    x, y = gps_to_tile(lat, lon, zoom)
    bounds = tile_bounds(zoom, x, y)

    out_path = Path(f"esri_{zoom}_{x}_{y}.png")

    print(f"\nInput GPS  : {lat:.6f}, {lon:.6f}")
    print(f"Zoom level : {zoom}")
    print(f"Tile       : z={zoom}  x={x}  y={y}")
    print(f"Tile NW    : {bounds['nw'][0]:.6f}, {bounds['nw'][1]:.6f}")
    print(f"Tile SE    : {bounds['se'][0]:.6f}, {bounds['se'][1]:.6f}")
    print(f"Tile center: {bounds['center'][0]:.6f}, {bounds['center'][1]:.6f}")
    print(f"Output file: {out_path}\n")

    ok = download_tile(zoom, x, y, out_path)
    if ok:
        size_kb = out_path.stat().st_size // 1024
        print(f"\nSaved {out_path}  ({size_kb} KB)")
        print(f"\nTo test localization:")
        print(f"  ./build/localize {out_path}")
        print(f"\nExpected GPS answer: {bounds['center'][0]:.6f}, {bounds['center'][1]:.6f}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
