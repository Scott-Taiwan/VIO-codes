"""
Download satellite map tiles for a region and save them to disk.

Usage:
    python download_tiles.py                    # full Taipei bbox from config
    python download_tiles.py --zoom 17          # override zoom
    python download_tiles.py --lat-min 25.03 --lat-max 25.08 \
                             --lon-min 121.50 --lon-max 121.56  # small test area
"""

import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import TAIPEI_BBOX, ZOOM_LEVEL, TILE_DIR, TILE_SERVER_URL
from tile_utils import bbox_to_tiles, meters_per_pixel

HEADERS = {
    'User-Agent': 'GpslessMapping/1.0 (research; jetson-orin-nano)',
    'Accept': 'image/png,image/*',
}
REQUEST_DELAY = 0.05   # seconds between requests — be polite to tile servers
REQUEST_TIMEOUT = 15   # seconds


def download_tile(x, y, z, tile_dir: Path, url_template: str, retries: int = 3) -> bool:
    path = tile_dir / str(z) / str(x) / f'{y}.png'
    if path.exists():
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    url = url_template.format(z=z, y=y, x=x)  # ESRI order: z/y/x

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            path.write_bytes(r.content)
            return True
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f'  Failed ({x},{y}): {e}')
                return False
    return False


def download_tiles(bbox=None, zoom=None, tile_dir=None, url=None):
    bbox = bbox or TAIPEI_BBOX
    zoom = zoom or ZOOM_LEVEL
    tile_dir = Path(tile_dir or TILE_DIR)
    url = url or TILE_SERVER_URL

    tiles = bbox_to_tiles(bbox['lat_min'], bbox['lat_max'],
                          bbox['lon_min'], bbox['lon_max'], zoom)
    lat_center = (bbox['lat_min'] + bbox['lat_max']) / 2
    res = meters_per_pixel(lat_center, zoom)

    print(f'Zoom {zoom}: {res:.2f} m/px  |  {len(tiles)} tiles to download')
    print(f'Tile directory: {tile_dir.resolve()}')

    ok = failed = 0
    with tqdm(total=len(tiles), unit='tile') as pbar:
        for x, y in tiles:
            if download_tile(x, y, zoom, tile_dir, url):
                ok += 1
            else:
                failed += 1
            pbar.update(1)
            pbar.set_postfix(ok=ok, failed=failed)
            time.sleep(REQUEST_DELAY)

    print(f'\nDone. {ok} downloaded, {failed} failed → {tile_dir}/{zoom}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download satellite tiles for Taipei')
    parser.add_argument('--zoom', type=int, default=None)
    parser.add_argument('--lat-min', type=float, default=None)
    parser.add_argument('--lat-max', type=float, default=None)
    parser.add_argument('--lon-min', type=float, default=None)
    parser.add_argument('--lon-max', type=float, default=None)
    parser.add_argument('--tile-dir', default=None)
    args = parser.parse_args()

    bbox = TAIPEI_BBOX.copy()
    if args.lat_min is not None: bbox['lat_min'] = args.lat_min
    if args.lat_max is not None: bbox['lat_max'] = args.lat_max
    if args.lon_min is not None: bbox['lon_min'] = args.lon_min
    if args.lon_max is not None: bbox['lon_max'] = args.lon_max

    download_tiles(bbox=bbox, zoom=args.zoom, tile_dir=args.tile_dir)
