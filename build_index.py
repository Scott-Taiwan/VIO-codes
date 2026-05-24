"""
Extract SIFT features from every downloaded tile and save a searchable index.

The index is a list of dicts, one per tile:
    {tile_x, tile_y, zoom, kps (serialisable tuples), descs (float32 ndarray)}

Usage:
    python build_index.py            # uses ZOOM_LEVEL from config
    python build_index.py --zoom 17
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import TILE_DIR, ZOOM_LEVEL, INDEX_DIR, SIFT_N_FEATURES


def kp_to_tuple(kp):
    return (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)


def build_index(tile_dir=None, zoom=None, index_dir=None):
    tile_dir = Path(tile_dir or TILE_DIR)
    zoom = zoom or ZOOM_LEVEL
    index_dir = Path(index_dir or INDEX_DIR)
    index_dir.mkdir(exist_ok=True)

    zoom_dir = tile_dir / str(zoom)
    if not zoom_dir.exists():
        raise FileNotFoundError(
            f'No tiles found at {zoom_dir}. Run download_tiles.py first.')

    tile_paths = sorted(zoom_dir.rglob('*.png'))
    print(f'Indexing {len(tile_paths)} tiles (zoom {zoom}) with SIFT ...')

    sift = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)
    index = []
    skipped = 0

    for path in tqdm(tile_paths, unit='tile'):
        # path structure: tiles/{z}/{x}/{y}.png
        x = int(path.parent.name)
        y = int(path.stem)

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        kps, descs = sift.detectAndCompute(img, None)
        if descs is None or len(kps) < 5:
            skipped += 1
            continue

        index.append({
            'tile_x': x,
            'tile_y': y,
            'zoom': zoom,
            'kps': [kp_to_tuple(k) for k in kps],
            'descs': descs.astype(np.float32),  # shape (N, 128)
        })

    out_path = index_dir / f'sift_index_z{zoom}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_kps = sum(len(e['kps']) for e in index)
    print(f'Index saved → {out_path}')
    print(f'  Tiles indexed  : {len(index)}  (skipped {skipped})')
    print(f'  Total keypoints: {total_kps:,}')
    print(f'  Approx. index size: {out_path.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build SIFT feature index from tiles')
    parser.add_argument('--zoom', type=int, default=None)
    parser.add_argument('--tile-dir', default=None)
    parser.add_argument('--index-dir', default=None)
    args = parser.parse_args()

    build_index(tile_dir=args.tile_dir, zoom=args.zoom, index_dir=args.index_dir)
