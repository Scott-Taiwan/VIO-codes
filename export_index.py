"""
Convert sift_index_z{zoom}.pkl to a flat binary file that C++ can read.

Binary format:
  int32  num_tiles
  per tile:
    int32  tile_x
    int32  tile_y
    int32  zoom
    int32  n_descs
    float32[n_descs * 128]  descriptors

Usage:
    python export_index.py
"""

import pickle
import struct
import numpy as np
from pathlib import Path
from config import INDEX_DIR, ZOOM_LEVEL


def export_index(zoom=ZOOM_LEVEL):
    pkl_path = Path(INDEX_DIR) / f'sift_index_z{zoom}.pkl'
    bin_path = Path(INDEX_DIR) / f'sift_index_z{zoom}.bin'

    print(f'Loading {pkl_path} ...')
    with open(pkl_path, 'rb') as f:
        index = pickle.load(f)

    print(f'Exporting {len(index)} tiles → {bin_path} ...')
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<i', len(index)))
        for entry in index:
            descs = entry['descs'].astype(np.float32)
            n = descs.shape[0]
            f.write(struct.pack('<iiii',
                                entry['tile_x'], entry['tile_y'], entry['zoom'], n))
            f.write(descs.tobytes())

    size_mb = bin_path.stat().st_size / 1e6
    print(f'Done  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zoom', type=int, default=ZOOM_LEVEL)
    args = parser.parse_args()
    export_index(args.zoom)
