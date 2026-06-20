#!/usr/bin/env python3
"""
convert_index.py — Convert Python pickle SIFT index to C++ binary format.

Binary layout (little-endian):
    int32   num_tiles
    for each tile:
        int32   tile_x
        int32   tile_y
        int32   zoom
        int32   n_descs
        float32 descs[n_descs * 128]

Usage:
    python3 convert_index.py
    python3 convert_index.py --zoom 19
    python3 convert_index.py --input index/sift_index_z19.pkl --output index/sift_index_z19.bin
"""

import argparse
import os
import pickle
import struct
import sys
import numpy as np

def convert(input_path, output_path):
    print(f"Loading  {input_path} …")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        print("ERROR: expected a list of tile entries in the pickle file.")
        sys.exit(1)

    print(f"  {len(data)} tiles found.")

    with open(output_path, "wb") as f:
        # num_tiles
        f.write(struct.pack("<i", len(data)))

        for i, entry in enumerate(data):
            tile_x = int(entry["tile_x"])
            tile_y = int(entry["tile_y"])
            zoom   = int(entry["zoom"])
            descs  = entry["descs"]

            # Ensure float32 and C-contiguous
            descs = np.ascontiguousarray(descs, dtype=np.float32)
            n_descs = descs.shape[0]

            if descs.ndim != 2 or descs.shape[1] != 128:
                print(f"  WARNING: tile {i} has unexpected descriptor shape {descs.shape} — skipping.")
                continue

            f.write(struct.pack("<i", tile_x))
            f.write(struct.pack("<i", tile_y))
            f.write(struct.pack("<i", zoom))
            f.write(struct.pack("<i", n_descs))
            f.write(descs.tobytes())

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Written  {output_path}  ({size_mb:.1f} MB)")
    print(f"Done — {len(data)} tiles converted.")


def main():
    parser = argparse.ArgumentParser(description="Convert pickle SIFT index to C++ binary")
    parser.add_argument("--zoom",   type=int, default=19)
    parser.add_argument("--input",  default=None,
                        help="Input  .pkl  (default: index/sift_index_z{zoom}.pkl)")
    parser.add_argument("--output", default=None,
                        help="Output .bin  (default: index/sift_index_z{zoom}.bin)")
    args = parser.parse_args()

    base    = os.path.dirname(os.path.abspath(__file__))
    in_path  = args.input  or os.path.join(base, "index", f"sift_index_z{args.zoom}.pkl")
    out_path = args.output or os.path.join(base, "index", f"sift_index_z{args.zoom}.bin")

    if not os.path.exists(in_path):
        print(f"ERROR: input file not found: {in_path}")
        sys.exit(1)

    convert(in_path, out_path)


if __name__ == "__main__":
    main()
