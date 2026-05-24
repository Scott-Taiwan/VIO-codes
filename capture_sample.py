#!/usr/bin/env python3
"""
capture_sample.py — Capture a single photo from the CSI camera on Jetson.

Usage:
    python3 capture_sample.py                    # saves sample_YYYYMMDD_HHMMSS.jpg
    python3 capture_sample.py --out my_photo.jpg # saves to custom filename
    python3 capture_sample.py --wait 5           # waits 5 seconds before capturing
                                                 # (time to reach altitude)
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate=30/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=True"
    )

def capture(out_path: str, wait_sec: int = 0):
    import cv2

    if wait_sec > 0:
        print(f"Waiting {wait_sec} seconds — fly to ~60 m altitude now …")
        for i in range(wait_sec, 0, -1):
            print(f"  {i}s …")
            time.sleep(1)

    pipeline = gstreamer_pipeline()
    print(f"Opening CSI camera …")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERROR: Could not open CSI camera. Check sensor connection.")
        sys.exit(1)

    # Discard first few frames — sensor needs time to settle exposure
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: Failed to capture frame.")
        sys.exit(1)

    cv2.imwrite(out_path, frame)
    h, w = frame.shape[:2]
    size_kb = Path(out_path).stat().st_size // 1024
    print(f"Saved {out_path}  ({w}×{h} px, {size_kb} KB)")
    print(f"\nTo localize:")
    print(f"  ./build/localize {out_path} --zoom 19")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None,
                        help="Output filename (default: sample_YYYYMMDD_HHMMSS.jpg)")
    parser.add_argument("--wait", type=int, default=0,
                        help="Seconds to wait before capturing (fly to altitude first)")
    args = parser.parse_args()

    if args.out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"sample_{timestamp}.jpg"

    capture(args.out, args.wait)
