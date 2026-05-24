# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

GPS-free visual localization for a drone (F450 frame) running on an NVIDIA Jetson Orin Nano. The system captures aerial photos and matches them to pre-downloaded map tiles to determine GPS coordinates, then forwards the position to a Pixhawk flight controller.

**Two-stage implementation (per [Specification.md](Specification.md)):**
- **Stage 1**: Download satellite/map tiles for Taipei, Taiwan; given an input photo, identify and output its GPS coordinates
- **Stage 2**: Live loop — CSI camera on the drone captures a frame at altitude (~30m), the system locates the frame on the map, and transmits the GPS fix to Pixhawk via MAVLink

## Platform Notes (Jetson Orin Nano)

**Python**: Use `/usr/bin/python3` (system Python) for any code that touches the CSI camera — pip-installed OpenCV lacks GStreamer support. Standard `python3` is fine for offline processing.

**PyTorch / CUDA**: Standard PyPI wheels do not work on Jetson. NVIDIA custom builds are required (see `../yolov5/CLAUDE.md` for the exact wheel URLs and torchvision build steps).

**GPU memory**: GPU and CPU share RAM on Jetson Orin. Desktop apps (Chromium, VS Code, gnome-shell) consume NVMM. For GPU inference, close desktop apps or default to CPU.

**CSI camera pipeline**: Uses GStreamer with `nvarguscamerasrc` → `nvvidconv` → `appsink`. Reference implementation at `../CSI-Camera/`. Key parameters: `sensor_id`, `flip_method`, `capture_width/height`, `display_width/height`.

## Expected Architecture

### Map Tile Pipeline
- Source: OpenStreetMap or a satellite tile provider (e.g., Mapbox, ESRI) using slippy map tile URLs (`/z/x/y.png`)
- Storage: tiles saved locally by zoom/x/y coordinates for offline use
- Coverage: Taipei, Taiwan — define bounding box and target zoom levels

### Visual Localization Approaches (choose one)
1. **Template matching**: slide the query photo over stitched map tiles using `cv2.matchTemplate` — simple but slow at scale
2. **Feature matching**: extract keypoints (ORB, SIFT) from both query and map tiles, match descriptors — faster for large areas
3. **Deep learning**: use a pre-trained geo-localization model (e.g., NetVLAD) — most accurate but heavier

### Pixhawk Integration (Stage 2)
- Protocol: MAVLink via `pymavlink` or `dronekit`
- Connection: serial UART from Jetson to Pixhawk (`/dev/ttyTHS*` or `/dev/ttyUSB*`)
- Message to send: `GPS_INPUT` or inject via `GLOBAL_POSITION_INT` depending on Pixhawk configuration

## Key Dependencies (anticipated)

```bash
pip install requests pillow numpy opencv-python-headless pymavlink
# For deep learning approaches, use NVIDIA PyTorch wheel (see ../yolov5/CLAUDE.md)
```

## Related Projects in This Repo

- `../CSI-Camera/` — GStreamer CSI camera capture patterns for Jetson
- `../yolov5/` — Jetson Orin PyTorch/CUDA setup, `detect_csi.py` for live camera inference
- `../yolov5/CLAUDE.md` — NVIDIA PyTorch wheel URLs, torchvision build from source, GPU memory notes
