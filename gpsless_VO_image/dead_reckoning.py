#!/usr/bin/env python3
"""
Anchor + dead-reckoning position tracker for GPS-less flight.

Holds a "last known good" absolute position (the anchor: lat/lon/alt +
true compass heading — typically from the SIFT/SuperPoint tile-matcher in
gpsless_mapping / gpsless_superpoint) and walks it forward leg by leg using
the frame-to-frame displacement from visual_odometry.process_pair(). Call
reset_anchor() whenever a fresh absolute fix succeeds, to collapse the
drift that has accumulated since the last one.

Horizontal accuracy (hacc_m) starts at the anchor fix's own accuracy and
grows with every dead-reckoned leg — more on a low-confidence or failed VO
leg than on a confident one — so ArduPilot's GPS_AUTO_SWITCH (which prefers
whichever GPS reports the lowest hacc) will naturally fall back to a fresh
absolute fix over a long, uncertain dead-reckoning chain instead of trusting
a run of guesses.
"""
import math
import time

import config


class PositionTracker:
    def __init__(self, lat, lon, alt_m, heading_deg, hacc_m=None, timestamp=None):
        self.reset_anchor(lat, lon, alt_m, heading_deg, hacc_m, timestamp)

    def reset_anchor(self, lat, lon, alt_m, heading_deg, hacc_m=None, timestamp=None):
        """Collapse all accumulated drift back to a fresh absolute fix.

        heading_deg: TRUE compass heading (deg, clockwise from North) of the
        airframe at the moment this fix was taken — e.g. from Pixhawk's
        ATTITUDE/VFR_HUD message, read at the same time as the anchor photo.
        """
        self.lat = lat
        self.lon = lon
        self.alt_m = alt_m
        self.heading_deg = heading_deg + config.CAMERA_MOUNT_YAW_OFFSET_DEG
        self.hacc_m = config.ANCHOR_HACC_M if hacc_m is None else hacc_m
        self.timestamp = time.time() if timestamp is None else timestamp
        self.legs_since_anchor = 0

    def apply_leg(self, vo_result, dt_s=None):
        """Advance the tracked position by one visual_odometry.process_pair()
        result. Always updates hacc_m (even on failure); only moves
        lat/lon/alt when the leg carries a usable displacement.

        dt_s: seconds elapsed for this leg, if known — used only to report
        vn/ve (m/s) for the caller to hand to send_gps_input(); position
        itself never depends on it.

        Returns a dict: lat, lon, alt_m, hacc_m, vn, ve.
        """
        self.legs_since_anchor += 1

        if not vo_result['ok']:
            self.hacc_m = min(config.HACC_MAX_M, self.hacc_m + config.HACC_PENALTY_NO_MATCH_M)
            return self._state(vn=0.0, ve=0.0)

        # Heading first — this leg's own yaw estimate feeds forward into how
        # its translation gets rotated into North/East below.
        self.heading_deg += vo_result['yaw_change_deg']
        th = math.radians(self.heading_deg)
        dx, dy = vo_result['dx_m'], vo_result['dy_m']  # image-right, image-down

        # Assumes image "up" = airframe forward at heading 0 — see
        # CAMERA_MOUNT_YAW_OFFSET_DEG if the camera isn't mounted that way.
        north_m = -dy * math.cos(th) - dx * math.sin(th)
        east_m = -dy * math.sin(th) + dx * math.cos(th)

        self.lat += north_m / 111111.0
        self.lon += east_m / (111111.0 * math.cos(math.radians(self.lat)))
        self.alt_m += vo_result['altitude_change_m']

        if vo_result['confident']:
            growth = config.HACC_GROWTH_FRACTION_CONFIDENT * vo_result['magnitude_m']
            self.hacc_m = min(config.HACC_MAX_M, math.hypot(self.hacc_m, growth))
        else:
            growth = config.HACC_GROWTH_FRACTION_LOW_CONF * vo_result['magnitude_m']
            self.hacc_m = min(
                config.HACC_MAX_M,
                math.hypot(self.hacc_m, growth) + config.HACC_PENALTY_LOW_CONFIDENCE_M,
            )

        vn = north_m / dt_s if dt_s else 0.0
        ve = east_m / dt_s if dt_s else 0.0
        return self._state(vn=vn, ve=ve)

    def _state(self, vn, ve):
        return {
            'lat': self.lat, 'lon': self.lon, 'alt_m': self.alt_m,
            'hacc_m': self.hacc_m, 'vn': vn, 've': ve,
        }


if __name__ == '__main__':
    # Synthetic self-test: a known 20m-north, confident leg from a known
    # anchor should land within ~1mm at this latitude (flat-earth offset is
    # only an approximation, but it's exact enough at this scale).
    t = PositionTracker(lat=25.0, lon=121.5, alt_m=60.0, heading_deg=0.0)
    fake_leg = {
        'ok': True, 'confident': True,
        'dx_m': 0.0, 'dy_m': -20.0,  # image-up = forward at heading 0 -> 20m north
        'yaw_change_deg': 0.0, 'magnitude_m': 20.0, 'altitude_change_m': 0.0,
    }
    state = t.apply_leg(fake_leg)
    expected_lat = 25.0 + 20.0 / 111111.0
    print(f"expected lat ~{expected_lat:.7f}, got {state['lat']:.7f}, "
          f"lon unchanged: {state['lon']:.7f}, hacc={state['hacc_m']:.2f}m")
    assert abs(state['lat'] - expected_lat) < 1e-6
    assert abs(state['lon'] - 121.5) < 1e-9
    print("OK")
