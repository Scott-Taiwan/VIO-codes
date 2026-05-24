import math


def deg2tile(lat_deg, lon_deg, zoom):
    """Convert WGS-84 lat/lon to slippy-map tile (x, y) at given zoom."""
    lat_rad = math.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile2deg(x, y, zoom):
    """Return the NW-corner lat/lon of tile (x, y) at given zoom."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def tile_bounds(x, y, zoom):
    """Return (lat_nw, lon_nw, lat_se, lon_se) for tile (x, y)."""
    lat_nw, lon_nw = tile2deg(x, y, zoom)
    lat_se, lon_se = tile2deg(x + 1, y + 1, zoom)
    return lat_nw, lon_nw, lat_se, lon_se


def pixel_to_latlon(px, py, tile_x, tile_y, zoom, tile_size=256):
    """
    Convert a pixel position within a tile to WGS-84 lat/lon.
    (0, 0) is the top-left (NW) corner of the tile.
    """
    lat_nw, lon_nw, lat_se, lon_se = tile_bounds(tile_x, tile_y, zoom)
    lat = lat_nw + (lat_se - lat_nw) * py / tile_size
    lon = lon_nw + (lon_se - lon_nw) * px / tile_size
    return lat, lon


def bbox_to_tiles(lat_min, lat_max, lon_min, lon_max, zoom):
    """
    Return all (x, y) tile coords that cover the given bounding box.
    Note: tile y increases southward, so lat_max → smaller y.
    """
    x_min, y_max = deg2tile(lat_min, lon_min, zoom)
    x_max, y_min = deg2tile(lat_max, lon_max, zoom)
    return [(x, y) for x in range(x_min, x_max + 1)
                   for y in range(y_min, y_max + 1)]


def meters_per_pixel(lat_deg, zoom, tile_size=256):
    """Ground resolution in metres/pixel at the given latitude and zoom."""
    return 156543.03 * math.cos(math.radians(lat_deg)) / (2 ** zoom)
