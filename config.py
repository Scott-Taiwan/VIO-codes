# Taipei city coverage — tighten the bbox to reduce download size for testing
TAIPEI_BBOX = {
    'lat_min': 25.02,
    'lat_max': 25.09,
    'lon_min': 121.43,
    'lon_max': 121.51,
}

# At Taipei latitude (~25°): zoom 17 ≈ 1.1 m/px, zoom 18 ≈ 0.54 m/px
# A drone at 30 m covers ~36 m × 27 m → ~33 × 25 px at z17, ~67 × 50 px at z18
# Use z18 for better matching fidelity; z17 if download size is a concern
ZOOM_LEVEL = 18
TILE_SIZE = 256  # px per tile (standard slippy-map)

TILE_DIR = 'tiles'
INDEX_DIR = 'index'

# ESRI World Imagery — free satellite tiles, no API key required
# Note: ESRI URL order is z/y/x (not z/x/y like OSM)
TILE_SERVER_URL = (
    'https://server.arcgisonline.com/ArcGIS/rest/services/'
    'World_Imagery/MapServer/tile/{z}/{y}/{x}'
)

# SIFT feature matching (much better cross-domain matching than ORB)
SIFT_N_FEATURES = 300   # keypoints per tile — 9k tiles × 100 × 128 × 4 B ≈ 470 MB all_descs
MATCH_RATIO = 0.75      # Lowe's ratio test threshold
MIN_INLIERS = 6         # minimum RANSAC inliers to accept a location fix
TOP_CANDIDATES = 20     # top vote-winning tiles to try homography on
