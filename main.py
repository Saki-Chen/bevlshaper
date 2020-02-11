# Import dependencies
from pcl_data import *
from pcl_filter import *
from pcl_plot import *
from adaptive_segmentation import *

# Constants
MAX_RADIUS = 1
POINTS_RATIO = 0.105

# Origin mask
ORIGIN_X = 0
ORIGIN_Y = 0
ORIGIN_CIRCLE_RADIUS = 6.1

# Full lane mask
PATCH_X = -2
PATCH_Y = -1.5
PATCH_WIDTH = 40
PATCH_HEIGHT = 12

# Debug values
COLORS = ['green', 'blue', 'yellow']

# Choose input data
date = '2011_09_26'
drive = '0001'

# Load dataset
dataset = load_dataset(date, drive)
dataset = list(dataset.velo)

# Set parameters
alpha = 0.4
beta = 1
min_cluster_len = 22

# Init empty filenames array
filenames = []

# Select frame
for frame in range(0, 41):
    # Log
    print('\n', "Frame:", frame)

    # Get point cloud data
    velo_frame = get_pcl_from_frame(dataset, frame, POINTS_RATIO)

    # Filter point cloud
    velo_frame = filter_ground_plane(velo_frame)
    velo_frame = apply_circular_mask(velo_frame, [ORIGIN_X, ORIGIN_Y], ORIGIN_CIRCLE_RADIUS)
    velo_frame = apply_rectangular_mask(velo_frame, [PATCH_X, PATCH_Y], PATCH_WIDTH, PATCH_HEIGHT)

    # Get points from filtered point cloud
    pcl = list(np.asarray(velo_frame[:, [0, 1]]))

    # Adaptively segment PCL data
    clusters = cluster_tree(pcl, alpha, beta, min_cluster_len, MAX_RADIUS)

    # Log
    print("Number of clusters:", len(clusters))

    # Determine L-shapes from clusters
    # Init empty array for L-shape points
    lpoints = []
    # Set angular step parameter
    delta = 0.02
    for cluster in clusters:
        # Get rectangle parameters from rectangle fit search
        [a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4] = search_rectangle_fit(cluster, delta)
        # Calc intersection points of lines represented by rectangle parameters
        [x1, y1] = calc_intersection_point(a1, b1, c1, a2, b2, c2)
        [x2, y2] = calc_intersection_point(a2, b2, c2, a3, b3, c3)
        [x3, y3] = calc_intersection_point(a3, b3, c3, a4, b4, c4)
        [x4, y4] = calc_intersection_point(a4, b4, c4, a1, b1, c1)
        # Collect intersection points
        lpoints.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # Print L-shape points
    print("L-shape points:", lpoints)

    # Render 2D BEV scene with colorized clusters and fitted L-shapes/rectangles
    filenames += [render_2Dbev(frame, velo_frame, clusters, lpoints, POINTS_RATIO, COLORS)]

# Render .gif file from scene
render_scene_gif(filenames, fps=5)