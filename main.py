# Import dependencies
from exploreKITTI.explore_kitti_lib import *
from adaptive_segmentation import *
from pcl_filter import *
import matplotlib.pyplot as plt

# Constants
MAX_RADIUS = 1
POINTS_RATIO = 0.105

# Origin mask
ORIGIN_X = 0
ORIGIN_Y = 0
ORIGIN_CIRCLE_RADIUS = 6.1

# Sample mask
# PATCH_X = 8
# PATCH_Y = 6
# PATCH_WIDTH = 8
# PATCH_HEIGHT = 6

# Full mask
# PATCH_X = -18
# PATCH_Y = -10.5
# PATCH_WIDTH = 80
# PATCH_HEIGHT = 21

# Full lane mask
PATCH_X = -2
PATCH_Y = -1.5
PATCH_WIDTH = 40
PATCH_HEIGHT = 12

# Debug values
COLORS = ['green', 'blue', 'yellow']

# Get PCL from frame function
def get_pcl_from_frame(data, frame, points=POINTS_RATIO):
    points_step = int(1. / points)
    data_range = range(0, data[frame].shape[0], points_step)
    xyz_values = data[frame][data_range, :]
    return xyz_values


# Render 2D bird's-eye-view scene from PCL data
def render_2Dbev(frame, data, clusters, lpoints=[], points=POINTS_RATIO, colors=COLORS):
    # Init figure, define points
    f = plt.figure(figsize=(12, 8))
    axis = f.add_subplot(111, xticks=[], yticks=[])
    point_size = 0.01 * (1. / points)

    # Extract coordinate ranges
    # print(data)
    xy_values = data[:, [0, 1]]
    # z_values = data[:, 3]
    # print(xy_values)

    # Draw scatter plot
    axis.scatter(*np.transpose(xy_values), s=point_size, c='black', cmap='gray')
    axis.set_xlim(*axes_limits[0])
    axis.set_ylim(*axes_limits[1])

    # Draw cluster points
    cnt = 0
    for cluster in clusters:
        cntc = np.mod(cnt, len(colors))
        axis.scatter(*np.transpose(cluster), s=point_size, c=colors[cntc], cmap='gray')
        cnt += 1

    # Draw L-shapes
    for lpoints_cluster in lpoints:
        lp1, lp2, lp3, lp4 = lpoints_cluster[0], lpoints_cluster[1], lpoints_cluster[2], lpoints_cluster[3]
        axis.plot([lp1[0], lp2[0]], [lp1[1], lp2[1]], 'red')
        axis.plot([lp2[0], lp3[0]], [lp2[1], lp3[1]], 'red')
        axis.plot([lp3[0], lp4[0]], [lp3[1], lp4[1]], 'red')
        axis.plot([lp4[0], lp1[0]], [lp4[1], lp1[1]], 'red')

    # Save frame and close plot
    filename = 'video/frame_{0:0>4}.png'.format(frame)
    plt.savefig(filename)
    plt.close(f)
    return filename


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

# Select frame
for frame in range(len(dataset)):
    # Filter point cloud
    velo_frame = get_pcl_from_frame(dataset, frame)
    velo_frame = filter_ground_plane(velo_frame)
    velo_frame = apply_circular_mask(velo_frame, [ORIGIN_X, ORIGIN_Y], ORIGIN_CIRCLE_RADIUS)
    velo_frame = apply_rectangular_mask(velo_frame, [PATCH_X, PATCH_Y], PATCH_WIDTH, PATCH_HEIGHT)

    # Get data
    pcl = list(np.asarray(velo_frame[:, [0, 1]]))

    # Adaptively segment PCL data
    clusters = cluster_kdtree(pcl, alpha, beta, min_cluster_len, MAX_RADIUS)

    # Determine L-shapes from clusters
    delta = 0.02
    # Init empty array for L-shape points
    lpoints = []
    for cluster in clusters:
        [a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4] = search_rectangle_fit(cluster, delta)
        # print("Rectangular parameters:")
        # print([a1, a2, a3, a4, b1, b2, b3, b4, c1, c2])
        [x1, y1] = calc_intersection_point(a1, b1, c1, a2, b2, c2)
        [x2, y2] = calc_intersection_point(a2, b2, c2, a3, b3, c3)
        [x3, y3] = calc_intersection_point(a3, b3, c3, a4, b4, c4)
        [x4, y4] = calc_intersection_point(a4, b4, c4, a1, b1, c1)
        lpoints.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # Print L-shape points
    print("L-shape points:", lpoints)

    # Render 2D BEV scene with colorized clusters
    render_2Dbev(frame, velo_frame, clusters, lpoints)

    # Log
    print("Number of clusters:", len(clusters))