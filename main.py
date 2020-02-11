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
COLORS = ['red', 'green', 'blue', 'yellow']

# Get PCL from frame function
def get_pcl_from_frame(data, frame, points=POINTS_RATIO):
    points_step = int(1. / points)
    data_range = range(0, data[frame].shape[0], points_step)
    xyz_values = data[frame][data_range, :]
    return xyz_values


# Render 2D bird's-eye-view scene from PCL data
def render_2Dbev(frame, data, clusters, bboxes=[], points=POINTS_RATIO, colors=COLORS):
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
# for frame in range(len(dataset)):
for frame in [32]:
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
    for cluster in clusters:
        [a1, a2, a3, a4, b1, b2, b3, b4, c1, c2] = search_rectangle_fit(cluster, delta)
        print("Cluster - rectangular fit:")
        print([a1, a2, a3, a4, b1, b2, b3, b4, c1, c2])

    # Render 2D BEV scene with colorized clusters
    bboxes = []
    render_2Dbev(frame, velo_frame, clusters, bboxes)

    # Log
    print("Number of clusters:", len(clusters))