# Import dependencies
from exploreKITTI.explore_kitti_lib import *
from adaptive_segmentation import *
from pcl_filter import *
import matplotlib.pyplot as plt

# Constants
COLORS = ['red', 'green', 'blue', 'yellow']
ALPHA = 0.0257
POINTS_RATIO = 0.1062
MIN_CLUSTER_LEN = 22
MAX_CLUSTER_LEN = 65

ORIGIN_X = 0
ORIGIN_Y = 0
ORIGIN_CIRCLE_RADIUS = 6.1
## Sample
# PATCH_X = 8
# PATCH_Y = 6
# PATCH_WIDTH = 8
# PATCH_HEIGHT = 6
## Full
# PATCH_X = -18
# PATCH_Y = -10.5
# PATCH_WIDTH = 80
# PATCH_HEIGHT = 21
## Full lane
PATCH_X = -2
PATCH_Y = -1.5
PATCH_WIDTH = 40
PATCH_HEIGHT = 12


# Get PCL from frame function
def get_pcl_from_frame(data, frame, points=POINTS_RATIO):
    points_step = int(1. / points)
    data_range = range(0, data[frame].shape[0], points_step)
    xyz_values = data[frame][data_range, :]
    return xyz_values


# Render 2D bird's-eye-view scene from PCL data
def render_2Dbev(frame, data, clusters, points=POINTS_RATIO, colors=COLORS):
    # Init figure, define points
    f = plt.figure(figsize=(12, 8))
    axis = f.add_subplot(111, xticks=[], yticks=[])
    point_size = 0.01 * (1. / points)

    # Extract coordinate ranges
    z_values = data[:, 3]
    xy_values = data[:, [0, 1]]

    # Draw scatter plot
    axis.scatter(*np.transpose(xy_values), s=point_size, c=z_values, cmap='gray')
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

# Select frame
for frame in [28, 38, 53]:
    # Filter point cloud
    velo_frame = get_pcl_from_frame(dataset, frame)
    velo_frame = filter_ground_plane(velo_frame)
    velo_frame = apply_circular_mask(velo_frame, [ORIGIN_X, ORIGIN_Y], ORIGIN_CIRCLE_RADIUS)
    velo_frame = apply_rectangular_mask(velo_frame, [PATCH_X, PATCH_Y], PATCH_WIDTH, PATCH_HEIGHT)

    # Get data
    pcl = list(np.asarray(velo_frame[:, [0, 1]]))

    # Adaptively segment PCL data
    clusters = cluster_kdtree(pcl, ALPHA, MIN_CLUSTER_LEN, MAX_CLUSTER_LEN)
    print("Number of clusters:", len(clusters))

    # Render 2D BEV scene with colorized clusters
    render_2Dbev(frame, velo_frame, clusters)