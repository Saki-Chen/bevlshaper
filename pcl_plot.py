# Import dependencies
from exploreKITTI.explore_kitti_lib import *
import matplotlib.pyplot as plt
import numpy as np

# Render 2D bird's-eye-view scene from PCL data
def render_2Dbev(frame, data, clusters, lpoints, points, colors):
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


# Render scene as .gif file function
def render_scene_gif(filenames, fps):
    # Render .gif file
    clip = ImageSequenceClip(filenames, fps=fps)
    clip.write_gif('pcl_data.gif', fps=fps)
    return 0