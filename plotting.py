""" Plotting joints and skeleton

First for loop plots joints one by one in order to check the order of the joints in the array
Second for loop plots the skeleton following the Human3.6M convention. 
Third for loop for plotting the 3D skeleton.
"""

import matplotlib.pyplot as plt
import numpy as np

plot_scatter = False
plot_skeleton = False
plot_3dskeleton = True

angle_x = 50 * (np.pi / 180)
angle_y = 140 * (np.pi / 180)
angle_z = -50 * (np.pi / 180)

rot_x = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
rot_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
rot_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])

data = np.load("data\\mhformer_predictions_S9_Directions1_cam0_original.npz")["data"][0]
# Hip joint is center of coordinates
data[:] -= data[0]

data = data @ rot_x.T @ rot_y.T @ rot_z.T

skeleton = [
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
    [0, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [8, 11],
    [11, 12],
    [12, 13],
    [8, 14],
    [14, 15],
    [15, 16],
]

if plot_scatter:
    fig, axs = plt.subplots(3, 6)
    fig.set_figheight(18)
    fig.set_figwidth(18)
    for i, ax in enumerate(axs.reshape(-1)):
        if i < data.shape[0]:
            ax.set_xlim(0, 960)
            ax.set_ylim(0, 540)
            ax.scatter(data[0 : i + 1, 0], data[0 : i + 1, 1], s=10)

    plt.savefig("scatter_keypoints_notnorm.png")
    plt.clf()

if plot_skeleton:
    fig = plt.figure()
    for joint in skeleton:
        plt.scatter(data[:, 0], data[:, 1])
        plt.plot(
            [data[joint[0], 0], data[joint[1], 0]],
            [data[joint[0], 1], data[joint[1], 1]],
        )
    plt.show()

    plt.savefig("skeleton.png")

if plot_3dskeleton:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for joint in skeleton:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        ax.plot(
            [data[joint[0], 0], data[joint[1], 0]],
            [data[joint[0], 1], data[joint[1], 1]],
            zs=[data[joint[0], 2], data[joint[1], 2]],
        )
    plt.show()

    # plt.savefig("3dskeleton.png")
