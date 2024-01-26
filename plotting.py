""" Plotting joints and skeleton

First for loop plots joints one by one in order to check the order of the joints in the array
Second for loop plots the skeleton following the Human3.6M convention. 
"""


import matplotlib.pyplot as plt
import numpy as np

plot_scatter = False
plot_skeleton = False
plot_3dskeleton = True

data = np.load("prediction.npz")["data"][0]
fig, axs = plt.subplots(3, 6)
fig.set_figheight(18)
fig.set_figwidth(18)

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
    for i, ax in enumerate(axs.reshape(-1)):
        if i < data.shape[0]:
            ax.set_xlim(0, 960)
            ax.set_ylim(0, 540)
            ax.scatter(data[0 : i + 1, 0], data[0 : i + 1, 1], s=10)

    plt.savefig("scatter_keypoints_notnorm.png")
    plt.clf()

if plot_3dskeleton:
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

    plt.savefig("3dskeleton.png")
