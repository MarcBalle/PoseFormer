import argparse

import numpy as np

from common.visualization import render_animation
from common.arguments import parse_args
from common.skeleton import Skeleton
from common.camera import *

h36m_skeleton = Skeleton(
    parents=[
        -1,
        0,
        1,
        2,
        3,
        4,
        0,
        6,
        7,
        8,
        9,
        0,
        11,
        12,
        13,
        14,
        12,
        16,
        17,
        18,
        19,
        20,
        19,
        22,
        12,
        24,
        25,
        26,
        27,
        28,
        27,
        30,
    ],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
)

h36m_skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
h36m_skeleton._parents[11] = 8
h36m_skeleton._parents[14] = 8


if __name__ == "__main__":
    kpts_filepath = (
        "data/data_2d_h36m_gt_altered.npz"  # Same format that the npz you get from VidePose3D/data/prepare_data_h36m.py
    )
    pred_filepath = (
        "data/mhformer_predictions_S9_Directions1_cam0_altered.npz"  # 3D poses with shape (nframes x 17 x 3)
    )

    # Visualization arguments
    viz_args = parse_args()

    keypoints = np.load(kpts_filepath, allow_pickle=True)
    keypoints_metadata = keypoints["metadata"].item()
    keypoints_metadata.update(layout_name="h36m")
    keypoints = keypoints["positions_2d"].item()

    # Select keypoinys from a specific subject, action and camera (change whenever needed)
    keypoints = keypoints["S9"]["Directions 1"][0]

    prediction = np.load(pred_filepath)["data"]

    # Extrinsics for the specific camera
    cam = {
        "orientation": [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
        "translation": [2044.45849609375, 4935.1171875, 1481.2275390625],
    }

    # Since I am not loading the ground truth data (I could, but it is not that necessary),
    # I follow the same method of visualization used by the author when there are no extrinsics
    # for the specific subject and camera.
    prediction = camera_to_world(prediction, R=cam["orientation"], t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    anim_output = {"Reconstruction": prediction}

    azimuth = np.array(70.0, dtype="float32")
    width, height = 1000, 1002  # Change whenever needed

    # Predictions might not be computed for the whole video sequence
    if keypoints.shape[0] != prediction.shape[0]:
        keypoints = keypoints[0 : prediction.shape[0]]

    render_animation(
        keypoints,
        keypoints_metadata,
        anim_output,
        h36m_skeleton,
        50,
        viz_args.viz_bitrate,
        azimuth,
        viz_args.viz_output,
        limit=viz_args.viz_limit,
        downsample=viz_args.viz_downsample,
        size=viz_args.viz_size,
        input_video_path=viz_args.viz_video,
        viewport=(width, height),
        input_video_skip=viz_args.viz_skip,
    )
