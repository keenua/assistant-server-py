import os
from typing import Tuple

import bvhio
import numpy as np
from torch import Tensor


def reset_pose(source: str, destination: str):
    print(f'Loading {os.path.basename(source)}')
    root = bvhio.readAsHierarchy(source)
    layout = root.layout()

    joint_map = {joint.Name: joint for joint, _, _ in layout}

    # rest pose correction
    root.loadRestPose()

    joint_map["LeftShoulder"].setEuler((   0,   0,  -90))
    joint_map["RightShoulder"].setEuler((   0,   0,  90))
    joint_map["LeftHandThumb1"].setEuler((   45,   0,  0))
    joint_map["RightHandThumb1"].setEuler((   45,   0,  0))

    root.writeRestPose(recursive=True, keep=['position', 'rotation', 'scale'])

    print('| Write file')
    bvhio.writeHierarchy(path=destination, root=root, frameTime=1/60)


def smooth_stiching(positions: Tensor, rotations: Tensor, prev_anim: dict, nframes: int = 30) -> Tuple[Tensor, Tensor]:
    if prev_anim is None:
        return positions, rotations

    prev_positions = prev_anim['positions']
    prev_rotations = prev_anim['rotations']

    nframes = min(nframes, positions.shape[0])
    interpolated_positions = positions[0:nframes] * (1 - 1.0 / nframes * (nframes - np.arange(nframes))[:, np.newaxis, np.newaxis])
    interpolated_prev_positions = prev_positions[-1] * (1.0 / nframes * (nframes - np.arange(nframes))[:, np.newaxis, np.newaxis])
    positions[0:nframes] = interpolated_positions + interpolated_prev_positions

    interpolated_rotations = rotations[0:nframes] * (1 - 1.0 / nframes * (nframes - np.arange(nframes))[:, np.newaxis, np.newaxis])
    interpolated_prev_rotations = prev_rotations[-1] * (1.0 / nframes * (nframes - np.arange(nframes))[:, np.newaxis, np.newaxis])
    rotations[0:nframes] = interpolated_rotations + interpolated_prev_rotations

    return positions, rotations


if __name__ == "__main__":
    source = "./data/zeggs/styles/relaxed.bvh"
    destination = "./data/zeggs/styles/relaxed_fixed.bvh"
    reset_pose(source, destination)
