import os

import bvhio


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


if __name__ == "__main__":
    source = "./data/zeggs/styles/old.bvh"
    destination = "./data/zeggs/styles/old_fixed.bvh"
    reset_pose(source, destination)
