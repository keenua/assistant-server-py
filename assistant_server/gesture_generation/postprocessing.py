import os

import bvhio


def reset_pose(source: str, destination: str):
    print(f'Loading {os.path.basename(source)}')
    root = bvhio.readAsHierarchy(source)
    layout = root.layout()

    joint_map = {layout[i][0].Name: layout[i][0] for i in range(len(layout))}

    # rest pose correction
    root.loadRestPose()
    # layout[ 0][0].setEuler((   0,  0,   0))  # Hips

    # layout[ 1][0].setEuler((   0,   0,  0))            # 1 LeftUpLeg
    # layout[ 2][0].setEuler((   0,   0,   0))            # 2 LeftLeg
    # layout[ 3][0].setEuler((   0,   0,   0))            # 3 LeftFoot
    # layout[ 4][0].setEuler((   0,   0,   0))            # 4 LeftToe

    # layout[ 5][0].setEuler((   0,   0,  0))            # 5 RightUpLeg
    # layout[ 6][0].setEuler((   0,   0,   0))            # 6 RightLeg
    # layout[ 7][0].setEuler((   0,   0,   0))            # 7 RightFoot
    # layout[ 8][0].setEuler((   0,   0,   0))            # 8 RightToe

    # layout[ 9][0].setEuler((   0,   0,   0))            # 9 Spine
    # layout[10][0].setEuler((   0,   0,   0))            # 10 Spine1
    # layout[11][0].setEuler((   0,   0,   0))  # 11 Spine2

    # layout[12][0].setEuler((   0,   0,   0))            # 12 Neck
    # layout[13][0].setEuler((   0,   0,   0))            # 13 Head

    joint_map["LeftShoulder"].setEuler((   0,   0,  -90))
    joint_map["RightShoulder"].setEuler((   0,   0,  90))
    joint_map["LeftHandThumb1"].setEuler((   45,   0,  0))
    joint_map["RightHandThumb1"].setEuler((   45,   0,  0))
    # layout[15][0].setEuler((   0,   0,  0))            # 15 LeftArm
    # layout[16][0].setEuler((   0,   0,   0))            # 16 LeftForeArm
    # layout[17][0].setEuler((   0,   0,   0))            # 17 LeftHand

    # layout[18][0].setEuler((   0,   0,  +90))            # 18 RightShoulder
    # layout[19][0].setEuler((   0,   0,  0))            # 19 RightArm
    # layout[20][0].setEuler((   0,   0,   0))            # 20 RightForeArm
    # layout[21][0].setEuler((   0,   0,   0))            # 21 RightHand

    # joint_map["Hips"].Position = (0, 0, 0)
    # for joint, index, depth in layout[1:]:
    #     joint.Position = (*joint.Position.xy, 0.0)

    root.writeRestPose(recursive=True, keep=['position', 'rotation', 'scale'])

    # key frame corrections, turns joints so than Z- axis always points forward
    # DELETE THIS LOOP IF THE BONE ROLL DOES NOT MATTER TO YOU.
    # print('| Correct bone roll')
    # for frame in range(*root.getKeyframeRange()):
    #     root.loadPose(frame, recursive=True)
    #     root.roll(-90, recursive=True)
    #     layout[14][0].roll(180, recursive=True)  # 14 LeftShoulder
    #     layout[18][0].roll(180, recursive=True)  # 18 RightShoulder

    #     layout[13][0].addEuler((-90, 0, 0))  # 13 Head
    #     layout[ 4][0].addEuler((-90, 0, 0))  # 4 LeftToe
    #     layout[ 8][0].addEuler((-90, 0, 0))  # 8 RightToe
    #     layout[17][0].addEuler((+90, 0, 0))  # 17 LeftHand
    #     layout[21][0].addEuler((+90, 0, 0))  # 21 RightHand

    #     for joint, index, depth in layout[1:]:
    #         joint.Position = (*joint.Position.xy, 0.0)

    #     root.writePose(frame, recursive=True)

    # scale to meters
    # print('| Correct scale')
    # root.RestPose.Scale = 0.01
    # root.applyRestposeScale(recursive=True, bake=False, bakeKeyframes=True)

    print('| Write file')
    bvhio.writeHierarchy(path=destination, root=root, frameTime=1/30)