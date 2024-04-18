import sys
import numpy as np
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import pdb
import math
from SMPLX2SMPL import JOINT_MAP_22
import os

# 提供的关节映射信息
JOINT_MAP_55 = {
    # 'BVH joint name': 'SMPLX joint index'
    'Hips': 0,
    'LeftUpLeg': 1,
    'RightUpLeg': 2,
    'Spine': 3,
    'LeftLeg': 4,
    'RightLeg': 5,
    'Spine1': 6,
    'LeftFoot': 7,
    'RightFoot': 8,
    'Spine2': 9,
    'LeftForeFoot': 10,
    'RightForeFoot': 11,
    'Spine3': 12,
    'LeftShoulder': 13,
    'RightShoulder': 14,
    'Neck': 15,
    'LeftArm': 16,
    'RightArm': 17,
    'LeftForeArm': 18,
    'RightForeArm': 19,
    'LeftHand': 20,
    'RightHand': 21,
    # 少了3个：对应smplx中的 'jaw','left_eye_smplhf','right_eye_smplhf'
    'LeftHandIndex1': 25,
    'LeftHandIndex2': 26,
    'LeftHandIndex3': 27,
    'LeftHandMiddle1': 28,
    'LeftHandMiddle2': 29,
    'LeftHandMiddle3': 30,
    'LeftHandPinky1': 31,
    'LeftHandPinky2': 32,
    'LeftHandPinky3': 33,
    'LeftHandRing1': 34,
    'LeftHandRing2': 35,
    'LeftHandRing3': 36,
    'LeftHandThumb1': 37,
    'LeftHandThumb2': 38,
    'LeftHandThumb3': 39,
    'RightHandIndex1': 40,
    'RightHandIndex2': 41,
    'RightHandIndex3': 42,
    'RightHandMiddle1': 43,
    'RightHandMiddle2': 44,
    'RightHandMiddle3': 45,
    'RightHandPinky1': 46,
    'RightHandPinky2': 47,
    'RightHandPinky3': 48,
    'RightHandRing1': 49,
    'RightHandRing2': 50,
    'RightHandRing3': 51,
    'RightHandThumb1': 52,
    'RightHandThumb2': 53,
    'RightHandThumb3': 54,
}


def bvh_to_smplx(bvh_file, n_frames=None):
    with open(bvh_file, 'r') as f:
        mocap = Bvh(f.read())

    if n_frames is None:
        num_frames = len(mocap.frames)
    else:
        num_frames = min(n_frames, len(mocap.frames))

    # 计算降采样后的帧数
    num_frames_downsampled = math.ceil(num_frames) #  / 6

    smplx_poses = np.zeros((num_frames_downsampled, 165))
    smplx_trans = np.zeros((num_frames_downsampled, 3))

    bvh_joint_names = set(mocap.get_joints_names())

    # 定义一个从Y轴负向到Z轴正向的旋转
    rotation_correction = R.from_euler('XYZ', [90, 0, 0], degrees=True)

    for i in range(0, num_frames): # , 6
        print('Processing frame {}/{}'.format(i, num_frames), end='\r')
        for joint_name, joint_index in JOINT_MAP_22.items():
            # print(joint_name, joint_index)
            # 检查关节是否存在于BVH文件中
            # if joint_name not in bvh_joint_names:
            #     continue

            # 提取关节旋转
            # rotation = R.from_euler('XYZ', mocap.frame_joint_channels(i, joint_name, ['Xrotation', 'Yrotation', 'Zrotation']), degrees=True)
            rotation = R.from_euler('ZYX', mocap.frame_joint_channels(i, joint_name, ['Zrotation', 'Yrotation', 'Xrotation']), degrees=True)

            # 仅对根关节（Hips）应用朝向校正
            if joint_name == 'Hips':
                # rotation = rotation * rotation_correction
                rotation = rotation_correction * rotation

            # smplx_poses[i//6, 3 * joint_index:3 * (joint_index + 1)] = rotation.as_rotvec() # 没有对应的则默认为0
            smplx_poses[i, 3 * joint_index:3 * (joint_index + 1)] = rotation.as_rotvec() # 没有对应的则默认为0

            # 提取根关节平移
            if joint_name == 'Hips':
                x, y, z = mocap.frame_joint_channels(i, joint_name, ['Xposition', 'Yposition', 'Zposition'])
                # smplx_trans[i // 6] = np.array([x, -z, y])
                smplx_trans[i] = np.array([x, -z, y])

                # smplx_trans[i] = mocap.frame_joint_channels(i, joint_name, ['Zposition', 'Yposition', 'Xposition'])

    # 应用朝向校正
    # smplx_trans = rotation_correction_trans.apply(smplx_trans)

    # 反转Y轴平移方向
    # smplx_trans[:, 1] *= -1

    # 应用整体缩放
    scale_factor = 1 # 0.009
    smplx_trans *= scale_factor

    return smplx_trans, smplx_poses


def save_npz(output_file, smplx_trans, smplx_poses, gender='neutral', model_type='smplx', frame_rate=30):
    np.savez(output_file, trans=smplx_trans, poses=smplx_poses, gender=gender, surface_model_type=model_type,
             mocap_frame_rate=frame_rate, betas=np.zeros(16))



def process_bvh_file(bvh_file, output_file):
    smplx_trans, smplx_poses = bvh_to_smplx(bvh_file)

    with open(bvh_file, 'r') as f:
        mocap = Bvh(f.read())
        frame_rate = 1.0 / mocap.frame_time
    print('\nframe_rate: ', frame_rate)
    save_npz(output_file, smplx_trans, smplx_poses, frame_rate=30)
    
    print(f"Processing {bvh_file}, output to {output_file}")
    
if __name__ == '__main__':
    '''
    pip install bvh
    cd process
    python bvh2smplx.py 2_scott_0_85_85.bvh 2_scott_0_85_85.npz
    '''
    bvh_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # import bvh
    # with open("10_kieks_0_96_96_1.bvh", "r") as f:
    #     mocap = bvh.Bvh(f.read())
    # joints = []
    # for joint in mocap.get_joints_names():
    #     joints.append(joint)
    # print(joints)
    
    if bvh_dir.endswith('.bvh'):
        # 获取文件名（带后缀）
        bvh_file = bvh_dir
        file_name_with_extension = os.path.basename(bvh_file)
        # 分割文件名和后缀
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        output_file = os.path.join(output_dir, file_name + '.npz')
        process_bvh_file(os.path.join(bvh_dir, bvh_file), output_file)
    else:
        bvh_files = [f for f in os.listdir(bvh_dir) if f.endswith('.bvh')]
        # 对每个bvh文件进行处理
        for bvh_file in bvh_files:
           # 构建输出文件路径
           output_file = os.path.join(output_dir, os.path.splitext(bvh_file)[0] + '.npz')
           # 调用process_bvh_file函数处理bvh文件
           process_bvh_file(os.path.join(bvh_dir, bvh_file), output_file)