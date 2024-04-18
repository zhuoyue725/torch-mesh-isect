import os
import pickle
import sys
sys.path.append('/usr/pydata/t2m/torch-mesh-isect')
from mesh_intersection.filter_faces import FilterFaces
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss

import argparse

from smplx import create

import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import time

from collections import defaultdict

# from mathutils import Vector, Quaternion # mathutils==2.81.2
# import quaternion
from utils.SMPLX2SMPL import *

NUM_SMPLX_BODYJOINTS = 21

def main(args):
    # 赋值参数
    model_folder = args.model_folder
    model_type = args.model_type
    param_fn = args.param_fn
    interactive = args.interactive
    delay = args.delay
    point2plane = args.point2plane
    #  optimize_shape = args.optimize_shape
    #  optimize_pose = args.optimize_pose
    lr = args.lr
    coll_loss_weight = args.coll_loss_weight
    pose_reg_weight = args.pose_reg_weight
    shape_reg_weight = args.shape_reg_weight
    max_collisions = args.max_collisions
    sigma = args.sigma
    optim = args.optim
    part_segm_fn = args.part_segm_fn
    print_timings = args.print_timings
    assign_frame_idx = args.assign_frame_idx
    patience = args.patience
    loss_thres = args.loss_thres
    output_folder = args.output_folder
    coll_thres = args.coll_thres
    smooth_loss_weight = args.smooth_loss_weight
    acc_thres = args.acc_thres
    grad_thres = args.grad_thres
    
    if interactive:
        import trimesh
        import pyrender
          
    device = torch.device('cuda')
    batch_size = len(param_fn)
    
    # 读取身体形状beta参数
    params_dict = defaultdict(lambda: [])
    data = np.load(param_fn[0], allow_pickle=True) # 原代码可以读入多个文件，这里取第一个文件
    assert 'betas' in data, \
        'No key for shape parameter in provided npz file'
    assert 'poses' in data, \
        'No key for poses parameter in provided npz file'
        
    for key, val in data.items():
        params_dict[key].append(val) # dict_keys(['poses', 'trans', 'betas', 'gender', 'mocap_framerate']) # 'dmpls' (3239,8)
    
    frame_num = 60#params_dict['poses'][0].shape[0]
    if 'mocap_frame_rate' in params_dict:
        fps = params_dict['mocap_frame_rate'][0].item()
    elif 'mocap_framerate' in params_dict:
        fps = params_dict['mocap_framerate'][0].item()
    else:
        fps = 30
    
    if assign_frame_idx >= 0:
        frame_num = 1
        
    params = {}
    for key in params_dict:
        if(key!='gender' and key!='model'):
            params[key] = params_dict[key] #np.stack(params_dict[key], axis=0).astype(np.float32)
    
    if part_segm_fn:
        # Read the part segmentation
        with open(part_segm_fn, 'rb') as faces_parents_file:
            data = pickle.load(faces_parents_file, encoding='latin1')
        faces_segm = data['segm'] # (20908, ) 每个面对应的身体部位，0到54，共55个部位
        faces_parents = data['parents'] # (20908, ) 每个面对应的身体部位的父部位
        # Create the module used to filter invalid collision pairs
        ign_part_pairs = ['16,9','9,17',         #,'13,16','14,17' ,'18,16','19,17'
                          '15,23','15,24',               # 头部眼睛
                          '1,2',                              # 大腿之间
                          '6,17','6,16','3,16','3,17', # 中胸、肚子与上臂
                        #   '6,17','6,16','3,16','3,17'  # 中胸、肚子与小臂
                          ] 
        # 9上胸，6中胸，3肚子
        # 13左肩，16左上臂，18 左小臂；14右肩，17右上臂，19 右小臂
        # 12颈部，15头部，23左眼，24右眼
        # 1左大腿，2右大腿
        filter_faces = FilterFaces(faces_parents = faces_parents,
                                   faces_segm = faces_segm,
                                   ign_part_pairs = ign_part_pairs).to(device=device) # 源代码中这里parents和segm反了
    
    # 创建相同的smplx模型
    body = create(model_folder, batch_size=batch_size,
                  model_type=model_type).to(device=device) # 创建一次大约3秒
        
    total_body_mesh = [copy.deepcopy(body) for _ in range(frame_num)] # 每帧对应一个mesh，是否可以优化
    
    if len(params['betas'][0]) > 10:
        model_type = 'smplh' # 参数为smplh
        params['betas'][0] = params['betas'][0][:10] # smplh有16个形状参数，这里只取前10个
        
    for cur_frame in range(frame_num):
        if assign_frame_idx >= 0: # 仅添加特定帧
            params['body_pose'] = params_dict['poses'][0][assign_frame_idx][3:66] # params_dict['poses']为list, 原55*3旋转，仅提取其中3:63
        else:
            params['body_pose'] = params_dict['poses'][0][cur_frame][3:66] # smplx的21个身体关节
        total_body_mesh[cur_frame].reset_params(**params) # 设置对应帧的姿势参数
    
    total_init_pose = [cur_body.body_pose.clone().detach() for cur_body in total_body_mesh] # 初始姿势

    search_tree = BVH(max_collisions=max_collisions) # 初始化搜索树
    
    pen_distance = \
        collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                     point2plane=point2plane,
                                                     vectorized=True)
    
    # 每帧穿模loss求和 or 均值
    def pen_distance_total(total_triangles, total_collision_idxs,coll_body_idx):
        pen_loss = torch.tensor(0, device=device,
                                      dtype=torch.float32)
        cur_coll_body_idx = []
        for i in range(len(coll_body_idx)): # 注意可能有的帧后面又有碰撞了，暂时指定帧每次都检测
            cur_loss = pen_distance(total_triangles[i], total_collision_idxs[i])[0] # 返回是一个list（原代码可能考虑多个网格）
            if cur_loss > coll_thres:
                pen_loss += cur_loss
                cur_coll_body_idx.append(coll_body_idx[i]) # 有碰撞的帧
        print(cur_coll_body_idx)
        if assign_frame_idx < 0:
            cur_coll_body_idx = range(43,50) # 每次都要检测，确保不会遗漏
        return pen_loss , cur_coll_body_idx# / frame_num
    
    def get_quat_from_rodrigues(rodrigues):
        rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
        angle_rad = rod.length
        axis = rod.normalized()
        quat = Quaternion(axis, angle_rad)
        return quat
    
    def get_quat_from_rodrigues_tensor(rodrigues):
        # 计算罗德里格斯向量的长度，即旋转角度的弧度值
        angle_rad = torch.norm(rodrigues, dim=0)

        # 归一化罗德里格斯向量，得到旋转轴
        axis = rodrigues / angle_rad

        # 根据罗德里格斯公式计算四元数
        # 这里我们使用公式: q = cos(θ/2) + sin(θ/2) * (x * i + y * j + z * k)
        half_angle = angle_rad / 2
        cos_half_angle = torch.cos(half_angle)
        sin_half_angle = torch.sin(half_angle)
        quat = torch.stack([
            cos_half_angle,
            sin_half_angle * axis[0],
            sin_half_angle * axis[1],
            sin_half_angle * axis[2]
        ], dim=0)

        return quat

    def quaternion_multiply(quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0.unbind(-1)
        w1, x1, y1, z1 = quaternion1.unbind(-1)
        result = torch.stack([
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ], dim=-1)
        return result
    
    # 定义四元数的逆函数
    def quaternion_inverse(q):
        w, x, y, z = q.unbind(-1)
        conjugate = torch.stack([w, -x, -y, -z], dim=-1)
        norm = torch.norm(q, dim=-1, keepdim=True)
        # if torch.allclose(norm, torch.Tensor([0],device=device)):
        #     return torch.Tensor([1, 0, 0, 0])
        return conjugate / (norm ** 2)

    def compute_angular_velocity(q1, q2, dt):
        dq = quaternion_multiply(q2, quaternion_inverse(q1))
        dx, dy, dz = dq[..., 1:]  # dq的虚部
        omega = (2 / dt) * torch.stack([dx, dy, dz], dim=-1)
        return omega
    
    # 计算一个骨骼的角加速度
    def calculate_acceleration_tensor(orientation): # 第一帧是nan,nan,nan,nan
        data_count = len(orientation)
        # time_delta = self.time[1:] - self.time[:-1] # time delta
        time_delta = 1 / fps
        ## Divided difference for dq_dt
        dq_dt = torch.zeros((data_count,4),device=device,
                                  dtype=torch.float32)
        dq_dt[0][0] = 1 # 第0帧角速度设置为(1,0,0,0) 否则第一帧梯度为计算为nan
        dq_dt[1:] = (orientation[1:] - orientation[:-1]) / time_delta # 第1帧到最后一帧计算速度，1阶导
        ## Divided difference for d2q_dt2
        d2q_dt2 = torch.zeros((data_count,4),device=device,
                                  dtype=torch.float32)
        d2q_dt2[2:] = (dq_dt[2:] - dq_dt[:-2]) / time_delta # 速度的第2帧到最后2帧计算加速度，2阶导
        ## Calculate velocity
        velocity = 2 * quaternion_multiply(dq_dt, quaternion_inverse(orientation))
        ## Calculate acceleration
        temp = quaternion_multiply(dq_dt, quaternion_inverse(orientation))
        acceleration = 2 * (quaternion_multiply(d2q_dt2, quaternion_inverse(orientation)) - quaternion_multiply(temp,temp))
        return acceleration, velocity
    
    def calculate_smooth_loss_total(body_poses_total):
        smooth_loss = torch.tensor(0, device=device,
                              dtype=torch.float32)
        # for bone_idx in range(NUM_SMPLX_BODYJOINTS):
        target_bone_idx = [13,16,18,14,11]
        for bone_idx in target_bone_idx: # 右手、手臂、肩膀
            orientation = torch.zeros((len(body_poses_total), 4),device=device, # 每个骨骼全部帧的四元数
                      dtype=torch.float32)
            for frame_index, frame_pose in enumerate(body_poses_total):
                current_pose = frame_pose.reshape(-1, 3)
                bone_quaternion = get_quat_from_rodrigues_tensor(current_pose[bone_idx])
                orientation[frame_index] = bone_quaternion
            acc, vel = calculate_acceleration_tensor(orientation)
            # 1. 直接求全部的角加速度之和
            # bone_loss = torch.sum(acc.pow(2), dim=1).sqrt().sum()
            
            # 2.求大于阈值的元素之和
            # 找出平方和大于阈值的元素的索引
            # indices = squared_sums > acc_thres
            # 计算选定行的平方和
            # bone_loss = torch.sum(squared_sums)
            # 计算acc张量中每个元素的平方
            squared_sums =  torch.norm(acc, dim=1)  # 角加速度最小化 torch.sum(acc.pow(2), dim=1).sqrt()
            # squared_sums = torch.sum(vel.pow(2), dim=1).sqrt() # 角速度最小化，旋转趋于静止
            # 除了该骨骼，除了该帧，其他是否有变化，其他帧有变化，其他骨骼没影响
            # 3.求前k个最大值的和
            k = 5
            # values, indices = torch.topk(squared_sums, k=k, largest=True, sorted=True)
            indices = range(30,60) # 仅优化特定的帧
            # 计算前5个最大值的和
            sum_of_top_values = torch.sum(squared_sums[indices])
            sum_of_top_values /= len(indices)
            # print(indices)
            smooth_loss += sum_of_top_values
        # smooth_loss /= NUM_SMPLX_BODYJOINTS
        return smooth_loss / len(target_bone_idx)
        
    # 计算整个序列的角速度loss
    def calculate_smooth_loss_total_2(body_poses_total,num_frames): # 相对父节点旋转角度
        delta_time = 1 / fps
        smooth_loss = torch.tensor(0, device=device,
                              dtype=torch.float32)
        
        total_pose_quat = []
        # 计算四元数
        for cur_frame in range(num_frames):
            current_pose_quat = []
            current_pose = body_poses_total[cur_frame].reshape(-1, 3) # 第0帧
            for index in range(NUM_SMPLX_BODYJOINTS):
                quat = get_quat_from_rodrigues_tensor(current_pose[index])
                current_pose_quat.append(quat) # 深拷贝or浅拷贝？
            total_pose_quat.append(current_pose_quat)
        
        total_pose_angular_velocity = []
        # 计算角速度
        for cur_frame in range(1, num_frames):
            # 计算上一帧的角速度
            prev_pose_quat = total_pose_quat[cur_frame-1]
            current_pose_quat = total_pose_quat[cur_frame]
            current_pose_angular_velocity = []
            for i in range(NUM_SMPLX_BODYJOINTS):
                # 计算四元数的共轭
                # conj_prev_quat = current_pose_quat[i].conj() # 已经归一化 q(k)=q-1(k)=q*(k)
                # 计算角速度
                dw = compute_angular_velocity(prev_pose_quat[i], current_pose_quat[i], delta_time)
                # dw = 2.0 * (current_pose_quat[i] - prev_pose_quat[i]) / delta_time
                # dw = dw * conj_prev_quat
                current_pose_angular_velocity.append(dw)
            total_pose_angular_velocity.append(current_pose_angular_velocity)
        
        total_pose_angular_acceleration = []
        # 计算角加速度
        for cur_frame in range(1, num_frames-1):
            # 计算上一帧的角速度
            prev_pose_angular_velocity = total_pose_angular_velocity[cur_frame-1]
            current_pose_angular_velocity = total_pose_angular_velocity[cur_frame]
            current_pose_angular_acceleration = []
            for i in range(NUM_SMPLX_BODYJOINTS):
                # 计算四元数的共轭
                # conj_prev_quat = current_pose_quat[i].conj() # 已经归一化 q(k)=q-1(k)=q*(k)
                da = (current_pose_angular_velocity[i] - prev_pose_angular_velocity[i]) / delta_time
                current_pose_angular_acceleration.append(da)
            total_pose_angular_acceleration.append(current_pose_angular_acceleration)
        
        # 遍历每一帧的角加速度
        for frame_acceleration in total_pose_angular_acceleration:
            acc_loss = torch.tensor(0, device=device,
                dtype=torch.float32)
            # 对每个角加速度分量求和
            for i in range(NUM_SMPLX_BODYJOINTS):
                acc_loss += frame_acceleration[i].pow(2).sum().sqrt()
            acc_loss /= NUM_SMPLX_BODYJOINTS
            smooth_loss += acc_loss
        return smooth_loss
    
    mse_loss = nn.MSELoss(reduction='sum').to(device=device)
    face_tensor = torch.tensor(body.faces.astype(np.int64), dtype=torch.long,
                               device=device).unsqueeze_(0).repeat([batch_size,
                                                                    1, 1]) # (1, 20908, 3)
    
    # 获取模型基本信息：顶点数、面数
    with torch.no_grad():
        output = body(get_skin=True)
        verts = output.vertices
                            
    bs, nv = verts.shape[:2]
    bs, nf = face_tensor.shape[:2]
    faces_idx = face_tensor + \
        (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None] # 为什么要+bs？
    
    if optim == 'SGD':
        optimizer = torch.optim.SGD([cur_body.body_pose for cur_body in total_body_mesh], lr=lr) # 随机梯度下降
    elif optim == 'Adam':
        optimizer = torch.optim.Adam([cur_body.body_pose for cur_body in total_body_mesh], lr=lr
                                     ,betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    
    # 显示模块初始化
    if interactive:
        # Plot the initial mesh
        with torch.no_grad():
            output = body(get_skin=True)
            verts = output.vertices

            np_verts = verts.detach().cpu().numpy()

        def create_mesh(vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                        wireframe=False):

            tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            rot = trimesh.transformations.rotation_matrix(np.radians(180),
                                                          [1, 0, 0])
            tri_mesh.apply_transform(rot)

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='BLEND',
                baseColorFactor=color)
            return pyrender.Mesh.from_trimesh(
                tri_mesh,
                material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                               ambient_light=(1.0, 1.0, 1.0))
        for bidx in range(np_verts.shape[0]):
            curr_verts = np_verts[bidx].copy()
            body_mesh = create_mesh(curr_verts, body.faces,
                                    color=(0.3, 0.3, 0.3, 0.99),
                                    wireframe=True)

            pose = np.eye(4)
            pose[0, 3] = bidx * 2
            scene.add(body_mesh,
                      name='body_mesh_{:03d}'.format(bidx),
                      pose=pose)

        viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                                 viewport_size=(1200, 800),
                                 cull_faces=False,
                                 run_in_thread=True)

    query_names = ['recv_mesh', 'intr_mesh', 'body_mesh']
    
    # 优化迭代
    if print_timings:
        start_optim = time.time()
    step = 0
    loss_old = 0
    count_loss = 0
    coll_body_idx = list(range(frame_num)) # 有碰撞的帧，初始设置为全部帧
    while True:
        # START OPTIMIZATION
        optimizer.zero_grad()
        
        if print_timings:
            start_step = time.time()
            start = time.time()
        
        if print_timings:
            torch.cuda.synchronize() # 等待之前发起的异步操作完成
        
        # 1.更新带有碰撞的body的mesh
        total_output_body_mesh = [total_body_mesh[i](get_skin=True) for i in coll_body_idx] # 仅更新有碰撞帧
        
        if print_timings:
            torch.cuda.synchronize()
            print('Body model forward: {:5f}'.format(time.time() - start))
        
        # 2.三角形顶点位置
        if print_timings:
            torch.cuda.synchronize()
            start = time.time()
        total_verts = [total_output_body_mesh[i].vertices for i in range(len(coll_body_idx))] #  (1, 10475, 3)
        
        if print_timings:
            torch.cuda.synchronize()
            print('Triangle indexing: {:5f}'.format(time.time() - start))
            
        total_triangles = [total_verts[i].view([-1, 3])[faces_idx] for i in range(len(coll_body_idx))]
        
        # 3.BVH碰撞检测
        with torch.no_grad():
            if print_timings:
                start = time.time()
            total_collision_idxs = [search_tree(total_triangles[i]) for i in range(len(coll_body_idx))] # triangles及三角形位置 (1, 20908, 3, 3)
            if print_timings:
                torch.cuda.synchronize()
                print('Collision Detection: {:5f}'.format(time.time() - start))
                
            # 4. 过滤部分身体部位or自交
            if part_segm_fn:
                if print_timings:
                    start = time.time()
                total_collision_idxs = [filter_faces(total_collision_idxs[i])[0] for i in range(len(coll_body_idx))] # 过滤部分碰撞
                if assign_frame_idx >= 0 : # 调试查看穿模的部位
                    coll_part = filter_faces(total_collision_idxs[0])[1]
                    part_msg = ""
                    for i in range(len(coll_part)):
                        part_msg += SMPLX_JOINT_NAMES_55[coll_part[i].item()] + ", "
                    print("collision body parts: {}".format(part_msg))
                
                if print_timings:
                    torch.cuda.synchronize()
                    print('Collision filtering: {:5f}'.format(time.time() - start))
                    
        # L1.穿模Loss
        if print_timings:
            start = time.time()
        pen_loss, coll_body_idx = pen_distance_total(total_triangles, total_collision_idxs, coll_body_idx)

        if coll_loss_weight > 0:
            pen_loss = coll_loss_weight * pen_loss
        else:
            pen_loss = torch.tensor(0, device=device,
                                      dtype=torch.float32)

        if print_timings:
            torch.cuda.synchronize()
            print('Penetration loss: {:5f}'.format(time.time() - start))
        
        # L2.Pose_Reg_Loss
        pose_reg_loss = torch.tensor(0, device=device,
                                     dtype=torch.float32)
        
        if pose_reg_weight > 0:
            for i in coll_body_idx:
                pose_reg_loss += mse_loss(total_body_mesh[i].body_pose, total_init_pose[i])
            pose_reg_loss /= len(coll_body_idx)
            pose_reg_loss = pose_reg_weight * pose_reg_loss
        
        # L3.平滑loss
        body_poses_total = [cur_body.body_pose for cur_body in total_body_mesh] # 获取每帧的pose
        # body_poses_location_total = [cur_body.body_pose for cur_body in total_body_mesh] # 获取每帧的pose
        
        if assign_frame_idx < 0 and smooth_loss_weight > 0: 
            smooth_loss = smooth_loss_weight * calculate_smooth_loss_total(body_poses_total)
            loss = pose_reg_loss + pen_loss + smooth_loss
        else: # 单帧测试 or 仅穿模
            smooth_loss = torch.tensor(0, device=device, dtype=torch.float32)
            loss = pose_reg_loss + pen_loss
        
        np_loss = loss.detach().cpu().squeeze().tolist()
        if type(np_loss) != list:
            np_loss = [np_loss]
        msg = '{:.5f} ' * len(np_loss)
        print('Loss total:', msg.format(*np_loss) ,
                'Pen : {:5f}'.format(pen_loss), 
                'Smooth : {:5f}'.format(smooth_loss),
                'Reg : {:5f}'.format(pose_reg_loss))
        
        # 判断是否结束优化：若干次loss下降小于阈值
        with torch.no_grad():
            count_loss = count_loss + 1 if abs(loss - loss_old) < loss_thres else 0
            if count_loss >= patience or loss == 0:
                break
            loss_old = loss
        
        if print_timings:
            start = time.time()
        
        # 7.反向传播    
        loss.backward(torch.ones_like(loss))
        
        topk = 6 # 仅优化梯度最大的前topk个关节
        
        for k in range(len(body_poses_total)): # len(body_poses_total)
            if type(body_poses_total[k].grad) == type(None):
                continue
            # print('Frame: {} is croping'.format(k))
            # non_zero_indices = body_poses_total[k].grad.reshape(-1,3).nonzero(as_tuple=False).unique()
            grad_reshaped = body_poses_total[k].grad.reshape(-1, 3)
            # non_zero_grads = grad_reshaped[non_zero_indices]
            sorted_indices = torch.argsort(torch.norm(grad_reshaped,dim=1), descending=True)
            # sorted_grads = grad_reshaped[sorted_indices]
            top_indices = sorted_indices[:topk]
            msg = ''
            for i in range(len(sorted_indices)):
                # if sorted_indices[i] not in [2,5,8,11,14,12,13,14,15,16,17,18,19,20]:
                #     if torch.norm(grad_reshaped[sorted_indices[i]]) > 0:
                #         msg += "{} {}; ".format(SMPLX_BODY_JOINT_NAMES[sorted_indices[i]],torch.norm(grad_reshaped[sorted_indices[i]])) # 约为0.0002左右
                if i not in top_indices:
                    # print(SMPLX_BODY_JOINT_NAMES[i], torch.norm(grad_reshaped[i]))
                    # body_poses_total[0].grad[i*3:(i+1)*3] = 0 # 有问题
                    grad_reshaped[i] = 0
                # else:
                #     msg += "{} {}; ".format(SMPLX_BODY_JOINT_NAMES[i],torch.norm(grad_reshaped[i])) # 大于 0.003
            # print(msg)
        if print_timings:
            torch.cuda.synchronize()
            print('Backward pass: {:5f}'.format(time.time() - start))
        optimizer.step()
        # END OPTIMIZATION
        
        assign_idx = 0 # 碰撞帧中的第1个
        if interactive:
            with torch.no_grad():
                output = total_body_mesh[coll_body_idx[assign_idx]](get_skin=True)
                verts = output.vertices

                np_verts = verts.detach().cpu().numpy()

            np_collision_idxs = total_collision_idxs[assign_idx].detach().cpu().numpy()
            np_receivers = np_collision_idxs[:, :, 0]
            np_intruders = np_collision_idxs[:, :, 1]
            viewer.render_lock.acquire()

            # 移除原先的mesh
            for node in scene.get_nodes():
                if node.name is None:
                    continue
                if any([query in node.name for query in query_names]):
                    scene.remove_node(node)

            # 显示mesh和穿模的三角形
            for bidx in range(batch_size):
                recv_faces_idxs = np_receivers[bidx][np_receivers[bidx] >= 0] # 接收者三角形面索引
                intr_faces_idxs = np_intruders[bidx][np_intruders[bidx] >= 0]
                recv_faces = body.faces[recv_faces_idxs]
                intr_faces = body.faces[intr_faces_idxs]

                curr_verts = np_verts[bidx].copy()
                body_mesh = create_mesh(curr_verts, body.faces,
                                        color=(0.3, 0.3, 0.3, 0.99),
                                        wireframe=True)

                pose = np.eye(4)
                pose[0, 3] = bidx * 2
                scene.add(body_mesh,
                          name='body_mesh_{:03d}'.format(bidx),
                          pose=pose)

                if len(intr_faces) > 0:
                    intr_mesh = create_mesh(curr_verts, intr_faces, # 穿模三角形（入侵者）的颜色
                                            color=(0.9, 0.0, 0.0, 1.0)) # 红色
                    scene.add(intr_mesh,
                              name='intr_mesh_{:03d}'.format(bidx),
                              pose=pose)

                if len(recv_faces) > 0:
                    recv_mesh = create_mesh(curr_verts, recv_faces, # 被穿模三角形（接收者）的颜色
                                            color=(0.0, 0.9, 0.0, 1.0))
                    scene.add(recv_mesh, name='recv_mesh_{:03d}'.format(bidx),
                              pose=pose)

            viewer.render_lock.release()
    
        if print_timings:
            torch.cuda.synchronize() 
            print('Step: {} Optimize Step Time: {:5f}'.format(step, time.time() - start_step))
        
        step += 1
    
    if print_timings:
        print('Step: {} Optimize Step Time: {:5f}'.format(step, time.time() - start_optim))
        
    # 保存动画为npz文件
    if output_folder:
        body_poses = [cur_body.body_pose for cur_body in total_body_mesh]
        # params_dict['poses'][0][0][3:66] = body_poses[47].cpu().detach()
        for i, pose in enumerate(body_poses):
            pose_cpu = pose.cpu()
            pose_numpy = pose_cpu.detach().numpy().flatten().tolist()
            params_dict['poses'][0][i][3:66] = pose_numpy
        if assign_frame_idx >= 0:
            params_dict['poses'][0][assign_frame_idx][3:66] = body_poses[0].cpu().detach().numpy().flatten().tolist()
        params_save = {}
        # params_dict中每个value都是list
        for key in params_dict:
            params_save[key] = params_dict[key][0] 
        file_count = len(os.listdir(output_folder))
        file_name = 'result{}.npz'.format(file_count)
        file_path = os.path.join(output_folder, file_name)
        np.savez_compressed(file_path, **params_save)
        print("result file save in {}".format(file_path))
    
    
if __name__=="__main__":
    description = 'Example script for untangling SMPL self intersections'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Batch SMPL-Untangle')
    parser.add_argument('--param_fn', type=str,
                        nargs='*',
                        required=True,
                        help='The pickle file with the model parameters')
    parser.add_argument('--interactive', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Display the mesh during the optimization' +
                        ' process')
    parser.add_argument('--delay', type=int, default=50,
                        help='The delay for the animation callback in ms')
    parser.add_argument('--model_folder', type=str,
                        default='models',
                        help='The path to the LBS model')
    parser.add_argument('--model_type', type=str,
                        default='smpl', choices=['smpl', 'smplx', 'smplh'],
                        help='The type of model to create')
    parser.add_argument('--point2plane', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use point to distance')
    parser.add_argument('--optimize_pose', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Enable optimization over the joint pose')
    parser.add_argument('--optimize_shape', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Enable optimization over the shape of the model')
    parser.add_argument('--sigma', default=0.5, type=float,
                        help='The height of the cone used to calculate the' +
                        ' distance field loss')
    parser.add_argument('--lr', default=1, type=float,
                        help='The learning rate for SGD')
    parser.add_argument('--coll_loss_weight', default=1e-4, type=float,
                        help='The weight for the collision loss')
    parser.add_argument('--smooth_loss_weight', default=1e-4, type=float,
                        help='The weight for the smooth loss')
    parser.add_argument('--pose_reg_weight', default=0, type=float,
                        help='The weight for the pose regularizer')
    parser.add_argument('--shape_reg_weight', default=0, type=float,
                        help='The weight for the shape regularizer')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--part_segm_fn', default='', type=str,
                        help='The file with the part segmentation for the' +
                        ' faces of the model')
    parser.add_argument('--optim', default='SGD', type=str,
                        help='optimizer type')
    parser.add_argument('--assign_frame_idx', default=-1, type=int,
                        help='assign_frame_idx for test')
    parser.add_argument('--print_timings', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Print timings for all the operations')
    parser.add_argument('--loss_thres', default=0.0001, type=float,
                        help='loss change threshold during optimization process')
    parser.add_argument('--grad_thres', default=0.0001, type=float,
                        help='loss change threshold during optimization process')
    parser.add_argument('--coll_thres', default=0, type=float,
                        help='Threshold of puncture, indicating collision')
    parser.add_argument('--acc_thres', default=15, type=float,
                        help='Threshold of accelerate')
    parser.add_argument('--patience', default=30, type=int,
                        help='If the loss is too small in several changes, the optimization is terminated')
    parser.add_argument('--output_folder', required=True, 
                        help='folder where the example npz files are written to')
    args = parser.parse_args()
    main(args)
    # 有点地方相交还要忽略
    # 不考虑平滑的效果
    # 时间有点慢
    # 整体的平滑
    
    # 仅优化部分骨骼的旋转
    # 输入已经穿模优化后的序列
    # 查看某个关节是否大部分帧的角加速度为0