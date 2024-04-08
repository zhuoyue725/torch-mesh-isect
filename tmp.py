# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import time

import pickle

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.append('/usr/pydata/t2m/torch-mesh-isect')
from mesh_intersection.filter_faces import FilterFaces
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss

from smplx import create


def main():
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
    parser.add_argument('--pose_reg_weight', default=0, type=float,
                        help='The weight for the pose regularizer')
    parser.add_argument('--shape_reg_weight', default=0, type=float,
                        help='The weight for the shape regularizer')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--part_segm_fn', default='', type=str,
                        help='The file with the part segmentation for the' +
                        ' faces of the model')
    parser.add_argument('--print_timings', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Print timings for all the operations')

    args = parser.parse_args()

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
    part_segm_fn = args.part_segm_fn
    print_timings = args.print_timings

    if interactive:
        import trimesh
        import pyrender

    device = torch.device('cuda')
    batch_size = len(param_fn)

    params_dict = defaultdict(lambda: [])
    for idx, fn in enumerate(param_fn):
        # with open(fn, 'rb') as param_file:
            # data = pickle.load(param_file, encoding='latin1')
        # data = np.load(fn)
        data = np.load(param_fn[0], allow_pickle=True)

        # assert 'betas' in data, \
        #     'No key for shape parameter in provided pickle file'
        # assert 'global_pose' in data, \
        #     'No key for the global pose in the given pickle file'
        # assert 'pose' in data, \
        #     'No key for the pose of the joints in the given pickle file'

        for key, val in data.items():
            params_dict[key].append(val)

    params = {}
    for key in params_dict:
        if(key!='gender' and key!='model'):
            params[key] = params_dict[key] #np.stack(params_dict[key], axis=0).astype(np.float32)
            if(key=='poses'):
                params['body_pose'] = params_dict['poses'][0][47][3:66] 
            # if(key=='body_pose'):
            #     params[key] = params_dict[key][0][3:66] # np.stack(params_dict[key][3:66], axis=0).astype(np.float32) 原先有87个，前三个
            # if len(params[key].shape) < 2:
            #     params[key] = params[key][np.newaxis]
    if 'global_pose' in params:
        params['global_orient'] = params['global_pose']
    if 'pose' in params: # 
        params['body_pose'] = params['pose']
    # if 'body_pose' in params: # body_pose
    #     params['body_pose'] = params['body_pose'][3:66]

    if part_segm_fn:
        # Read the part segmentation
        with open(part_segm_fn, 'rb') as faces_parents_file:
            data = pickle.load(faces_parents_file, encoding='latin1')
        faces_segm = data['segm'] # (20908, ) 每个面对应的身体部位，0到54，共55个部位
        faces_parents = data['parents'] # (20908, ) 每个面对应的身体部位的父部位
        # Create the module used to filter invalid collision pairs
        ign_part_pairs = ['16,9','9,17' #,'13,16','14,17' ,'18,16','19,17'
                          ,'15,23','15,24'] 
        # 9上胸；13左肩，16左上臂，18 左小臂；14右肩，17右上臂，19 右小臂
        # 12颈部，15头部，23左眼，24右眼
        filter_faces = FilterFaces(faces_parents = faces_parents,
                                   faces_segm = faces_segm,
                                   ign_part_pairs = ign_part_pairs).to(device=device) # 源代码中这里parents和segm反了

    # Create the body model smplx基准模型
    body = create(model_folder, batch_size=batch_size,
                  model_type=model_type).to(device=device)
    body.reset_params(**params) 
    # dict_keys(['betas' (10,), 'vertices'(10475,3), 'body_pose'(63,)]) 

    # Clone the given pose to use it as a target for regularization
    init_pose = body.body_pose.clone().detach()

    # Create the search tree
    search_tree = BVH(max_collisions=max_collisions) # 初始化搜索树

    pen_distance = \
        collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                     point2plane=point2plane,
                                                     vectorized=True)

    mse_loss = nn.MSELoss(reduction='sum').to(device=device)

    face_tensor = torch.tensor(body.faces.astype(np.int64), dtype=torch.long,
                               device=device).unsqueeze_(0).repeat([batch_size,
                                                                    1, 1]) # (1, 20908, 3)
    with torch.no_grad():
        output = body(get_skin=True)
        verts = output.vertices

    bs, nv = verts.shape[:2]
    bs, nf = face_tensor.shape[:2]
    faces_idx = face_tensor + \
        (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None] # 为什么要+bs？

    optimizer = torch.optim.SGD([body.body_pose], lr=lr) # 随机梯度下降，使用其他优化器？

    # pyrender渲染相关初始化
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
    step = 0
    while True:
        optimizer.zero_grad()

        if print_timings:
            start = time.time()

        if print_timings:
            torch.cuda.synchronize()
        output = body(get_skin=True) # 更新当前body姿势参数？
        verts = output.vertices #  (1, 10475, 3)

        if print_timings:
            torch.cuda.synchronize()
            print('Body model forward: {:5f}'.format(time.time() - start))

        if print_timings:
            torch.cuda.synchronize()
            start = time.time()
        triangles = verts.view([-1, 3])[faces_idx] # triangles及三角形位置 (1, 20908, 3, 3)
        if print_timings:
            torch.cuda.synchronize()
            print('Triangle indexing: {:5f}'.format(time.time() - start))

        with torch.no_grad():
            if print_timings:
                start = time.time()
            collision_idxs = search_tree(triangles) # 0.01秒左右 通过BVH检测碰撞，碰撞顶点对 (1,167264,2)，为什么16万个碰撞？
            if print_timings:
                torch.cuda.synchronize()
                print('Collision Detection: {:5f}'.format(time.time() -
                                                          start))
            if part_segm_fn:
                if print_timings:
                    start = time.time()
                collision_idxs = filter_faces(collision_idxs) # 过滤部分碰撞
                if print_timings:
                    torch.cuda.synchronize()
                    print('Collision filtering: {:5f}'.format(time.time() -
                                                              start))

        if print_timings:
            start = time.time()
        pen_loss = coll_loss_weight * \
            pen_distance(triangles, collision_idxs) # 计算穿模损失，输入三角形及顶点位置、碰撞顶点对
        if print_timings:
            torch.cuda.synchronize()
            print('Penetration loss: {:5f}'.format(time.time() - start))

        shape_reg_loss = torch.tensor(0, device=device,
                                      dtype=torch.float32)
        if shape_reg_weight > 0:
            shape_reg_loss = shape_reg_weight * torch.sum(body.betas ** 2) # 为什么直接计算平方和
        pose_reg_loss = torch.tensor(0, device=device,
                                     dtype=torch.float32)
        if pose_reg_weight > 0:
            pose_reg_loss = pose_reg_weight * \
                mse_loss(body.pose, init_pose)

        loss = pen_loss + pose_reg_loss + shape_reg_loss # 添加contact loss？

        np_loss = loss.detach().cpu().squeeze().tolist()
        if type(np_loss) != list:
            np_loss = [np_loss]
        msg = '{:.5f} ' * len(np_loss)
        print('Loss per model:', msg.format(*np_loss))

        if print_timings:
            start = time.time()
        loss.backward(torch.ones_like(loss))
        if print_timings:
            torch.cuda.synchronize()
            print('Backward pass: {:5f}'.format(time.time() - start))

        # 显示模型和穿模顶点
        if interactive:
            with torch.no_grad():
                output = body(get_skin=True)
                verts = output.vertices

                np_verts = verts.detach().cpu().numpy()

            np_collision_idxs = collision_idxs.detach().cpu().numpy()
            np_receivers = np_collision_idxs[:, :, 0]
            np_intruders = np_collision_idxs[:, :, 1]

            viewer.render_lock.acquire()

            for node in scene.get_nodes():
                if node.name is None:
                    continue
                if any([query in node.name for query in query_names]):
                    scene.remove_node(node)

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
                
                # 测试一个三角形
                # test_faces = body.faces[[4882]]
                # test_mesh = create_mesh(curr_verts, test_faces, # 被穿模三角形（接收者）的颜色
                #                             color=(0.9, 0.0, 0.0, 1.0))
                # scene.add(test_mesh, name='test_mesh_{:03d}'.format(bidx),
                #               pose=pose)
                
                # 显示特定身体部位的面              
                # idx = 16  # 16
                # print(idx)
                # mask = faces_segm == idx
                # # mask = np.isin(faces_segm, idx_list)
                # segm_faces = body.faces[mask, :]
                # segm_mesh = create_mesh(curr_verts, segm_faces, # 被穿模三角形（接收者）的颜色
                #                             color=(0.0, 0.0, 0.9, 1.0))
                # scene.add(segm_mesh, name='segm_mesh_{:03d}'.format(bidx),
                #               pose=pose)
                
                # idx = faces_parents[4882]  # 16
                # print(idx)
                # mask = faces_segm == idx
                # # mask = np.isin(faces_segm, idx_list)
                # segm_faces = body.faces[mask, :]
                # segm_mesh = create_mesh(curr_verts, segm_faces, # 被穿模三角形（接收者）的颜色
                #                             color=(0.0, 0.9, 0.9, 1.0))
                # scene.add(segm_mesh, name='segm_parent_mesh_{:03d}'.format(bidx),
                #               pose=pose)
                    
            viewer.render_lock.release()

            if not viewer.is_active:
                break

            time.sleep(delay / 1000)
        optimizer.step()

        step += 1


if __name__ == '__main__':
    main()
