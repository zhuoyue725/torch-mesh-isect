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
    
    
    if interactive:
        import trimesh
        import pyrender
          
    device = torch.device('cuda')
    batch_size = len(param_fn)
    
    # 读取身体形状beta参数
    params_dict = defaultdict(lambda: [])
    data = np.load(param_fn[0], allow_pickle=True)
    assert 'betas' in data, \
        'No key for shape parameter in provided npz file'
    assert 'poses' in data, \
        'No key for poses parameter in provided npz file'
        
    for key, val in data.items():
        params_dict[key].append(val) # dict_keys(['poses', 'trans', 'betas', 'gender', 'mocap_framerate'])
    
    frame_num = params_dict['poses'][0].shape[0]
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
    
    for cur_frame in range(frame_num):
        if assign_frame_idx >= 0: # 仅添加特定帧
            params['body_pose'] = params_dict['poses'][0][assign_frame_idx][3:66] # params_dict['poses']为list, 原55*3旋转，仅提取其中3:63
        else:
            params['body_pose'] = params_dict['poses'][0][cur_frame][3:66]
        total_body_mesh[cur_frame].reset_params(**params) # 设置对应帧的姿势参数
    
    total_init_pose = [cur_body.body_pose.clone().detach() for cur_body in total_body_mesh] # 初始姿势

    search_tree = BVH(max_collisions=max_collisions) # 初始化搜索树
    
    pen_distance = \
        collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                     point2plane=point2plane,
                                                     vectorized=True)
    def pen_distance_total(total_triangles, total_collision_idxs):
        pen_loss = torch.tensor(0, device=device,
                                      dtype=torch.float32)
        coll_body_idx.clear()
        for i in range(frame_num):
            cur_loss = pen_distance(total_triangles[i], total_collision_idxs[i])[0] # 返回是一个list（原代码可能考虑多个网格）
            if cur_loss > 0:
                pen_loss += cur_loss
                coll_body_idx.append(i)
        print(coll_body_idx)
        return pen_loss # / frame_num
    
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
        optimizer = torch.optim.Adam([cur_body.body_pose for cur_body in total_body_mesh], lr=lr)
                                    #  betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    
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
    step = 0
    loss_old = 0
    count_loss = 0
    coll_body_idx = [] # 有碰撞的帧
    while True:
        # START OPTIMIZATION
        optimizer.zero_grad()
        
        if print_timings:
            start_step = time.time()
            start = time.time()
        
        if print_timings:
            torch.cuda.synchronize() # 等待之前发起的异步操作完成
        
        # 1.更新当前步的body的mesh
        total_output_body_mesh = [cur_body(get_skin=True) for cur_body in total_body_mesh] # 耗时
        
        if print_timings:
            torch.cuda.synchronize()
            print('Body model forward: {:5f}'.format(time.time() - start))
        
        # 2.三角形顶点位置
        if print_timings:
            torch.cuda.synchronize()
            start = time.time()
        total_verts = [cur_body_mesh.vertices for cur_body_mesh in total_output_body_mesh] #  (1, 10475, 3)
        
        if print_timings:
            torch.cuda.synchronize()
            print('Triangle indexing: {:5f}'.format(time.time() - start))
            
        total_triangles = [verts.view([-1, 3])[faces_idx]for verts in total_verts] # triangles及三角形位置 (1, 20908, 3, 3)
        
        # 3.BVH碰撞检测
        with torch.no_grad():
            if print_timings:
                start = time.time()
            total_collision_idxs = [search_tree(triangles) for triangles in total_triangles] # 耗时
            if print_timings:
                torch.cuda.synchronize()
                print('Collision Detection: {:5f}'.format(time.time() - start))
                
            # 4. 过滤部分身体部位or自交
            if part_segm_fn:
                if print_timings:
                    start = time.time()
                total_collision_idxs = [filter_faces(collision_idxs) for collision_idxs in total_collision_idxs] # 过滤部分碰撞
                if print_timings:
                    torch.cuda.synchronize()
                    print('Collision filtering: {:5f}'.format(time.time() - start))
                    
        # 5.穿模Loss
        if print_timings:
            start = time.time()
        pen_loss = coll_loss_weight * \
            pen_distance_total(total_triangles, total_collision_idxs)
        if print_timings:
            torch.cuda.synchronize()
            print('Penetration loss: {:5f}'.format(time.time() - start))
        
        # 6.Pose_Reg_Loss
        pose_reg_loss = torch.tensor(0, device=device,
                                     dtype=torch.float32)
        
        if pose_reg_weight > 0:
            for i in range(frame_num):
                pose_reg_loss += mse_loss(total_body_mesh[i].pose, total_init_pose[i])
            pose_reg_loss = pose_reg_weight * \
                pose_reg_loss
        
        loss = pen_loss + pose_reg_loss # 添加contact loss？
        
        np_loss = loss.detach().cpu().squeeze().tolist()
        if type(np_loss) != list:
            np_loss = [np_loss]
        msg = '{:.5f} ' * len(np_loss)
        print('Loss total model:', msg.format(*np_loss))
        
        # 判断是否结束优化：若干次loss下降小于阈值
        with torch.no_grad():
                count_loss = count_loss + 1 if abs(loss - loss_old) < loss_thres else 0
                if count_loss >= patience:
                    break
                loss_old = loss
        
        if print_timings:
            start = time.time()
        
        # 7.反向传播    
        loss.backward(torch.ones_like(loss))
        
        if print_timings:
            torch.cuda.synchronize()
            print('Backward pass: {:5f}'.format(time.time() - start))
        # END OPTIMIZATION
        assign_idx = 0
        
        if interactive:
            with torch.no_grad():
                output = total_body_mesh[assign_idx](get_skin=True)
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
    
        optimizer.step()
        if print_timings:
            torch.cuda.synchronize() # 等待之前发起的异步操作完成
            print('Step: {} Optimize Step Time: {:5f}'.format(step, time.time() - start_step))
        
        step += 1
    
    print("Optimization Finish")
    
    # 保存动画为npz文件
    if output_folder:
        body_poses = [cur_body.body_pose for cur_body in total_body_mesh]
        for i, pose in enumerate(body_poses):
            pose_cpu = pose.cpu()
            pose_numpy = pose_cpu.detach().numpy().flatten().tolist()
            params_dict['poses'][0][i][3:66] = pose_numpy
        params_save = {}
        # params_dict中每个value都是list
        for key in params_dict:
            params_save[key] = params_dict[key][0]
        np.savez_compressed(osp.join(output_folder, 'result6.npz'), **params_save)
    
    
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
    parser.add_argument('--loss_thres', default=0.00001, type=float,
                        help='loss change threshold during optimization process')
    parser.add_argument('--patience', default=10, type=int,
                        help='If the loss is too small in several changes, the optimization is terminated')
    parser.add_argument('--output_folder', required=True, 
                        help='folder where the example npz files are written to')
    args = parser.parse_args()
    main(args)
    # 有点地方相交还要忽略
    # 不考虑平滑的效果
    # 时间有点慢
    # 整体的平滑