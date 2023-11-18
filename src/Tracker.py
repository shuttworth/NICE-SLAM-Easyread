import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    # 相机姿态的迭代优化，通过采样像素get_samples，渲染深度、不确定性和颜色render_batch_ray，计算loss损失并进行反向传播来完成
    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """

        # 初始化准备，内参和外参
        device = self.device
        # 内参
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        # 外参c2w，也是相机的位姿矩阵，从相机张量转换而来
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        if self.nice:
            # should pre-filter those out of bounding box depth value
            # 过滤掉不在预定边界 (self.bound) 内的深度值，以确保处理的光线在场景的边界内
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        # 调用render_batch_ray()得到预测的深度depth、不确定性uncertainty(depth_var)和颜色color
        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()
        if self.handle_dynamic:
            # 启用了动态处理 (self.handle_dynamic)，使用中位数滤波来识别和处理动态对象。
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        # 计算基于深度的loss（预测深度和真值深度之间的差异）
        loss = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()

        # 启用了颜色跟踪 (self.use_color_in_tracking)，则加入颜色损失，这是预测颜色与真实颜色之间的差异
        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss

        # 触发反向传播，计算梯度
        loss.backward()
        optimizer.step()
        # 再次清空梯度，为下一次迭代做准备
        optimizer.zero_grad()
        # 返回计算出的loss
        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        # 检查当前的映射索引和之前保存的映射索引是否一致，不一致则需要更新
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            # 使用深拷贝(copy.deepcopy)更新解码器decoders
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                # 更新特征网格feature_grid(self.c)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)

            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                if self.const_speed_assumption and idx-2 >= 0:
                    # 基于恒定速度假设的预测
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                    # 未启用恒定速度假设时的处理，新的相机姿态估计就简单地使用前一个姿态作为预测
                    estimated_new_cam_c2w = pre_c2w

                # get_tensor_from_camera(): 从位姿pose转变为张量tensor，其逆操作是get_camera_from_tensor()函数
                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                if self.seperate_LR:
                    # 是否分离学习率，即将旋转和平移的学习分离
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else:
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr)

                # 计算估计的相机姿态和真实相机姿态之间的平均绝对误差，在print打印信息的时候使用
                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                # candidate用于在for循环迭代中存储最优相机姿态
                candidate_cam_tensor = None
                # 用于在for循环迭代中存储最小loss，初值设的很大是方便后续更新赋值
                current_min_loss = 10000000000.
                
                # 相机姿态优化for循环
                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        # 处理独立的学习率情况
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    # 可视化当前迭代
                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                    # 优化相机pose
                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)

                    # 记录初始损失
                    if cam_iter == 0:
                        initial_loss = loss

                    # 计算相机姿态误差
                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                # 更新最优相机姿态
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            # 在经历一整个位姿的迭代更新过程后，本次tracker更新的最后的位姿结果，存储到estimate_c2w_list[idx]，后续在Mapper.py中的cur_c2w变量也会访问estimate_c2w_list
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            # 此处给pre_c2w赋值，pre_c2w用在新相机姿态的更新处（恒定速度假设 or not）
            pre_c2w = c2w.clone()
            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
