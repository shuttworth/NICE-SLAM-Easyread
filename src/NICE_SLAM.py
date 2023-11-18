import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')


class NICE_SLAM():
    """
    NICE_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):
        # 初始化配置和参数
        self.cfg = cfg
        self.args = args
        self.nice = args.nice

        # 从配置中读取各种设置
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']

        # 设置输出目录
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)

        # 读取相机配置
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        # 初始化模型
        model = config.get_model(cfg,  nice=self.nice)
        self.shared_decoders = model

        # 加载其他配置
        self.scale = cfg['scale']

        self.load_bound(cfg)
        if self.nice:
            self.load_pretrain(cfg)
            self.grid_init(cfg)
        else:
            self.shared_c = {}

        # need to use spawn
        # 设置多进程启动方法
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # 初始化帧读取器
        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)

        # 初始化估计的相机位姿列表
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()

        # 初始化真实的相机位姿列表
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()

        # 初始化其他共享内存变量
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()

        # 将共享变量移至指定设备并共享内存
        for key, val in self.shared_c.items():
            val = val.to(self.cfg['mapping']['device'])
            val.share_memory_()
            self.shared_c[key] = val
        
        # 初始化渲染器、网格生成器、日志记录器
        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)

        # 初始化映射器和追踪器
        if self.coarse:
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True)
        self.tracker = Tracker(cfg, args, self)

        # 打印输出描述
        self.print_output_desc()

    def print_output_desc(self):
        # 打印输出信息，在上方的__init__里调用
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    # 根据预处理配置更新相机的内参，这可能包括调整图像大小或裁剪边缘，在__init__里调用
    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch) 
        # 检查配置中是否有 crop_size 参数。如果有，它将调整相机的焦距和主点坐标以适应新的图像尺寸
        # sx 和 sy 是宽度和高度的缩放比例，分别用于调整焦距（fx, fy）和主点坐标（cx, cy）。最后，更新图像的宽度（W）和高度（H）为新的裁剪尺寸
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        # 检查配置中是否有 crop_edge 参数，用于裁剪图像边缘，如果 crop_edge 大于0(nice_slam.yaml里的crop_edge值是0)，它将从图像的宽度和高度中减去两倍的 crop_edge 值，并相应地调整主点坐标
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    # 加载和设置场景的边界参数，在__init__里调用
    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        # 从配置中读取边界参数，并将其转换为一个PyTorch张量。边界参数被乘以一个全局缩放因子 self.scale(nice_slam.yaml里的scale值是1)，用于调整场景的大小
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound'])*self.scale)
        bound_divisible = cfg['grid_len']['bound_divisible']
        # enlarge the bound a bit to allow it divisible by bound_divisible
        # 调整边界的上限，使其可以被 bound_divisible 整除
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisible).int()+1)*bound_divisible+self.bound[:, 0]
        # 如果执行的是nice-slam的算法
        if self.nice:
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            # 如果粗层场景表达是coarse，给乘以一个额外的扩大因子 self.coarse_bound_enlarge，粗粒度解码器需要处理更大范围的场景数据
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound*self.coarse_bound_enlarge

    # 加载预先训练的ConvOnet参数，在__init__里调用
    # ConvONet论文:https://arxiv.org/pdf/2003.04618.pdf
    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        if self.coarse:
            # ckpt加载coarse权重(从yaml的pretrained_decoders里)
            ckpt = torch.load(cfg['pretrained_decoders']['coarse'],
                              map_location=cfg['mapping']['device'])
            # 初始化一个空字典，用于存储调整后的权重
            coarse_dict = {}
            # 遍历模型权重，只处理解码器的权重，排除编码器的权重
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            # 加载权重到解码器
            self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict)

        # ckpt加载middle_fine权重(从yaml的pretrained_decoders里)
        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict)
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict)

    # 分层特征网格初始化
    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids.

        Args:
            cfg (dict): parsed config dict.
        """
        # 各项grid_len参数设置见yaml里的值
        if self.coarse:
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        c = {}
        # 特征向量维度c_dim和场景边界xyz_len
        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        # If you have questions regarding the swap of axis 0 and 2,
        # please refer to https://github.com/cvg/nice-slam/issues/24

        if self.coarse:
            coarse_key = 'grid_coarse'
            coarse_val_shape = list(
                map(int, (xyz_len*self.coarse_bound_enlarge/coarse_grid_len).tolist()))
            coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]
            self.coarse_val_shape = coarse_val_shape
            val_shape = [1, c_dim, *coarse_val_shape]
            # 初始化一个具有特定形状和尺寸的零张量，并用标准正态分布填充，mid fine color同理；标准正态分布是深度学习中常见的权重初始化方法，有助于模型的训练和收敛
            coarse_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
            c[coarse_key] = coarse_val
        
        middle_key = 'grid_middle'
        middle_val_shape = list(map(int, (xyz_len/middle_grid_len).tolist()))
        middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
        self.middle_val_shape = middle_val_shape
        val_shape = [1, c_dim, *middle_val_shape]
        middle_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[middle_key] = middle_val

        fine_key = 'grid_fine'
        fine_val_shape = list(map(int, (xyz_len/fine_grid_len).tolist()))
        fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        self.fine_val_shape = fine_val_shape
        val_shape = [1, c_dim, *fine_val_shape]
        fine_val = torch.zeros(val_shape).normal_(mean=0, std=0.0001) # 精细网格使用更小的标准差进行初始化
        c[fine_key] = fine_val

        color_key = 'grid_color'
        color_val_shape = list(map(int, (xyz_len/color_grid_len).tolist()))
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[color_key] = color_val

        # 所有初始化的网格（粗糙、中等、精细、颜色）被存储在一个字典 c 中，每个网格对应一个键（例如，'grid_coarse', 'grid_middle' 等）
        # 这个字典随后被赋值给 self.shared_c，使得这些网格可以在整个类的其他方法中被共享和访问
        self.shared_c = c

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        # 一定要进行了初始化、确定了世界坐标系，才能够进行Tracking；
        # 而NICE-SLAM这样的NeRF based SLAM初始化的办法就是把第一帧图像拍摄的相机位置作为世界坐标系原点，然后先建图再去跟踪；
        # 在Tracking中，只优化camera pose，不优化hierarchical scene representation
        while (1):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def coarse_mapping(self, rank):
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(3):
            # 当 rank 为 0 时，创建一个tracking进程；为1时，创建一个mapping进程；为2时，进行self.coarse的判断，通过则执行coarse_mapping线程
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            elif rank == 2:
                if self.coarse:
                    p = mp.Process(target=self.coarse_mapping, args=(rank, ))
                else:
                    continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
