import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf


class Renderer(object):
    # 初始化
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_importance = cfg['rendering']['N_importance']

        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        self.nice = slam.nice
        self.bound = slam.bound

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    # 评估一组空间点的属性信息，根据输入的信息输出一个多维tensor；函数在render_batch_ray中调用
    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)
        bound = self.bound
        rets = []
        for pi in p_split:
            # mask for points out of bound
            # 对于每批点坐标pi，方法首先检查这些点是否在预设的边界bound内
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            # 若点坐标在边界内，它们将被送入解码器decoders进行处理。
            mask = mask_x & mask_y & mask_z

            pi = pi.unsqueeze(0)
            # 如果类实例的nice属性为真，则在调用解码器时会使用特征网格c；否则，不使用特征网格。
            if self.nice:
                # 和decoders.py里360多行的middle_decoder(p, c_grid) color_decoder(p, c_grid)等调用的传参基本是一致的
                # 我们可以合理的猜测，ret的值是occupancy的值或者是四通道的（三通道RGB+一通道OCC）
                ret = decoders(pi, c_grid=c, stage=stage)
            else:
                ret = decoders(pi, c_grid=None)
            ret = ret.squeeze(0)
            # 结合这里对一维张量且长度为4的处理
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    # 最重要函数，作用是渲染一批采样光线的颜色、深度和不确定性；此函数也作为外层调用的接口，在Mapper.py Tracker.py Mesher.py中都进行了调用
    # 方法逻辑：
    # 1. 根据光线的起点和方向，以及设定的参数，计算出一系列采样点。
    # 2. 如果存在真实深度图像（gt_depth），则在该深度附近采样更多点（N_surface）以提高渲染精度。
    # 3. 使用线性插值或透视插值（根据self.lindisp的值）计算采样点的深度值（z_vals）。
    # 4. 如果设定了扰动（self.perturb），则对采样点进行随机扰动以增加渲染的多样性。
    # 5. 根据采样点的空间位置（通过光线方程计算得到），生成一个点集（pointsf）。
    def render_batch_ray(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids. 特征网格
            decoders (nn.module): decoders. 解码器
            rays_d (tensor, N*3): rays direction. 光线方向
            rays_o (tensor, N*3): rays origin. 光线起点
            device (str): device name to compute on. 计算设备名称
            stage (str): query stage. 处于哪个阶段
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """

        N_samples = self.N_samples
        N_surface = self.N_surface
        N_importance = self.N_importance

        N_rays = rays_o.shape[0]

        # coarse粗渲染阶段用不到深度的真值
        if stage == 'coarse':
            gt_depth = None
        if gt_depth is None:
            # 无深度的真值，就不需要在表面附近采点 N_surface = 0
            N_surface = 0
            near = 0.01
        else:
            # 分层采样：N_samples->near->z_vals
            gt_depth = gt_depth.reshape(-1, 1)
            gt_depth_samples = gt_depth.repeat(1, N_samples)
            near = gt_depth_samples*0.01

        # 计算了每个光线与边界相交的最远距离（far_bb）
        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) -
                 det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
        else:
            far = far_bb
        
        # 表面采样：N_surface > 0 判断通过，开始在表面附近采样
        if N_surface > 0:
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
            else:
                # 在渲染过程中对每个像素点进行颜色处理，无论这些像素是否有深度传感器的读数。
                # 对于有深度信息的像素，采样点集中在实际深度附近；对于没有深度信息的像素，则在一个较大范围内均匀分布采样点
                
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.
                gt_none_zero_mask = gt_depth > 0
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).double().to(device)
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(
                    gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask,
                               :] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = near_surface * \
                    (1.-t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(
                    0).repeat((~gt_none_zero_mask).sum(), 1)
                z_vals_surface[~gt_none_zero_mask,
                               :] = z_vals_surface_depth_zero

        # 生成了一个在0到1之间的均匀分布的采样系数（t_vals）
        t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

        # 在nice_slam.yaml里lindisp默认是False，这里是对采样点的深度值z_vals应用不同的线性插值方式
        # 注意，这里依旧是分层采样，即原论文的Nstrat，z_vals里存储的是沿着每条光线采样的点的深度值
        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        # 是否有扰动
        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        # 合并额外的表面采样点
        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        # NICE_SLAM的点位置公式体现了： P = O + dr
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        pointsf = pts.reshape(-1, 3)

        # 将pointsf送入评估模块
        raw = self.eval_points(pointsf, decoders, c, stage, device)
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)

        # 这个 raw2outputs_nerf_color 函数是将从神经辐射场（NeRF）模型得到的原始预测转换为有用的深度图、深度方差、RGB颜色和权重。
        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
        
        # 判断是否需要重要性采样；在nice_slam.yaml里，N_importance=0,在原论文里也只有 Nstrat + Nimp
        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + \
                rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples+N_importance+N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
            return depth, uncertainty, color

        return depth, uncertainty, color

    # 仅在可视化中使用
    def render_img(self, c, decoders, c2w, device, stage, gt_depth=None):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=gt_depth_batch)

                depth, uncertainty, color = ret
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, uncertainty, color

    # this is only for imap*
    def regulation(self, c, decoders, rays_d, rays_o, gt_depth, device, stage='color'):
        """
        Regulation that discourage any geometry from the camera center to 0.85*depth.
        For imap, the geometry will not be as good if this loss is not added.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            gt_depth (tensor): sensor depth image
            device (str): device name to compute on.
            stage (str, optional):  query stage. Defaults to 'color'.

        Returns:
            sigma (tensor, N): volume density of sampled points.
        """
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, self.N_samples)
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        near = 0.0
        far = gt_depth*0.85
        z_vals = near * (1.-t_vals) + far * (t_vals)
        perturb = 1.0
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # (N_rays, N_samples, 3)
        pointsf = pts.reshape(-1, 3)
        raw = self.eval_points(pointsf, decoders, c, stage, device)
        sigma = raw[:, -1]
        return sigma
