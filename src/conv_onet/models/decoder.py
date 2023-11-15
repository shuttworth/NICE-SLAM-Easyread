import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate


# 实现高斯傅里叶特征映射，在class MLP的__init__中被调用
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)

# 原Nerf的位置编码方式，在class MLP的__init__中被调用
class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


# 实现一个标准的全连接层，带有自定义的权重初始化和激活函数
class DenseLayer(nn.Linear):
    # out_dim：输出维度
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


# 创建一个简单的神经网络模块，其唯一作用是修改输入张量的形状
class Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(0)
        return x


# 多层感知机，解码点坐标，根据输入的点坐标等内容，生成颜色等输出，在class NICE(nn.Module)的__init__中供middle fine color调用
# ConvONet代码: https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/conv_onet/models/decoder.py
class MLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    # 初始化内部层和配置
    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        # 从顶部的几种positional embedding method方法中，选取其一，传参默认是fourier(高斯傅里叶特征映射)
        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            # 如果是color decoder，输出维度为4，对应3维度的RGB颜色值和另一个维度的occupancy值
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            # 如果不是color decoder，输出维度为1，对应occupancy值
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    # 将点坐标标准化并使用 grid_sample 方法进行采样
    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    # 前向传播，如果有特征维度，从特征网格中采样特征，对点坐标应用位置嵌入，通过一系列全连接层处理嵌入后的点坐标，可选择性地与采样的特征结合，生成最终输出
    def forward(self, p, c_grid=None):
        # 如果 c_dim 不为零（即存在特征维度），则从特征网格 c_grid 中采样特征
        if self.c_dim != 0:
            c = self.sample_grid_feature(
                p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

            # 仅fine层会concat特征
            if self.concat_feature:
                # only happen to fine decoder, get feature from middle level and concat to the current feature
                with torch.no_grad():
                    c_middle = self.sample_grid_feature(
                        p, c_grid['grid_middle']).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_middle], dim=1)

        p = p.float()

        embedded_pts = self.embedder(p)
        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            # h 的维度在每个 pts_linears 层后保持不变，因为这些层是全连接层，其输出大小由 hidden_size 决定
            h = F.relu(h)
            if self.c_dim != 0:
                h = h + self.fc_c[i](c)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out = self.output_linear(h)
        # out 的维度将是 (N, D)，其中 N 是输入点 p 的数量，D 是 output_linear 层的输出维度（4 或 1）
        # 如果 color=false，D 会被压缩，使得 out 的维度变为 (N,)
        if not self.color:
            out = out.squeeze(-1)
        return out


# 简化版MLP，点坐标仅用于从特征网格中采样特征，不作为 MLP 的输入，也没有positional embedding步骤，在class NICE(nn.Module)的__init__中供coarse调用
class MLP_no_xyz(nn.Module):
    """
    Decoder. Point coordinates only used in sampling the feature grids, not as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connection.
        grid_len (float): voxel length of its corresponding feature grid.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False,
                 sample_mode='bilinear', color=False, skips=[2], grid_len=0.16):
        super().__init__()
        self.name = name
        self.no_grad_feature = False
        self.color = color
        self.grid_len = grid_len
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + c_dim, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, grid_feature):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        c = F.grid_sample(grid_feature, vgrid, padding_mode='border',
                          align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid, **kwargs):
        c = self.sample_grid_feature(
            p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)
        h = c
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([c, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class NICE(nn.Module):
    """    
    Neural Implicit Scalable Encoding.

    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        coarse_grid_len (float): voxel length in coarse grid.
        middle_grid_len (float): voxel length in middle grid.
        fine_grid_len (float): voxel length in fine grid.
        color_grid_len (float): voxel length in color grid.
        hidden_size (int): hidden size of decoder network
        coarse (bool): whether or not to use coarse level.
        pos_embedding_method (str): positional embedding method.
    """

    def __init__(self, dim=3, c_dim=32,
                 coarse_grid_len=2.0,  middle_grid_len=0.16, fine_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=32, coarse=False, pos_embedding_method='fourier'):
        super().__init__()

        # coarse_decoder和middle_decoder等等decoder与MLP和MLP_no_xyz之间的关系
        if coarse:
            self.coarse_decoder = MLP_no_xyz(
                name='coarse', dim=dim, c_dim=c_dim, color=False, hidden_size=hidden_size, grid_len=coarse_grid_len)

        self.middle_decoder = MLP(name='middle', dim=dim, c_dim=c_dim, color=False,
                                  skips=[2], n_blocks=5, hidden_size=hidden_size,
                                  grid_len=middle_grid_len, pos_embedding_method=pos_embedding_method) # 默认embedding方法都是fourier
        self.fine_decoder = MLP(name='fine', dim=dim, c_dim=c_dim*2, color=False,
                                skips=[2], n_blocks=5, hidden_size=hidden_size,
                                grid_len=fine_grid_len, concat_feature=True, pos_embedding_method=pos_embedding_method)
        # 与上方decoder在传参上的区别: name='color', color=True, grid_len=color_grid_len
        self.color_decoder = MLP(name='color', dim=dim, c_dim=c_dim, color=True,
                                 skips=[2], n_blocks=5, hidden_size=hidden_size,
                                 grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)

    def forward(self, p, c_grid, stage='middle', **kwargs):
        """
            Output occupancy/color in different stage.
        """
        device = f'cuda:{p.get_device()}'
        # 使用粗糙解码器输出占用率
        if stage == 'coarse':
            # 输入参数是(p,c_grid)，decoder内部调用的是forward，输出一维张量，对应occupancy值
            occ = self.coarse_decoder(p, c_grid)
            occ = occ.squeeze(0)
            # raw的四个通道初始时都是0
            raw = torch.zeros(occ.shape[0], 4).to(device).float()
            # raw的最后一个通道值用occ赋值
            raw[..., -1] = occ
            return raw
        # 使用中等解码器输出占用率
        elif stage == 'middle':
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
            raw[..., -1] = middle_occ
            return raw
        # 使用精细解码器输出占用率，并将其与中等解码器的输出相加
        elif stage == 'fine':
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(fine_occ.shape[0], 4).to(device).float()
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ+middle_occ
            return raw
        # 使用颜色解码器输出颜色
        elif stage == 'color':
            # 使用 fine_decoder 获取精细层级的占用率（occupancy）信息: fine_occ
            fine_occ = self.fine_decoder(p, c_grid)
            # 使用 color_decoder 从相同的输入(p, c_grid)中获取颜色信息: raw
            # 注意此时raw是四列张量，前三列是RGB颜色，第四列对应occupancy值；
            raw = self.color_decoder(p, c_grid)
            # 使用 middle_decoder 获取中层级的占用率（occupancy）信息: middle_occ
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            # raw的RGB颜色值不变，第四列的occupancy值=fine_occ+middle_occ
            raw[..., -1] = fine_occ+middle_occ
            return raw
