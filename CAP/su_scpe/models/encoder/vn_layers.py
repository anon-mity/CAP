import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def conv1x1(in_channels, out_channels, dim):
    if dim == 3:
        return nn.Conv1d(in_channels, out_channels, 1, bias=False)
    elif dim == 4:
        return nn.Conv2d(in_channels, out_channels, 1, bias=False)
    elif dim == 5:
        return nn.Conv3d(in_channels, out_channels, 1, bias=False)
    else:
        raise NotImplementedError(f'{dim}D 1x1 Conv is not supported')


class VNSimpleLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNSimpleLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                    mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNNewLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNNewLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                mask * x + (1 - mask) * (x - (d / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, 20]
        '''
        # Linear (2,21)
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1) #x: [B, 21, 3, N_samples, 20]
        # BatchNorm
        p = self.batchnorm(p)

        # LeakyReLU
        # 应用 map_to_dir 线性层来计算方向向量 d
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)  # 论文中的q
        mask = (dotprod >= 0).float()            # 点积 > 0的mask
        d_norm_sq = (d * d).sum(2, keepdims=True) # # 论文中的k

        # 当 dotprod 非负时，输出 p；
        # 当 dotprod 为负时，输出调整后的 p，该调整考虑了 d 的方向和大小，以及负斜率。
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
                    mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm= 'none',
                 negative_slope=0.2):
        super(VNLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out




class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x

class VNBatchNorm_oavnn(nn.Module):
    def __init__(self, num_features, dim, mode='norm'):
        super(VNBatchNorm_oavnn, self).__init__()
        self.dim = dim
        self.mode = mode
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        norm = torch.sqrt((x * x).sum(2) + EPS)
        if self.mode == 'norm':
            norm_bn = self.bn(norm)
        elif self.mode == 'norm_log':
            norm_log = torch.log(norm)
            norm_log_bn = self.bn(norm_log)
            norm_bn = torch.exp(norm_log_bn)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / (norm + EPS) * norm_bn

        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''

        # 应用 map_to_dir 线性层来计算方向向量 d
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        # 方向向量d与特征向量的内积
        dotprod = (x * d).sum(2, keepdims=True)
        # 计算 dotprod 的最大值，并获取最大值的索引 idx。这代表了每个点在特征维度上最相关的方向。
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        # 使用 torch.meshgrid 创建一个索引网格，这个网格覆盖了 x 的所有前几个维度（除了最后一个维度）。
        # 将索引网格与 idx 连接起来，形成一个索引元组，用于从 x 中选择特定的元素。
        # 使用索引元组 index_tuple 从 x 中选择出最大点特征 x_max。
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)

class VNSimpleLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNSimpleLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNSimpleLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm_oavnn(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNSimpleStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, share_nonlinearity=False, use_batchnorm=True, combine_lin_nonlin=True,
                 reflection_variant=True, b_only=False):
        super(VNSimpleStdFeature, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm

        self.vn1 = VNSimpleLinearAndLeakyReLU(in_channels, in_channels // 2, dim=dim,
                                              share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
        self.vn2 = VNSimpleLinearAndLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                              share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
        self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0

class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point gloab features of shape [B, 1024, 3]
        先使用LBR LBR L学习出旋转矩阵，再@这个矩阵得到不变表示
        '''
        z0 = x
        z0 = self.vn1(z0)  # LBR [B, 1024, 3]->[B, 512, 3]
        z0 = self.vn2(z0)  # LBR [B, 512, 3]->[B, 256, 3]
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1) # L [B, 256, 3]->[B, 3, 3]

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)

            # compute the cross product of the two output vectors
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)  # 求解z0的逆

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0


class VNInFeature(nn.Module):
    """VN-Invariant layer."""

    def __init__(
            self,
            in_channels,
            dim=4,
            share_nonlinearity=False,
            negative_slope=0.2,
            use_rmat=False,
    ):
        super().__init__()

        self.dim = dim
        self.use_rmat = use_rmat
        self.vn1 = VNLinearBNLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn2 = VNLinearBNLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn_lin = conv1x1(
            in_channels // 4, 2 if self.use_rmat else 3, dim=dim)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]
        Returns:
            rotation invariant features of the same shape
        """
        z = self.vn1(x)
        z = self.vn2(z)
        z = self.vn_lin(z)  # [B, 3, 3, N] or [B, 2, 3, N]
        if self.use_rmat:
            z = z.flatten(1, 2).transpose(1, 2).contiguous()  # [B, N, 6]
            z = rot6d_to_matrix(z)  # [B, N, 3, 3]
            z = z.permute(0, 2, 3, 1)  # [B, 3, 3, N]
        z = z.transpose(1, 2).contiguous()

        if self.dim == 4:
            x_in = torch.einsum('bijm,bjkm->bikm', x, z)
        elif self.dim == 3:
            x_in = torch.einsum('bij,bjk->bik', x, z)
        elif self.dim == 5:
            x_in = torch.einsum('bijmn,bjkmn->bikmn', x, z)
        else:
            raise NotImplementedError(f'dim={self.dim} is not supported')

        return x_in


class VNLinearBNLeakyReLU(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            dim=5,
            share_nonlinearity=False,
            negative_slope=0.2,
    ):
        super().__init__()

        self.linear = VNLinear(in_channels, out_channels)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        self.leaky_relu = VNLeakyReLU(
            out_channels,
            # dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C_in, 3, N, ...]
        Returns:
            [B, C_out, 3, N, ...]
        """
        # Linear
        p = self.linear(x)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        p = self.leaky_relu(p)
        return p