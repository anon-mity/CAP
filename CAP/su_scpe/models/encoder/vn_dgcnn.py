import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
import os
import sys
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from models.encoder.vn_layers import *

def knn(x, k):
    # x:(b, 3, 1024)
    # k:20
    # -2 * 每个点与所有点的内积  (-2ab)
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # inner:(b,1024,1024)

    # 矩阵的**2表示矩阵每个元素值平方，矩阵维度不变
    # xx(b,1,1024)表示1024个点中 每个点的模长平方 (a**2 , b**2)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    # qurt(2ab - a**2 - b**2)
    # pairwise_distance:(b,1024,1024) 每个点与所有点的距离的平方
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # idx: (b, 1024 ,20) 每个点的k个最近邻点的索引
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, x_coord=None):
    # (b, 1, 3, 1024)
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points) # (b, 3, 1024)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            # idx: (b, 1024 ,20) 每个点的k个最近邻点的索引
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 基础索引 idx_base(b,1,1)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx.to(device)
    idx_base.to(device)
    # 将基础索引加到最近邻索引 idx 上，以便正确地索引整个批次的点
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous() # x:(b,3,1024) >>(b,1024,3)
    # 根据最近邻索引 idx 从 x 中取出每个点最近邻20个点的坐标，得到 (b * 1024, 3 * 20) 形状的特征矩阵
    feature = x.view(batch_size * num_points, -1)[idx, :]
    # feature >> (b, 1024, 20, 1, 3)
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    # x >> (b, 1024, 20, 1, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    # 从最近邻的坐标中减去了中心点的坐标，然后把中心点的坐标额外cat在旁边;
    # 对于每个最近邻群，每个点都拥有2x3的矩阵，其中分别表示中心点的坐标和该点减去中心点的残差坐标
    # feature : (b, 1024, 20, 2, 3) >> (b, 2, 3, 1024, 20)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

def get_graph_feature_oavnn(x, k=20, idx=None, x_coord=None, use_x_coord=False):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if not use_x_coord:  # dynamic knn graph
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:  # fixed knn graph with input point coordinates
            x_coord = x_coord.view(batch_size, -1, num_points)
            # if we just do idx = knn(x_coord, k=k), we get nan loss
            idx = knn(x_coord, k=k + 1)
            idx = idx[:, :, 1:]  # find k nearest neighbors for each point (excluding self as negihbor)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

def sq_dist_mat(source):
    # source = target = B x C x 3
    # a2
    r0 = source * source  # B x C x 3
    r0 = torch.sum(r0, dim=2, keepdim=True)  # B x C x 1
    # b2
    r1 = r0.permute(0, 2, 1)  # B x 1 x C
    # a2b2- 2ab
    sq_distance_mat = r0 - 2. * torch.matmul(source, source.permute(0, 2, 1)) + r1  # B x C x C

    return sq_distance_mat

def compute_patches(source, sq_distance_mat, num_samples):
    # source = target = B x 1024 x 3
    # sq_distance_mat = B x 1024 x 1024
    # num_samples : 1024/4
    batch_size = source.size()[0]
    num_points_source = source.size()[1]
    assert (num_samples <= num_points_source)

    sq_patches_dist, patches_idx = torch.topk(-sq_distance_mat, k=num_samples, dim=-1)  # B x C x k
    sq_patches_dist = -sq_patches_dist

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points_source
    patches_idx = patches_idx + idx_base
    patches_idx = patches_idx.view(-1)
    feature = source.view(batch_size * num_points_source, -1)[patches_idx, :]
    feature = feature.view(batch_size, num_points_source, num_samples, 3)  # feature (b, 1024, k, 3)
    return feature


class VN_DGCNN(nn.Module):
    def __init__(self, feat_dim):
        super(VN_DGCNN, self).__init__()
        self.n_knn = 20
        # num_part = feat_dim  # 原版是做partseg,所以num_part=feat_dim

        pooling = 'mean'

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(2 * feat_dim)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, feat_dim, dim=4, share_nonlinearity=True)

    def forward(self, x):

        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)
        batch_size = x.size(0)
        num_points = x.size(2)
        l = x[:, 0, 0:16].reshape(batch_size, 1, 16)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123)
        x = self.pool4(x)  # [batch, feature_dim, 3]
        return x


class VN_DGCNN_CAP(nn.Module):

    def __init__(self, feat_dim):
        super(VN_DGCNN_CAP, self).__init__()
        self.n_knn = 20
        # feat_dim = 512
        pooling = 'mean'

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv1_normvec = VNLinearLeakyReLU(2 * 2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.VnInv = VNStdFeature(2 * feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling层没有得到梯度
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(2 * feat_dim)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, feat_dim, dim=4, share_nonlinearity=True)  # feat_dim:512
        self.linear0 = nn.Linear(3, 2* feat_dim)

    def forward(self, x):

        # x: (batch_size, 3, num_points)
        batch = x.size(0)
        channel = x.size(1)
        npoint = x.size(2)

        x = x.unsqueeze(1)  # (b, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn)  # (b, 2(4), 3, 1024, 20)

        x = self.conv1(x)  # (b, 21, 3, 1024, 20)
        x = self.conv2(x)  # (b, 21, 3, 1024, 20)
        x1 = self.pool1(x)  # (b, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)  # (b, 42, 3, 1024, 20)
        x = self.conv3(x)  # (b, 21, 3, 1024, 20)
        x = self.conv4(x)  # (b, 21, 3, 1024, 20)
        x2 = self.pool2(x)  # (b, 21, 3, 1024)

        x = get_graph_feature(x2, k=self.n_knn)  # (b, 42, 3, 1024, 20)
        x = self.conv5(x)  # (b, 21, 3, 1024, 20)
        x3 = self.pool3(x)  # (b, 21, 3, 1024)

        x123 = torch.cat((x1, x2, x3), dim=1)  #

        x = self.conv6(x123)  # b, 64, 3 , 1024  -> b, embd , 1024
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)  # b, embd//3 *2 ,3 , 1024

        # globa pose
        x = self.pool4(x)  # [batch, 1024, 3] #

        # bubian tezheng ()
        x_inv, z0 = self.VnInv(x)    # [batch, 1024 , 3] #
        x_inv = self.linear0(x_inv)  # [batch, 1024 , 1024]
        return x, x_inv
