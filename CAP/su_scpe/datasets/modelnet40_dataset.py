'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import open3d as o3d
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from .augment import rotate_point_cloud

def vis_pc(data):
    # 提取 XYZ 坐标和法线
    points = data[:, :3]
    normals = data[:, 3:]

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # 运行可视化窗口的事件循环
    vis.run()
    vis.destroy_window()


def vis_pc_mtp(point_clouds):
    # point_clouds 是一个包含三个 (1024, 3) 数组的列表
    # 多个 1024x3 在坐标系中的比较
    # 创建一个图形窗口
    fig = plt.figure(figsize=(20, 5))  # 可以调整图形大小

    # 为每个点云创建一个子图 (1行3列)
    for i, point_cloud in enumerate(point_clouds):
        # 添加子图 (1行3列), 第 i+1 个图
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')

        # 绘制点云
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

        # 设置坐标轴标签
        ax.set_xlabel('X', fontsize=18)
        ax.set_ylabel('Y', fontsize=18)
        ax.set_zlabel('Z', fontsize=18)

        # 设置坐标轴范围
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # 设置子图标题
        ax.set_title(f'Point Cloud {i + 1}')

        # 绘制坐标轴 (xyz轴)
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X-Axis')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y-Axis')
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z-Axis')

        # 可选：隐藏坐标轴
        # ax.axis('off')

        # 可选：添加图例
        ax.legend()

    # 调整子图间距
    plt.tight_layout()

    # 显示图形
    plt.show()

# 归一化点云到[-1,1], 均值为0 方差不变
def pc_normalize_11(pc):
    # normalize [-1,1]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

# 按照边界框中心归一化点云到[-0.5,0.5],均值不变 方差不变
def pc_normalize_0505(pc):
    # 计算点云的边界、中心点和边界长度。
    boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
    center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
    length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

    # all normalize into 0
    pc_canon = (pc - center_pt.reshape(1, 3)) / length_bb
    pc = np.copy(pc_canon)  # centered at 0
    # 规范化后的点云转换到 NOCS 空间，通过加 0.5 使点云位于 [0, 1] 范围内。
    return pc


# Max-Min归一化点云到[0,1],均值不变 方差不变
def pc_normalize_01(pc):
    # Max-Min [0,1]
    # 计算每个维度的最大值和最小值
    max_vals, _ = pc.max(dim=0)  # 最大值
    min_vals, _ = pc.min(dim=0)  # 最小值

    # 归一化点云
    normalized_point_cloud = (pc - min_vals) / (max_vals - min_vals + 1e-7)

    return normalized_point_cloud


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNet40(Dataset):
    def __init__(self, data_root_dir, num_pc_points, split='train', process_data=False, use_uniform_sample=False, use_normals=False,num_category=40,anchor_path=''):
        self.root = data_root_dir
        self.npoints = num_pc_points
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category
        self.anchor_path = anchor_path
        self.anchors = get_anchors.get_rotation_anchors(anchor_path=self.anchor_path, k=60, returnRs=True)

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        # self.cat 是一个包含了数据集中所有类别的列表
        self.cat = [line.rstrip() for line in open(self.catfile)]
        # self.classes 是一个字典，其中每个类别对应一个唯一的整数标签[0 ~ N-1]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # shape_ids包含了所有训练或测试数据名称的列表
        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train_copy.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test_copy.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]

        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        # 如果最远点采样
        if self.uniform:
            self.save_path = os.path.join(data_root_dir,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(data_root_dir, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # if process_data ==true
        # fps采样1024个点(包括法线向量)放入dat文件中
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                # 直接读取1024个点的dat文件
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    # self.list_of_points : [list,1024,6]
                    self.list_of_points, self.list_of_labels = pickle.load(f)


    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, category_label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            category_label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize_0505(point_set[:, 0:3])  # [-0.5,0.5]
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        pc_norm = point_set + 0.5  # [0,1]norm
        pc_aug, R_aug = rotate_point_cloud(point_set)
        anchors = self.anchors
        _, R_label, R_delta = get_anchors.rotation_distance_np(R_aug, anchors)

        return {
            'pc_aug' : torch.from_numpy(pc_aug.astype(np.float32)),
            'pc_norm': torch.from_numpy(pc_norm.astype(np.float32)),
            'R_aug'  : torch.from_numpy(R_aug.astype(np.float32)),
            'anchors': torch.from_numpy(anchors.astype(np.float32)),
            'R_delta': torch.from_numpy(R_delta.astype(np.float32)),
            'R_label': torch.Tensor([R_label]).long(),
            'category_label': category_label[0]
        }

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data_path = 'data/modelnet40_normal_resampled/'
    data = ModelNet40(data_path, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    train_batches = enumerate(DataLoader, 0)
    for train_batch_ind, batch in train_batches:
        batch = batch  
        pc_aug = batch['pc_aug'][0]
        pc_norm = batch['pc_norm'][0]
        anchors = batch['anchors'][0].double()
        R_label = batch['R_label'][0]
        R_delta = batch['R_delta'][0]
        R_delta = R_delta[R_label][0]
        anchor = anchors[R_label]
        pc_anchors = torch.mm(pc_aug, R_delta.T)
        pc_delta = torch.mm(pc_anchors, anchor[0].T)
        vis_list = [pc_aug, pc_anchors, pc_delta, pc_norm]
        # vis_pc(point_set)
        vis_pc_mtp(vis_list)


    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
