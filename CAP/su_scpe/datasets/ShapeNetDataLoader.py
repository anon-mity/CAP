# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import torch
from .augment import rotate_point_cloud
from .anchors import get_anchors
import open3d as o3d

def vis_point_cloud_with_normals(points_normals):
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()

    # 将点和法线数据赋值给点云对象
    pcd.points = o3d.utility.Vector3dVector(points_normals[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(points_normals[:, 3:])

    # 可视化点云和法线
    o3d.visualization.draw_geometries([pcd])

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


# 归一化点云到[-1,1],均值=0 方差不变
def pc_normalize_11(pc):
    # normalize [-1,1]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

# 按照边界框中心归一化点云到[-0.5,0.5],均值不=0 方差不变
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

class Shapenet_v0(Dataset):
    def __init__(self, data_root_dir, num_pc_points, split='train', class_choice='', use_normals=False, anchor_path=''):
        self.npoints = num_pc_points
        self.root = data_root_dir
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = use_normals

        # self.cat是一个字典，键表示类的名称，值表示对应文件夹的序号，比如"airplane" : 02691156 , ......
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        #if not class_choice is None:
        #    self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        # print(self.cat)
        # 通过train_test_split中的txt文件最后一位获得用来训练或测试的实例ID
        # train_ids  , val_ids , test_ids表示实例ID
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        # 把每个类中的训练、测试或验证数据路径划分到self.meta列表字典中,字典的键表示类，值是该类下所有用来train或test的txt文件路径列表
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            #os.listdir返回指定路径下的文件和文件夹的列表。
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            # 查找该类下面的id在不在train或test、val列表中
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
        #self.datapath和self.meta相似的一个元组列表，每个元组中有两个元素 一个是类别 一个是txt'文件路径
        #self.datapath是用于dataloder索引的数据列表
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.loadtxt(fn[1]).astype(np.float32)
            # point_set use xyz or xyz&normals
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]

        # FPS
        point_set = farthest_point_sample(point_set, self.npoints)

        # xyz was been normalize
        point_set[:, 0:3] = pc_normalize_0505(point_set[:, 0:3])  # 0505
        pc_norm = point_set  # 0505

        # R augment
        pc_aug, R_aug = rotate_point_cloud(point_set[:, 0:3])

        # if category = airplane
        upright_vec_norm = np.array([[0, 1, 0]])
        symmtry_vec_norm = np.array([[0, 0, 1]])
        upright_vec_aug = np.array([[0, 1, 0]]) @ R_aug
        symmtry_vec_aug = np.array([[0, 0, 1]]) @ R_aug

        if self.normal_channel:
            n = point_set[:, 3:] @ R_aug
            pc_aug = np.concatenate((pc_aug, n), axis=1)

        #Rt = R_aug.transpose(1, 0)
        #Rt1 = R_aug.T
        #R_inverse = np.linalg.inv(R_aug)
        return {
            'pc_aug' : torch.from_numpy(pc_aug.astype(np.float32)),
            'pc_norm': torch.from_numpy(pc_norm.astype(np.float32)),
            'R_aug'  : torch.from_numpy(R_aug.astype(np.float32)),
            'upright_vec_norm': torch.from_numpy(upright_vec_norm.astype(np.float32)),
            'symmtry_vec_norm': torch.from_numpy(symmtry_vec_norm.astype(np.float32)),
            'upright_vec_aug': torch.from_numpy(upright_vec_aug.astype(np.float32)),
            'symmtry_vec_aug': torch.from_numpy(symmtry_vec_aug.astype(np.float32)),
            #'anchors': torch.from_numpy(anchors.astype(np.float32)),
            #'R_delta': torch.from_numpy(R_delta.astype(np.float32)),
            #'R_label': torch.Tensor([R_label]).long(),
            #'category_label': category_label[0]
        }

    def __len__(self):
        return len(self.datapath)



