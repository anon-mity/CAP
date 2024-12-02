import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .augment import rotate_point_cloud
from pytorch3d.loss import chamfer_distance
from mpl_toolkits.mplot3d import Axes3D
import csv
from sklearn.neighbors import NearestNeighbors

def visualize_point_clouds(original_X, clipped_X,augori_X,augclip_X):
    # 创建一个 3D 图形对象
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(121, projection='3d')

    # Visualize original point cloud in gray
    ax.scatter(original_X[:, 0], original_X[:, 1], original_X[:, 2], c='gray',  label='Original Point Cloud')

    # Visualize clipped point cloud in red
    ax.scatter(clipped_X[:, 0], clipped_X[:, 1], clipped_X[:, 2], c='red',  label='Clipped Point Cloud')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # 添加坐标轴箭头
    ax.quiver(0, 0, 0, 1, 0, 0, color='blue', arrow_length_ratio=0.1, label='X-axis')
    ax.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1, label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, 1, color='red', arrow_length_ratio=0.1, label='Z-axis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax1 = fig.add_subplot(122, projection='3d')

    # Visualize original point cloud in gray
    ax1.scatter(augori_X[:, 0], augori_X[:, 1], augori_X[:, 2], c='gray', label='augori_X Point Cloud')

    # Visualize clipped point cloud in red
    ax1.scatter(augclip_X[:, 0], augclip_X[:, 1], augclip_X[:, 2], c='red', label='augclip_X Point Cloud')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    # 添加坐标轴箭头
    ax1.quiver(0, 0, 0, 1, 0, 0, color='blue', arrow_length_ratio=0.1, label='X-axis')
    ax1.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1, label='Y-axis')
    ax1.quiver(0, 0, 0, 0, 0, 1, color='red', arrow_length_ratio=0.1, label='Z-axis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    plt.show()


def vis(point_cloud,vector,save):
    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 可视化点云
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='b', s=1, label='Point Cloud')

    # 可视化向量
    origin = np.array([[0, 0, 0]])  # 向量的起点
    ax.quiver(*origin[0], *vector[0], length=0.2, color='r', arrow_length_ratio=0.1, label='Vector')

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud and Vector')
    ax.legend()

    # 显示图形
    #plt.show()
    # 保存图像而不是显示
    plt.savefig(save)

    plt.close(fig)
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


def clipping_operator(X,v):
    N = X.shape[0]

    # Compute the dot product of each point with v
    dot_products = np.dot(X, v)

    # Find indices of the top N/2 points with highest dot(x, v) values
    N_half = N // 2
    sorted_indices = np.argsort(-dot_products)
    indices_to_remove = sorted_indices[:N_half]

    # Remove the selected points from X
    mask = np.ones(N, dtype=bool)
    mask[indices_to_remove] = False
    X_clipped = X[mask]

    return X_clipped


def is_orthogonal_batch(matrices, tol=1e-6):
    """ Check if each matrix in a batch is orthogonal """
    B = matrices.shape[0]
    results = np.zeros(B, dtype=bool)

    for i in range(B):
        matrix = matrices[i]
        R_transpose = matrix.T
        product = np.dot(R_transpose, matrix)
        identity_matrix = np.eye(3)
        results[i] = np.allclose(product, identity_matrix, atol=tol)

    return results

def add_gaussian_noise(point_cloud, noise_std=0.1):

    # 生成高斯噪声
    noise = np.random.normal(loc=0.0, scale=noise_std, size=point_cloud.shape)

    # 添加噪声到原始点云
    noisy_point_cloud = point_cloud + noise

    return noisy_point_cloud

def upsample_point_cloud(points, target_num=1024):
    num_points = points.shape[0]
    num_new_points = target_num - num_points

    # 使用KNN找到每个点的最近邻
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    new_points = []
    for i in range(num_new_points):
        idx = i % num_points
        neighbor_idx = indices[idx, 1]
        # 计算中点
        new_point = (points[idx] + points[neighbor_idx]) / 2
        new_points.append(new_point)

    new_points = np.array(new_points)
    upsampled_points = np.vstack((points, new_points))
    return upsampled_points

class ShapeNetH5Loader(Dataset):
    def __init__(self, data_path, mode='train', category='', num_pts=1024, transform=None):
        """
        Args:
            h5_file_path (str): Path to the h5 file containing the dataset.
            num_pts (int): Number of points to sample from each point cloud.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_path = os.path.join(data_path, f'{mode}_{category}.h5')
        self.num_pts = num_pts
        self.transform = transform

        # Load file
        with h5py.File(self.file_path, 'r') as f:
            self.data = f['data'][:]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        point_cloud = self.data[idx]

        # Randomly sample points
        if self.num_pts is not None and len(point_cloud) > self.num_pts:
            point_cloud = farthest_point_sample(point_cloud, self.num_pts)

        # part_caiyang
        y = np.array([[0, 1, 0]])
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        pc_part = clipping_operator(point_cloud, v)
        pc_part_1024_norm = upsample_point_cloud(pc_part)

        # R augment
        pc_norm = point_cloud
        pc_aug, R_aug = rotate_point_cloud(point_cloud[:, 0:3])
        pc_part_1024 = pc_part_1024_norm @ R_aug

        # 添加高斯噪声
        #pc_aug = add_gaussian_noise(pc_aug, noise_std=0.01)
        y_aug = y @ R_aug
        #pc_part_1024 = add_gaussian_noise(pc_part_1024, noise_std=0.01)
        pc_siamese, R_siam = rotate_point_cloud(point_cloud[:, 0:3])
        #pc_siamese = add_gaussian_noise(pc_siamese, noise_std=0.01)
        y_aug_siam = y @ R_siam

        return {
            'pc_norm': torch.from_numpy(pc_norm.astype(np.float32)),
            'pc_aug': torch.from_numpy(pc_aug.astype(np.float32)),  # 主分支
            'pc_siamese': torch.from_numpy(pc_siamese.astype(np.float32)),  # 等变一致性分支
            'pc_part': torch.from_numpy(pc_part.astype(np.float32)),  # 不完整观测分支
            'pc_part_1024': torch.from_numpy(pc_part_1024.astype(np.float32)),
            'pc_part_1024_norm': torch.from_numpy(pc_part_1024_norm.astype(np.float32)),
            'y_aug': torch.from_numpy(y_aug.astype(np.float32)),         # 不完整观测的方向，用于后面计算loss
            'y_aug_siam': torch.from_numpy(y_aug_siam.astype(np.float32)),
            'R_aug': torch.from_numpy(R_aug.astype(np.float32)),
            #"v": torch.from_numpy(v.astype(np.float32))  # 裁切部分点云时的观测向量
        }


def compute_pca_torch(points):
    import torch

    # 确保 points 是二维张量
    points = torch.atleast_2d(points)
    if points.shape[1] != 3:
        raise ValueError("输入的点云数据应具有三个坐标 (N, 3)")

    # 检查点的数量
    if points.shape[0] < 2:
        raise ValueError("点云数据必须包含至少两个点才能进行 PCA 分析。")

    # 1. 数据中心化
    mean_point = torch.mean(points, dim=0)
    centered_points = points - mean_point

    # 2. 计算协方差矩阵
    cov_matrix = torch.matmul(centered_points.T, centered_points) / (centered_points.shape[0] - 1)

    # 3. 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # 4. 特征值和特征向量排序（从大到小）
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. 计算旋转矩阵 R
    R = eigenvectors.T  # R 为旋转矩阵

    # 6. 将旋转矩阵应用于点云
    rotated_points = torch.matmul(centered_points, R.T)

    return R, eigenvalues, eigenvectors, mean_point, rotated_points


def vis_pts(inputx, X0, Xc, save_path):
    # 创建一个 3D 图形对象
    fig = plt.figure(figsize=(30, 15))

    # 创建3个子图用于可视化3个点云
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232, projection='3d')
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    # 可视化 inputx
    ax1.scatter(inputx[:, 0].cpu().numpy(), inputx[:, 1].cpu().numpy(), inputx[:, 2].cpu().numpy(), c=(0,1,0), marker='o', label='norm_x')
    ax1.set_title('inputx')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    # 可视化 'X0'
    ax2.scatter(X0[:, 0].cpu().numpy(), X0[:, 1].cpu().numpy(), X0[:, 2].cpu().numpy(), c=(0,1,0), marker='o', label='X0')
    ax2.set_title('X0')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    # 可视化 Xc
    ax3.scatter(Xc[:, 0].cpu().numpy(), Xc[:, 1].cpu().numpy(), Xc[:, 2].cpu().numpy(), c=(0,1,0), marker='o', label='Xc')
    ax3.set_title('Xc')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])

    plt.tight_layout()
    plt.show()

    # 保存图像而不是显示
    #plt.savefig(save_path)
    #plt.close(fig)
def visualize_pca(centered_points, eigenvalues, eigenvectors, rotated_points, save_path):
    """
    可视化原始点云、主成分轴以及旋转后的点云，并将图片保存到指定路径。

    参数：
    - centered_points: 中心化后的原始点云数据。
    - eigenvalues: 主成分对应的特征值。
    - eigenvectors: 主成分对应的特征向量。
    - rotated_points: 旋转后的点云数据。
    - save_path: 保存图片的文件路径。
    """
    # 可视化原始点云和主成分轴
    fig = plt.figure(figsize=(12, 6))

    # 原始点云和主成分轴
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(centered_points[:, 0], centered_points[:, 1], centered_points[:, 2], alpha=0.5, label='原始点云')

    # 绘制主成分轴
    origin = np.array([0, 0, 0])
    scale = np.sqrt(eigenvalues) * 2  # 根据特征值大小调整轴的长度

    V = eigenvectors  # 为方便理解

    ax1.quiver(origin[0], origin[1], origin[2],
               V[0, 0], V[1, 0], V[2, 0],
               length=scale[0], color='r', label='PC1')

    ax1.quiver(origin[0], origin[1], origin[2],
               V[0, 1], V[1, 1], V[2, 1],
               length=scale[1], color='g', label='PC2')

    ax1.quiver(origin[0], origin[1], origin[2],
               V[0, 2], V[1, 2], V[2, 2],
               length=scale[2], color='b', label='PC3')

    ax1.set_title('原始点云与主成分轴')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    ax1.legend()

    # 旋转后的点云与世界坐标系轴
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], alpha=0.5, label='旋转后的点云')

    # 绘制世界坐标系轴
    axis_length = np.max(rotated_points) - np.min(rotated_points)

    ax2.quiver(0, 0, 0, axis_length, 0, 0, color='r', label='X轴')
    ax2.quiver(0, 0, 0, 0, axis_length, 0, color='g', label='Y轴')
    ax2.quiver(0, 0, 0, 0, 0, axis_length, color='b', label='Z轴')

    ax2.set_title('旋转后的点云与世界坐标系')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # 关闭图形，释放内存


if __name__ == "__main__":
    # dataset_path = "../../../DATA/ShapeNetAtlasNetH5_1024"
    dataset_path = "/home/hanbing/datasets/ShapeNetAtlasNetH5_1024"
    shuffle = True
    category = 'plane'
    mode = 'train'
    batch_size = 100
    data_set = ShapeNetH5Loader(data_path=dataset_path, mode=mode, category=category, num_pts=1024)
    loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    output_dir = f"/home/hanbing/公共的/datasets/ShapeNetCore_Upright/{category}"  # 你希望保存的文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num = 0
    sin_err = []
    for l, batch in enumerate(loader):
        # pc_aug = pc_norm  @ R_aug
        batch_pc_aug = batch['pc_aug']  # b,1024,3
        for i in range(10):
            pc_aug = batch_pc_aug[i]
            R, eigenvalues, eigenvectors, mean_point, rotated_points = compute_pca_torch(pc_aug)

            centered_points = pc_aug - mean_point  # 确保传递中心化后的点云
            save = f'/home/hanbing/paper_code/SUPE/vis/PCA/pca_{i}.png'
            centered_points = centered_points.numpy()
            eigenvalues = eigenvalues.numpy()
            eigenvectors = eigenvectors.numpy()
            rotated_points = rotated_points.numpy()
            visualize_pca(centered_points, eigenvalues, eigenvectors, rotated_points, save)








