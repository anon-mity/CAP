import os
import argparse
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import matplotlib
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../su_scpe'))
from config import get_cfg_defaults
from datasets.baseline.modelnet40_dataset import ModelNet40
from datasets.baseline.ShapeNetH5Dataloader import ShapeNetH5Loader
from models.baseline.network_vnn import CAP
from tensorboardX import SummaryWriter
import csv
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F
import math





def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

def recover_R_from_6d( R_6d):
    # R_6d:  b, 6 >>>> B, 3, 2
    R_6d = R_6d.reshape(-1, 2, 3).permute(0, 2, 1)
    R = bgs(R_6d)
    # R is batch * 3 * 3
    return R
def computeCD(pred_pts,Xc):
    cd_loss, _ = chamfer_distance(pred_pts, Xc)
    return cd_loss

def is_orthogonal(matrix, tol=1e-6):
    """ Check if the matrix is orthogonal """
    R_transpose = matrix.T
    product = np.dot(R_transpose, matrix)
    identity_matrix = np.eye(3)
    check = np.allclose(product, identity_matrix, atol=tol)
    return check
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
def write_to_csv(file_path, category, total_X_dis_mean):
    # 打开或创建一个 CSV 文件并附加写入数据
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 写入一行数据
        writer.writerow([category, total_X_dis_mean])

def rotation_angle(A, B):
    trace = torch.trace(torch.mm(A.T, B))  # 计算矩阵的迹

    trace = torch.clamp(trace, min=-1.0 + 1e-6, max=3.0 - 1e-6)

    angle = (trace - 1) / 2
    if angle < -1 or angle > 1:
        print(f"Warning: angle value out of range: {angle}")
    hudu = torch.acos(angle)

    jiaodu = hudu * 180 / math.pi

    return jiaodu

def compute_pca_torch_batch(points):

    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("输入的点云数据应具有形状 (B, N, 3)")
    B, N, _ = points.shape
    mean_point_batch = torch.mean(points, dim=1, keepdim=True)  # (B, 1, 3)
    centered_points = points - mean_point_batch  # (B, N, 3)

    centered_points_T = centered_points.transpose(1, 2)  # (B, 3, N)

    cov_matrices = torch.matmul(centered_points_T, centered_points) / (N - 1)
    cov_matrices_cpu = cov_matrices.cpu()
    if torch.isnan(cov_matrices_cpu).any() or torch.isinf(cov_matrices_cpu).any():
        raise ValueError("协方差矩阵包含 NaN 或 Inf 值。")

    eigenvalues_batch_cpu, eigenvectors_batch_cpu = torch.linalg.eigh(cov_matrices_cpu)
    device = points.device
    eigenvalues_batch = eigenvalues_batch_cpu.to(device)
    eigenvectors_batch = eigenvectors_batch_cpu.to(device)
    idx = torch.argsort(eigenvalues_batch, descending=True)  # (B, 3)

    eigenvalues_batch = torch.gather(eigenvalues_batch, 1, idx)
    idx_expanded = idx.unsqueeze(2).expand(-1, -1, 3)  # (B, 3, 3)
    eigenvectors_batch = torch.gather(eigenvectors_batch, 1, idx_expanded)

    v1 = eigenvectors_batch[:, :, 0]  # (B, 3)
    v2 = eigenvectors_batch[:, :, 1]  # (B, 3)
    v3 = eigenvectors_batch[:, :, 2]  # (B, 3)
    cross = torch.cross(v1, v2, dim=1)  # (B, 3)
    dot_product = torch.sum(cross * v3, dim=1)  # (B,)

    mask = dot_product < 0
    v3[mask] = -v3[mask]
    eigenvectors_batch[:, :, 2] = v3

    R_batch = eigenvectors_batch.transpose(1, 2)  # (B, 3, 3)

    rotated_points_batch = torch.matmul(centered_points, R_batch)  # (B, N, 3)

    return R_batch, rotated_points_batch

# batchsize
def test(conf, network, val_dataloader, category):

    # >>>model
    network = nn.DataParallel(network, device_ids=[0,1])
    device = torch.device('cuda:0')
    network.to(device)

    # >>>>>> load checkpoints
    load_path = cfg.exp.checkpoint_dir
    if not os.path.exists(load_path):
        raise ValueError("Checkpoint {} not exists.".format(load_path))

    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
    if isinstance(network, torch.nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
        network.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        network.load_state_dict(checkpoint['model_state_dict'])
    print("Loading checkpoint from {} ...".format(load_path))

    # eval
    network.eval()
    num_batch = len(val_dataloader)
    val_batches = enumerate(val_dataloader, 0)
    total_consis = []

    for i, val_batch in val_batches:
        aug_data = val_batch['pc_part_1024_norm']
        aug_R = val_batch['R_aug']
        aug_data = aug_data.to(device)
        aug_R = aug_R.to(device)
        with torch.no_grad():
            # PCM
            pred_dict = network.module.forward_pcm(aug_data.float())
            pred_r6d = pred_dict['rp']
            pred_r = recover_R_from_6d(pred_r6d)
            pred_r = pred_r.cpu()
            #pca_r, rotated_points_batch = compute_pca_torch_batch(aug_data)

            R_mean = torch.mean(pred_r, dim=0).cpu()

            U, _, Vt = torch.svd(R_mean)
            R_mean = torch.mm(U, Vt.t())  # R_mean = U * V^T

            '''orthogonal_check3 = is_orthogonal(R_mean.detach().numpy())'''  # True
            angle_diffs = torch.tensor([rotation_angle(pred_r[i], R_mean) for i in range(10)])

            d_consis = torch.sqrt(torch.mean(angle_diffs ** 2))

            total_consis.append(d_consis)

    total_consis_mean = sum(total_consis) / len(total_consis)
    cate = category
    save_path = os.path.join(cfg.exp.log_dir, 'consistency_part.csv')
    write_to_csv(save_path, cate, total_consis_mean)

def main(cfg):


    cfg.defrost()
    cfg.set_new_allowed(True)

    cfg.exp.log_dir = os.path.join(cfg.exp.log_dir, cfg.exp.name)

    if not os.path.exists(cfg.exp.log_dir):
        os.makedirs(cfg.exp.log_dir)

    # >>>>>>>model
    model = CAP(cfg=cfg).cuda()
    #
    categorys = ['bench', 'cabinet', 'car', 'cellphone', 'chair', 'couch', 'firearm',
                 'lamp', 'monitor', 'plane', 'speaker', 'table', 'watercraft']
    for i in range(len(categorys)):
        category = categorys[i]
        cfg.exp.batch_size = 10
        val_set = ShapeNetH5Loader(data_path=cfg.data.root_dir,
                                num_pts=cfg.data.num_pc_points,
                                mode='val',
                                category=category)

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=cfg.exp.batch_size,
            num_workers=cfg.exp.num_workers,
            # persistent_workers=True,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        # Create checkpoint directory for save
        cfg.exp.checkpoint_dir = os.path.join('/home/hanbing/paper_code/CAP/log_dir_train_CHP', cfg.exp.name, category, 'ckpts/latest.pth')

        # training
        test(cfg, model, val_loader, category)

        print("Done eval...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval script")
    parser.add_argument('--cfg_file', default='/home/hanbing/paper_code/CAP/config/eval_vnn_pn.yml', type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)

    args = parser.parse_args()
    # args.cfg_file = './config/train.yml'
    #args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # if args.gpus == -1:
    #     args.gpus = [0, 1, 2, 3]

    cfg.freeze()
    print(cfg)
    main(cfg)
