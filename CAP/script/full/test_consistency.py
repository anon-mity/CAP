import os
import argparse
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
def write_to_csv(file_path, category, total_X_dis_mean,angle):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([category, total_X_dis_mean,angle])
def rotation_angle(A, B):
    trace = torch.trace(torch.mm(A.T, B))  # 计算矩阵的迹

    trace = torch.clamp(trace, min=-1.0 + 1e-6, max=3.0 - 1e-6)

    angle = (trace - 1) / 2
    if angle < -1 or angle > 1:
        print(f"Warning: angle value out of range: {angle}")
    hudu = torch.acos(angle)
    jiaodu = hudu * 180 / math.pi

    return jiaodu

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
        aug_data = val_batch['pc_aug']
        aug_R = val_batch['R_aug']
        aug_data = aug_data.to(device)
        aug_R = aug_R.to(device)
        with torch.no_grad():
            # CCM
            pred_dict = network.module.forward(aug_data.float())
            pred_r6d = pred_dict['t0']
            pred_r = recover_R_from_6d(pred_r6d)

            # HCM
            Xc = torch.einsum('bij,bjk->bik', aug_data, pred_r)
            pred_dict_hcm = network.module.forward_hcm(Xc.float())
            pred_r6d_hcm = pred_dict_hcm['r0']
            pred_r_hcm = recover_R_from_6d(pred_r6d_hcm)
            pred_r_hcm = torch.einsum('bij,bjk->bik', pred_r, pred_r_hcm)

            # pca_r, rotated_points_batch = compute_pca_torch_batch(aug_data)

            pred_r = pred_r.cpu()
            pred_r = pred_r_hcm.cpu()

            '''aug_R = aug_R.cpu()
            pred_r = pred_r.cpu()
            orthogonal_check = is_orthogonal_batch(aug_R.detach().numpy())  # True
            orthogonal_check1 = is_orthogonal_batch(pred_r.detach().numpy())  # True
            orthogonal_check2 = is_orthogonal_batch(RRT.detach().numpy())  # True'''

            aug_R = aug_R.cpu()
            # test CCM
            RRT = torch.bmm(aug_R, pred_r).cpu()
            # test HCM
            #RRT = torch.bmm(aug_R, pred_r_hcm).cpu()

            R_mean = torch.mean(RRT, dim=0).cpu()

            U, _, Vt = torch.svd(R_mean)
            R_mean = torch.mm(U, Vt.t())  # R_mean = U * V^T

            '''orthogonal_check3 = is_orthogonal(R_mean.detach().numpy())'''  # True

            angle_diffs = torch.tensor([rotation_angle(RRT[i], R_mean) for i in range(10)])

            d_consis = torch.sqrt(torch.mean(angle_diffs ** 2))

            total_consis.append(d_consis)

    total_consis_mean = sum(total_consis) / len(total_consis)
    cate = category
    save_path = os.path.join(cfg.exp.log_dir, 'consistency.csv')
    write_to_csv(save_path, cate, total_consis_mean, total_consis_mean)

def main(cfg):

    cfg.defrost()
    cfg.set_new_allowed(True)

    cfg.exp.log_dir = os.path.join(cfg.exp.log_dir, cfg.exp.name)
    if not os.path.exists(cfg.exp.log_dir):
        os.makedirs(cfg.exp.log_dir)

    # >>>>>>>model
    model = CAP(cfg=cfg).cuda()

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
