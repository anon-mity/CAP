import os
import argparse
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../su_scpe'))
from config import get_cfg_defaults
from datasets.baseline.ShapeNetH5Dataloader import ShapeNetH5Loader
from models.baseline.network_vnn import CAP
import csv
from pytorch3d.loss import chamfer_distance
from scipy.spatial.transform import Rotation as sciR
import math
import torch.nn.functional as F

def R_from_euler_np(angles):
 
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(angles[0]), -math.sin(angles[0]) ],
                    [0,         math.sin(angles[0]), math.cos(angles[0])  ]
                    ])
    R_y = np.array([[math.cos(angles[1]),    0,      math.sin(angles[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(angles[1]),   0,      math.cos(angles[1])  ]
                    ])

    R_z = np.array([[math.cos(angles[2]),    -math.sin(angles[2]),    0],
                    [math.sin(angles[2]),    math.cos(angles[2]),     0],
                    [0,                     0,                      1]
                    ])
    return np.dot(R_z, np.dot( R_y, R_x ))
def rotate_point_cloud(data, R = None, max_degree = None):
   
    if R is not None:
      rotation_angle = R
    elif max_degree is not None:
      rotation_angle = np.random.randint(0, max_degree, 3) * np.pi / 180.0
    else:
      rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or  rotation_angle.ndim == 1:
      rotation_matrix = R_from_euler_np(rotation_angle)
    else:
      assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
      rotation_matrix = rotation_angle[:3, :3]

    if data is None:
      return None, rotation_matrix
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data, rotation_matrix # return [N, 3],

def computeCD(pred_pts,Xc):
    cd_loss, _ = chamfer_distance(pred_pts, Xc)
    return cd_loss

def write_to_csv(file_path, category, total_X_dis_mean):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([category, total_X_dis_mean])
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

def rotation_angle(A, B):
    trace = torch.trace(torch.mm(A.T, B)) 
    trace = torch.clamp(trace, min=-1.0 + 1e-6, max=3.0 - 1e-6)

    angle = (trace - 1) / 2
    if angle < -1 or angle > 1:
        print(f"Warning: angle value out of range: {angle}")
    hudu = torch.acos(angle)
    jiaodu = hudu * 180 / math.pi
    return jiaodu


def test(conf, network, val_dataloader, category):
    # >>>model
    network = nn.DataParallel(network, device_ids=[0, 1])
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

    # on cuda
    network.eval()
    num_batch = len(val_dataloader)
    val_batches = enumerate(val_dataloader, 0)
    total_sta = []
    # img_path = 'vis'
    for i, val_batch in val_batches:
        batch_data = np.empty((10, 1024, 3))  # 120,1024,3
        batch_R = np.empty((10, 3, 3))  # 120,3,3
        val_pc = val_batch['pc_norm'][0]

        for j in range(0,10):  # 100
            rotate_pc, R_aug = rotate_point_cloud(val_pc)
            batch_data[j] = rotate_pc
            batch_R[j] = R_aug

        # 将 NumPy 数组转换为 Tensor
        batch_data = torch.from_numpy(batch_data).float()
        batch_R = torch.from_numpy(batch_R).float()
        batch_data = batch_data.to(device)
        batch_R = batch_R.to(device)

        with torch.no_grad():
            # CCM
            pred_dict = network.module.forward(batch_data)
            # pred_data:{'t0': t0,'x0': x0 }
            pred_r6d = pred_dict["t0"]  # 100,6
            pred_r = recover_R_from_6d(pred_r6d)  # 100,3,3
            Xc = torch.einsum('bij,bjk->bik', batch_data, pred_r)

            # HCM
            pred_dict_hcm = network.module.forward_hcm(Xc.float())
            pred_r6d_hcm = pred_dict_hcm['r0']
            pred_r_hcm = recover_R_from_6d(pred_r6d_hcm)
            pred_r_hcm = torch.einsum('bij,bjk->bik', pred_r, pred_r_hcm)

            #pca_r, rotated_points_batch = compute_pca_torch_batch(batch_data)
            # test CCM
            RRT = torch.bmm(batch_R, pred_r).cpu()
            # test HCM
            RRT = torch.bmm(batch_R, pred_r_hcm).cpu()

            R_mean = torch.mean(RRT, dim=0)  # 3,3

            R_mean_cpu = R_mean.cpu()
            U, _, Vt = torch.svd(R_mean_cpu)
            R_mean = torch.mm(U, Vt.t())  # R_mean = U * V^T

            angle_diffs = torch.tensor([rotation_angle(RRT[i], R_mean) for i in range(10)])

            d_stability = torch.sqrt(torch.mean(angle_diffs ** 2))

            total_sta.append(d_stability)

    total_sta_mean = sum(total_sta) / len(total_sta)
    cate = category
    sta_save_path = os.path.join(cfg.exp.log_dir, 'stability.csv')
    write_to_csv(sta_save_path, cate, total_sta_mean)


def main(cfg):
    # 将 cfg 设置为可变，允许添加新属性
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.exp.log_dir = os.path.join(cfg.exp.log_dir, cfg.exp.name)
    # 确保目录存在
    if not os.path.exists(cfg.exp.log_dir):
        os.makedirs(cfg.exp.log_dir)

    # >>>>>>>model
    model = CAP(cfg=cfg).cuda()

    categorys = ['bench', 'cabinet', 'car', 'cellphone', 'chair', 'couch', 'firearm',
                 'lamp', 'monitor', 'plane', 'speaker', 'table', 'watercraft']
    for i in range(len(categorys)):
        category = categorys[i]
        cfg.exp.batch_size = 64
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
    # args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # if args.gpus == -1:
    #     args.gpus = [0, 1, 2, 3]

    cfg.freeze()
    print(cfg)
    main(cfg)
