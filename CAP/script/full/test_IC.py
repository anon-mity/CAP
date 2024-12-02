import os
import argparse
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,5"
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import matplotlib
matplotlib.use('Agg')
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
    '''
    angles: [(b, )3]
    '''
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

def write_to_csv(file_path, category, total_X_dis_mean, total_T_dis_mean, total_T_hcm_dis_mean):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([category, total_X_dis_mean, total_T_dis_mean, total_T_hcm_dis_mean])
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

def test(conf, network, val_dataloader, category):

    # >>>model
    network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
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
    total_X_dis = []
    total_T_dis = []
    total_T_hcm_dis = []
    # img_path = 'vis'
    for i, val_batch in val_batches:
        batch_data = np.empty((101, 1024, 3))  # 120,1024,3
        val_pc = val_batch['pc_aug'][0]
        batch_data[0] = val_pc
        for j in range(1,101):  # 121
            rotate_pc, _ = rotate_point_cloud(val_pc)
            batch_data[j] = rotate_pc
        # 将 NumPy 数组转换为 Tensor
        batch_data = torch.from_numpy(batch_data).float()
        batch_data = batch_data.to(device)
        with torch.no_grad():
            # CCM
            pred_dict = network.module.forward(batch_data)
            # pred_data:{'t0': t0,'x0': x0 }
            pred_r6d = pred_dict["t0"]  # 121,6
            pred_pts = pred_dict['x0']  # 121,n,3
            pred_r = recover_R_from_6d(pred_r6d)

            # HCM
            xc = torch.einsum('bij,bjk->bik', batch_data, pred_r)
            pred_dict_hcm = network.module.forward_hcm(xc.float())
            pred_r6d_hcm = pred_dict_hcm['r0']
            pred_r_hcm = recover_R_from_6d(pred_r6d_hcm)


            inputx = batch_data[0]
            X0 = pred_pts[0]
            T0 = pred_r[0]
            R0 = pred_r_hcm[0]
            Xc = inputx @ T0
            Xh = Xc @ R0

            distance = torch.empty(100)  # 120
            distanceT = torch.empty(100)  # 120
            distanceT_hcm = torch.empty(100)  # 120

            for i in range(1,101):  # 121
                inputxi = batch_data[i]
                X0i = pred_pts[i]
                T0i = pred_r[i]
                R0i = pred_r_hcm[i]
                Xci = inputxi @ T0i
                Xhi = Xci @ R0i

                dis = computeCD(X0i.unsqueeze(0),X0.unsqueeze(0))
                distance[i-1] = dis
                dis_T = computeCD(Xc.unsqueeze(0), Xci.unsqueeze(0))
                distanceT[i - 1] = dis_T
                dis_T_hcm = computeCD(Xh.unsqueeze(0), Xhi.unsqueeze(0))
                distanceT_hcm[i-1] = dis_T_hcm

            oneX_dis = distance.mean(dim=0)
            oneT_dis = distanceT.mean(dim=0)
            oneT_hcm_dis = distanceT_hcm.mean(dim=0)

            total_X_dis.append(oneX_dis.item())
            total_T_dis.append(oneT_dis.item())
            total_T_hcm_dis.append(oneT_hcm_dis.item())
    total_X_dis_mean = sum(total_X_dis) / len(total_X_dis)
    total_T_dis_mean = sum(total_T_dis) / len(total_T_dis)
    total_T_hcm_dis_mean = sum(total_T_hcm_dis) / len(total_T_hcm_dis)

    cate = category
    IC_save_path = os.path.join(cfg.exp.log_dir, 'IC.csv')
    write_to_csv(IC_save_path, cate, total_X_dis_mean, total_T_dis_mean, total_T_hcm_dis_mean)


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
    #
    categorys = ['bench', 'cabinet', 'car', 'cellphone', 'chair', 'couch', 'firearm',
                 'lamp', 'monitor', 'plane', 'speaker', 'table', 'watercraft']
    for i in range(len(categorys)):
        category = categorys[i]
        cfg.exp.batch_size = 80
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
