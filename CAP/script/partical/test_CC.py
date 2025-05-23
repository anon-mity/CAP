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


def write_to_csv(file_path, category,  total_T_dis_mean):
    # 打开或创建一个 CSV 文件并附加写入数据
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 写入一行数据
        writer.writerow([category, total_T_dis_mean])

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
    total_T_dis = []

    for i, val_batch in val_batches:
        batch_data = val_batch['pc_part_1024_norm']
        norm_data = val_batch['pc_aug']
        batch_data = batch_data.to(device)  # b,n,3 (120,n,3)
        norm_data = norm_data.to(device)
        with torch.no_grad():
            # CCM
            pred_dict = network.module.forward_pcm(batch_data.float())
            # pred_data:{'rp': rp }
            pred_r6d = pred_dict["rp"]  # 120,n,3
            pred_r = recover_R_from_6d(pred_r6d)

            distanceT = torch.empty(conf.exp.batch_size, device='cuda')

            for i in range(conf.exp.batch_size):
                T0 = pred_r[i]    # 1024,3,3
                inputx = norm_data[i]
                Xc = inputx @ T0  # 1024,3

                for j in range(conf.exp.batch_size):
                    T0i = pred_r[j]
                    inputxi = norm_data[j]
                    Xci = inputxi @ T0i  # 1024,3

                    dis_T = computeCD(Xc.unsqueeze(0), Xci.unsqueeze(0))

                    distanceT[j] = dis_T

                oneT_dis = distanceT.mean(dim=0)  # 一个点云的CC损失

                total_T_dis.append(oneT_dis.item())


    total_T_dis_mean = sum(total_T_dis) / len(total_T_dis)

    cate = category
    CC_save_path = os.path.join(cfg.exp.log_dir, 'CC_part.csv')
    write_to_csv(CC_save_path, cate,  total_T_dis_mean)

def main(cfg):


    cfg.defrost()
    cfg.set_new_allowed(True)

    cfg.exp.log_dir = os.path.join(cfg.exp.log_dir, cfg.exp.name)

    if not os.path.exists(cfg.exp.log_dir):
        os.makedirs(cfg.exp.log_dir)

    # >>>>>>>model
    model = CAP(cfg=cfg).cuda()
    #
    categorys = ['plane', 'car', 'chair']
    #categorys = ['bench', 'cabinet', 'car', 'cellphone', 'chair', 'couch', 'firearm',
    #             'lamp', 'monitor', 'plane', 'speaker', 'table', 'watercraft']

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
    #args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # if args.gpus == -1:
    #     args.gpus = [0, 1, 2, 3]

    cfg.freeze()
    print(cfg)
    main(cfg)
