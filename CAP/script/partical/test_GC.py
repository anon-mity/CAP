import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../su_scpe'))
from config import get_cfg_defaults
from datasets.baseline.modelnet40_dataset import ModelNet40
from datasets.baseline.ShapeNetH5Dataloader import ShapeNetH5Loader
from models.baseline.network_vnn import CAP
from tensorboardX import SummaryWriter
import csv
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F

def computeCD(pred_pts,Xc):
    cd_loss, _ = chamfer_distance(pred_pts, Xc)
    return cd_loss

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
def write_to_csv(file_path, category, total_X_dis_mean,total_T_dis_mean):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([category, total_X_dis_mean, total_T_dis_mean])

# batchsize
def test(conf, network, val_dataloader, category):
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
    network.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.eval()
    num_batch = len(val_dataloader)
    val_batches = enumerate(val_dataloader, 0)
    total_X_dis = []
    total_T_dis = []
    for i, val_batch in val_batches:
        batch_data = val_batch
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(device)
        with torch.no_grad():
            pred_dict = network.forward(batch_data['pc_aug'].float())
            # pred_dict = {'pred_r': pred_r, 'norm_recon_pts': norm_recon_pts }
            pred_pts = pred_dict["norm_recon_pts"]  # 120,n,3
            pred_r6d = pred_dict['pred_r']
            gt_pts = batch_data['pc_norm']
            distance = computeCD(pred_pts,gt_pts)

            pred_r = recover_R_from_6d(pred_r6d)
            # norm_r_pc = gt_pc @ pred_r
            transforms_pc = torch.einsum('bij,bjk->bik', batch_data['pc_aug'], pred_r)
            distance_T = computeCD(transforms_pc, gt_pts)
            total_T_dis.append(distance_T.item())
            total_X_dis.append(distance.item())
    total_X_dis_mean = sum(total_X_dis) / len(total_X_dis)
    total_T_dis_mean = sum(total_T_dis) / len(total_T_dis)
    cate = category
    CC_save_path = os.path.join(cfg.exp.log_dir, 'GC.csv')
    write_to_csv(CC_save_path, cate, total_X_dis_mean,total_T_dis_mean)

def main(cfg):
    all_gpus = list(cfg.gpus)
    if len(all_gpus) == 1:
        torch.cuda.set_device(all_gpus[0])

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
        cfg.exp.checkpoint_dir = os.path.join('/home/hanbing/公共的/paper_code/SU_for_SCPE/log_dir_training', cfg.exp.name, category, 'ckpts/best.pth')

        # training
        test(cfg, model, val_loader, category)

        print("Done eval...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval script")
    parser.add_argument('--cfg_file', default='/home/hanbing/公共的/paper_code/SU_for_SCPE/config/eval_vnn_pn.yml', type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)

    args = parser.parse_args()
    # args.cfg_file = './config/train.yml'
    #args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # if args.gpus == -1:
    #     args.gpus = [0, 1, 2, 3]
    cfg.gpus = cfg.exp.gpus

    cfg.freeze()
    print(cfg)
    main(cfg)
