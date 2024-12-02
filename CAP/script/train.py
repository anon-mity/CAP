import os
import argparse
import torch
os.environ["CUDA_VISIBLE_DEVICES"]= "6,7,0,2"
from torch.utils.data import DataLoader
import torch.nn as nn
import importlib.util
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../su_scpe'))
from config import get_cfg_defaults
from datasets.baseline.ShapeNetH5Dataloader import ShapeNetH5Loader
from models.baseline.network_vnn import CAP
from tensorboardX import SummaryWriter


def import_module_from_path(module_name, file_path):

    module_dir = os.path.dirname(file_path)
    if module_dir not in sys.path:
        sys.path.append(module_dir)
        sys.path.append('/home/hanbing/paper_code/Point-M2AE-main')

    spec = importlib.util.spec_from_file_location(module_name, file_path)

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    sys.modules[module_name] = module
    return module


def train(conf, network, seg_network, train_dataloader, val_dataloader, category):

    # >>>>>>>> setting optimizer
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.optimizer.lr, weight_decay=conf.optimizer.weight_decay)
    train_num_batch = len(train_dataloader) # batchsize * train_num_batch = train data num of one epoch
    val_num_batch = len(val_dataloader) # batchsize * val_num_batch = val data num of one epoch

    # >>>>>>> load ckpt
    if cfg.exp.use_ckpt:
        load_path = cfg.exp.checkpoint_dir
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        if isinstance(network, torch.nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
            network.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            network.load_state_dict(checkpoint)
        print("Loading checkpoint from {} ...".format(load_path))

    # use TensorBoard
    train_writer = SummaryWriter(os.path.join(conf.exp.log_dir, "tb_logs", 'train'))
    val_writer = SummaryWriter(os.path.join(conf.exp.log_dir, "tb_logs", 'val'))
    csv_path = {}
    csv_path['loss'] = os.path.join(conf.exp.log_dir, "loss.csv")
    csv_path['metric'] = os.path.join(conf.exp.log_dir, "metric.csv")

    # train and eval for every epoch
    best_metric = 0.25

    for epoch in range(conf.exp.num_epochs):
        train_batches = enumerate(train_dataloader, 0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # train and val
        # >>>>>>>>> train for every batch
        for train_batch_ind, batch in train_batches:
            # set models to training mode
            network.train()
            # set data to device
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            # CCM
            Xc, t0_matrxic, losses_c = network.module.forward_pass_ccm(batch_data=batch)
            ccm_loss = losses_c["ccm_loss"]
            #loss = ccm_loss

            # detach()
            Xc = Xc.detach()
            t0_matrxic = t0_matrxic.detach()

            # HCM
            losses_h = network.module.forward_pass_hcm(Xc, batch, seg_network, t0_matrxic)
            hcm_loss = losses_h['hcm_loss']

            # if PCM
            if conf.model.use_part:
                part_loss = losses_h['part_loss']
                loss = ccm_loss + hcm_loss + part_loss
            else:
                loss = ccm_loss + hcm_loss

            # optimize one step
            network_opt.zero_grad()
            loss.backward()
            network_opt.step()

            total_norm = 0
            for name, param in network.named_parameters():
                if param.grad is not None:
                    train_writer.add_scalar(f'Gradients/{name}', param.grad.norm().item(), epoch)

            # tensorboard
            train_step = epoch * train_num_batch + train_batch_ind
            for key, value in losses_c.items():
                train_writer.add_scalar(f'{key}', value.item(), train_step)
            #for key, value in losses_h.items():
                #train_writer.add_scalar(f'{key}', value.item(), train_step)

            # log string
            if train_batch_ind % 50 == 0:
                print('epoch: ',epoch,'; train_batch_ind: ', train_batch_ind)
                print('train_loss: ', loss.detach().cpu().numpy())

            if epoch == conf.exp.num_epochs-1:
                with torch.no_grad():
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': network.module.state_dict()},
                         os.path.join(conf.exp.log_dir, 'ckpts', 'latest.pth'))


        # >>>>>>>>>>>>>> validate one batch
        val_batches = enumerate(val_dataloader, 0)
        for val_batch_ind, val_batch in val_batches:
            network.eval()
            for key in val_batch.keys():
                val_batch[key] = val_batch[key].to(device)

            with torch.no_grad():
                # CCM
                Xc, t0_matrxic, losses_c = network.module.forward_pass_ccm(batch_data=val_batch)
                ccm_loss = losses_c["ccm_loss"]
                #loss = ccm_loss
                # HCM
                losses_h = network.module.forward_pass_hcm(Xc, val_batch, seg_network, t0_matrxic)
                hcm_loss = losses_h['hcm_loss']

                #if PCM
                if conf.model.use_part:
                    part_loss = losses_h['part_loss']
                    loss = ccm_loss + hcm_loss + part_loss
                else:
                    loss = ccm_loss + hcm_loss

            # tensorboard
            val_fraction_done = (val_batch_ind + 1) / val_num_batch
            val_step = (epoch + val_fraction_done) * train_num_batch - 1
            for key, value in losses_c.items():
                val_writer.add_scalar(f'Metric_{key}',value.item(), val_step)
            #for key, value in losses_h.items():
           #     val_writer.add_scalar(f'Metric_{key}',value.item(), val_step)


            # log string
            if val_batch_ind % 10 == 0:
                print("val:", loss.detach().cpu().numpy())

            # save the best model
            if loss.item() < best_metric:
                best_metric = loss.item()
                with torch.no_grad():
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': network.module.state_dict()},
                        os.path.join(conf.exp.log_dir, 'ckpts', 'best.pth'))
            else:
                best_metric = best_metric


def main(cfg):

    seg_module = import_module_from_path('Point_SEG',cfg.exp.segmodel_path)

    categorys = ['bench', 'cabinet', 'car', 'cellphone', 'chair', 'couch', 'firearm',
                 'lamp', 'monitor', 'plane', 'speaker', 'table', 'watercraft']


    for i in range(len(categorys)):
        category = categorys[i]
        
        # >>>>>>>model
        model = CAP(cfg=cfg).cuda()
        # Parallel
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

        if cfg.exp.use_seg_model:
            #　init seg_model
            seg_model = seg_module.Point_M2AE_SEG(1).cuda()

            # forzen
            for param in seg_model.parameters():
                param.requires_grad = False
            # Parallel
            seg_model = nn.DataParallel(seg_model, device_ids=[0, 1, 2, 3])

            # load ckpt
            seg_ckpt = os.path.join('/home/hanbing/paper_code/Point-M2AE-main/segmentation/log/upright_focal', category, 'checkpoints/latest_model.pth')
            if not os.path.exists(seg_ckpt):
                raise ValueError("Seg Checkpoint {} not exists.".format(seg_ckpt))
            checkpoint = torch.load(seg_ckpt, map_location='cuda:0')
            seg_model.module.load_state_dict(checkpoint['model_state_dict'])

            print("Loading seg checkpoint from {} ...".format(seg_ckpt))
        else:
            seg_model = None

        # >>>>>>>dataloader
        train_set = ShapeNetH5Loader(data_path=cfg.data.root_dir,
                                num_pts=cfg.data.num_pc_points,
                                mode='train',
                                category=category)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=cfg.exp.batch_size,
            num_workers=cfg.exp.num_workers,
            # persistent_workers=True,
            pin_memory=False,
            shuffle=True,
            drop_last=True
        )

        val_set = ShapeNetH5Loader(data_path=cfg.data.root_dir,
                                num_pts=cfg.data.num_pc_points,
                                mode='val',
                                category=category)

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=cfg.exp.batch_size,
            num_workers=cfg.exp.num_workers,
            # persistent_workers=True,
            pin_memory=False,
            shuffle=True,
            drop_last=True
        )

        cfg.defrost()
        cfg.set_new_allowed(True)

        # Create checkpoint directory for save
        cfg.exp.log_dir = os.path.join(cfg.exp.log_dir_source, category)
        if not os.path.exists(cfg.exp.log_dir):
            os.makedirs(cfg.exp.log_dir)
        if not os.path.exists(os.path.join(cfg.exp.log_dir, "tb_logs")):
            os.makedirs(os.path.join(cfg.exp.log_dir, "tb_logs"))
        checkpoint_dir = os.path.join(cfg.exp.log_dir, "ckpts")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        cfg.freeze()

        # training
        train(cfg, model, seg_model, train_loader, val_loader, category)

        print(f"Done training {category}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg_file', default='/home/hanbing/paper_code/SUPE/config/train_vnn_pn.yml', type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    print(cfg)
    main(cfg)
