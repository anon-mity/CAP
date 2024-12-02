import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#from transforms3d.quaternions import quat2mat
from scipy.spatial.transform import Rotation as R
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))
from models.decoder.MLPDecoder import MLPDecoder, MLPDecoder_assembly
from models.encoder.vn_dgcnn import *
from models.encoder.vn_layers import *
from models.baseline.regressor_CR import Regressor_CR, Regressor_6d, VN_Regressor_6d
import utilss
from pytorch3d.loss import chamfer_distance
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


class CAP(nn.Module):
    def __init__(self, cfg):
        super(CAP, self).__init__()
        self.cfg = cfg
        # CCM
        self.encoder = self.init_encoder()
        self.encoder = torch.nn.DataParallel(self.encoder)
        self.predictor = self.init_pose_predictor()
        self.predictor = torch.nn.DataParallel(self.predictor)
        self.decoder = self.init_decoder()
        self.decoder = torch.nn.DataParallel(self.decoder)
        # HCM
        self.encoder_hcm = self.init_encoder()
        self.encoder_hcm = torch.nn.DataParallel(self.encoder_hcm)
        self.predictor_hcm = self.init_pose_predictor()
        self.predictor_hcm = torch.nn.DataParallel(self.predictor_hcm)

        # PCM
        if self.cfg.model.use_part:
            self.encoder_pcm = self.init_encoder()
            self.encoder_pcm = torch.nn.DataParallel(self.encoder_pcm)
            self.predictor_pcm = self.init_pose_predictor()
            self.predictor_pcm = torch.nn.DataParallel(self.predictor_pcm)
            # self.predictor_t_pcm = self.init_pose_predictor_trans()
            # self.predictor_t_pcm = torch.nn.DataParallel(self.predictor_t_pcm)


    def init_encoder(self):
        if self.cfg.model.encoder == 'vn_dgcnn':
            encoder = VN_DGCNN_New(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.model.encoder == 'vn_dgcnn_cap':
            encoder = VN_DGCNN_CAP(feat_dim=self.cfg.model.pc_feat_dim)

        return encoder

    def init_pose_predictor(self):
        pose_predictor = Regressor_CR(pc_feat_dim=self.cfg.model.pc_feat_dim * 2 * 3, out_dim=6)
        return pose_predictor

    def init_pose_predictor_trans(self):
        pose_predictor_trans = Regressor_CR(pc_feat_dim=self.cfg.model.pc_feat_dim * 2 * 3, out_dim=3)
        return pose_predictor_trans

    def init_decoder(self):
        if self.cfg.model.encoder == 'vn_dgcnn_cap':
            decoder = MLPDecoder_assembly(feat_dim=self.cfg.model.pc_feat_dim, num_points=self.cfg.data.num_pc_points)
        else:
            decoder = MLPDecoder(feat_dim=self.cfg.model.pc_feat_dim, num_points=self.cfg.data.num_pc_points)
        return decoder

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        return optimizer

    def forward(self, pc_aug):
        # pc (b,1024,3) >>>> (b, 3, 1024)
        src_pc = torch.permute(pc_aug, dims=(0, 2, 1))
        batch_size = src_pc.shape[0]

        # encoder
        if self.cfg.model.encoder == 'dgcnn':
            src_point_feat = self.encoder(src_pc)  # (batch_size, pc_feat_dim(512), num_point(1024))# (batch_size, pc_feat_dim(512), num_point(1024))
            src_feat = torch.mean(src_point_feat, dim=2)
            Equ_feat = src_feat
        else:
            # x_equ: [batch, 1024, 3],  x_inv:[batch, 1024, 1024]
            Equ_feat, Inv_feat = self.encoder(src_pc)

        # pose_predictor_R
        t0 = self.predictor(Equ_feat.reshape(batch_size, -1))  # 6d

        # recons
        if self.cfg.model.encoder == 'vn_dgcnn_cap':
            Inv_feat = torch.sum(Inv_feat, dim=1)  # b,1,1024
        x0 = self.decoder(Inv_feat)

        pred_dict = {
            't0': t0,  # 6d
            'x0': x0
        }
        return pred_dict


    def forward_hcm(self, Xc):
        # Xc (b,1024,3) >>>> (b, 3, 1024)
        src_pc = torch.permute(Xc, dims=(0, 2, 1))
        batch_size = src_pc.shape[0]

        # encoder
        # x_equ: [batch, 1024, 3],  x_inv:[batch, 1024, 1024]
        Equ_feat, _ = self.encoder_hcm(src_pc)

        # pose_predictor_R
        r0 = self.predictor_hcm(Equ_feat.reshape(batch_size, -1))  # 6d

        pred_dict = {
            'r0': r0  # 6d
        }
        return pred_dict

    def forward_pcm(self, xp):
        # xp (b,1024,3) >>>> (b, 3, 1024)
        src_pc = torch.permute(xp, dims=(0, 2, 1))
        batch_size = src_pc.shape[0]

        # encoder
        # x_equ: [batch, 1024, 3],  x_inv:[batch, 1024, 1024]
        Equ_feat, _ = self.encoder_pcm(src_pc)

        # pose_predictor_R
        r0 = self.predictor_pcm(Equ_feat.reshape(batch_size, -1))  # 6d

        pred_dict = {
            'rp': r0  # 6d
        }
        return pred_dict

    # 6D -> 3x3
    def recover_R_from_6d(self, R_6d):
        # R_6d:  b, 6 >>>> B, 3, 2
        R_6d = R_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        R = utilss.bgs(R_6d)
        # R is batch * 3 * 3
        return R

    def quat_to_eular(self, quat):
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])

        r = R.from_quat(quat)
        euler0 = r.as_euler('xyz', degrees=True)

        return euler0

    def Batch_SVD(self, Xc_seg):
        # Xc_seg: (B, N, 4)
        B, N, D = Xc_seg.shape
        assert D == 4

        device = Xc_seg.device
        dtype = Xc_seg.dtype

        # 提取坐标和 sup_flag
        coords = Xc_seg[:, :, :3]  # (B, N, 3)
        sup_flag = Xc_seg[:, :, 3]  # (B, N)

        normals = []
        for b in range(B):
            sup_flag_b = sup_flag[b]  # (N,)
            coords_b = coords[b]  # (N, 3)
            mask_sup = sup_flag_b > 0.5  # (N,)
            mask_non_sup = sup_flag_b <= 0.5  # (N,)

            if mask_sup.sum() >= 3:
                # 使用满足条件的支撑点
                coords_sup = coords_b[mask_sup]  # (num_sup_points, 3)
            else:
                # 使用 sup_flag 最大的三个点
                topk = min(3, N)
                sup_flag_topk, indices = torch.topk(sup_flag_b, topk)
                coords_sup = coords_b[indices]  # (topk, 3)

            num_points_b = coords_sup.shape[0]

            if num_points_b < 3:
                # 无法计算法向量，返回默认值
                normal_b = torch.tensor([0, 0, 1], device=device, dtype=dtype)
                normals.append(normal_b)
                continue

            # 将 coords_sup 移动到 CPU
            coords_sup_cpu = coords_sup.cpu()

            # 计算支撑点的质心
            centroid_sup = coords_sup_cpu.mean(dim=0, keepdim=True)  # (1, 3)

            # 中心化坐标
            coords_centered_b = coords_sup_cpu - centroid_sup  # (num_points_b, 3)

            # 计算协方差矩阵
            cov_b = coords_centered_b.t().mm(coords_centered_b) / num_points_b  # (3, 3)

            # 在 CPU 上计算 SVD
            try:
                u_b, s_b, v_b = torch.svd(cov_b)  # u_b: (3, 3), s_b: (3,), v_b: (3, 3)
                # 法向量对应于最小奇异值的右奇异向量
                normal_b = v_b[:, -1]  # (3,)
            except RuntimeError as e:
                # 处理可能的错误
                normal_b = torch.tensor([0, 0, 1], dtype=dtype)
                normals.append(normal_b)
                continue

            # 将法向量移动回原设备
            normal_b = normal_b.to(device=device)

            # 归一化法向量
            normal_b = normal_b / normal_b.norm()

            # **确定法向量的方向**

            # 如果非支撑点存在，计算其质心
            if mask_non_sup.sum() > 0:
                coords_non_sup = coords_b[mask_non_sup]  # (num_non_sup_points, 3)
                coords_non_sup_cpu = coords_non_sup.cpu()
                centroid_non_sup = coords_non_sup_cpu.mean(dim=0, keepdim=True)  # (1, 3)
            else:
                # 如果没有非支撑点，使用一个默认值，例如质心上方的一个点
                centroid_non_sup = centroid_sup + torch.tensor([[0, 0, 1]], dtype=dtype)

            # 计算从支撑点质心指向非支撑点质心的向量
            direction_vector = (centroid_non_sup - centroid_sup).squeeze(0)  # (3,)
            direction_vector = direction_vector.to(device=device)
            direction_vector = direction_vector / direction_vector.norm()

            # 计算法向量与方向向量的点积
            dot_product = torch.dot(normal_b, direction_vector)

            # 如果点积为负，翻转法向量
            if dot_product < 0:
                normal_b = -normal_b

            normals.append(normal_b)

        # 将法向量列表转换为张量
        normals = torch.stack(normals, dim=0)  # (B, 3)

        return normals

    def align_gravity(self,normals):

        normals = normals.squeeze(1)
        device = normals.device
        dtype = normals.dtype
        batch_size = normals.size(0)

        gravity_vector = torch.tensor([0, 1, 0], device=device, dtype=dtype)
        gravity_vector = gravity_vector / gravity_vector.norm()

        gravity_vectors = gravity_vector.unsqueeze(0).expand(batch_size, -1)  # (B, 3)

        cosine_similarity = F.cosine_similarity(normals, gravity_vectors, dim=1)  # (B,)

        loss = 1 - cosine_similarity  # (B,)

        loss = loss.mean()

        return loss


    def compute_ccm_loss(self, pc_input, pc_siamese, pred_data, pred_data_siam):
        # pred_data:{'t0': t0,'x0': x0 }
        t0_6d = pred_data['t0']
        t0_siam_6d = pred_data_siam['t0']
        t0_matrxic = self.recover_R_from_6d(t0_6d)
        t0_siam_matrxic = self.recover_R_from_6d(t0_siam_6d)

        # xc = x @ t0
        Xc = torch.einsum('bij,bjk->bik', pc_input, t0_matrxic)

        # xc = x @ t0
        Xc_siam = torch.einsum('bij,bjk->bik', pc_siamese, t0_siam_matrxic)

        # Le
        e_loss, _ = chamfer_distance(Xc, Xc_siam, batch_reduction='mean', point_reduction='mean')

        X0 = pred_data['x0']  # batch x 1024 x 3
        X0_siam = pred_data_siam['x0']  # batch x 1024 x 3

        # compute disentanglement Loss
        dis_loss1, _ = chamfer_distance(Xc, X0, batch_reduction='mean', point_reduction='mean')
        dis_loss2, _ = chamfer_distance(Xc_siam, X0_siam, batch_reduction='mean', point_reduction='mean')
        dis_loss = dis_loss1 + dis_loss2

        #ccm_loss = dis_loss + e_loss
        ccm_loss = dis_loss
        loss = {"ccm_loss": ccm_loss,
                "disentangle_loss": dis_loss,
                "le_loss": e_loss
               }
        return Xc, t0_matrxic, loss

    def compute_hcm_loss(self, xc, batch_data, pred_data, seg_network, t0_matrxic, pred_data_part):

        # pred_data:{'r0': r0}
        r0_6d = pred_data['r0']
        r0_matrxic = self.recover_R_from_6d(r0_6d)

        # xh = xc @ r0
        xh = torch.einsum('bij,bjk->bik', xc, r0_matrxic)

        # Ls + Lu
        # symm
        reflection_matrix = torch.diag(torch.tensor([-1, 1, 1], dtype=xh.dtype, device=xh.device))
        xh_symm = xh @ reflection_matrix
        symm_loss, _ = chamfer_distance(xh, xh_symm, batch_reduction='mean', point_reduction='mean')

        hcm_loss = 0.2*symm_loss

        # upright
        xh_seg = xh.transpose(2, 1)  # B, N, 3 -> B,3,N
        sup_flag = seg_network.forward(xh, cls_label=None)  # B, n, 1 是否为支撑的标签
        xh_seg = torch.cat((xh, sup_flag), dim=2)  # B, N, 4

        normals = self.Batch_SVD(xh_seg)  # B, 1, 3 法线向量

        # (1-cosine_similar)
        up_loss = self.align_gravity(normals)

        hcm_loss = 0.1 * symm_loss + 0.5 * up_loss

        return {"hcm_loss": hcm_loss, "symm_loss": symm_loss,  "up_loss": up_loss}



    def forward_pass_ccm(self, batch_data):
        pred_data = self.forward(batch_data['pc_aug'].float())
        pred_data_siam = self.forward(batch_data['pc_siamese'].float())

        Xc, t0_matrxic, loss_dict = self.compute_ccm_loss(batch_data['pc_aug'], batch_data['pc_siamese'], pred_data, pred_data_siam)
        #{"ccm_loss": ccm_loss, "disentangle_loss": dis_loss, "le_loss": e_loss}
        return Xc, t0_matrxic, loss_dict


    def forward_pass_hcm(self, xc, batch_data, seg_network, t0_matrxic):
        pred_data = self.forward_hcm(xc.float())

        # pred_data:{'r0': r0}
        if self.cfg.model.use_part:
            pred_data_part = self.forward_pcm(batch_data['pc_part_1024'].float())
            # pred_data:{'rp': rp}
        else:
            pred_data_part = None

        loss_dict = self.compute_hcm_loss(xc, batch_data, pred_data, seg_network, t0_matrxic, pred_data_part)
        # {"hcm_loss": hcm_loss,  "symm_loss": symm_loss,  "up_loss": up_loss}

        return loss_dict



