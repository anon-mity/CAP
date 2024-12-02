from yacs.config import CfgNode as CN

# Miscellaneous configs
_C = CN()

# Experiment related
_C.exp = CN()
_C.exp.only_val       = False
_C.exp.name           = ''
_C.exp.checkpoint_dir = ''
_C.exp.weight_file    = ''
_C.exp.num_workers    = 8
_C.exp.batch_size     = 1
_C.exp.num_epochs     = 1000
_C.exp.log_dir        = ''
_C.exp.log_dir_source = ''
_C.exp.segmodel_path  = ''
_C.exp.use_seg_model  = False
_C.exp.load_from      = ''
_C.exp.use_ckpt      = False
_C.exp.category = ''

# Model related
_C.model = CN()
_C.model.encoder      = ''
_C.model.encoder_geo  = ''
_C.model.pose_predictor_quat = ''
_C.model.pose_predictor_rot = ''
_C.model.pose_predictor_trans = ''
_C.model.corr_module  = ''
_C.model.sdf_predictor= ''
_C.model.aggregator   = ''
_C.model.pc_feat_dim = 512 # 这里调feature的dimision
_C.model.transformer_feat_dim = 1024
_C.model.num_heads   = 4
_C.model.num_blocks  = 1

_C.model.use_part = False
_C.model.symmetry_loss = ''
_C.model.upright_loss = 'yes'


# Data related
_C.data = CN()
_C.data.root_dir       = ''
_C.data.num_pc_points  = 1024
_C.data.use_uniform_sample = True
_C.data.use_normals = False
_C.data.num_category = 40
_C.data.process_data = True


# Optimizer related
_C.optimizer = CN()
_C.optimizer.lr           = 1e-3
_C.optimizer.lr_decay     = 0.7
_C.optimizer.decay_step   = 2e4
_C.optimizer.weight_decay = 1e-6
_C.optimizer.lr_clip      = 1e-5

def get_cfg_defaults():
    return _C.clone()