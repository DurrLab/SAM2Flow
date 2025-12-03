from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = 'SAMFlow_base'
_CN.suffix = 'MGH_samflownet'
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 16
_CN.sum_freq = 100
_CN.val_freq = 10000
_CN.image_size = [512, 512]
_CN.add_noise = False
_CN.use_smoothl1 = False
_CN.filter_epe = False
_CN.critical_params = []

_CN.network = 'SAM2Flow'

_CN.restore_steps = 0
_CN.mixed_precision = True
_CN.val_decoder_depth = 15

###############################################
# Mem

_CN.input_frames = 3
_CN.num_ref_frames = 2
_CN.train_avg_length = (512 * 512 // 64) * 3 / 2
_CN.mem_every = 1
_CN.top_k = None
_CN.enable_long_term = False
_CN.enable_long_term_count_usage = True
_CN.max_mid_term_frames = 7
_CN.min_mid_term_frames = 1
_CN.num_prototypes = 128
_CN.max_long_term_elements = 10000

################################################
################################################
_CN.SAMFlow_base = CN()
_CN.SAMFlow_base.pretrain = True

_CN.SAMFlow_base.image_size = _CN.image_size[0]
_CN.SAMFlow_base.down_ratio = 8 # downsample ratio for fnet & cnet

_CN.SAMFlow_base.feature_dim = 256
_CN.SAMFlow_base.context_dim = 256
_CN.SAMFlow_base.mem_dim     = 64
_CN.SAMFlow_base.net_dim     = 128
_CN.SAMFlow_base.inp_dim     = 128
_CN.SAMFlow_base.motion_dim  = 128

################################################
_CN.SAMFlow_base.cnet = 'SAM2'
_CN.SAMFlow_base.fnet = 'resnet34'
_CN.SAMFlow_base.resume = False
_CN.SAMFlow_base.restore_ckpt = None
_CN.SAMFlow_base.use_var = True
_CN.SAMFlow_base.var_min = 0
_CN.SAMFlow_base.var_max = 10

# SAM2 encoder
_CN.SAMFlow_base.encoder = CN()
_CN.SAMFlow_base.encoder.scalp = 1
_CN.SAMFlow_base.encoder.pretrained_fondation = ''

_CN.SAMFlow_base.encoder.trunk = CN()
_CN.SAMFlow_base.encoder.trunk.embed_dim = 144
_CN.SAMFlow_base.encoder.trunk.num_heads = 2
_CN.SAMFlow_base.encoder.trunk.stages = [2, 6, 36, 4]
_CN.SAMFlow_base.encoder.trunk.global_att_blocks = [23, 33, 43]
_CN.SAMFlow_base.encoder.trunk.window_pos_embed_bkg_spatial_size = [7, 7]
_CN.SAMFlow_base.encoder.trunk.window_spec = [8, 4, 16, 8]

_CN.SAMFlow_base.encoder.PE = CN()
_CN.SAMFlow_base.encoder.PE.num_pos_feats = _CN.SAMFlow_base.context_dim
_CN.SAMFlow_base.encoder.PE.normalize = True
_CN.SAMFlow_base.encoder.PE.scale = None
_CN.SAMFlow_base.encoder.PE.temperature = 10000

_CN.SAMFlow_base.encoder.neck = CN()
_CN.SAMFlow_base.encoder.neck.d_model = _CN.SAMFlow_base.context_dim
_CN.SAMFlow_base.encoder.neck.backbone_channel_list = [1152, 576, 288, 144]
_CN.SAMFlow_base.encoder.neck.fpn_top_down_levels = [2, 3]
_CN.SAMFlow_base.encoder.neck.fpn_interp_model = 'nearest'


_CN.SAMFlow_base.encoder.cnet = CN()
_CN.SAMFlow_base.encoder.cnet.block_dims = [64, 128, 256]
_CN.SAMFlow_base.encoder.cnet.initial_dim = 64
_CN.SAMFlow_base.encoder.cnet.pretrain = 'resnet34'

_CN.SAMFlow_base.encoder.fnet = CN()
_CN.SAMFlow_base.encoder.fnet.block_dims = [64, 128, 256]
_CN.SAMFlow_base.encoder.fnet.initial_dim = 64
_CN.SAMFlow_base.encoder.fnet.pretrain = 'resnet34'


################################################
# prompt encoder
_CN.SAMFlow_base.prompt_encoder = CN()
_CN.SAMFlow_base.prompt_encoder.embed_dim = 256
_CN.SAMFlow_base.prompt_encoder.image_embedding_size = [64, 64]
_CN.SAMFlow_base.prompt_encoder.input_image_size = [512, 512]
_CN.SAMFlow_base.prompt_encoder.mask_in_chans = 16
_CN.SAMFlow_base.roi_decoder = CN()
_CN.SAMFlow_base.roi_decoder.iou_prediction_use_sigmoid = True
_CN.SAMFlow_base.roi_decoder.pred_obj_scores = True
_CN.SAMFlow_base.roi_decoder.pred_obj_scores_mlp = True
_CN.SAMFlow_base.roi_decoder.use_multimask_token_for_obj_ptr = True
_CN.SAMFlow_base.roi_decoder.num_multimask_outputs = 3
_CN.SAMFlow_base.roi_decoder.transformer_cfg = CN()
_CN.SAMFlow_base.roi_decoder.transformer_cfg.depth = 2
_CN.SAMFlow_base.roi_decoder.transformer_cfg.embedding_dim = _CN.SAMFlow_base.prompt_encoder.embed_dim
_CN.SAMFlow_base.roi_decoder.transformer_cfg.mlp_dim = 2048
_CN.SAMFlow_base.roi_decoder.transformer_cfg.num_heads = 8

_CN.SAMFlow_base.motion_memory = CN()
_CN.SAMFlow_base.motion_memory.out_dim = _CN.SAMFlow_base.mem_dim
_CN.SAMFlow_base.motion_memory.mask_downsampler = CN()
_CN.SAMFlow_base.motion_memory.mask_downsampler.kernel_size = 3
_CN.SAMFlow_base.motion_memory.mask_downsampler.stride = 2
_CN.SAMFlow_base.motion_memory.mask_downsampler.padding = 1
_CN.SAMFlow_base.motion_memory.mask_downsampler.total_stride = 16
_CN.SAMFlow_base.motion_memory.fuser = CN()
_CN.SAMFlow_base.motion_memory.fuser.kernel_size = 7
_CN.SAMFlow_base.motion_memory.fuser.padding = 3
_CN.SAMFlow_base.motion_memory.fuser.layer_scale_init_value = 1e-6
_CN.SAMFlow_base.motion_memory.fuser.use_dwconv = True
_CN.SAMFlow_base.motion_memory.fuser.num_layers = 2
_CN.SAMFlow_base.motion_memory.position_encoding = CN()
_CN.SAMFlow_base.motion_memory.position_encoding.num_pos_feats = _CN.SAMFlow_base.mem_dim
_CN.SAMFlow_base.motion_memory.position_encoding.normalize = True
_CN.SAMFlow_base.motion_memory.position_encoding.scale = None
_CN.SAMFlow_base.motion_memory.position_encoding.temperature = 10000

_CN.SAMFlow_base.context_memory = CN()
_CN.SAMFlow_base.context_memory.out_dim = _CN.SAMFlow_base.mem_dim
_CN.SAMFlow_base.context_memory.mask_downsampler = _CN.SAMFlow_base.motion_memory.mask_downsampler.clone()
_CN.SAMFlow_base.context_memory.fuser = _CN.SAMFlow_base.motion_memory.fuser.clone()
_CN.SAMFlow_base.context_memory.position_encoding = _CN.SAMFlow_base.motion_memory.position_encoding.clone()

_CN.SAMFlow_base.motion_attention_layer = CN()
_CN.SAMFlow_base.motion_attention_layer.activation = 'relu'
_CN.SAMFlow_base.motion_attention_layer.dim_feedforward = int(_CN.SAMFlow_base.motion_dim * 8)
_CN.SAMFlow_base.motion_attention_layer.dropout = 0.1
_CN.SAMFlow_base.motion_attention_layer.pos_enc_at_attn = False
_CN.SAMFlow_base.motion_attention_layer.pos_enc_at_cross_attn_keys = True
_CN.SAMFlow_base.motion_attention_layer.pos_enc_at_cross_attn_queries = False
_CN.SAMFlow_base.motion_attention_layer.num_heads = 1
_CN.SAMFlow_base.motion_attention_layer.num_layers = 2

_CN.SAMFlow_base.context_attention_layer = CN()
_CN.SAMFlow_base.context_attention_layer.activation = 'relu'
_CN.SAMFlow_base.context_attention_layer.dim_feedforward = int(_CN.SAMFlow_base.inp_dim * 8)
_CN.SAMFlow_base.context_attention_layer.dropout = 0.1
_CN.SAMFlow_base.context_attention_layer.pos_enc_at_attn = False
_CN.SAMFlow_base.context_attention_layer.pos_enc_at_cross_attn_keys = True
_CN.SAMFlow_base.context_attention_layer.pos_enc_at_cross_attn_queries = False
_CN.SAMFlow_base.context_attention_layer.num_heads = 1
_CN.SAMFlow_base.context_attention_layer.num_layers = 2
_CN.SAMFlow_base.corr_fn = 'attended'
_CN.SAMFlow_base.corr_levels = 4
_CN.SAMFlow_base.corr_radius = 4

_CN.SAMFlow_base.gma = 'SEA-RAFT'
_CN.SAMFlow_base.gma_args = CN()
_CN.SAMFlow_base.gma_args.corr_channel = _CN.SAMFlow_base.corr_levels * (_CN.SAMFlow_base.corr_radius * 2 + 1) ** 2
_CN.SAMFlow_base.gma_args.num_blocks = 2
_CN.SAMFlow_base.decoder_depth = 8#12


_CN.SAMFlow_base.critical_params = ["cnet", "fnet", "pretrain", "corr_levels", "decoder_depth", "train_avg_length"]

_CN.SAMFlow_base.train_avg_length = _CN.train_avg_length


################################################
################################################
### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 1.75e-4
_CN.trainer.adamw_decay = 1e-3
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 400000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
# _CN.trainer.context_lr_factor = 1. #0.1
_CN.trainer.feature_lr_factor = 0.2

def get_cfg():
    return _CN.clone()
