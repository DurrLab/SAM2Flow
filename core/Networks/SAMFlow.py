import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.init import trunc_normal_

from .RAFT.cnn import BasicEncoder
from .RAFT.corr import CorrBlock
from .RAFT.extractor import ResNetFPN
from .RAFT.utils.utils import coords_grid, InputPadder
from .SAM2.modeling.encoder import SAM2_encoder_adapted
from .SAM2.modeling.mask_decoder import MaskDecoder
from .SAM2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from .SAM2.modeling.memory_encoder import (
    CXBlock,
    Fuser,
    MaskDownSampler,
    MemoryEncoder,
)
from .SAM2.modeling.position_encoding import PositionEmbeddingSine
from .SAM2.modeling.prompt_encoder import PromptEncoder
from .SAM2.modeling.transformer import Attention, RoPEAttention, TwoWayTransformer
from .update import BasicUpdateBlock, GMAUpdateBlock, SEAUpdateBlock


"""
Base model: two-frame optical flow estimation
**[No prompt, No memory]**
- Feature encoder (SEA-RAFT): ResNet34 or BasicEncoder
- Context encoder (SEA-RAFT / SAM2): ResNet34 or HieraNet
- Optical flow estimation (SEA-RAFT): GMA
"""
class SAMFlow_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.down_ratio = cfg.down_ratio
        # For encoders
        self.feature_dim = cfg.feature_dim
        self.context_dim = cfg.context_dim
        
        # For RNN
        self.net_dim = cfg.net_dim
        self.inp_dim = cfg.inp_dim
        self.motion_dim = cfg.motion_dim

        #########################################################################################
        # Image Encoding
        # 1.1 context network
        assert cfg.cnet in ['SAM2', 'SEA-RAFT']  # support SAM2 or SEA-RAFT encoder
        if cfg.cnet == 'SEA-RAFT':
            if cfg.fnet == 'basicencoder':
                print("[Using basicencoder as context encoder]")
                self.cnet = BasicEncoder(output_dim=self.context_dim, norm_fn='instance')
            elif cfg.fnet == 'resnet34':
                print("[Using ResNet34 as context encoder]")
                self.cnet = ResNetFPN(args=cfg.encoder.cnet, 
                                    input_dim=6, 
                                    output_dim=self.context_dim, 
                                    norm_layer=nn.BatchNorm2d, 
                                  init_weight=True)
            # elif cfg.fnet == 'twins':
            #     print("[Using Twins-SVT as context encoder]")
            #     self.cnet = twins_svt_large(pretrained=True)
            #     self.proj = nn.Conv2d(256, self.context_dim, 1)
        elif cfg.cnet == 'SAM2':
            print("[Using SAM2_encoder as context encoder]")
            self.cnet = SAM2_encoder_adapted(cfg.encoder)
        self.init_conv = nn.Conv2d(self.context_dim, self.net_dim + self.inp_dim, kernel_size=1)

        # 1.2 feature network
        assert cfg.fnet in ['basicencoder', 'resnet34', 'twins']
        if cfg.fnet == 'basicencoder':
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=self.feature_dim, norm_fn='instance')
        elif cfg.fnet =='resnet34':
            print("[Using ResNet34 as feature encoder]")
            self.fnet = ResNetFPN(args=cfg.encoder.fnet, 
                                  input_dim=3, 
                                  output_dim=self.feature_dim, 
                                  norm_layer=nn.BatchNorm2d, 
                                  init_weight=True)
        # elif cfg.fnet == 'twins':
        #     print("[Using Twins-SVT as feature encoder]")
        #     self.fnet = twins_svt_large(pretrained=True)
        #     self.channel_convertor = nn.Conv2d(256, self.feature_dim, kernel_size=1)

        #########################################################################################
        # 2. Optical Flow Estimation
        # 2.1 correlation volume
        print("[Using corr_fn {}]".format(self.cfg.corr_fn))
        # 2.2 update block: RNN
        if self.cfg.gma == "GMA":
            print("[Using GMA]")
            self.update_block = GMAUpdateBlock(
                self.cfg, hidden_dim=self.net_dim)
        elif self.cfg.gma == 'GMA-SK':
            print("[Using GMA-SK]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
                args=self.cfg, hidden_dim=self.net_dim)
        elif self.cfg.gma == 'GMA-SK2':
            print("[Using GMA-SK2]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_skflow(
                args=self.cfg, hidden_dim=self.net_dim)
        elif self.cfg.gma == 'SEA-RAFT':
            print("[Using SEA-RAFT RNN]")
            self.update_block = SEAUpdateBlock(
                args=self.cfg.gma_args, hdim=self.net_dim, cdim=self.motion_dim)
        self.roi_threshold = -1

        # 2.2 paierd encoders for motion encoding
        self.flow_head = nn.Sequential(
            nn.Conv2d(self.net_dim, 2 * self.net_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.net_dim, 6, 3, padding=1)
        )
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.net_dim, self.net_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.net_dim * 2, 64 * 9, 1, padding=0)
        )
        # 5. Training Setttings
        self.train_avg_length = cfg.train_avg_length
        self.use_var = True
        self.var_max = 10
        self.var_min = 0
        
    def load_pretrained_weights(self, ckpt_path):
        """ Load pretrained weights """
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            self.load_state_dict(ckpt_model, strict=True)
        else:
            self.load_state_dict(ckpt_model, strict=True)
        
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        # NOTE: this assumes img is of shape NCHW
        # NOTE: should we use a conv layer to initialize the flow?
        N, C, H, W = img.shape
        coords0 = coords_grid(1, H // self.down_ratio, W // self.down_ratio).to(img.device)
        coords1 = coords_grid(1, H // self.down_ratio, W // self.down_ratio).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1    
    
    def forward_feature_encoder(self, frame):
        """ 
        Inputs:
        - frame: B * C * H * W
        outputs:
        - fmaps: B * self.feature_dim * H//8 * W//8 
        - coords0: B * 2 * H//8 * W//8
        - coords1: B * 2 * H//8 * W//8
        """
        # Determine input shape
        if len(frame.shape) == 5:  # B, T, C, H, W
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:  # B, C, H, W
            need_reshape = False
        else:
            raise RuntimeError("Input shape not supported")

        # Normalization for feature network
        frame = 2 * (frame / 255.0) - 1.0
        frame = frame.float().contiguous()
        
        fmaps = self.fnet(frame)
        if need_reshape:
            # B*T*C*H*W
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])
            # frame = frame.view(b, t, *frame.shape[-3:])
        return fmaps
    
    def forward_context_encoder(self, frames):
        if self.cfg.cnet == 'SAM2':
            # SAM2 encoder
            context = self.forward_SAM_context_encoder(frames[:-1])
        else:
            # SEA-RAFT encoder
            context = self.forward_SEARAFT_context_encoder(frames[:-1], frames[1:])
        # init conv
        context = self.init_conv(context)
        # split into net and context
        net, context = torch.split(context, [self.net_dim, self.inp_dim], dim=1)
        return net, context

    def forward_SEARAFT_context_encoder(self, img1, img2):
        img1 = 2 * (img1 / 255.0) - 1.0
        img2 = 2 * (img2 / 255.0) - 1.0
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        N, _, H, W = img1.shape
        # dilation = torch.ones(N, 1, H//8, W//8, device=img1.device)
        # run the context network
        img_pair = torch.cat([img1, img2], dim=1)
        context = self.cnet(img_pair)
        return context
    
    def forward_SAM_context_encoder(self,
                                frame,
                                img_mean=(0.485, 0.456, 0.406),
                                img_std=(0.229, 0.224, 0.225),
                                ):
        """ 
        Inputs:
        - frame: B * C * H * W #NOTE Do we need to adapt to B * T * C * H * W?
        outputs:
        - context (dict):
        -- "context_feats": B * self.context_dim * H//8 * W//8
        -- "vision_pos_enc" (List): [(H//4 * W//4), (H//8 * W//8)]
        -- "backbone_fpn" (List): [(H//4 * W//4), (H//8 * W//8)])
        """
        # Determine input shape
        if len(frame.shape) == 5:  # B, T, C, H, W
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:  # B, C, H, W
            need_reshape = False
        else:
            raise RuntimeError("Input shape not supported")

        #############################
        # Normalization for Context network?
        #############################
        normalize = T.Normalize(mean=img_mean, std=img_std)
        frame = normalize(frame / 255.0)

        context = self.cnet(frame, return_dict=True)
        if need_reshape:
            # B*T*C*H*W
            context["context_feats"] = context["context_feats"].view(
                b, t, *context["context_feats"].shape[-3:])
            context["backbone_fpn"] = [
                x.view(b, t, *x.shape[-3:]) for x in context["backbone_fpn"]]
            context["vision_pos_enc"] = [
                x.view(b, t, *x.shape[-3:]) for x in context["vision_pos_enc"]]

        return context
    
    def upsample_flow(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)
    
    def forward(self, video, iters=None, flow_gt=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        T, _, H, W = video.shape
        N = T-1
        if iters is None:
            iters = self.cfg.decoder_depth
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=video.device)
        
        # context encoder
        net, context = self.forward_context_encoder(video)
        # init flow
        flow_predictions, info_predictions = [], []
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_flow(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)
            
        # run the feature network
        fmap_8x = self.forward_feature_encoder(video)
        corr_fn = CorrBlock(fmap_8x[:-1], fmap_8x[1:], 
                            num_levels=self.cfg.corr_levels, 
                            radius=self.cfg.corr_radius)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=video.device) + flow_8x).detach()
            corr = corr_fn(coords2)
            motion_features,_ = self.update_block.encode_motion_features(flow_8x, corr)
            net = self.update_block(net, context, motion_features)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_flow(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        if test_mode == False:
            # exlude invalid pixels and extremely large diplacements
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.cfg.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.var_max
                    var_min = self.var_min
                    
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                # Large b Component                
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                # Small b Component
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                # term2: [N, 2, m, H, W]
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                # term1: [N, m, H, W]
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)

            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': None}


@dataclass
class MemoryEntry:
    motion_feature: torch.Tensor
    motion_pos: torch.Tensor
    context_feature: torch.Tensor
    context_pos: torch.Tensor
    obj_ptr: torch.Tensor


class DualMemoryBank:
    """Per-sample FIFO storage for motion & context memories."""

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self.storage: Dict[int, deque] = {}
        self.batch_size = 0

    def ensure_batch(self, batch_size: int):
        self.batch_size = max(self.batch_size, batch_size)
        for idx in range(batch_size):
            if idx not in self.storage:
                self.storage[idx] = deque(maxlen=self.max_entries)

    def add(self, batch_idx: int, entry: MemoryEntry):
        if batch_idx not in self.storage:
            self.storage[batch_idx] = deque(maxlen=self.max_entries)
        self.storage[batch_idx].append(entry)

    def get(self, batch_idx: int) -> List[MemoryEntry]:
        return list(self.storage.get(batch_idx, []))


class SAM2Flow(SAMFlow_base):
    """
    Full SAM2Flow network with prompt-conditioned ROI decoding and dual memory.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prompt_cfg = cfg.prompt_encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.prompt_cfg.embed_dim,
            image_embedding_size=tuple(self.prompt_cfg.image_embedding_size),
            input_image_size=tuple(self.prompt_cfg.input_image_size),
            mask_in_chans=self.prompt_cfg.mask_in_chans,
        )
        transformer_cfg = cfg.roi_decoder.transformer_cfg
        transformer = TwoWayTransformer(
            depth=transformer_cfg.depth,
            embedding_dim=transformer_cfg.embedding_dim,
            num_heads=transformer_cfg.num_heads,
            mlp_dim=transformer_cfg.mlp_dim,
        )
        self.roi_decoder = MaskDecoder(
            transformer_dim=cfg.prompt_encoder.embed_dim,
            transformer=transformer,
            num_multimask_outputs=cfg.roi_decoder.num_multimask_outputs,
            iou_prediction_use_sigmoid=cfg.roi_decoder.iou_prediction_use_sigmoid,
            pred_obj_scores=cfg.roi_decoder.pred_obj_scores,
            pred_obj_scores_mlp=cfg.roi_decoder.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=cfg.roi_decoder.use_multimask_token_for_obj_ptr,
        )

        self.motion_pos_enc = PositionEmbeddingSine(self.motion_dim)
        self.context_pos_enc = PositionEmbeddingSine(self.inp_dim)

        self.motion_memory_encoder = self._build_memory_encoder(cfg.motion_memory, in_dim=self.motion_dim)
        self.context_memory_encoder = self._build_memory_encoder(cfg.context_memory, in_dim=self.inp_dim)
        self.motion_attention = self._build_memory_attention(cfg.motion_attention_layer, d_model=self.motion_dim, kv_dim=cfg.motion_memory.out_dim)
        self.context_attention = self._build_memory_attention(cfg.context_attention_layer, d_model=self.inp_dim, kv_dim=cfg.context_memory.out_dim)
        self.obj_ptr_proj = nn.Linear(self.prompt_cfg.embed_dim, self.inp_dim)

        self.max_memory_entries = cfg.max_mid_term_frames

    def _build_memory_encoder(self, cfg, in_dim: int) -> MemoryEncoder:
        mask_cfg = cfg.mask_downsampler
        mask_downsampler = MaskDownSampler(
            embed_dim=cfg.out_dim,
            kernel_size=mask_cfg.kernel_size,
            stride=mask_cfg.stride,
            padding=mask_cfg.padding,
            total_stride=mask_cfg.total_stride,
        )
        fuser_layer = CXBlock(
            dim=in_dim,
            kernel_size=cfg.fuser.kernel_size,
            padding=cfg.fuser.padding,
            layer_scale_init_value=cfg.fuser.layer_scale_init_value,
            use_dwconv=cfg.fuser.use_dwconv,
        )
        fuser = Fuser(fuser_layer, num_layers=cfg.fuser.num_layers, dim=in_dim, input_projection=True)
        pos_cfg = cfg.position_encoding
        pos_enc = PositionEmbeddingSine(
            num_pos_feats=pos_cfg.num_pos_feats,
            normalize=pos_cfg.normalize,
            scale=pos_cfg.scale,
            temperature=pos_cfg.temperature,
        )
        return MemoryEncoder(
            out_dim=cfg.out_dim,
            mask_downsampler=mask_downsampler,
            fuser=fuser,
            position_encoding=pos_enc,
            in_dim=in_dim,
        )

    def _build_memory_attention(self, cfg, d_model: int, kv_dim: int) -> MemoryAttention:
        self_attn = Attention(d_model, num_heads=cfg.num_heads)
        cross_attn = RoPEAttention(embedding_dim=d_model, num_heads=cfg.num_heads, kv_in_dim=kv_dim)
        layer = MemoryAttentionLayer(
            activation=cfg.activation,
            cross_attention=cross_attn,
            d_model=d_model,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            pos_enc_at_attn=cfg.pos_enc_at_attn,
            pos_enc_at_cross_attn_keys=cfg.pos_enc_at_cross_attn_keys,
            pos_enc_at_cross_attn_queries=cfg.pos_enc_at_cross_attn_queries,
            self_attention=self_attn,
        )
        return MemoryAttention(
            d_model=d_model,
            pos_enc_at_input=False,
            layer=layer,
            num_layers=cfg.num_layers,
            batch_first=False,
        )

    def _encode_context(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Runs the context encoder and returns a dictionary of feature tensors.
        """
        if self.cfg.cnet == 'SAM2':
            sam_out = self.forward_SAM_context_encoder(frames[:, :-1])
            feats = sam_out["context_feats"]
            backbone = sam_out["backbone_fpn"]
        else:
            feats = self.forward_SEARAFT_context_encoder(frames[:, :-1], frames[:, 1:])
            backbone = [feats, feats]

        fused = self.init_conv(feats)
        net, context = torch.split(fused, [self.net_dim, self.inp_dim], dim=1)
        return {
            "net": net,
            "context_tokens": context,
            "image_embeddings": feats,
            "backbone": backbone,
        }

    def _format_prompts(self, prompts, batch_size: int):
        if prompts is None:
            return [[] for _ in range(batch_size)]
        if isinstance(prompts, dict):
            prompts = [prompts]
        if len(prompts) == 1 and batch_size > 1:
            prompts = prompts * batch_size
        if len(prompts) != batch_size:
            raise ValueError("Prompt batch size mismatch.")

        formatted = []
        for sample in prompts:
            if sample is None:
                formatted.append([])
                continue
            if isinstance(sample, list):
                formatted.append(sample)
                continue
            pts = sample.get("points")
            labels = sample.get("labels")
            if pts is None or labels is None:
                formatted.append([])
                continue
            if pts.ndim == 2:
                pts = pts.unsqueeze(0)
                labels = labels.unsqueeze(0)
            rois = []
            for roi_pts, roi_lbl in zip(pts, labels):
                rois.append({"points": roi_pts, "labels": roi_lbl})
            formatted.append(rois)
        return formatted

    def _decode_prompts(self, context_dict, prompts, device):
        feats = context_dict["image_embeddings"]
        backbone = context_dict["backbone"]
        batch_size = feats.shape[0]
        prompts = self._format_prompts(prompts, batch_size)
        dense_pe = self.prompt_encoder.get_dense_pe().to(device)

        roi_logits = []
        roi_masks_context = []
        obj_ptrs = []
        batch_indices = []
        roi_counts = []

        for b in range(batch_size):
            if not prompts[b]:
                # default ROI: whole frame
                roi_counts.append(1)
                height, width = feats.shape[-2], feats.shape[-1]
                default_mask = torch.ones(1, 1, height, width, device=device)
                roi_logits.append(default_mask)
                roi_masks_context.append(default_mask)
                obj_ptrs.append(torch.zeros(1, self.prompt_cfg.embed_dim, device=device))
                batch_indices.append(b)
                continue

            roi_counts.append(len(prompts[b]))
            for roi in prompts[b]:
                coords = roi["points"].to(device).float().unsqueeze(0)
                labels = roi["labels"].to(device).float().unsqueeze(0)
                sparse, dense = self.prompt_encoder((coords, labels), None, None)
                high_res = [lvl[b:b+1] for lvl in backbone[:2]]
                masks, _, sam_tokens, _ = self.roi_decoder(
                    image_embeddings=feats[b:b+1],
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res,
                )
                roi_logits.append(masks)
                obj_ptrs.append(sam_tokens[:, 0, :])
                ctx_mask = F.interpolate(
                    torch.sigmoid(masks),
                    size=context_dict["context_tokens"].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                roi_masks_context.append(ctx_mask)
                batch_indices.append(b)

        roi_logits = torch.cat(roi_logits, dim=0)
        roi_masks_context = torch.cat(roi_masks_context, dim=0)
        obj_ptrs = torch.cat(obj_ptrs, dim=0)
        batch_indices = torch.tensor(batch_indices, device=device, dtype=torch.long)

        return {
            "roi_logits": roi_logits,
            "context_masks": roi_masks_context,
            "obj_ptrs": obj_ptrs,
            "batch_indices": batch_indices,
            "counts": roi_counts,
        }

    def _stack_memory(self, entries: List[MemoryEntry], attr: str, pos_attr: str):
        seqs = []
        poss = []
        for entry in entries:
            feat = getattr(entry, attr).unsqueeze(0)
            seqs.append(feat.flatten(2).permute(2, 0, 1))
            pos = getattr(entry, pos_attr).unsqueeze(0)
            poss.append(pos.flatten(2).permute(2, 0, 1))
        return torch.cat(seqs, dim=0), torch.cat(poss, dim=0)

    def _apply_motion_memory(self, motion_feat, motion_pos, batch_indices, memory_state: Optional[DualMemoryBank]):
        if memory_state is None:
            return motion_feat
        outputs = []
        for idx, sample in enumerate(batch_indices.tolist()):
            entries = memory_state.get(sample)
            if not entries:
                outputs.append(motion_feat[idx:idx+1])
                continue
            mem_seq, mem_pos = self._stack_memory(entries, "motion_feature", "motion_pos")
            curr_seq = motion_feat[idx:idx+1].flatten(2).permute(2, 0, 1)
            curr_pos = motion_pos[idx:idx+1].flatten(2).permute(2, 0, 1)
            attn_out = self.motion_attention(
                curr_seq,
                mem_seq,
                curr_pos=curr_pos,
                memory_pos=mem_pos,
                num_obj_ptr_tokens=0,
            )
            outputs.append(attn_out.permute(1, 2, 0).reshape(1, *motion_feat.shape[1:]))
        return torch.cat(outputs, dim=0)

    def _apply_context_memory(self, context_feat, context_pos, batch_indices, obj_ptrs, memory_state: Optional[DualMemoryBank]):
        if memory_state is None:
            return context_feat
        outputs = []
        for idx, sample in enumerate(batch_indices.tolist()):
            entries = memory_state.get(sample)
            if not entries:
                outputs.append(context_feat[idx:idx+1])
                continue
            mem_seq, mem_pos = self._stack_memory(entries, "context_feature", "context_pos")
            obj_tokens = self.obj_ptr_proj(torch.cat([entry.obj_ptr for entry in entries], dim=0)).unsqueeze(1)
            obj_pos = torch.zeros_like(obj_tokens)
            mem_seq = torch.cat([obj_tokens, mem_seq], dim=0)
            mem_pos = torch.cat([obj_pos, mem_pos], dim=0)
            curr_seq = context_feat[idx:idx+1].flatten(2).permute(2, 0, 1)
            curr_pos = context_pos[idx:idx+1].flatten(2).permute(2, 0, 1)
            attn_out = self.context_attention(
                curr_seq,
                mem_seq,
                curr_pos=curr_pos,
                memory_pos=mem_pos,
                num_obj_ptr_tokens=obj_tokens.shape[0],
            )
            outputs.append(attn_out.permute(1, 2, 0).reshape(1, *context_feat.shape[1:]))
        return torch.cat(outputs, dim=0)

    def _store_memory(self, memory_state, batch_indices, motion_feat, context_feat, obj_ptrs, roi_masks):
        if memory_state is None:
            return
        for idx, sample in enumerate(batch_indices.tolist()):
            mask = roi_masks[idx:idx+1].detach()
            motion_mem = self.motion_memory_encoder(motion_feat[idx:idx+1].detach(), mask)
            context_mem = self.context_memory_encoder(context_feat[idx:idx+1].detach(), mask)
            entry = MemoryEntry(
                motion_feature=motion_mem["vision_features"][0].detach(),
                motion_pos=motion_mem["vision_pos_enc"][0].detach(),
                context_feature=context_mem["vision_features"][0].detach(),
                context_pos=context_mem["vision_pos_enc"][0].detach(),
                obj_ptr=obj_ptrs[idx:idx+1].detach(),
            )
            memory_state.add(sample, entry)

    def forward(self, video, prompts=None, flow_gt=None, iters=None, test_mode=False, memory_state: Optional[DualMemoryBank]=None):
        assert video.dim() == 5, "Expecting input shape [B, T, C, H, W]"
        batch_size, num_frames = video.shape[:2]
        if num_frames < 2:
            raise ValueError("SAM2Flow requires at least two frames.")

        device = video.device
        frames = video[:, :2]
        context_dict = self._encode_context(frames)
        roi_info = self._decode_prompts(context_dict, prompts, device)
        roi_indices = roi_info["batch_indices"]

        net = context_dict["net"][roi_indices]
        context_tokens = context_dict["context_tokens"][roi_indices] * roi_info["context_masks"]
        obj_ptrs = roi_info["obj_ptrs"]

        fmap = self.forward_feature_encoder(frames)
        fmap1 = fmap[:, 0][roi_indices]
        fmap2 = fmap[:, 1][roi_indices]
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)

        height = frames.shape[-2] // self.down_ratio
        width = frames.shape[-1] // self.down_ratio
        coords0 = coords_grid(net.shape[0], height, width).to(device)
        flow_8x = torch.zeros_like(coords0)

        if iters is None:
            iters = self.cfg.decoder_depth
        flow_predictions, info_predictions = [], []

        if memory_state is None:
            memory_state = DualMemoryBank(self.max_memory_entries)
        memory_state.ensure_batch(batch_size)

        # initial prediction
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_8x + flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_flow(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        flow_masks = F.interpolate(
            roi_info["context_masks"],
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )

        for _ in range(iters):
            coords1 = coords0 + flow_8x
            corr = corr_fn(coords1)
            motion_features, _ = self.update_block.encode_motion_features(flow_8x, corr)
            motion_features = motion_features * flow_masks
            motion_pos = self.motion_pos_enc(motion_features)
            motion_features = self._apply_motion_memory(motion_features, motion_pos, roi_indices, memory_state)

            context_pos = self.context_pos_enc(context_tokens)
            context_tokens = self._apply_context_memory(context_tokens, context_pos, roi_indices, obj_ptrs, memory_state)

            net = self.update_block(net, context_tokens, motion_features)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            flow_up, info_up = self.upsample_flow(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        self._store_memory(memory_state, roi_indices, motion_features, context_tokens, obj_ptrs, roi_info["context_masks"])

        if test_mode:
            nf_predictions = None
        else:
            if flow_gt is None:
                flow_gt = torch.zeros_like(flow_predictions[-1])
            nf_predictions = []
            for pred, info in zip(flow_predictions, info_predictions):
                var_max = self.var_max
                var_min = self.var_min
                raw_b = info[:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info[:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                term2 = ((flow_gt - pred).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)

        return {
            'final': flow_predictions[-1],
            'flow': flow_predictions,
            'info': info_predictions,
            'nf': nf_predictions,
            'roi_masks': roi_info['roi_logits'],
            'roi_batch_indices': roi_indices,
             'roi_counts': roi_info['counts'],
            'memory_state': memory_state,
        }
