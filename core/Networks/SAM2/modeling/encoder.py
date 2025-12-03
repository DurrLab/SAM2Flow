import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .backbones import image_encoder
from .position_encoding import PositionEmbeddingSine

from .backbones.hieradet_adapt import Hiera_adapt

class SAM2_encoder_adapted(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        trunk_cfg = cfg.trunk
        self.trunk = Hiera_adapt(**trunk_cfg)
        PE_cfg = cfg.PE
        position_encoding = PositionEmbeddingSine(**PE_cfg)
        neck_cfg = cfg.neck
        self.neck_context = image_encoder.FpnNeck(position_encoding, **neck_cfg)
        self.scalp = cfg.scalp
        if cfg.pretrained_fondation is not None:
            print(f"Loading pretrained SAM2 model ...")
            self.load_pretrained_weights(cfg.pretrained_fondation)

    def load_pretrained_weights(self, path):
        state_dict = torch.load(path, weights_only=True, map_location='cpu')['model']
        # block weights
        block_weights = {k.split('trunk.')[-1]: v for k, v in state_dict.items() if 'trunk.' in k}
        result = self.trunk.load_state_dict(block_weights, strict=False)
        # print(result.missing_keys)
        # adaptor weights
        adaptor_weights = {k.split('prompt_generator.')[-1]: v for k, v in state_dict.items() if 'prompt_generator' in k}
        if len(adaptor_weights) > 0:
            self.trunk.context_prompt_generator.load_state_dict(adaptor_weights, strict=True)
        # neck weights
        neck_weights = {k.split('neck.')[-1]: v for k, v in state_dict.items() if 'neck' in k}
        self.neck_context.load_state_dict(neck_weights, strict=True)
        # self.neck_feature.load_state_dict(neck_weights, strict=True)    
        print(f"Loaded pretrained SAM2 model from {path}")
        
        
    def forward(self, images, return_dict=False):
        contexts = self.trunk(images)
        contexts, pos_context = self.neck_context(contexts)

        if self.scalp > 0:
            # Discard the lowest resolution features
            contexts, pos_context = contexts[: -self.scalp], pos_context[: -self.scalp]
        if return_dict:
            output = {
            "context_feats": contexts[-1],
            "vision_pos_enc": pos_context,
            "backbone_fpn": contexts,
            }  
            return output 
        return contexts[-1]
