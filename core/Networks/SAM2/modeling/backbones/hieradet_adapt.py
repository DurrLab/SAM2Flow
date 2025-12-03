from functools import partial
from typing import List, Tuple, Union
from itertools import repeat

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from .hieradet import (
    do_pool,
    MultiScaleAttention,
    MultiScaleBlock,
)

from .adapter_utils import PromptGenerator

class Hiera_adapt(nn.Module): # hiera_L
    def __init__(
        self,
        #[!]
        in_chans=3,
        embed_dims=[144, 288, 576, 1152], 
        img_size: int = 512,
        patch_size: int = 4,
        
        embed_dim: int = 144,# initial embed dim
        num_heads: int = 2,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 6, 36, 4),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (7, 7),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (8, 4, 16, 8),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (23, 33, 43),
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )

        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        self.embed_dim = embed_dims
        self.depth = stages
        self.scale_factor = 32
        self.prompt_type = 'highpass'
        self.tuning_stage = "1234"
        self.input_type = 'fft'
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        self.adaptor = 'adaptor'
        self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dim,
                                                self.tuning_stage, self.depth,
                                                self.input_type, self.freq_nums,
                                                self.handcrafted_tune, self.embedding_tune, self.adaptor,
                                                img_size)

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        inp = x
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(inp)

        self.block1 = []
        self.block2 = []
        self.block3 = []
        self.block4 = []
        outputs = []

        for i, blk in enumerate(self.blocks):
            if i < 3:
                self.block1.append(blk)  # 第一个块包含前3个元素
            elif 3 <= i < 9:
                self.block2.append(blk)  # 第二个块包含接下来的6个元素
            elif 9 <= i < 45:
                self.block3.append(blk)  # 第三个块包含接下来的36个元素
            elif 45 <= i:
                self.block4.append(blk)  # 其余元素组成第四个块

        if '1' in self.tuning_stage:
            prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)
        for i, blk in enumerate(self.block1):
            if '1' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt1, 1, i)
            x = blk(x)
        # x = self.norm1(x)
            if i == 1:
                feat = x.permute(0, 3, 1, 2)
                outputs.append(feat)

        if '2' in self.tuning_stage:
            prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, 2)
        for i, blk in enumerate(self.block2):
            if '2' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
            x = blk(x)
        # x = self.norm2(x)
            if i == 4:
                feat = x.permute(0, 3, 1, 2)
                outputs.append(feat)

        if '3' in self.tuning_stage:
            prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, 3)
        for i, blk in enumerate(self.block3):
            if '3' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x,prompt3, 3, i)
            x = blk(x)
        # x = self.norm3(x)
            if i == 34:
                feat = x.permute(0, 3, 1, 2)
                outputs.append(feat)

        if '4' in self.tuning_stage:
            prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, 4)
        for i, blk in enumerate(self.block4):
            if '4' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
            x = blk(x)
        # x = self.norm4(x)
            if i == 2:
                feat = x.permute(0, 3, 1, 2)
                outputs.append(feat)

        return outputs