import math
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .position_encoding import apply_rotary_enc, compute_axial_cis
from .sam2_utils import MLP


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.downsample_rate = downsample_rate
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, num_heads, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, n, num_heads * c)
        return x

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(k.shape[-1])
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(Attention):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        kv_in_dim: int = None,
        num_text_classes: int = 0,
    ) -> None:
        super().__init__(embedding_dim, num_heads, kv_in_dim=kv_in_dim)
        self.num_text_classes = num_text_classes

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        freqs_cis_img: Tensor,
        freqs_cis_txt: Tensor,
        num_k_exclude_rope: int = 0,
    ) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        q_img, q_txt = q[:, :, :, :-self.num_text_classes], q[:, :, :, -self.num_text_classes:]
        k_img, k_txt = k[:, :, :, :-self.num_text_classes], k[:, :, :, -self.num_text_classes:]

        q_img, k_img = apply_rotary_enc(q_img, k_img, freqs_cis_img)
        if freqs_cis_txt is not None:
            q_txt, k_txt = apply_rotary_enc(q_txt, k_txt, freqs_cis_txt)

        q = torch.cat([q_img, q_txt], dim=-2)
        k = torch.cat([k_img, k_txt], dim=-2)

        if num_k_exclude_rope > 0:
            k_img_td = k[:, :, :, :-num_k_exclude_rope]
            k_img_mem = k[:, :, :, -num_k_exclude_rope:]
            attn = torch.matmul(q, k_img_td.transpose(-2, -1))
            attn = attn / math.sqrt(k_img_td.shape[-1])
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v[:, :, :, :-num_k_exclude_rope])
            attn_mem = torch.matmul(q, k_img_mem.transpose(-2, -1))
            attn_mem = attn_mem / math.sqrt(k_img_mem.shape[-1])
            attn_mem = torch.softmax(attn_mem, dim=-1)
            attn_mem = self.dropout(attn_mem)
            out_mem = torch.matmul(attn_mem, v[:, :, :, -num_k_exclude_rope:])
            out = out + out_mem
        else:
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = attn / math.sqrt(k.shape[-1])
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out
