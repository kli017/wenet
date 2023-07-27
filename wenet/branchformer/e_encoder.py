# Copyright (c) 2022 Yifan Peng (Carnegie Mellon University)
#               2023 Voicecomm Inc (Kai Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Encoder definition."""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from wenet.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from wenet.transformer.embedding import (
    RelPositionalEncoding,
    PositionalEncoding,
    NoPositionalEncoding,
)
from wenet.transformer.subsampling import (
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)

from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.branchformer.e_encoder_layer import EBranchformerEncoderLayer
from wenet.branchformer.cgmlp import ConvolutionalGatingMLP
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask
from wenet.utils.common import get_activation

class EBranchformerEncoder(nn.Module):
    """EBranchformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        attention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        use_cgmlp: bool = True,
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        causal: bool = False,
        use_ffn: bool = False,
        macaron_ffn: bool = False,
        ffn_activation_type: str = "swish",
        positionwise_layer_type: str = "linear",
        linear_units: int = 2048,
        merge_conv_kernel: int = 3,
    ):
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling4(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif attention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        activation = get_activation(ffn_activation_type)
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        else:
            raise ValueError("Support only linear.")
        
        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        self.encoders = torch.nn.ModuleList([EBranchformerEncoderLayer(
            output_size,
            encoder_selfattn_layer(*encoder_selfattn_layer_args),
            cgmlp_layer(*cgmlp_layer_args),
            positionwise_layer(*positionwise_layer_args) if use_ffn else None,
            positionwise_layer(*positionwise_layer_args)
            if use_ffn and macaron_ffn
            else None,
            dropout_rate,
            merge_conv_kernel,
            ) for _ in range(num_blocks)
        ])
        self.after_norm = nn.LayerNorm(output_size)
        self.static_chunk_size = static_chunk_size
        self.global_cmvn = global_cmvn
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
    
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            xs (torch.Tensor): Input tensor (B, T, D).
            ilens (torch.Tensor): Input length (#batch).
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks

        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """

        T = xs.size(1)
        masks = ~make_pad_mask(ilens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        for layer in self.encoders:
            xs, chunk_masks, _  = layer(xs, chunk_masks, pos_emb, mask_pad)

        xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        subsampling_cache: Optional[torch.Tensor] = None,
        elayers_output_cache: Optional[List[torch.Tensor]] = None,
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor],
               List[torch.Tensor]]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            subsampling_cache (Optional[torch.Tensor]): subsampling cache
            elayers_output_cache (Optional[List[torch.Tensor]]):
                transformer/conformer encoder layers output cache
            conformer_cnn_cache (Optional[List[torch.Tensor]]): conformer
                cnn cache

        Returns:
            torch.Tensor: output of current input xs
            torch.Tensor: subsampling cache required for next chunk computation
            List[torch.Tensor]: encoder layers output cache required for next
                chunk computation
            List[torch.Tensor]: conformer cnn cache

        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        if subsampling_cache is not None:
            cache_size = subsampling_cache.size(1)
            xs = torch.cat((subsampling_cache, xs), dim=1)
        else:
            cache_size = 0
        pos_emb = self.embed.position_encoding(offset - cache_size, xs.size(1))
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = xs.size(1)
        else:
            next_cache_start = max(xs.size(1) - required_cache_size, 0)
        r_subsampling_cache = xs[:, next_cache_start:, :]
        # Real mask for transformer/conformer layers
        masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        r_elayers_output_cache = []
        r_conformer_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            if elayers_output_cache is None:
                attn_cache = None
            else:
                attn_cache = elayers_output_cache[i]
            if conformer_cnn_cache is None:
                cnn_cache = None
            else:
                cnn_cache = conformer_cnn_cache[i]
            xs, _, new_cnn_cache = layer(xs,
                                         masks,
                                         pos_emb,
                                         output_cache=attn_cache,
                                         cnn_cache=cnn_cache)
            r_elayers_output_cache.append(xs[:, next_cache_start:, :])
            r_conformer_cnn_cache.append(new_cnn_cache)

        xs = self.after_norm(xs)

        return (xs[:, cache_size:, :], r_subsampling_cache,
                r_elayers_output_cache, r_conformer_cnn_cache)

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        subsampling_cache: Optional[torch.Tensor] = None
        elayers_output_cache: Optional[List[torch.Tensor]] = None
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self.forward_chunk(chunk_xs, offset,
                                                       required_cache_size,
                                                       subsampling_cache,
                                                       elayers_output_cache,
                                                       conformer_cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones(1, ys.size(1), device=ys.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        return ys, masks