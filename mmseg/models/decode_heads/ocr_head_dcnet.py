# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from ..utils import resize
from .cascade_decode_head import BaseCascadeDecodeHead

from collections import OrderedDict
from .base_contrast import Encode

class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super().__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super().forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output

@MODELS.register_module()
class OCRHead_DC(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, proj_channels=128, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.proj_channels = proj_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        # >>> project contrast
        self.projector_decode = Encode(self.channels, self.channels, self.proj_channels,
                                                conv_cfg=self.conv_cfg,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=self.act_cfg)
        self.projector_layer3 = Encode(1024, self.channels, self.proj_channels,
                                                conv_cfg=self.conv_cfg,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=self.act_cfg)
        self.de_projector = ConvModule(
                proj_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        # self.cov1 = ConvModule(
        #         self.channels,
        #         self.channels,
        #         1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)

    def forward(self, inputs, prev_output):
        """Forward function."""
        prev_output = prev_output['out']
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        
        output = OrderedDict()
        layer = OrderedDict()
        # >>> project contrast
        temp = self.projector_decode(object_context)
        proj_decode = F.normalize(temp, dim=1)
        proj_layer3 = F.normalize(self.projector_layer3(inputs[2]), dim=1)

        output["decode"] = proj_decode
        layer['layer_3'] = proj_layer3
        output["proj"] = layer

        contrast = self.de_projector(temp)
        # object_context = self.cov1(object_context)
        object_context =  self.relu(object_context + contrast)
        # project contrast <<<

        ocr = self.cls_seg(object_context)
        output["out"] = ocr

        return output
