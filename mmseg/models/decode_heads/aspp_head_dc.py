# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

from collections import OrderedDict
# from .base_contrast import EncodeProjector

class Encode(nn.Module):
    def __init__(self, layer_channels, ocr_channels, proj_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.bottleneck = ConvModule(
            layer_channels,
            ocr_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.projector = ConvModule(
            ocr_channels,
            proj_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
    def forward(self, feats):
        feats = self.bottleneck(feats)
        feats = self.projector(feats)

        return feats

class EncodeProjector(nn.Module):
    def __init__(self, decode_channels, layers, proj_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.layers = layers
        self.decode = Encode(decode_channels, decode_channels, proj_channels,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.layers_proj = OrderedDict()
        for layer in self.layers:
            if layer == 'layer_4':
                self.layers_proj['layer_4'] = Encode(2048, decode_channels, proj_channels,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            if layer == 'layer_3':
                self.layers_proj['layer_3'] = Encode(1024, decode_channels, proj_channels,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            if layer == 'layer_2':
                self.layers_proj['layer_2'] = Encode(512, decode_channels, proj_channels,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            if layer == 'layer_1':
                self.layers_proj['layer_1'] = Encode(256, 256, proj_channels,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.de_projector = ConvModule(
                proj_channels,
                decode_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        # self.cov1 = ConvModule(
        #         self.channels,
        #         self.channels,
        #         1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)
        
    def forward(self, decode, inputs):
        proj = OrderedDict()
        decode = self.decode(decode)
        con = self.de_projector(decode)

        for layer in self.layers:
            index = int(layer.split('_')[-1]) - 1
            proj[layer] = self.layers_proj[layer](inputs[index])

        return decode, proj, con

class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@MODELS.register_module()
class ASPPHeadDC(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super().__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        # >>> project contrast
        self.relu = nn.ReLU(inplace=True)
        if self.projector:
            self.project = EncodeProjector(self.channels, self.projector, self.proj_channels,
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg,
                                            act_cfg=self.act_cfg)
        

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        aspp = self._forward_feature(inputs)
            
        decode, proj, con= self.project(aspp, inputs)
        con = self.relu(aspp + con)
        out = self.cls_seg(con)

        output = OrderedDict()
        output['out'] = out
        output['decode'] = decode
        output['proj'] = proj
        return output
