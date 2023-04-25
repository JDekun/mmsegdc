# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

from collections import OrderedDict
from .base_contrast import Encode, Encode_Down
import torch.nn.functional as F
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
        self.projector_decode = Encode(self.channels, self.channels, self.proj_channels,
                                                conv_cfg=self.conv_cfg,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=self.act_cfg)
        for layer in self.projector:
            if layer == "layer_4":
                self.projector_layer4 = Encode(2048, self.channels, self.proj_channels,
                                                        conv_cfg=self.conv_cfg,
                                                        norm_cfg=self.norm_cfg,
                                                        act_cfg=self.act_cfg)
            elif layer == "layer_3":
                self.projector_layer3 = Encode(1024, self.channels, self.proj_channels,
                                                        conv_cfg=self.conv_cfg,
                                                        norm_cfg=self.norm_cfg,
                                                        act_cfg=self.act_cfg)
            elif layer == "layer_2":
                self.projector_layer2 = Encode(512, self.channels, self.proj_channels,
                                                        conv_cfg=self.conv_cfg,
                                                        norm_cfg=self.norm_cfg,
                                                        act_cfg=self.act_cfg)
            elif layer == "layer_1":
                self.projector_layer1 = Encode_Down(256, 256, self.proj_channels,
                                                        conv_cfg=self.conv_cfg,
                                                        norm_cfg=self.norm_cfg,
                                                        act_cfg=self.act_cfg)
                
        self.de_projector = ConvModule(
                self.proj_channels,
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

        output = OrderedDict()
        layer = OrderedDict()
        # >>> project contrast
        temp = self.projector_decode(aspp)
        proj_decode = F.normalize(temp, dim=1)
        output["decode"] = proj_decode
        for lay in self.projector:
            if lay == 'layer_4':
                proj_layer4 = F.normalize(self.projector_layer4(inputs[3]), dim=1)
                layer['layer_4'] = proj_layer4
            elif lay == 'layer_3':
                proj_layer3 = F.normalize(self.projector_layer3(inputs[2]), dim=1)
                layer['layer_3'] = proj_layer3
            elif lay == 'layer_2':
                proj_layer2 = F.normalize(self.projector_layer2(inputs[1]), dim=1)
                layer['layer_2'] = proj_layer2
            elif lay == 'layer_1':
                proj_layer1 = F.normalize(self.projector_layer1(inputs[0]), dim=1)
                layer['layer_1'] = proj_layer1

        
        output["proj"] = layer

        contrast = self.de_projector(temp)
        # object_context = self.cov1(object_context)
        aspp =  self.relu(aspp + contrast)
        # project contrast <<<
        out = self.cls_seg(aspp)
        
        output['out'] = out
        
        return output
