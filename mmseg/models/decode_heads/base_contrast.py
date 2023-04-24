import torch.nn as nn
from mmcv.cnn import ConvModule

class EncodeProjector(nn.Module):
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