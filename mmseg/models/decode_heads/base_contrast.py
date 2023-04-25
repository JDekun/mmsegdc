import torch.nn as nn
from mmcv.cnn import ConvModule
from collections import OrderedDict

class EncodeProjector(nn.Module):
    def __init__(self, decode_channels, layers, proj_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.layers = layers
        self.decode = nn.Sequential(
            ConvModule(decode_channels, decode_channels, 1,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(decode_channels, proj_channels, 1,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)) 
        self.layers_proj = OrderedDict()
        for layer in self.layers:
            if layer == 'layer_4':
                self.layers_proj['layer_4'] = nn.Sequential(
                    ConvModule(2048, decode_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(decode_channels, proj_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            if layer == 'layer_3':
                self.layers_proj['layer_3'] = nn.Sequential(
                    ConvModule(1024, decode_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(decode_channels, proj_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            if layer == 'layer_2':
                self.layers_proj['layer_2'] = nn.Sequential(
                    ConvModule(512, decode_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(decode_channels, proj_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            if layer == 'layer_1':
                self.layers_proj['layer_1'] = nn.Sequential(
                    ConvModule(256, 256, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    ConvModule(256, proj_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        
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