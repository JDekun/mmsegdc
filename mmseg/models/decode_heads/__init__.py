# Copyright (c) OpenMMLab. All rights reserved.
from .aspp_head import ASPPHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .ocr_head import OCRHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead

from .ocr_head_dcnet import OCRHead_DC
from .aspp_head_dc import ASPPHeadDC

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'ISAHead', 'STDCHead', 'IterativeDecodeHead',
    'KernelUpdateHead', 'KernelUpdator', 'MaskFormerHead', 'Mask2FormerHead',
    'LightHamHead', 'PIDHead', 'OCRHead_CON', 'OCRHead_DC', 'ASPPHeadDC'
]
