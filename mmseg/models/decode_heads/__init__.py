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
    'FCNHead', 'PSPHead', 'ASPPHead', 'FPNHead', 'OCRHead',
    'PSAHead', 'DepthwiseSeparableASPPHead', 'DepthwiseSeparableFCNHead',
    'OCRHead_DC', 'ASPPHeadDC'
]
