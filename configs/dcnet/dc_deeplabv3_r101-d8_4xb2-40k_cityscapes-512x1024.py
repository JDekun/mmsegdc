_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
             pretrained='open-mmlab://resnet101_v1c', 
             backbone=dict(depth=101),
             decode_head=dict(
                type='ASPPHeadDC',
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=norm_cfg,
                align_corners=False,
                projector = ['layer_3'],
                proj_channels= 128,
                loss_decode=dict(
                    type='CrossEntropyLossDC', use_sigmoid=False, loss_weight=1.0,
                    layer_weight = [0, 0, 0.1, 0])))