_base_ = [
    '../_base_/models/ocrnet_r50-d8_dc.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=[
        dict(
            type='FCNHead',
            num_classes=150
            ),
        dict(
            type='OCRHead_DC',
            num_classes=150
            )
    ]
    )
