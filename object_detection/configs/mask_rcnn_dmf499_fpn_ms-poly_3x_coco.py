_base_ = [
    'common/mstrain-poly_3x_coco_instance.py',
    '_base_/models/mask_rcnn_r50_fpn.py'
]
custom_imports = dict(
    imports=['mmdet.models.backbones.dmf'],
    allow_failed_imports=False)
model = dict(
    backbone=dict(
        type='dmf_499',
        style='pytorch',
        init_cfg=dict(type = "Pretrained", checkpoint='configs/pretrained/checkpoint-441.pth')),
    neck=dict(
        type='FPN',
        in_channels=[40, 72, 176, 240],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)